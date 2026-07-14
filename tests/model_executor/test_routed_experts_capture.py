# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import types
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm.distributed.eplb.eplb_state import EplbLayerState
from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
    RoutedExpertsCapturer,
)
from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter

pytestmark = pytest.mark.cpu_test

_REC_MODULE = "vllm.model_executor.layers.fused_moe.routed_experts_capturer"


def _capturer_with_buffer(
    *,
    max_tokens: int = 8,
    num_layers: int = 4,
    num_experts_per_tok: int = 2,
    dp_rank: int = 0,
    tp_size: int = 1,
    with_scores: bool = False,
) -> RoutedExpertsCapturer:
    # Bypass __init__ so the test can use a CPU buffer and skip the
    # VllmConfig dependency. The CUDA device-tensor allocation in the
    # real constructor is not what we are exercising here.
    c = RoutedExpertsCapturer.__new__(RoutedExpertsCapturer)
    c.dp_rank = dp_rank
    c.tp_size = tp_size
    c.device_buffer = torch.full(
        (max_tokens, num_layers, num_experts_per_tok),
        -1,
        dtype=torch.int32,
    )
    c.capture_scores = with_scores
    c.score_device_buffer = (
        torch.full(
            (max_tokens, num_layers, num_experts_per_tok),
            -1.0,
            dtype=torch.float16,
        )
        if with_scores
        else None
    )
    return c


class DummyRouter(BaseRouter):
    @property
    def routing_method_type(self) -> RoutingMethodType:
        return RoutingMethodType.FUSED_TOPK

    def _compute_routing(
        self, hidden_states, router_logits, indices_type, *, input_ids=None
    ):
        topk_ids = torch.tensor([[1, 2], [3, 4]], dtype=torch.int64)
        topk_weights = torch.tensor([[0.75, 0.25], [0.5, 0.5]], dtype=torch.float32)
        return topk_weights, topk_ids

    def _apply_eplb_mapping(self, topk_ids: torch.Tensor) -> torch.Tensor:
        # Make mapping observable without requiring CUDA EPLB path.
        return topk_ids + 10


def _make_router(eplb_state: EplbLayerState | None = None) -> DummyRouter:
    return DummyRouter(
        top_k=2,
        global_num_experts=16,
        eplb_state=eplb_state,
        indices_type_getter=None,
    )


def test_base_router_capture_pre_eplb_mapping():
    router = _make_router()
    captured = []

    def capture_fn(ids, weights):
        captured.append((ids.clone(), weights.clone()))

    router.set_capture_fn(capture_fn)
    topk_weights, topk_ids = router.select_experts(
        hidden_states=torch.empty(1),
        router_logits=torch.empty(1),
    )

    assert topk_weights.shape == topk_ids.shape
    assert len(captured) == 1
    cap_ids, cap_weights = captured[0]
    assert torch.equal(cap_ids, torch.tensor([[1, 2], [3, 4]]))
    # Weights captured alongside the ids, row-aligned.
    assert torch.equal(
        cap_weights, torch.tensor([[0.75, 0.25], [0.5, 0.5]], dtype=torch.float32)
    )
    assert torch.equal(topk_ids, torch.tensor([[11, 12], [13, 14]]))


def test_base_router_capture_with_eplb_enabled():
    eplb_state = EplbLayerState()
    eplb_state.expert_load_view = torch.zeros(32, dtype=torch.int64)
    eplb_state.logical_to_physical_map = torch.arange(32).view(32, 1)
    eplb_state.logical_replica_count = torch.ones(32, dtype=torch.int64)
    eplb_state.should_record_tensor = torch.ones((), dtype=torch.bool)
    router = _make_router(eplb_state=eplb_state)

    captured = []

    def capture_fn(ids, weights):
        captured.append(ids.clone())

    router.set_capture_fn(capture_fn)
    _, topk_ids = router.select_experts(
        hidden_states=torch.empty(1),
        router_logits=torch.empty(1),
    )

    assert len(captured) == 1
    # Capture should see logical ids pre-EPLB mapping.
    assert torch.equal(captured[0], torch.tensor([[1, 2], [3, 4]]))
    # Our DummyRouter mapping adds +10.
    assert torch.equal(topk_ids, torch.tensor([[11, 12], [13, 14]]))


def test_gpu_model_runner_binds_router_capture(monkeypatch):
    from vllm.v1.worker import gpu_model_runner as gmr

    class DummyFusedMoE:
        def __init__(self):
            self.layer_id = 7
            self.router = _make_router()

    class DummyCapturer:
        def __init__(self):
            self.calls = []

        def capture(self, layer_id, topk_ids, topk_weights=None):
            self.calls.append((layer_id, topk_ids, topk_weights))

    dummy_module = DummyFusedMoE()

    # Patch the runtime import inside _bind_routed_experts_capturer.
    import vllm.model_executor.layers.fused_moe.layer as fused_moe_layer

    monkeypatch.setattr(fused_moe_layer, "FusedMoE", DummyFusedMoE)

    dummy_self = types.SimpleNamespace(
        compilation_config=types.SimpleNamespace(
            static_forward_context={"dummy": dummy_module}
        )
    )

    capturer = DummyCapturer()
    gmr.GPUModelRunner._bind_routed_experts_capturer(dummy_self, capturer)

    assert dummy_module.router.capture_fn is not None
    dummy_module.router.capture_fn(
        torch.tensor([[5, 6]]), torch.tensor([[0.6, 0.4]])
    )

    assert len(capturer.calls) == 1
    layer_id, topk_ids, topk_weights = capturer.calls[0]
    assert layer_id == 7
    assert torch.equal(topk_ids, torch.tensor([[5, 6]]))
    assert torch.equal(topk_weights, torch.tensor([[0.6, 0.4]]))


def test_gpu_model_runner_binding_stage(monkeypatch):
    from vllm.v1.worker import gpu_model_runner as gmr

    class DummyFusedMoE:
        def __init__(self):
            self.layer_id = 11
            self.router = _make_router()

    class DummyCapturer:
        def __init__(self):
            self.calls = []

        def capture(self, layer_id, topk_ids, topk_weights=None):
            self.calls.append((layer_id, topk_ids, topk_weights))

    dummy_module = DummyFusedMoE()

    import vllm.model_executor.layers.fused_moe.layer as fused_moe_layer

    monkeypatch.setattr(fused_moe_layer, "FusedMoE", DummyFusedMoE)

    dummy_self = types.SimpleNamespace(
        compilation_config=types.SimpleNamespace(
            static_forward_context={"dummy": dummy_module}
        )
    )

    # Before binding, no capture hook.
    assert dummy_module.router.capture_fn is None

    capturer = DummyCapturer()
    gmr.GPUModelRunner._bind_routed_experts_capturer(dummy_self, capturer)

    # After binding, hook should exist and be callable.
    assert callable(dummy_module.router.capture_fn)
    dummy_module.router.capture_fn(torch.tensor([[9, 10]]), torch.tensor([[0.9, 0.1]]))
    assert len(capturer.calls) == 1


def test_routed_experts_capturer_single_dp_no_metadata():
    """dp_metadata is None: capture writes the full topk_ids rows."""
    capturer = _capturer_with_buffer(dp_rank=0)
    topk = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.int32)
    ctx = SimpleNamespace(dp_metadata=None)
    with patch(f"{_REC_MODULE}.get_forward_context", return_value=ctx):
        capturer.capture(layer_id=0, topk_ids=topk)
    assert torch.equal(capturer.device_buffer[:3, 0, :], topk)
    assert capturer.device_buffer[3, 0, 0].item() == -1


def test_routed_experts_capturer_dp_naive_concatenated_all_ranks():
    """n == sum(num_tokens_dp): slice this rank's segment from concatenated topk."""
    capturer = _capturer_with_buffer(dp_rank=1)
    num_tokens_dp = torch.tensor([2, 3], dtype=torch.int32)
    ctx = SimpleNamespace(
        dp_metadata=SimpleNamespace(num_tokens_across_dp_cpu=num_tokens_dp)
    )
    # Concatenated order: rank0 rows then rank1 rows.
    topk = torch.tensor(
        [[0, 1], [2, 3], [10, 11], [12, 13], [14, 15]], dtype=torch.int32
    )
    with patch(f"{_REC_MODULE}.get_forward_context", return_value=ctx):
        capturer.capture(layer_id=0, topk_ids=topk)
    want = topk[2:5]
    assert torch.equal(capturer.device_buffer[:3, 0, :], want)


def test_routed_experts_capturer_dp_modular_local_tokens():
    """n == token_num_per_dp: topk is already local to this DP rank."""
    capturer = _capturer_with_buffer(dp_rank=1)
    num_tokens_dp = torch.tensor([2, 3], dtype=torch.int32)
    ctx = SimpleNamespace(
        dp_metadata=SimpleNamespace(num_tokens_across_dp_cpu=num_tokens_dp)
    )
    topk = torch.tensor([[10, 11], [12, 13], [14, 15]], dtype=torch.int32)
    with patch(f"{_REC_MODULE}.get_forward_context", return_value=ctx):
        capturer.capture(layer_id=0, topk_ids=topk)
    assert torch.equal(capturer.device_buffer[:3, 0, :], topk)


def test_routed_experts_capturer_dp_unexpected_batch_raises():
    """Mismatch between topk batch dim and DP layout: fail fast."""
    capturer = _capturer_with_buffer(dp_rank=0)
    num_tokens_dp = torch.tensor([2, 3], dtype=torch.int32)
    ctx = SimpleNamespace(
        dp_metadata=SimpleNamespace(num_tokens_across_dp_cpu=num_tokens_dp)
    )
    # total=5, local=2: n=1 matches neither naive (5) nor modular (2).
    topk = torch.tensor([[1, 2]], dtype=torch.int32)
    with (
        patch(f"{_REC_MODULE}.get_forward_context", return_value=ctx),
        pytest.raises(AssertionError, match="unexpected topk_ids batch dim"),
    ):
        capturer.capture(layer_id=0, topk_ids=topk)
    assert capturer.device_buffer[0, 0, 0].item() == -1


def test_routed_experts_capturer_scores_single_dp():
    """Score buffer receives the same rows as the id buffer."""
    capturer = _capturer_with_buffer(dp_rank=0, with_scores=True)
    topk = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.int32)
    weights = torch.tensor(
        [[0.9, 0.1], [0.7, 0.3], [0.5, 0.5]], dtype=torch.float32
    )
    ctx = SimpleNamespace(dp_metadata=None)
    with patch(f"{_REC_MODULE}.get_forward_context", return_value=ctx):
        capturer.capture(layer_id=1, topk_ids=topk, topk_weights=weights)
    assert torch.equal(capturer.device_buffer[:3, 1, :], topk)
    assert capturer.score_device_buffer is not None
    assert torch.equal(
        capturer.score_device_buffer[:3, 1, :], weights.to(torch.float16)
    )
    # Untouched rows stay at the sentinel value.
    assert capturer.score_device_buffer[3, 1, 0].item() == -1.0


def test_routed_experts_capturer_scores_dp_naive_slicing():
    """Scores are sliced with the same DP span as the ids."""
    capturer = _capturer_with_buffer(dp_rank=1, with_scores=True)
    num_tokens_dp = torch.tensor([2, 3], dtype=torch.int32)
    ctx = SimpleNamespace(
        dp_metadata=SimpleNamespace(num_tokens_across_dp_cpu=num_tokens_dp)
    )
    topk = torch.tensor(
        [[0, 1], [2, 3], [10, 11], [12, 13], [14, 15]], dtype=torch.int32
    )
    weights = torch.arange(10, dtype=torch.float32).reshape(5, 2) / 10.0
    with patch(f"{_REC_MODULE}.get_forward_context", return_value=ctx):
        capturer.capture(layer_id=0, topk_ids=topk, topk_weights=weights)
    assert torch.equal(capturer.device_buffer[:3, 0, :], topk[2:5])
    assert capturer.score_device_buffer is not None
    assert torch.equal(
        capturer.score_device_buffer[:3, 0, :], weights[2:5].to(torch.float16)
    )


def test_routed_experts_capturer_scores_disabled_ignores_weights():
    """When score capture is off, passing weights is a no-op."""
    capturer = _capturer_with_buffer(dp_rank=0, with_scores=False)
    topk = torch.tensor([[1, 2]], dtype=torch.int32)
    weights = torch.tensor([[0.6, 0.4]], dtype=torch.float32)
    ctx = SimpleNamespace(dp_metadata=None)
    with patch(f"{_REC_MODULE}.get_forward_context", return_value=ctx):
        capturer.capture(layer_id=0, topk_ids=topk, topk_weights=weights)
    assert capturer.score_device_buffer is None
    assert torch.equal(capturer.device_buffer[:1, 0, :], topk)


def test_routed_experts_capturer_scores_missing_weights_ok():
    """Score capture enabled but router gave no weights: ids still land."""
    capturer = _capturer_with_buffer(dp_rank=0, with_scores=True)
    topk = torch.tensor([[1, 2]], dtype=torch.int32)
    ctx = SimpleNamespace(dp_metadata=None)
    with patch(f"{_REC_MODULE}.get_forward_context", return_value=ctx):
        capturer.capture(layer_id=0, topk_ids=topk)
    assert torch.equal(capturer.device_buffer[:1, 0, :], topk)
    # Score rows untouched (sentinel), no crash.
    assert capturer.score_device_buffer is not None
    assert capturer.score_device_buffer[0, 0, 0].item() == -1.0


def test_routed_experts_manager_store_and_get_scores():
    """store_batch/get_scores round-trip through the slot buffer."""
    import numpy as np

    from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
        RoutedExpertsManager,
    )

    mgr = RoutedExpertsManager.__new__(RoutedExpertsManager)
    mgr.block_size = 4
    num_slots, num_layers, top_k = 16, 2, 2
    mgr.routed_experts_by_slot = np.zeros(
        (num_slots, num_layers, top_k), dtype=np.uint8
    )
    mgr.capture_scores = True
    mgr.routed_expert_scores_by_slot = np.zeros(
        (num_slots, num_layers, top_k), dtype=np.float16
    )

    data = np.arange(3 * num_layers * top_k, dtype=np.uint8).reshape(
        3, num_layers, top_k
    )
    scores = (
        np.arange(3 * num_layers * top_k, dtype=np.float16).reshape(
            3, num_layers, top_k
        )
        / 16.0
    )
    # Block 2 -> slots 8,9,10.
    slot_mapping = np.array([8, 9, 10])
    mgr.store_batch(data, slot_mapping, scores=scores)

    got_ids = mgr.get([2], num_tokens=3)
    got_scores = mgr.get_scores([2], num_tokens=3)
    assert np.array_equal(got_ids, data)
    assert got_scores is not None
    assert np.array_equal(got_scores, scores)
    # token_start slicing aligned between ids and scores.
    got_scores_tail = mgr.get_scores([2], num_tokens=3, token_start=1)
    assert np.array_equal(got_scores_tail, scores[1:])


def test_routed_experts_manager_get_scores_disabled_returns_none():
    import numpy as np

    from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
        RoutedExpertsManager,
    )

    mgr = RoutedExpertsManager.__new__(RoutedExpertsManager)
    mgr.block_size = 4
    mgr.routed_experts_by_slot = np.zeros((8, 1, 1), dtype=np.uint8)
    mgr.capture_scores = False
    mgr.routed_expert_scores_by_slot = None

    # store_batch with scores=None must not raise.
    mgr.store_batch(
        np.ones((2, 1, 1), dtype=np.uint8), np.array([0, 1]), scores=None
    )
    assert mgr.get_scores([0], num_tokens=2) is None
