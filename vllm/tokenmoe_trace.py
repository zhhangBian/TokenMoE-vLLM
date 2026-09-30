# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only, opt-in engine trace recorder (raw format version 1)."""

from __future__ import annotations

import hashlib
import json
import os
import queue
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import regex as re


def content_id(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(data.encode()).hexdigest()[:16]


def validate_config(config: Any) -> None:
    """Reject configurations whose execution cannot be attributed exactly."""
    checks = [
        (
            config.model_config.enable_return_routed_experts,
            "enable_return_routed_experts: use --enable-return-routed-experts",
        ),
        (
            config.cache_config.enable_prefix_caching,
            "enable_prefix_caching: use --enable-prefix-caching",
        ),
        (config.speculative_config is None, "speculative_config must be unset"),
        (
            not config.scheduler_config.async_scheduling,
            "async_scheduling: use --no-async-scheduling",
        ),
        (
            config.parallel_config.data_parallel_size == 1,
            "data_parallel_size must be 1",
        ),
    ]
    for valid, message in checks:
        if not valid:
            raise ValueError(f"TokenMoE trace: {message}")


def prepare_engine(config: Any) -> Path | None:
    """Validate before loading weights and give workers this engine's directory."""
    root = os.environ.get("TOKENMOE_TRACE_DIR")
    if not root:
        return None
    validate_config(config)
    engine_id = f"eng_{time.time_ns():016x}{uuid.uuid4().hex[:16]}"
    path = Path(root).expanduser().resolve() / engine_id
    os.environ["TOKENMOE_TRACE_ENGINE_DIR"] = str(path)
    return path


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def write_layer_map(layers: list[dict[str, Any]]) -> None:
    """Write a worker's map; the caller selects TP rank zero."""
    if not os.environ.get("TOKENMOE_TRACE_DIR"):
        return
    directory = Path(os.environ["TOKENMOE_TRACE_ENGINE_DIR"])
    error = []

    def write() -> None:
        try:
            _atomic_json(
                directory / "layer_map.json",
                sorted(layers, key=lambda x: x["layer_id"]),
            )
        except BaseException as exc:
            error.append(exc)

    thread = threading.Thread(target=write, name="tokenmoe-layer-map")
    thread.start()
    thread.join()
    if error:
        raise RuntimeError("TokenMoE layer map write failed") from error[0]


def engine_metadata(config: Any, path: Path, block_size: int, version: str) -> dict:
    model, cache = config.model_config, config.cache_config
    parallel, sched = config.parallel_config, config.scheduler_config
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    effective = {
        "vllm_commit": commit,
        "model": model.model,
        "revision": model.revision,
        "tensor_parallel": parallel.tensor_parallel_size,
        "expert_parallel": parallel.enable_expert_parallel,
        "expert_parallel_size": (
            parallel.tensor_parallel_size * parallel.data_parallel_size
            if parallel.enable_expert_parallel
            else 1
        ),
        "data_parallel": parallel.data_parallel_size,
        "dtype": str(model.dtype),
        "quantization": model.quantization,
        "max_model_len": model.max_model_len,
        "gpu_memory_utilization": cache.gpu_memory_utilization,
        "max_num_seqs": sched.max_num_seqs,
        "max_num_batched_tokens": sched.max_num_batched_tokens,
        "block_size": block_size,
        "enable_prefix_caching": cache.enable_prefix_caching,
        "async_scheduling": sched.async_scheduling,
        "model_runner_version": 2 if config.use_v2_model_runner else 1,
        "scheduling_policy": sched.policy,
        "capture_routed_experts": model.enable_return_routed_experts,
        "speculative_decoding": False,
    }
    return {
        "raw_format_version": 1,
        "engine_instance_id": path.name,
        "vllm_version": version,
        "vllm_commit": commit,
        "engine_config": effective,
        "engine_config_id": content_id(effective),
        "clock_domain_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "monotonic_anchor_ns": time.monotonic_ns(),
        "wall_anchor_ns": time.time_ns(),
        "pid": os.getpid(),
    }


def classify_range(
    start: int, end: int, hwm: int, prompt: int
) -> list[tuple[str, int, int]]:
    """Split one scheduled interval at recompute and prompt boundaries."""
    result = []
    if start < min(end, hwm):
        result.append(("recompute", start, min(end, hwm)))
        start = min(end, hwm)
    if start < min(end, prompt):
        result.append(("new_prefill", start, min(end, prompt)))
        start = min(end, prompt)
    if start < end:
        result.append(("decode", start, end))
    return result


def classify_coverage(
    start: int, end: int, computed: list[tuple[int, int]], prompt: int
) -> list[tuple[str, int, int]]:
    """Classify actual coverage, including holes from later prefix-cache hits."""
    boundaries = {start, end}
    boundaries.update(x for pair in computed for x in pair if start < x < end)
    if start < prompt < end:
        boundaries.add(prompt)
    points = sorted(boundaries)
    result = []
    for a, b in zip(points, points[1:]):
        covered = any(left <= a < right for left, right in computed)
        phase = "recompute" if covered else "new_prefill" if a < prompt else "decode"
        if result and result[-1][0] == phase:
            result[-1] = (phase, result[-1][1], b)
        else:
            result.append((phase, a, b))
    return result


@dataclass
class RequestTrace:
    llm_request_id: str | None
    vllm_request_id: str
    prompt_length: int
    seed: int | None
    received: int
    row_start: int | None = None
    row_end: int = 0
    hwm: int = 0
    computed: list[tuple[int, int]] = field(default_factory=list)
    first_scheduled: int | None = None
    first_token: int | None = None
    preemptions: int = 0
    chunks: list[tuple[int, np.ndarray, int]] = field(default_factory=list)
    finish: dict | None = None
    token_ids: list[int] = field(default_factory=list)


class TraceRecorder:
    """Accumulate step slices in the engine, serialize on one writer thread."""

    def __init__(self, directory: Path, metadata: dict, num_experts: int, top_k: int):
        self.directory = Path(directory)
        self.metadata = metadata
        self.dtype = np.uint8 if num_experts <= 256 else np.uint16
        self.top_k = top_k
        self.requests: dict[str, RequestTrace] = {}
        self.used_ids: set[str] = set()
        self.pending: dict[str, tuple[int, int, list]] = {}
        self.step_index = 0
        self.step: dict | None = None
        self.captured = False
        self.output_time: int | None = None
        self.layer_ids: list[int] = []
        self._queue: queue.Queue = queue.Queue()
        self._error: BaseException | None = None
        self._ready = threading.Event()
        self._closed = False
        self._thread = threading.Thread(
            target=self._write_loop, name="tokenmoe-writer", daemon=True
        )
        self._thread.start()
        self._ready.wait()
        self._check_writer()

    @classmethod
    def from_config(cls, config: Any, block_size: int) -> TraceRecorder | None:
        if not os.environ.get("TOKENMOE_TRACE_DIR"):
            return None
        from vllm.version import __version__

        path = Path(os.environ["TOKENMOE_TRACE_ENGINE_DIR"])
        return cls(
            path,
            engine_metadata(config, path, block_size, __version__),
            config.model_config.get_num_experts(),
            config.model_config.get_num_experts_per_tok(),
        )

    def _check_writer(self) -> None:
        if self._error is not None:
            raise RuntimeError("TokenMoE trace writer failed") from self._error

    def _write_loop(self) -> None:
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            layers = json.loads((self.directory / "layer_map.json").read_text())
            self.layer_ids = [entry["layer_id"] for entry in layers]
            if not self.layer_ids or self.layer_ids != sorted(set(self.layer_ids)):
                raise ValueError(
                    "TokenMoE layer_map must contain unique sorted bound layers"
                )
            (self.directory / "routing").mkdir()
            _atomic_json(self.directory / "engine_meta.json", self.metadata)
            self._ready.set()
            with (
                (self.directory / "steps.jsonl").open("x") as steps,
                (self.directory / "requests.jsonl").open("x") as requests,
            ):
                while (item := self._queue.get()) is not None:
                    kind, value = item
                    if kind == "step":
                        steps.write(json.dumps(value) + "\n")
                        steps.flush()
                    else:
                        record, state = value
                        if record["routing_file"]:
                            if state.chunks:
                                chunks = sorted(state.chunks, key=lambda x: x[0])
                                experts = np.concatenate(
                                    [chunk for _, chunk, _ in chunks]
                                )
                                indices = np.concatenate(
                                    [
                                        np.full(len(chunk), step, dtype=np.int32)
                                        for _, chunk, step in chunks
                                    ]
                                )
                                positions = np.concatenate(
                                    [
                                        np.arange(a, a + len(chunk), dtype=np.int32)
                                        for a, chunk, _ in chunks
                                    ]
                                )
                            else:
                                experts = np.empty(
                                    (0, len(self.layer_ids), self.top_k),
                                    dtype=self.dtype,
                                )
                                indices = np.empty(0, dtype=np.int32)
                                positions = np.empty(0, dtype=np.int32)
                            target = self.directory / record["routing_file"]
                            with target.with_suffix(".tmp").open("wb") as out:
                                np.savez(
                                    out,
                                    token_ids=np.asarray(
                                        state.token_ids, dtype=np.int32
                                    ),
                                    row_start=record["row_start"],
                                    row_end=record["row_end"],
                                    routing_complete=record["routing_complete"],
                                    experts=experts,
                                    token_positions=positions,
                                    step_index=indices,
                                    layer_ids=np.asarray(
                                        self.layer_ids, dtype=np.int32
                                    ),
                                )
                            target.with_suffix(".tmp").replace(target)
                        requests.write(json.dumps(record) + "\n")
                        requests.flush()
        except BaseException as exc:
            self._error = exc
        finally:
            self._ready.set()

    def add_request(self, request: Any) -> None:
        self._check_writer()
        params = request.sampling_params
        extra = (params.extra_args or {}) if params else {}
        identifier = extra.get("tokenmoe_llm_request_id")
        if identifier is not None:
            if not isinstance(identifier, str) or not re.fullmatch(
                r"req_[A-Za-z0-9_-]+", identifier
            ):
                raise ValueError(
                    "tokenmoe_llm_request_id must be a safe req_ identifier"
                )
            if identifier in self.used_ids:
                raise ValueError("duplicate tokenmoe_llm_request_id")
            self.used_ids.add(identifier)
        self.requests[request.request_id] = RequestTrace(
            identifier,
            request.request_id,
            request.num_prompt_tokens,
            params.seed if params else None,
            time.monotonic_ns(),
        )

    def begin_step(self, timestamp: int) -> None:
        self._check_writer()
        if self.pending or self.step is not None:
            raise RuntimeError("TokenMoE trace requires synchronous engine steps")
        self.captured = False
        self.output_time = None
        self.step = {"step_index": self.step_index, "t_sched": timestamp, "entries": []}

    def schedule(self, request: Any, count: int) -> None:
        state = self.requests[request.request_id]
        start, end = request.num_computed_tokens, request.num_computed_tokens + count
        if state.row_start is None:
            state.row_start = state.row_end = state.hwm = start
            state.computed = [(0, start)] if start else []
        phases = classify_coverage(start, end, state.computed, state.prompt_length)
        self.pending[request.request_id] = (start, end, phases)
        assert self.step is not None
        self.step["entries"].extend(
            [
                [state.llm_request_id, state.vllm_request_id, phase, a, b]
                for phase, a, b in phases
            ]
        )
        state.hwm = max(state.hwm, end)
        merged: list[tuple[int, int]] = []
        for a, b in sorted([*state.computed, (start, end)]):
            if merged and a <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], b))
            else:
                merged.append((a, b))
        state.computed = merged

    def dispatch(
        self, timestamp: int, running: int, waiting: int, usage: float
    ) -> None:
        assert self.step is not None
        self.step.update(
            step_started_at=timestamp,
            num_running=running,
            num_waiting=waiting,
            kv_cache_usage=usage,
            num_tokens_total=sum(e - s for s, e, _ in self.pending.values()),
        )
        for rid in self.pending:
            if self.requests[rid].first_scheduled is None:
                self.requests[rid].first_scheduled = timestamp

    def output_ready(self, timestamp: int) -> None:
        self.output_time = timestamp
        assert self.step is not None
        self.step["step_finished_at"] = timestamp

    def capture_step(self, routing: np.ndarray | None, offsets: dict[str, int]) -> None:
        if self.pending and routing is None:
            raise RuntimeError("TokenMoE trace: model output has no routing")
        for rid, (start, end, phases) in self.pending.items():
            state = self.requests[rid]
            for phase, a, b in phases:
                if phase == "recompute":
                    continue
                if state.llm_request_id:
                    assert routing is not None
                    offset = offsets[rid]
                    chunk = routing[
                        offset + a - start : offset + b - start, self.layer_ids, :
                    ]
                    if chunk.shape != (b - a, len(self.layer_ids), self.top_k):
                        raise RuntimeError(
                            "TokenMoE trace: missing routing rows or layers"
                        )
                    state.chunks.append(
                        (a, chunk.astype(self.dtype, copy=True), self.step_index)
                    )
                state.row_end = max(state.row_end, b)
        self.captured = True
        for rid in list(self.pending):
            if self.requests[rid].finish is not None:
                self._emit_finish(rid)

    def first_token(self, request: Any) -> None:
        state = self.requests[request.request_id]
        if state.first_token is None:
            state.first_token = self.output_time

    def preempt(self, request: Any) -> None:
        self.requests[request.request_id].preemptions += 1

    def finish_request(self, request: Any) -> None:
        state = self.requests[request.request_id]
        status = str(request.status)
        normal = status in {
            "FINISHED_STOPPED",
            "FINISHED_LENGTH_CAPPED",
            "FINISHED_REPETITION",
        }
        timestamp = (
            self.output_time
            if normal and self.output_time is not None
            else time.monotonic_ns()
        )
        state.token_ids = list(request.all_token_ids)
        state.finish = {
            "engine_finish_status": status,
            "inference_finished_at": timestamp,
            "num_output_tokens": request.num_output_tokens,
            "normal_finish": normal,
        }
        if request.request_id not in self.pending or self.captured:
            self._emit_finish(request.request_id)

    def _emit_finish(self, rid: str) -> None:
        state = self.requests.pop(rid)
        assert state.finish is not None
        finish = dict(state.finish)
        normal = finish.pop("normal_finish")
        complete = normal and state.row_end == len(state.token_ids) - 1
        record = {
            "llm_request_id": state.llm_request_id,
            "vllm_request_id": rid,
            "engine_instance_id": self.metadata["engine_instance_id"],
            "engine_received_at": state.received,
            "first_scheduled_at": state.first_scheduled,
            "first_token_at": state.first_token,
            "num_prompt_tokens": state.prompt_length,
            "num_cached_tokens": state.row_start or 0,
            "num_preemptions": state.preemptions,
            "sampling_seed": state.seed,
            "row_start": state.row_start or 0,
            "row_end": state.row_end,
            "routing_complete": complete,
            "routing_file": f"routing/{state.llm_request_id}.npz"
            if state.llm_request_id
            else None,
            **finish,
        }
        self._queue.put(("request", (record, state)))

    def end_step(self, timestamp: int) -> None:
        assert self.step is not None
        self.step["t_end"] = timestamp
        if self.pending:
            self._queue.put(("step", self.step))
            self.step_index += 1
        self.step = None
        self.pending = {}
        self.output_time = None
        self._check_writer()

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._queue.put(None)
            self._thread.join()
        self._check_writer()
