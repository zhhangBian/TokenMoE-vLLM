# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU behavior tests: real recorder files, with small synthetic engine requests.

Only the recorder is loaded, so these also run before installing the fork wheel:
python -m unittest discover -s tests/tokenmoe
"""

import ast
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np

FORK = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "tokenmoe_trace_cpu", FORK / "vllm/tokenmoe_trace.py"
)
trace = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = trace
spec.loader.exec_module(trace)


def request(name="a", prompt=6, cached=0, tagged=True):
    return NS(
        request_id=name,
        num_prompt_tokens=prompt,
        sampling_params=NS(
            seed=7,
            extra_args={"tokenmoe_llm_request_id": f"req_{name}"} if tagged else {},
        ),
        num_computed_tokens=cached,
        all_token_ids=list(range(prompt)),
        num_output_tokens=0,
        status="FINISHED_STOPPED",
    )


class RecorderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name)
        (self.path / "layer_map.json").write_text(
            json.dumps(
                [
                    {"layer_id": 1, "capture_path": "router"},
                    {"layer_id": 3, "capture_path": "monolithic"},
                ]
            )
        )
        self.recorder = None

    def tearDown(self):
        if self.recorder is not None:
            self.recorder.close()
        self.tmp.cleanup()

    def create(self, experts=128):
        self.recorder = trace.TraceRecorder(
            self.path, {"engine_instance_id": "eng_test"}, experts, 2
        )
        return self.recorder

    def step(self, requests, order=None, fill=1, abort=None):
        recorder = self.recorder
        recorder.begin_step(10 + recorder.step_index * 10)
        for req, count in requests:
            recorder.schedule(req, count)
        recorder.dispatch(11 + recorder.step_index * 10, len(requests), 0, 0.25)
        recorder.output_ready(12 + recorder.step_index * 10)
        order = order or [req.request_id for req, _ in requests]
        counts = {req.request_id: count for req, count in requests}
        offsets, chunks, offset = {}, [], 0
        for i, rid in enumerate(order):
            offsets[rid] = offset
            offset += counts[rid]
            chunks.append(np.full((counts[rid], 4, 2), fill + i, dtype=np.int32))
        if abort:
            abort.status = "FINISHED_ABORTED"
            recorder.finish_request(abort)
        recorder.capture_step(np.concatenate(chunks), offsets)
        for req, count in requests:
            req.num_computed_tokens += count
        recorder.end_step(13 + recorder.step_index * 10)

    def test_phase_boundaries_split_without_losing_tokens(self):
        self.assertEqual(
            trace.classify_range(0, 9, 3, 6),
            [("recompute", 0, 3), ("new_prefill", 3, 6), ("decode", 6, 9)],
        )
        self.assertEqual(
            trace.classify_range(5, 10, 8, 6), [("recompute", 5, 8), ("decode", 8, 10)]
        )

    def test_chunked_prefill_keeps_first_rows_across_preemption(self):
        r = self.create()
        req = request(cached=2)
        r.add_request(req)
        self.step([(req, 2)], fill=4)
        r.preempt(req)
        req.num_computed_tokens = 0
        self.step([(req, 6)], fill=9)
        req.all_token_ids.append(99)
        req.num_output_tokens = 1
        r.finish_request(req)
        r.close()
        with np.load(self.path / "routing/req_a.npz") as data:
            self.assertEqual(data["experts"].shape, (4, 2, 2))
            self.assertEqual(data["experts"].dtype, np.uint8)
            np.testing.assert_array_equal(data["experts"][:, 0, 0], [4, 4, 9, 9])
            np.testing.assert_array_equal(data["step_index"], [0, 0, 1, 1])
            np.testing.assert_array_equal(data["layer_ids"], [1, 3])
            self.assertEqual(data["row_start"], 2)
            self.assertEqual(data["row_end"], 6)
            self.assertTrue(data["routing_complete"])
        record = json.loads((self.path / "requests.jsonl").read_text())
        self.assertEqual(record["num_cached_tokens"], 2)
        self.assertEqual(record["num_preemptions"], 1)
        steps = [
            json.loads(line)
            for line in (self.path / "steps.jsonl").read_text().splitlines()
        ]
        self.assertEqual(steps[1]["entries"][0][2:], ["recompute", 0, 4])
        for step in steps:
            self.assertEqual(
                sum(entry[4] - entry[3] for entry in step["entries"]),
                step["num_tokens_total"],
            )

    def test_model_runner_order_controls_routing_slices(self):
        r = self.create(experts=300)
        a, b = request("a", prompt=2), request("b", prompt=3)
        for req in [a, b]:
            r.add_request(req)
        self.step([(a, 2), (b, 3)], order=["b", "a"], fill=257)
        for req in [a, b]:
            req.all_token_ids.append(42)
            req.num_output_tokens = 1
            r.finish_request(req)
        r.close()
        with np.load(self.path / "routing/req_a.npz") as data:
            self.assertEqual(data["experts"].dtype, np.uint16)
            self.assertTrue((data["experts"] == 258).all())
        with np.load(self.path / "routing/req_b.npz") as data:
            self.assertTrue((data["experts"] == 257).all())

    def test_abort_waits_for_current_prefill_slice(self):
        r = self.create()
        req = request(prompt=10)
        r.add_request(req)
        self.step([(req, 3)], abort=req)
        r.close()
        with np.load(self.path / "routing/req_a.npz") as data:
            self.assertEqual(data["row_end"], 3)
            self.assertEqual(len(data["experts"]), 3)
            self.assertFalse(data["routing_complete"])
        record = json.loads((self.path / "requests.jsonl").read_text())
        self.assertEqual(record["engine_finish_status"], "FINISHED_ABORTED")

    def test_untagged_request_is_in_step_without_routing_file(self):
        r = self.create()
        req = request(tagged=False)
        r.add_request(req)
        self.step([(req, 6)])
        r.finish_request(req)
        r.close()
        step = json.loads((self.path / "steps.jsonl").read_text())
        self.assertIsNone(step["entries"][0][0])
        self.assertEqual(list((self.path / "routing").iterdir()), [])

    def test_never_scheduled_abort_writes_zero_rows(self):
        r = self.create()
        req = request()
        req.status = "FINISHED_ABORTED"
        r.add_request(req)
        r.finish_request(req)
        r.close()
        with np.load(self.path / "routing/req_a.npz") as data:
            self.assertEqual(data["experts"].shape, (0, 2, 2))
            self.assertFalse(data["routing_complete"])

    def test_abort_during_decode_preserves_last_computed_token(self):
        r = self.create()
        req = request(prompt=2)
        r.add_request(req)
        self.step([(req, 2)])
        req.all_token_ids.append(42)
        req.num_output_tokens = 1
        self.step([(req, 1)], abort=req)
        r.close()
        with np.load(self.path / "routing/req_a.npz") as data:
            self.assertEqual(data["row_end"], len(data["token_ids"]))
            self.assertEqual(len(data["experts"]), 3)
            self.assertFalse(data["routing_complete"])

    def test_lifecycle_uses_dispatch_and_output_ready_times(self):
        r = self.create()
        req = request(prompt=2)
        r.add_request(req)
        r.begin_step(100)
        r.schedule(req, 2)
        r.dispatch(110, 1, 0, 0.1)
        r.output_ready(120)
        r.capture_step(np.ones((2, 4, 2)), {"a": 0})
        req.all_token_ids.append(42)
        req.num_output_tokens = 1
        r.first_token(req)
        r.finish_request(req)
        r.end_step(130)
        r.close()
        record = json.loads((self.path / "requests.jsonl").read_text())
        self.assertEqual(record["first_scheduled_at"], 110)
        self.assertEqual(record["first_token_at"], 120)
        self.assertEqual(record["inference_finished_at"], 120)

    def test_later_cache_hit_preserves_only_actual_token_positions(self):
        r = self.create()
        req = request(prompt=10)
        r.add_request(req)
        self.step([(req, 2)])
        r.preempt(req)
        # Another request has since populated a longer cached prefix.
        req.num_computed_tokens = 6
        r.begin_step(100)
        r.schedule(req, 2)
        r.capture_step(np.ones((2, 4, 2)), {"a": 0})
        r.end_step(130)
        req.status = "FINISHED_ABORTED"
        r.finish_request(req)
        r.close()
        with np.load(self.path / "routing/req_a.npz") as data:
            np.testing.assert_array_equal(data["token_positions"], [0, 1, 6, 7])
            self.assertEqual(data["experts"].shape[0], 4)
            self.assertEqual(data["row_end"], 8)

    def test_first_computation_of_a_previous_cache_gap_is_not_recompute(self):
        r = self.create()
        req = request(prompt=10)
        r.add_request(req)
        self.step([(req, 2)], fill=1)
        r.preempt(req)
        req.num_computed_tokens = 6
        self.step([(req, 2)], fill=2)
        r.preempt(req)
        req.num_computed_tokens = 0
        self.step([(req, 10)], fill=3)
        req.num_output_tokens = 1
        req.all_token_ids.append(42)
        r.finish_request(req)
        r.close()
        with np.load(self.path / "routing/req_a.npz") as data:
            np.testing.assert_array_equal(data["token_positions"], range(10))
            np.testing.assert_array_equal(
                data["step_index"], [0, 0, 2, 2, 2, 2, 1, 1, 2, 2]
            )
            np.testing.assert_array_equal(
                data["experts"][:, 0, 0], [1, 1, 3, 3, 3, 3, 2, 2, 3, 3]
            )

    def test_missing_capture_fails_instead_of_fabricating_rows(self):
        r = self.create()
        req = request()
        r.add_request(req)
        r.begin_step(1)
        r.schedule(req, 2)
        with self.assertRaisesRegex(RuntimeError, "no routing"):
            r.capture_step(None, {})

    def test_disabled_recorder_does_not_validate_or_write(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(trace.prepare_engine(NS()))
            self.assertIsNone(trace.TraceRecorder.from_config(NS(), 16))

    def test_startup_rejects_each_unsupported_setting(self):
        config = NS(
            model_config=NS(enable_return_routed_experts=True),
            cache_config=NS(enable_prefix_caching=True),
            speculative_config=None,
            scheduler_config=NS(async_scheduling=False),
            parallel_config=NS(data_parallel_size=1),
        )
        for obj, name, bad in [
            (config.model_config, "enable_return_routed_experts", False),
            (config.cache_config, "enable_prefix_caching", False),
            (config, "speculative_config", {}),
            (config.scheduler_config, "async_scheduling", True),
            (config.parallel_config, "data_parallel_size", 2),
        ]:
            original = getattr(obj, name)
            setattr(obj, name, bad)
            with self.subTest(setting=name), self.assertRaisesRegex(ValueError, name):
                trace.validate_config(config)
            setattr(obj, name, original)
        trace.validate_config(config)

    def test_api_assembly_is_suppressed_only_with_recorder(self):
        # Execute the actual scheduler assembly block without importing CUDA.
        module = ast.parse((FORK / "vllm/v1/core/sched/scheduler.py").read_text())
        method = next(
            n
            for n in ast.walk(module)
            if isinstance(n, ast.FunctionDef) and n.name == "update_from_output"
        )
        assembly = next(
            n
            for n in ast.walk(method)
            if isinstance(n, ast.If)
            and "self.tokenmoe_trace is None" in ast.unparse(n.test)
        )
        code = compile(
            ast.fix_missing_locations(ast.Module(body=[assembly], type_ignores=[])),
            "scheduler_assembly",
            "exec",
        )

        def forbidden(*args, **kwargs):
            raise AssertionError("slot buffer must not be read")

        scheduler = NS(
            enable_return_routed_experts=True,
            tokenmoe_trace=object(),
            routed_experts_mgr=NS(get=forbidden),
            _re_block_ids={},
        )
        scope = {
            "self": scheduler,
            "routing_data": np.ones((2, 2, 2)),
            "new_token_ids": [1],
            "routed_experts": None,
            "req_id": "a",
            "routing_offsets": {"a": 0},
            "num_tokens_scheduled": 2,
            "num_output_tokens_before": 1,
            "scheduled_spec_token_ids": None,
        }
        exec(code, scope)
        self.assertIsNone(scope["routed_experts"])
        scheduler.tokenmoe_trace = None
        exec(code, scope)
        self.assertEqual(scope["routed_experts"].shape, (1, 2, 2))


if __name__ == "__main__":
    unittest.main()
