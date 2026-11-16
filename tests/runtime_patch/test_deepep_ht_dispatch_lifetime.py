# SPDX-License-Identifier: Apache-2.0
"""CPU-only contracts at the DeepEP HT dispatch boundary.

Load the real dispatch function through AST, as other runtime patch tests do,
so these ordering tests cannot import optional GPU libraries or create a
device context.
"""
import ast
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest

_env_spec = importlib.util.spec_from_file_location(
    "step5_dispatch_test_envs", Path(__file__).parents[2] / "vllm_hcu/platforms/envs.py")
HENVS = importlib.util.module_from_spec(_env_spec)
_env_spec.loader.exec_module(HENVS)


def _load_dispatch():
    path = (Path(__file__).parents[2]
            / "vllm_hcu/model_executor/layers/fused_moe/deepep_runtime.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = {"_apply_step5_ht_runtime_config", "_uses_post_dispatch_channel_fp8_quant", "ht_do_dispatch"}
    functions = [node for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(functions) == len(names)
    namespace = dict(henvs=HENVS, os=os, json=json,
                     _step5_ht_runtime_config=None, _step5_ht_runtime_applied=None)
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"),
         namespace)
    return namespace["ht_do_dispatch"]


class DispatchLifetimeTest(unittest.TestCase):
    def setUp(self):
        self.dispatch = _load_dispatch()
        self.events = []
        self.state = {"dbo": True, "uid": 0, "fail_sync": False}
        self.tokens, self.scales, self.ids, self.weights = (
            object(), object(), object(), object())
        self.previous = object()
        self.received_ids, self.received_weights = object(), object()
        self.num_tokens = object()
        self.handle = object()
        self.event = SimpleNamespace(event=None)
        self.old_handles = [object(), object()]
        self.pf = SimpleNamespace(
            handles=list(self.old_handles), async_prepare=True,
            _get_dispatch_config=lambda: "dispatch-config",
        )

        def capture():
            self.assertIs(self.pf._hcu_dbo_private_output, self.state["dbo"])
            self.events.append("capture")
            return self.previous

        def layout(**kwargs):
            self.events.append("layout")
            self.assertIs(kwargs["topk_idx"], self.ids)
            self.assertIs(kwargs["previous_event"], self.previous)
            return ("rank-counts", "rdma-counts", self.num_tokens,
                    "token-rank-mask", self.event)

        def send(**kwargs):
            self.events.append("dispatch")
            self.sent = kwargs
            self.assertIs(self.pf._hcu_dbo_private_output, self.state["dbo"])
            return (kwargs["x"], self.received_ids, self.received_weights,
                    [1, 2], self.handle, self.event)

        def sync():
            self.events.append("sync" if self.state["dbo"] else "sync-noop")
            # Publication of this dispatch's handle must follow its fence.
            uid = self.state["uid"]
            self.assertIs(self.pf.handles[uid], self.expected_old_handle)
            if self.state["fail_sync"]:
                raise RuntimeError("communication fence failed")

        def handoff():
            self.events.append("compute-handoff")
            self.assertIs(self.pf.handles[self.state["uid"]], self.handle)
            self.assertIn("sync" if self.state["dbo"] else "sync-noop", self.events)

        def receiver(*args, **kwargs):
            self.events.append("receiver")
            return args, kwargs

        self.pf.buffer = SimpleNamespace(
            capture=capture, get_dispatch_layout=layout, dispatch=send)
        self.pf._sync_dbo_comm_if_needed = sync
        self.pf._receiver = receiver
        self.module = SimpleNamespace(
            dbo_enabled=lambda: self.state["dbo"],
            dbo_current_ubatch_id=lambda: self.state["uid"],
            dbo_get_previous_event=lambda capture_fn: capture_fn(),
            dbo_yield_and_switch_from_compute_to_comm=lambda: self.events.append("yield-to-comm"),
            dbo_switch_to_compute_sync=handoff,
        )

    def _call(self, token_scales=None, *, post_dispatch=False):
        self.expected_old_handle = self.pf.handles[self.state["uid"]]
        self.quant_config = SimpleNamespace(
            _vllm_hcu_channel_fp8_deepgemm=post_dispatch,
            use_fp8_w8a8=True, use_int8_w8a8=False,
            is_per_act_token=True, is_block_quantized=False,
        )
        return self.dispatch(
            self.module, self.pf, self.tokens, token_scales, self.ids,
            self.weights, 352, None, self.quant_config, False)

    def test_both_dbo_microbatches_fence_before_handle_and_handoff(self):
        for uid in (0, 1):
            with self.subTest(uid=uid):
                self.events.clear()
                self.state.update(dbo=True, uid=uid)
                receiver = self._call(self.scales)
                self.assertIs(self.pf._hcu_dbo_private_output, True)
                self.assertEqual(self.events, ["capture", "yield-to-comm", "layout",
                                               "dispatch", "sync", "compute-handoff"])
                self.assertFalse(self.sent["async_finish"])
                self.assertEqual(self.sent["expert_alignment"], 256)
                received, kwargs = receiver()
                self.assertIs(received[0], self.event)
                self.assertTrue(received[1])
                self.assertIs(received[2][0], self.tokens)
                self.assertIs(received[2][1], self.scales)
                self.assertIs(received[3], self.received_ids)
                self.assertIs(received[6], self.received_weights)
                self.assertIs(received[8], self.quant_config)
                self.assertEqual(kwargs, {"defer_input_quant": False})
        self.assertIs(self.pf.handles[0], self.handle)
        self.assertIs(self.pf.handles[1], self.handle)

    def test_unsplit_call_resets_private_output_marker(self):
        self._call(self.scales)
        self.assertIs(self.pf._hcu_dbo_private_output, True)
        self.events.clear()
        self.state.update(dbo=False, uid=0)
        self._call(self.scales)
        self.assertIs(self.pf._hcu_dbo_private_output, False)
        self.assertEqual(self.events[-3:], ["dispatch", "sync-noop", "compute-handoff"])
        self.assertTrue(self.sent["async_finish"])

    def test_fence_failure_does_not_publish_handle_or_handoff(self):
        self.state["fail_sync"] = True
        with self.assertRaisesRegex(RuntimeError, "communication fence failed"):
            self._call(self.scales)
        self.assertIs(self.pf.handles[0], self.old_handles[0])
        self.assertNotIn("compute-handoff", self.events)
        self.assertIs(self.pf._hcu_dbo_private_output, True)

    def test_payload_and_quant_alignment_contract_stays_unchanged(self):
        self.state["dbo"] = False
        for scales, post_dispatch, alignment in (
            (self.scales, False, 256), (None, True, 1),
        ):
            with self.subTest(post_dispatch=post_dispatch):
                receiver = self._call(scales, post_dispatch=post_dispatch)
                sent = self.sent
                self.assertIs(sent["topk_idx"], self.ids)
                self.assertIs(sent["topk_weights"], self.weights)
                self.assertIs(sent["num_tokens_per_expert"], self.num_tokens)
                self.assertEqual(sent["expert_alignment"], alignment)
                self.assertEqual(sent["config"], "dispatch-config")
                self.assertIs(sent["previous_event"], self.previous)
                self.assertFalse(sent["allocate_on_comm_stream"])
                received, _ = receiver()
                if scales is None:
                    self.assertIs(sent["x"], self.tokens)
                    self.assertIs(received[2], self.tokens)
                    self.assertFalse(received[1])
                else:
                    self.assertIs(sent["x"][0], self.tokens)
                    self.assertIs(sent["x"][1], self.scales)
                    self.assertTrue(received[1])


if __name__ == "__main__":
    unittest.main()
