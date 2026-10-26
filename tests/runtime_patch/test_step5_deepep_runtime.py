# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for opt-in routing and asynchronous return lifetimes."""
import ast
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


def load_functions(*names):
    path = (Path(__file__).parents[2]
            / "vllm_hcu/model_executor/layers/fused_moe/deepep_runtime.py")
    nodes = [n for n in ast.parse(path.read_text(encoding="utf-8")).body
             if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(nodes) == len(names)
    scope = dict(os=os, json=json, _step5_ht_runtime_config=None,
                 _step5_ht_runtime_applied=None)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    return scope


def logger_modules():
    logger = ModuleType("vllm.logger")
    logger.init_logger = lambda _: SimpleNamespace(
        info=lambda *a, **k: None, info_once=lambda *a, **k: None)
    return {"vllm.logger": logger}


class RuntimeConfigurationTest(unittest.TestCase):
    def setUp(self):
        self.scope = load_functions("_apply_step5_ht_runtime_config",
                                    "step5_ht_launch_config")

        class Buffer:
            num_sms = 18
            num_nvl_bytes = 1536 * 1024**2

            @classmethod
            def set_num_sms(cls, value):
                cls.num_sms = value

        self.prepare = SimpleNamespace(
            buffer=Buffer(), num_dispatchers_=16,
            _hcu_step5_runtime_tuning_allowed=True,
            _hcu_step5_buffer_pair=(object(), object()))

    def apply(self, config):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "communication.json"
            path.write_text(json.dumps(config))
            with patch.dict(os.environ, {"HCU_STEP5_DEEPEP_RUNTIME_CONFIG": str(path)}), \
                    patch.dict("sys.modules", logger_modules()):
                self.scope["_apply_step5_ht_runtime_config"](self.prepare)

    def test_non_step5_never_reads_profile_path(self):
        self.prepare._hcu_step5_runtime_tuning_allowed = False
        with patch.dict(os.environ, {"HCU_STEP5_DEEPEP_RUNTIME_CONFIG": "/missing"}):
            self.scope["_apply_step5_ht_runtime_config"](self.prepare)
        self.assertFalse(hasattr(self.prepare, "_hcu_step5_overlap"))

    def test_step5_profile_keeps_other_instances_and_library_defaults_unchanged(self):
        other = SimpleNamespace(buffer=type(self.prepare.buffer)(),
                                _hcu_step5_runtime_tuning_allowed=False)
        original = object()
        self.apply(dict(version=1, num_sms=16, dispatch_send=8,
                        combine_send=8, overlap=True, async_layout=True))
        self.assertEqual(type(other.buffer).num_sms, 18)
        self.assertIs(self.scope["step5_ht_launch_config"](other, original), original)
        self.assertFalse(hasattr(other, "_hcu_step5_num_sms"))

    def test_disabled_profile_returns_original_config_object(self):
        self.prepare._hcu_step5_runtime_tuning_allowed = False
        original = object()
        self.assertIs(self.scope["step5_ht_launch_config"](
            self.prepare, original), original)

    def test_runtime_budget_and_overlap_are_applied_together(self):
        self.apply(dict(version=1, num_sms=24, dispatch_send=8,
                        combine_send=32, overlap=True, async_layout=True))
        self.assertEqual(self.prepare._hcu_step5_num_sms, 24)
        self.assertEqual(type(self.prepare.buffer).num_sms, 18)
        self.assertTrue(self.prepare._hcu_step5_overlap)
        self.assertTrue(self.prepare._hcu_step5_async_layout)
        self.assertEqual(self.prepare._hcu_step5_combine_send, 32)

    def test_overlap_cannot_use_single_shared_buffer(self):
        self.prepare._hcu_step5_buffer_pair = (object(),)
        with self.assertRaisesRegex(ValueError, "two preallocated"):
            self.apply(dict(version=1, num_sms=18, overlap=True))

    def test_mnvl_sm16_profile_is_accepted(self):
        self.apply(dict(version=1, num_sms=16, dispatch_send=8,
                        combine_send=8, overlap=True, async_layout=True))
        self.assertEqual(self.prepare._hcu_step5_num_sms, 16)
        self.assertEqual(type(self.prepare.buffer).num_sms, 18)
        self.assertTrue(self.prepare._hcu_step5_overlap)

    def test_invalid_budget_is_rejected(self):
        for config in [dict(version=1, num_sms=6),
                       dict(version=1, num_sms=16.0),
                       dict(version=1, num_sms=16, dispatch_send=8.0),
                       dict(version=1, num_sms=18, combine_send=128),
                       dict(version=1, num_sms=18, overlap="true"),
                       dict(version=1, num_sms=18, recv_tokens=64)]:
            with self.subTest(config=config), self.assertRaises(ValueError):
                self.apply(config)

    def test_custom_geometry_cannot_exceed_reserved_buffer(self):
        class Config:
            def __init__(self, *args):
                pass

            def get_nvl_buffer_size_hint(self, *args):
                return self.required

        self.prepare._hcu_step5_combine_send = 32
        self.prepare._hcu_step5_num_sms = 16
        Config.required = self.prepare.buffer.num_nvl_bytes + 1
        with self.assertRaisesRegex(ValueError, "reserved NVL buffer"):
            self.scope["step5_ht_launch_config"](
                self.prepare, Config(), combine=True)
        Config.required = self.prepare.buffer.num_nvl_bytes
        self.assertIsInstance(self.scope["step5_ht_launch_config"](
            self.prepare, Config(), combine=True), Config)


class ConstructionIsolationTest(unittest.TestCase):
    def setUp(self):
        self.configure = load_functions("configure_step5_ht_kernel")[
            "configure_step5_ht_kernel"]

    def test_default_off_does_not_import_optional_runtime(self):
        kernel = object()
        with patch.dict(os.environ, {}, clear=True):
            self.assertIs(self.configure(kernel, None, None), kernel)

    def modules(self):
        config = ModuleType("vllm.config")
        config.get_current_vllm_config_or_none = lambda: SimpleNamespace(
            parallel_config=SimpleNamespace(enable_dbo=True))
        step5 = ModuleType("vllm_hcu.runtime_compat.step5")
        step5.is_step5_owner = lambda owner: owner.step5
        return {"vllm.config": config,
                "vllm_hcu.runtime_compat.step5": step5}

    def test_other_models_ignore_enabled_step5_overlap_flags(self):
        prepare = type("DeepEPHTPrepareAndFinalize", (), {})()
        kernel = SimpleNamespace(prepare_finalize=prepare,
                                 fused_experts=SimpleNamespace())
        with patch.dict(os.environ, {
            "HCU_STEP5_DEEPEP_PROFILE_DBO_OUTPUT": "1",
            "HCU_STEP5_DEEPEP_DBO_BUFFER_PAIR": "1",
        }), patch.dict("sys.modules", self.modules()):
            self.assertIs(self.configure(kernel, None,
                                         SimpleNamespace(step5=False)), kernel)
        self.assertFalse(prepare._hcu_step5_runtime_tuning_allowed)
        self.assertFalse(prepare._hcu_step5_profile_dbo_output_requested)
        self.assertFalse(hasattr(prepare, "_hcu_step5_buffer_pair"))

    def test_unsupported_ep_group_rejected_before_optional_import_or_allocation(self):
        prepare = type("DeepEPHTPrepareAndFinalize", (), {})()
        prepare.num_dispatchers_ = 8
        kernel = SimpleNamespace(prepare_finalize=prepare,
                                 fused_experts=SimpleNamespace())
        with patch.dict(os.environ, {
            "HCU_STEP5_DEEPEP_PROFILE_DBO_OUTPUT": "1",
            "HCU_STEP5_DEEPEP_DBO_BUFFER_PAIR": "1",
            "DEEP_EP_NORMAL_MNVL": "1",
        }), patch.dict("sys.modules", self.modules()):
            with self.assertRaisesRegex(ValueError, "EP16 NORMAL MNVL"):
                self.configure(kernel, None, SimpleNamespace(step5=True))
        self.assertFalse(hasattr(prepare, "_hcu_step5_buffer_pair"))


class FinalizeLifetimeTest(unittest.TestCase):
    def test_non_step5_finalizer_delegates_without_switching_streams(self):
        finalize = load_functions("ht_finalize_with_buffer_pair")[
            "ht_finalize_with_buffer_pair"]
        module = SimpleNamespace(dbo_current_ubatch_id=lambda: 0)
        arguments = []

        def original(*args):
            arguments.append(args)
            return "legacy-receiver"

        values = [object() for _ in range(6)]
        owner = SimpleNamespace()
        result = finalize(module, owner, *values[:4], False, values[4],
                          True, original)
        self.assertEqual(result, "legacy-receiver")
        self.assertIs(arguments[0][0], owner)
        self.assertIs(arguments[0][1], values[0])

    def test_return_waits_on_copy_stream_and_retains_inputs_until_receiver(self):
        finalize = load_functions("ht_finalize_with_buffer_pair")[
            "ht_finalize_with_buffer_pair"]
        events = []
        dtype = object()

        class Tensor:
            is_cuda = True

            def __init__(self, name):
                self.name, self.dtype = name, dtype

            def numel(self):
                return 1

            def copy_(self, other, **kwargs):
                events.append(("copy", other.name))

            def record_stream(self, stream):
                events.append(("retain", self.name))

        source, output, returned = Tensor("source"), Tensor("output"), Tensor("returned")

        class Delegate:
            pass

        class NoOp:
            def apply(self, **kwargs):
                return kwargs["fused_expert_output"]

        class Buffer:
            def capture(self):
                return object()

            def combine(self, **kwargs):
                self.arguments = kwargs
                events.append(("combine", kwargs["x"].name))
                event = SimpleNamespace(event=object(), current_stream_wait=
                                        lambda: events.append(("wait", "communication")))
                return returned, None, event

        buffer = Buffer()
        owner = SimpleNamespace(_hcu_step5_pair_active={0: True},
                                _hcu_step5_buffer_pair=(buffer, object()),
                                _hcu_step5_inflight={0: [source]},
                                handles=[object()], _get_combine_config=lambda: None)
        module = SimpleNamespace(
            dbo_current_ubatch_id=lambda: 0, dbo_enabled=lambda: True,
            dbo_get_previous_event=lambda capture: capture(),
            dbo_yield_and_switch_from_compute_to_comm=lambda: events.append(("yield", "communication")),
            dbo_switch_to_compute=lambda: events.append(("switch", "compute")),
            dbo_switch_to_comm=lambda: events.append(("switch", "communication")),
            dbo_yield_and_switch_from_comm_to_compute=lambda: events.append(("yield", "compute")),
            TopKWeightAndReduceDelegate=Delegate,
            TopKWeightAndReduceContiguous=NoOp,
            torch=SimpleNamespace(Tensor=Tensor, bfloat16=dtype,
                                  cuda=SimpleNamespace(current_stream=lambda: "communication")))
        receiver = finalize(module, owner, output, source, None, None,
                            False, NoOp(), True, None)
        self.assertIn(0, owner._hcu_step5_inflight)
        self.assertNotIn(("copy", "returned"), events)
        receiver()
        self.assertLess(events.index(("switch", "communication")),
                        events.index(("wait", "communication")))
        self.assertLess(events.index(("wait", "communication")),
                        events.index(("copy", "returned")))
        self.assertIn(("retain", "source"), events)
        self.assertIn(("retain", "returned"), events)
        self.assertNotIn(0, owner._hcu_step5_inflight)
        self.assertFalse(owner._hcu_step5_pair_active[0])
        self.assertTrue(buffer.arguments["async_finish"])


if __name__ == "__main__":
    unittest.main()
