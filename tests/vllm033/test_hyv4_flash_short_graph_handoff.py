# SPDX-License-Identifier: Apache-2.0
"""Production runner methods with explicit Core control doubles, no GPU math."""
from enum import Enum
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch


class Mode(Enum):
    NONE = 0
    PIECEWISE = 1
    FULL = 2
    FULL_AND_PIECEWISE = 3


class IndexerDouble:
    def __init__(self, budget=511, ratio=4, max_model_len=1048576):
        self.group_budget, self.ratio = budget, ratio
        self.max_model_len = max_model_len
        self._hcu_fullgraph_capture_bound = None


class ManagerDouble:
    """No capture/attention implementation: descriptor and delegation only."""
    def __init__(self, cfg, device, mode, **kwargs):
        self.cfg, self.device, self.cudagraph_mode = cfg, device, mode
        self.lora_capture_cases = kwargs.get("lora_capture_cases", [0])
        self.varlen_decode = kwargs.get("varlen_decode", True)
        self.arguments = kwargs
        self._capture_descs = {Mode.FULL: [6, 384], Mode.PIECEWISE: [128]}
        self.pool = "core-global-pool"

    def dispatch(self, *args, **kwargs):
        return NS(cg_mode=Mode.FULL)

    def run_fullgraph(self, descriptor):
        return self


class BudgetsDouble:
    def __init__(self, values, mapping):
        self.values, self.mapping = values, mapping

    def __getitem__(self, mapping):
        assert mapping is self.mapping, "wrong batch-to-request mapping"
        return self.values


class BaseRunnerDouble:
    def __init__(self):
        self.events = []
        self.owners = [IndexerDouble(), IndexerDouble(31, 128)]
        self.model = NS(modules=lambda: iter(self.owners))
        self.vllm_config, self.device = object(), object()
        self.max_model_len, self.decode_query_len = 1048576, 6
        self.req_states = NS(max_seq_len=object())
        self.mapping = object()
        self.manager_type = ManagerDouble
        self.resolved_mode = Mode.FULL_AND_PIECEWISE
        self.empty_full = False

    def initialize_kv_cache(self, *args, **kwargs):
        self.events.append(("initialize", args, kwargs))
        self.cudagraph_manager = self.manager_type(
            self.vllm_config, self.device, Mode.FULL_AND_PIECEWISE,
            decode_query_len=self.decode_query_len,
        )
        self.cudagraph_manager.cudagraph_mode = self.resolved_mode
        if self.empty_full:
            self.cudagraph_manager._capture_descs.pop(Mode.FULL)
        self.original_manager = self.cudagraph_manager

    def gather_batch_req_state(self, next_budgets, dummy_run):
        self.events.append(("gather", next_budgets, dummy_run))
        self.req_states.max_seq_len = BudgetsDouble(next_budgets, self.mapping)
        return (None if dummy_run else NS(idx_mapping_np=self.mapping)), 6


class RunnerHandoffContracts(unittest.TestCase):
    def setUp(self):
        source = Path(__file__).resolve().parents[2] / (
            "adapter_extension/src/hcu_vllm_das/models/hy_v4_flash"
        )
        package = "_flash_bank_handoff_control"
        modules = {}
        for name in [package, "vllm", "vllm.platforms", "vllm.config", "vllm.config.compilation",
                     "vllm.v1", "vllm.v1.worker", "vllm.v1.worker.gpu",
                     "vllm.v1.worker.gpu.cudagraph_utils",
                     "vllm.v1.worker.gpu.model_runner",
                     package + ".compressed_indexer"]:
            modules[name] = ModuleType(name)
            modules[name].__path__ = []
        modules["vllm.config.compilation"].CUDAGraphMode = Mode
        modules["vllm.platforms"].current_platform = NS(graph_pool_handle=object)
        modules["vllm.v1.worker.gpu.cudagraph_utils"].ModelCudaGraphManager = ManagerDouble
        modules["vllm.v1.worker.gpu.model_runner"].GPUModelRunner = BaseRunnerDouble
        modules[package + ".compressed_indexer"].FlashCompressedIndexer = IndexerDouble
        self.modules = patch.dict(sys.modules, modules)
        self.modules.start()
        self.addCleanup(self.modules.stop)
        for leaf in ["fullgraph_banks", "fullgraph_runner"]:
            name = package + "." + leaf
            spec = importlib.util.spec_from_file_location(name, source / (leaf + ".py"))
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            setattr(self, leaf, module)
        self.runner = self.fullgraph_runner.runner_class()()

    def test_initialize_preserves_mode_arguments_and_original_manager(self):
        marker = object()
        self.runner.initialize_kv_cache(marker, is_profiling=True)
        banks = self.runner.cudagraph_manager
        self.assertIsInstance(banks, self.fullgraph_banks.FlashFullGraphBanks)
        self.assertEqual(self.runner.events, [("initialize", (marker,), {"is_profiling": True})])
        self.assertIs(banks._short.cudagraph_mode, Mode.FULL_AND_PIECEWISE)
        self.assertIs(banks._short.cfg, self.runner.vllm_config)
        self.assertIs(banks._short.device, self.runner.device)
        self.assertEqual(banks._short.arguments["decode_query_len"], 6)
        self.assertTrue(banks._short.varlen_decode)
        self.assertEqual(banks._short.lora_capture_cases, banks._wide.lora_capture_cases)
        self.assertIsNone(banks._short.arguments["ubatch_runner"])
        self.assertNotIn(Mode.PIECEWISE, banks._short._capture_descs)
        self.assertIn(Mode.PIECEWISE, banks._wide._capture_descs)
        self.assertNotEqual(banks._short.pool, banks._wide.pool)

    def test_budget_reads_core_update_and_resets_between_batches(self):
        self.runner.initialize_kv_cache()
        banks = self.runner.cudagraph_manager
        for limits, dummy, expected in [([100, 2042], False, "short"),
                                        ([100, 2043], False, "wide"),
                                        ([100], False, "short"),
                                        ([100], True, "wide")]:
            state, uniform = self.runner.gather_batch_req_state(limits, dummy)
            self.assertEqual(uniform, 6)
            self.assertEqual(state is None, dummy)
            descriptor = banks.dispatch(2, 12, 6, 0)
            self.assertIs(banks.run_fullgraph(descriptor), getattr(banks, "_" + expected))

    def test_missing_or_inconsistent_indexers_keep_original_wide_manager(self):
        for owners in [[], [IndexerDouble(max_model_len=8192)], [IndexerDouble(ratio=0)]]:
            self.runner.owners = owners
            self.runner.initialize_kv_cache()
            self.assertIs(type(self.runner.cudagraph_manager), ManagerDouble)

    def test_unexpected_core_manager_type_fails_before_wrapping(self):
        class OtherManager(ManagerDouble):
            pass
        self.runner.manager_type = OtherManager
        with self.assertRaisesRegex(RuntimeError, "admitted Core manager"):
            self.runner.initialize_kv_cache()

    def test_resolved_mode_downgrade_preserves_original_manager(self):
        for mode in [Mode.NONE, Mode.PIECEWISE, Mode.FULL]:
            self.runner.resolved_mode = mode
            self.runner.initialize_kv_cache()
            self.assertIs(self.runner.cudagraph_manager, self.runner.original_manager)

    def test_empty_effective_full_candidates_preserves_original_manager(self):
        self.runner.empty_full = True
        self.runner.initialize_kv_cache()
        self.assertIs(self.runner.cudagraph_manager, self.runner.original_manager)


if __name__ == "__main__":
    unittest.main()
