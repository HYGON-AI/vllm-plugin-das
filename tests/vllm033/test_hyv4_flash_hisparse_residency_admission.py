# SPDX-License-Identifier: Apache-2.0
"""Check that only the owned target/draft construction delegates to Adapter."""

from contextlib import ExitStack
from dataclasses import dataclass, replace
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


SOURCE = (Path(__file__).resolve().parents[2] / "adapter_extension" / "src" / "hcu_vllm_das"
          / "models" / "hy_v4_flash" / "hisparse_planner_compat.py")
MODULE = spec_from_file_location("_hisparse_admission_under_test.planner", SOURCE)
PLANNER = module_from_spec(MODULE)
MODULE.loader.exec_module(PLANNER)
_ADAPTER = "hcu_adapter.kernels.hisparse"
_CONFIG = "_hisparse_admission_under_test.hisparse_config_compat"


@dataclass(frozen=True)
class ResolvedConfig:
    top_k: int
    device_buffer_size: int
    max_union_rows: int
    eager_host_mirror: bool = True


TARGET = ResolvedConfig(2048, 14336, 12288)
DRAFT = ResolvedConfig(2176, 15232, 13056)


class TestHiSparseResidencyAdmission(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.device = SimpleNamespace(type="cuda", index=3)
        self.properties = SimpleNamespace(gcnArchName="gfx936:sramecc+", multi_processor_count=80)
        self.cfg = SimpleNamespace(admitted=True)
        self.current_config = self.cfg
        self.stock = Mock(side_effect=lambda device, cfg: self.calls.append(("stock", device.index, cfg)))
        self.query = Mock(side_effect=lambda *geometry: self.calls.append(geometry))
        self.validator = Mock()
        modules = {
            "torch": SimpleNamespace(cuda=SimpleNamespace(get_device_properties=lambda i: self.properties)),
            "vllm.config": SimpleNamespace(get_current_vllm_config_or_none=lambda: self.current_config),
            _ADAPTER: SimpleNamespace(validate_hisparse_residency_capacity=self.query),
            _CONFIG: SimpleNamespace(_graph_requested=lambda cfg, **_: cfg.admitted,
                                     validate_hisparse_runtime=self.validator),
        }
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch.dict(sys.modules, modules))
        stack.enter_context(patch.object(PLANNER, "check_health", return_value={}))
        self.check = PLANNER._make_residency_check(self.stock, ResolvedConfig)

    def test_ordinary_and_unrelated_constructors_retain_stock_check(self):
        self.check(self.device, TARGET)
        with PLANNER.hisparse_residency_scope(SimpleNamespace(admitted=False)):
            self.check(self.device, TARGET)
        with PLANNER.hisparse_residency_scope(self.cfg):
            self.check(self.device, DRAFT)
            self.properties.gcnArchName = "gfx938"
            self.check(self.device, TARGET)
        self.assertEqual(self.calls, [
            ("stock", 3, TARGET), ("stock", 3, TARGET),
            ("stock", 3, DRAFT), ("stock", 3, TARGET),
        ])

    def test_target_and_draft_use_exact_provider_geometry(self):
        with PLANNER.hisparse_residency_scope(self.cfg):
            self.check(self.device, TARGET)
            draft_cfg = SimpleNamespace(admitted=False)
            with PLANNER.hisparse_residency_scope(self.cfg, construction_config=draft_cfg, draft=True):
                self.check(self.device, DRAFT)
                self.current_config = draft_cfg
                self.check(self.device, DRAFT)
            self.current_config = self.cfg
            self.check(self.device, replace(TARGET, max_union_rows=1))
        self.check(self.device, TARGET)
        self.assertEqual(self.calls, [
            (3, 14336, 12288), ("stock", 3, DRAFT), (3, 15232, 13056), (3, 14336, 13056),
            ("stock", 3, replace(TARGET, max_union_rows=1)), ("stock", 3, TARGET),
        ])
        self.assertEqual(self.validator.call_count, 2)

    def test_provider_failure_never_enters_stock_and_scope_resets(self):
        self.query.side_effect = ValueError("residency workspace unavailable")
        with self.assertRaisesRegex(ValueError, "workspace unavailable"):
            with PLANNER.hisparse_residency_scope(self.cfg):
                self.check(self.device, TARGET)
        self.check(self.device, TARGET)
        self.assertEqual(self.calls, [("stock", 3, TARGET)])

    def test_old_adapter_fails_for_the_admitted_geometry(self):
        with patch.dict(sys.modules, {_ADAPTER: SimpleNamespace()}):
            with self.assertRaisesRegex(RuntimeError, "requires the HCU residency capacity provider"):
                with PLANNER.hisparse_residency_scope(self.cfg):
                    self.check(self.device, TARGET)
        self.assertEqual(self.calls, [])


if __name__ == "__main__":
    unittest.main()
