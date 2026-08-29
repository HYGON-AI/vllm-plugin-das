# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType

import pytest

import vllm_hcu.patch.worker as worker_dispatcher
from vllm_hcu.patch.import_coordinator import ExactImportCoordinator
from vllm_hcu.patch.runtime_state import LatchedPatchError, PatchRegistry, PatchStatus


REPO = Path(__file__).resolve().parents[2]
TARGET_VLLM_ROOT = Path(
    os.environ.get("VLLM_V0251_SOURCE_ROOT", REPO.parent / "vllm_0251")
).resolve()
if not (TARGET_VLLM_ROOT / "vllm" / "__init__.py").is_file():
    raise RuntimeError(
        f"VLLM_V0251_SOURCE_ROOT does not contain vllm: {TARGET_VLLM_ROOT}"
    )

_TARGET_SOURCE_ASSERTION = r'''
import os as _vllm_hcu_os
from pathlib import Path as _VllmHcuPath
import vllm as _vllm_hcu_target
_vllm_hcu_root = _VllmHcuPath(
    _vllm_hcu_os.environ["VLLM_V0251_SOURCE_ROOT"]
).resolve()
_vllm_hcu_file = _VllmHcuPath(_vllm_hcu_target.__file__).resolve()
assert _vllm_hcu_file.is_relative_to(_vllm_hcu_root), (
    f"vllm resolved outside target root: {_vllm_hcu_file} not under {_vllm_hcu_root}"
)
'''


def _run_fresh(code: str, *, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["VLLM_PLUGINS"] = "__disabled__"
    env["VLLM_V0251_SOURCE_ROOT"] = str(TARGET_VLLM_ROOT)
    env["PYTHONPATH"] = os.pathsep.join((str(TARGET_VLLM_ROOT), str(REPO)))
    return subprocess.run(
        [sys.executable, "-c", _TARGET_SOURCE_ASSERTION + code],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        timeout=timeout,
    )


def test_worker_inventory_is_complete_explicit_and_dependency_ordered():
    replacements = worker_dispatcher.worker_module_exchange_names()
    callbacks = worker_dispatcher.worker_callback_names()
    pending_callbacks = worker_dispatcher.worker_pending_callback_names()

    assert replacements == (
        (
            "worker.op_opt.moe.runner.shared_experts",
            "vllm.model_executor.layers.fused_moe.runner.shared_experts",
            "vllm_hcu.model_executor.layers.fused_moe.shared_experts",
        ),
        (
            "worker.op_opt.moe.runner",
            "vllm.model_executor.layers.fused_moe.runner.moe_runner",
            "vllm_hcu.model_executor.layers.fused_moe.moe_runner",
        ),
        (
            "worker.op_opt.aiter_ops.hcu_runtime",
            "vllm._aiter_ops",
            "vllm_hcu.model_executor.layers.fused_moe.aiter_ops",
        ),
        (
            "worker.framework_opt.communicator.hcu_custom_allreduce_exchange",
            "vllm.distributed.device_communicators.custom_all_reduce",
            "vllm_hcu.distributed.device_communicators.custom_all_reduce",
        ),
    )
    assert callbacks
    assert len({patch_id for patch_id, _ in callbacks}) == len(callbacks)
    assert pending_callbacks == (
        (
            "worker.op_opt.compressed_tensors.moe_wna16",
            "vllm.model_executor.layers.quantization.compressed_tensors."
            "compressed_tensors_moe.compressed_tensors_moe_wna16",
        ),
    )
    assert {patch_id for patch_id, _ in callbacks}.isdisjoint(
        patch_id for patch_id, _ in pending_callbacks
    )

    positions = {patch_id: index for index, (patch_id, _) in enumerate(callbacks)}
    assert positions["worker.op_opt.moe.config"] < positions[
        "worker.op_opt.moe.prepare_finalize.deepep_ht"
    ]
    assert positions["worker.op_opt.moe.utils.int8_expert_quant"] < positions[
        "worker.op_opt.moe.all2all_utils"
    ]
    assert positions["worker.op_opt.attention.hcu_layout_and_fused_qkv"] < positions[
        "worker.op_opt.attention.fused_qkv_public_export"
    ]
    framework_order = (
        "worker.framework_opt.dp.deepep_low_latency",
        "worker.framework_opt.forward_context.hcu_runtime_fields",
        "worker.framework_opt.communicator.base_custom_sp",
        "worker.framework_opt.communicator.pynccl_wrapper_all_to_all",
        "worker.framework_opt.communicator.pynccl_all_to_all",
        "worker.framework_opt.spec_decode.hcu_proposer",
        "worker.framework_opt.spec_decode.eagle_topk_buffer",
        "worker.framework_opt.dbo.deep_gemm_sms_capability",
        "worker.framework_opt.dbo.ubatch_metadata",
        "worker.framework_opt.communicator.deep_ep_runtime",
    )
    assert tuple(positions[name] for name in framework_order) == tuple(
        sorted(positions[name] for name in framework_order)
    )

    callback_specs = {
        spec.adapter
        for group in worker_dispatcher._ALL_CALLBACK_GROUPS
        for spec in group
    }
    replacement_specs = {
        spec.adapter
        for spec in (
            *worker_dispatcher._MOE_REPLACEMENTS,
            *worker_dispatcher._OP_REPLACEMENTS,
        )
    }
    pending_specs = {
        spec.adapter for spec in worker_dispatcher._PENDING_CALLBACKS
    }
    inventoried_adapters = callback_specs | replacement_specs | pending_specs
    adapter_files = {
        ".".join(path.with_suffix("").parts)
        for root in (
            Path("vllm_hcu/patch/worker/core_fix"),
            Path("vllm_hcu/patch/worker/op_opt"),
            Path("vllm_hcu/patch/worker/op_opt/moe"),
            Path("vllm_hcu/patch/worker/framework_opt"),
        )
        for path in root.glob("patch_*.py")
    }
    assert inventoried_adapters == adapter_files

    source = Path(worker_dispatcher.__file__).read_text(encoding="utf-8")
    assert ".glob(" not in source
    assert ".rglob(" not in source
    assert "iterdir(" not in source
    assert "os.walk(" not in source


def test_cold_replacement_metadata_matches_lazy_adapter_contracts():
    for spec in worker_dispatcher._COLD_REPLACEMENTS:
        adapter = importlib.import_module(spec.adapter)
        assert spec.patch_id == adapter.PATCH_ID
        if spec is worker_dispatcher._CUSTOM_ALLREDUCE_REPLACEMENT:
            assert spec.target_module == adapter.CUSTOM_ALLREDUCE_MODULE
            assert spec.replacement_module == adapter.HCU_CUSTOM_ALLREDUCE_MODULE
            assert spec.targets == (
                adapter.CUSTOM_ALLREDUCE_MODULE,
                adapter.HCU_CUSTOM_ALLREDUCE_MODULE,
            )
            assert spec.validate_with_adapter is False
        else:
            assert spec.target_module == adapter.TARGET_MODULE
            assert spec.replacement_module == adapter.REPLACEMENT_MODULE
            assert spec.targets == adapter.TARGETS
            assert spec.validate_with_adapter is True


def test_independent_worker_cold_prepare_installs_inside_atomic_batch(
    monkeypatch: pytest.MonkeyPatch,
):
    events: list[str] = []

    class FakeCoordinator:
        in_batch = False

        @contextmanager
        def registration_batch(self):
            self.in_batch = True
            events.append("batch-enter")
            try:
                yield self
            finally:
                events.append("batch-exit")
                self.in_batch = False

        def install(self):
            assert self.in_batch
            events.append("install")

    coordinator = FakeCoordinator()

    def register(value):
        assert value is coordinator and coordinator.in_batch
        events.append("cold")
        return []

    monkeypatch.setattr(
        worker_dispatcher,
        "_register_worker_cold_replacements",
        register,
    )
    assert worker_dispatcher._prepare_worker_cold_replacements(coordinator) == ()
    assert events == ["batch-enter", "install", "cold", "batch-exit"]


def test_prepare_is_lazy_narrow_idempotent_and_keeps_main_role():
    expected_callbacks = len(worker_dispatcher.worker_callback_names())
    expected_replacements = len(worker_dispatcher.worker_module_exchange_names())
    expected_total = expected_callbacks + expected_replacements
    result = _run_fresh(
        "import builtins,json,sys; "
        "from vllm_hcu.patch.worker import (prepare_worker_patches,"
        "worker_callback_names,worker_module_exchange_names); "
        "from vllm_hcu.patch.runtime_state import patch_report; "
        "old=builtins.__import__; "
        "targets={name for _,name in worker_callback_names()}|"
        "{name for _,name,_ in worker_module_exchange_names()}; "
        "business={'vllm_hcu.model_executor.layers.fused_moe.config_runtime',"
        "'vllm_hcu.model_executor.layers.fused_moe.router_runtime',"
        "'vllm_hcu.model_executor.layers.fused_moe.deepep_runtime',"
        "'vllm_hcu.model_executor.layers.fused_moe.aiter_runtime',"
        "'vllm_hcu.model_executor.layers.fused_moe.aiter_ops'}; "
        "assert not (targets & sys.modules.keys()); "
        "assert not (business & sys.modules.keys()); "
        "first=prepare_worker_patches(); second=prepare_worker_patches(); "
        "print(json.dumps({'first':len(first),'second':len(second),"
        "'replacements':sum(item.action.value=='replacement' for item in first),"
        "'callbacks':sum(item.action.value=='callback' for item in first),"
        "'statuses':sorted({item.status for item in first}),"
        "'targets_loaded':sorted(targets & sys.modules.keys()),"
        "'business_loaded':sorted(business & sys.modules.keys()),"
        "'runner_loaded':'vllm_hcu.model_executor.layers.fused_moe.moe_runner' "
        "in sys.modules,'shared_loaded':"
        "'vllm_hcu.model_executor.layers.fused_moe.shared_experts' in sys.modules,"
        "'builtins_same':old is builtins.__import__,"
        "'role':patch_report()['process_role']}))"
    )
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload == {
        "first": expected_total,
        "second": expected_total,
        "replacements": expected_replacements,
        "callbacks": expected_callbacks,
        "statuses": ["armed"],
        "targets_loaded": [],
        "business_loaded": [],
        "runner_loaded": False,
        "shared_loaded": False,
        "builtins_same": True,
        "role": "Main",
    }


def test_apply_binds_pickled_sidecar_feature_state_and_worker_report():
    expected_worker_patches = len(worker_dispatcher.worker_callback_names()) + len(
        worker_dispatcher.worker_module_exchange_names()
    )
    result = _run_fresh(
        "import json,os,pickle; from types import SimpleNamespace; "
        "CompilationConfig=type('CompilationConfig',(),{}); "
        "from vllm_hcu.patch.worker import apply_worker_patches; "
        "from vllm_hcu.patch.runtime_state import patch_report; "
        "config=SimpleNamespace(additional_config={'hcu':{"
        "'enable_lightly_cp':True,'enable_lightly_cplb':True,"
        "'enable_custom_sp':True,'enable_multi_layers_mtp':True,"
        "'moe_backend':'deep_gemm'}},"
        "compilation_config=CompilationConfig(),parallel_config=SimpleNamespace("
        "all2all_backend='deepep_low_latency')); "
        "config=pickle.loads(pickle.dumps(config)); "
        "apply_worker_patches(config); apply_worker_patches(config); "
        "report=patch_report(); patches=report['patches']; "
        "worker_patches={name:value for name,value in patches.items() if "
        "name.startswith('worker.')}; "
        "selected=['worker.op_opt.mla.lightly_cp_wrapper',"
        "'worker.framework_opt.communicator.base_custom_sp',"
        "'worker.framework_opt.spec_decode.eagle_topk_buffer',"
        "'worker.op_opt.moe.oracle.fp8_dpsk',"
        "'worker.op_opt.moe.prepare_finalize.deepep_ll']; "
        "print(json.dumps({'pid':report['pid'],'actual_pid':os.getpid(),"
        "'role':report['process_role'],'count':len(worker_patches),"
        "'selected':{name:[patches[name]['status'],"
        "patches[name]['feature_enabled']] for name in selected},"
        "'pynccl':[patches["
        "'worker.framework_opt.communicator.pynccl_wrapper_all_to_all']['status'],"
        "patches['worker.framework_opt.communicator.pynccl_wrapper_all_to_all']"
        "['feature_enabled']]}))"
    )
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["pid"] == payload["actual_pid"]
    assert payload["role"] == "Worker"
    assert payload["count"] == expected_worker_patches
    assert set(map(tuple, payload["selected"].values())) == {("armed", True)}
    assert payload["pynccl"] == ["armed", False]


def test_pickled_worker_config_rebinds_custom_sp_before_real_adjustment():
    result = _run_fresh(
        "import json,pickle; from types import SimpleNamespace; "
        "from vllm_hcu.patch import apply_platform_patches; "
        "apply_platform_patches(); "
        "from vllm.config.compilation import CompilationConfig; "
        "from vllm_hcu.patch.config import HcuFeatureConfig; "
        "from vllm_hcu.patch.worker import apply_worker_patches; "
        "sizes=[1,2,3,4,5,6,7,8,9,10,12,16]; "
        "make=lambda enabled:pickle.loads(pickle.dumps(SimpleNamespace("
        "additional_config={'hcu':HcuFeatureConfig("
        "enable_custom_sp=enabled).to_dict()},"
        "compilation_config=CompilationConfig(cudagraph_capture_sizes=list(sizes),"
        "max_cudagraph_capture_size=16),parallel_config=SimpleNamespace("
        "all2all_backend='allgather_reducescatter')))); "
        "enabled=make(True); apply_worker_patches(enabled); "
        "enabled.compilation_config.adjust_cudagraph_sizes_for_spec_decode(2,4); "
        "disabled=make(False); apply_worker_patches(disabled); "
        "disabled.compilation_config.adjust_cudagraph_sizes_for_spec_decode(2,4); "
        "print(json.dumps({'enabled':enabled.compilation_config."
        "cudagraph_capture_sizes,'disabled':disabled.compilation_config."
        "cudagraph_capture_sizes,'enabled_sp_after':enabled.compilation_config."
        "pass_config.enable_sp,'disabled_sp_after':disabled.compilation_config."
        "pass_config.enable_sp}))"
    )
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload == {
        "enabled": [4, 8, 12, 16],
        "disabled": [2, 4, 6, 8, 10, 12, 16],
        "enabled_sp_after": False,
        "disabled_sp_after": None,
    }


def test_apply_orders_platform_role_sidecar_prepare_and_feature_binding(
    monkeypatch: pytest.MonkeyPatch,
):
    events: list[str] = []

    class FakeCoordinator:
        def set_feature_enabled(self, patch_id: str, enabled: bool):
            assert patch_id == "worker.test.always"
            assert enabled is True
            events.append("bind")

    fake_platform = ModuleType("vllm_hcu.patch.platform")
    fake_platform.apply_platform_patches = lambda: events.append("platform")
    monkeypatch.setitem(
        sys.modules,
        "vllm_hcu.patch.platform",
        fake_platform,
    )
    monkeypatch.setattr(
        worker_dispatcher,
        "set_process_role",
        lambda role: events.append(f"role:{role}"),
    )

    def get_config(value: object):
        events.append("sidecar")
        return worker_dispatcher.HcuFeatureConfig()

    monkeypatch.setattr(worker_dispatcher, "get_hcu_config", get_config)
    monkeypatch.setattr(
        worker_dispatcher,
        "_bind_deserialized_hcu_config",
        lambda value: events.append("sidecar-bind")
        or worker_dispatcher.HcuFeatureConfig(),
    )
    monkeypatch.setattr(
        worker_dispatcher,
        "prepare_worker_patches",
        lambda: events.append("prepare"),
    )

    def patch_features():
        events.append("inventory")
        return {"worker.test.always": "always"}

    monkeypatch.setattr(worker_dispatcher, "_patch_features", patch_features)
    monkeypatch.setattr(
        worker_dispatcher,
        "_raise_latched_or_required_failures",
        lambda coordinator: events.append("validate"),
    )
    monkeypatch.setattr(worker_dispatcher, "IMPORT_COORDINATOR", FakeCoordinator())

    worker_dispatcher.apply_worker_patches(object())
    assert events == [
        "platform",
        "role:Worker",
        "sidecar",
        "sidecar-bind",
        "prepare",
        "inventory",
        "bind",
        "validate",
    ]


def test_terminal_validation_rejects_enabled_armed_but_allows_feature_off():
    patch_id = "worker.op_opt.mla.lightly_cp_wrapper"
    coordinator = ExactImportCoordinator(registry=PatchRegistry())
    coordinator.register_callback(
        patch_id,
        "hcu_worker_dispatcher_terminal_target",
        lambda module: None,
        targets="hcu_worker_dispatcher_terminal_target.MLA",
        feature_enabled=True,
    )
    try:
        with pytest.raises(RuntimeError, match="did not reach.*required terminal"):
            worker_dispatcher.validate_worker_patches(
                require_applied=True, coordinator=coordinator
            )
        # Before model import the same enabled callback is allowed to be armed.
        worker_dispatcher.validate_worker_patches(
            require_applied=False, coordinator=coordinator
        )
        coordinator.set_feature_enabled(patch_id, False)
        worker_dispatcher.validate_worker_patches(
            require_applied=True, coordinator=coordinator
        )
    finally:
        coordinator.reset_for_tests()


def test_cold_moe_replacements_validate_once_and_preempt_official_modules():
    result = _run_fresh(
        "import builtins,importlib,json; "
        "from vllm_hcu.patch.worker import apply_worker_patches; "
        "from vllm_hcu.patch.runtime_state import patch_report; "
        "old=builtins.__import__; apply_worker_patches(); "
        "shared=importlib.import_module("
        "'vllm.model_executor.layers.fused_moe.runner.shared_experts'); "
        "runner=importlib.import_module("
        "'vllm.model_executor.layers.fused_moe.runner.moe_runner'); "
        "patches=patch_report()['patches']; "
        "print(json.dumps({'shared_name':shared.__name__,"
        "'runner_name':runner.__name__,'shared_status':patches["
        "'worker.op_opt.moe.runner.shared_experts']['status'],"
        "'runner_status':patches['worker.op_opt.moe.runner']['status'],"
        "'forward_status':patches["
        "'worker.framework_opt.forward_context.hcu_runtime_fields']['status'],"
        "'builtins_same':old is builtins.__import__}))",
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload == {
        "shared_name": "vllm_hcu.model_executor.layers.fused_moe.shared_experts",
        "runner_name": "vllm_hcu.model_executor.layers.fused_moe.moe_runner",
        "shared_status": "applied",
        "runner_status": "applied",
        "forward_status": "applied",
        "builtins_same": True,
    }


def test_aiter_cold_replacement_preempts_official_registration():
    result = _run_fresh(
        """
import importlib
import importlib.abc
import importlib.util
import json
import sys

from vllm_hcu.patch.import_coordinator import (
    IMPORT_COORDINATOR,
    ModuleReloadBlockedError,
)
from vllm_hcu.patch.runtime_state import patch_report
from vllm_hcu.patch.platform import apply_platform_patches


class TrapLoader(importlib.abc.Loader):
    def create_module(self, spec):
        return None

    def exec_module(self, module):
        trap.executed += 1
        raise AssertionError("official vllm._aiter_ops body executed")


class TrapFinder(importlib.abc.MetaPathFinder):
    consulted = 0
    executed = 0

    def find_spec(self, fullname, path=None, target=None):
        if fullname != "vllm._aiter_ops":
            return None
        self.consulted += 1
        return importlib.util.spec_from_loader(fullname, TrapLoader())


# Platform discovery must pre-arm this import-time custom-op module even when
# no general/worker plugin has configured post-import callbacks yet.
apply_platform_patches()
replacement = "vllm_hcu.model_executor.layers.fused_moe.aiter_ops"
assert replacement not in sys.modules
trap = TrapFinder()
index = sys.meta_path.index(IMPORT_COORDINATOR)
sys.meta_path.insert(index + 1, trap)
module = importlib.import_module("vllm._aiter_ops")
replacement_name = module.__name__
replacement_file = module.__file__
blocked = False
try:
    importlib.reload(module)
except ModuleReloadBlockedError:
    blocked = True
record = patch_report()["patches"]["worker.op_opt.aiter_ops.hcu_runtime"]
print(
    json.dumps(
        {
            "name": replacement_name,
            "file": replacement_file,
            "marker": module._vllm_hcu_aiter_ops_replacement,
            "validated": module._vllm_hcu_aiter_ops_replacement_validated,
            "register_calls": module._HCU_REGISTER_OPS_CALLS,
            "official_consulted": trap.consulted,
            "official_executed": trap.executed,
            "blocked": blocked,
            "status": record["status"],
        }
    )
)
""",
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["name"] == (
        "vllm_hcu.model_executor.layers.fused_moe.aiter_ops"
    )
    assert payload["file"].endswith(
        "vllm_hcu/model_executor/layers/fused_moe/aiter_ops.py"
    )
    assert payload["marker"] is True
    assert payload["validated"] is True
    assert payload["register_calls"] == 1
    assert payload["official_consulted"] == 0
    assert payload["official_executed"] == 0
    assert payload["blocked"] is True
    assert payload["status"] == "applied"


def test_aiter_late_official_import_fails_closed():
    result = _run_fresh(
        """
import importlib
import json

from vllm_hcu.patch.import_coordinator import LateModuleReplacementError

official = importlib.import_module("vllm._aiter_ops")
failed = False
try:
    from vllm_hcu.patch.worker import prepare_worker_patches

    prepare_worker_patches()
except LateModuleReplacementError:
    failed = True
print(
    json.dumps(
        {
            "official_name": official.__name__,
            "failed": failed,
            "hcu_marker": getattr(
                official, "_vllm_hcu_aiter_ops_replacement", False
            ),
        }
    )
)
""",
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload == {
        "official_name": "vllm._aiter_ops",
        "failed": True,
        "hcu_marker": False,
    }


def test_optional_capability_is_skipped_then_explicit_request_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
):
    patch_id = "worker.framework_opt.communicator.pynccl_wrapper_all_to_all"
    module_name = "hcu_worker_dispatcher_optional_target"
    target = ModuleType(module_name)
    sys.modules[module_name] = target
    coordinator = ExactImportCoordinator(registry=PatchRegistry())
    adapter = ModuleType("hcu_worker_dispatcher_optional_adapter")

    def apply_to_module(module: ModuleType, *, required: bool = False) -> bool:
        module._vllm_hcu_pynccl_all_to_all_probe = "RCCL symbol absent"
        if required:
            raise RuntimeError("explicitly requested but unavailable")
        return False

    adapter.apply_to_module = apply_to_module
    try:
        with monkeypatch.context() as scoped:
            scoped.setattr(worker_dispatcher, "_load_adapter", lambda name: adapter)
            callback = worker_dispatcher._callback_for(
                "hcu_worker_dispatcher_optional_adapter",
                True,
                coordinator,
                patch_id,
            )
            registration = coordinator.register_callback(
                patch_id,
                module_name,
                callback,
                targets=f"{module_name}.ncclAllToAll",
                feature_enabled=False,
            )
        assert registration.status == PatchStatus.SKIPPED.value
        assert "RCCL symbol absent" in (
            coordinator._registry.get(patch_id).failure_reason or ""
        )
        coordinator.set_feature_enabled(patch_id, True)
        with pytest.raises(RuntimeError, match="requires patch"):
            worker_dispatcher._raise_latched_or_required_failures(coordinator)
    finally:
        coordinator.reset_for_tests()
        sys.modules.pop(module_name, None)
        worker_dispatcher._callback_for.cache_clear()


def test_required_callback_post_validation_failure_is_latched(
    monkeypatch: pytest.MonkeyPatch,
):
    patch_id = "worker.core_fix.gpt_oss.rocm_unquantized_gemm_layout"
    module_name = "hcu_worker_dispatcher_bad_marker_target"
    target = ModuleType(module_name)
    sys.modules[module_name] = target
    coordinator = ExactImportCoordinator(registry=PatchRegistry())
    adapter = ModuleType("hcu_worker_dispatcher_bad_marker_adapter")
    calls = 0

    def apply_to_module(module: ModuleType) -> bool:
        nonlocal calls
        calls += 1
        return True

    adapter.apply_to_module = apply_to_module
    try:
        with monkeypatch.context() as scoped:
            scoped.setattr(worker_dispatcher, "_load_adapter", lambda name: adapter)
            callback = worker_dispatcher._callback_for(
                adapter.__name__, False, coordinator, patch_id
            )
            with pytest.raises(RuntimeError, match="reapplied"):
                coordinator.register_callback(
                    patch_id,
                    module_name,
                    callback,
                    targets=f"{module_name}.required_symbol",
                )
        assert calls == 2
        record = coordinator._registry.get(patch_id)
        assert record is not None and record.status is PatchStatus.FAILED
        with pytest.raises(LatchedPatchError, match="required worker patch"):
            worker_dispatcher._raise_latched_or_required_failures(coordinator)
    finally:
        coordinator.reset_for_tests()
        sys.modules.pop(module_name, None)
        worker_dispatcher._callback_for.cache_clear()


def test_invalid_spawned_sidecar_fails_before_worker_registration():
    result = _run_fresh(
        "from types import SimpleNamespace; "
        "from vllm_hcu.patch.worker import apply_worker_patches; "
        "apply_worker_patches(SimpleNamespace(additional_config={'hcu':{"
        "'enable_lightly_cp':False,'enable_lightly_cplb':True}}))"
    )
    assert result.returncode != 0
    assert "enable_lightly_cplb requires enable_lightly_cp" in result.stderr
