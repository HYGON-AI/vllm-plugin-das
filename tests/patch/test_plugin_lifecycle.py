# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from __future__ import annotations

import builtins
import importlib
import inspect
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Iterator

import pytest

# Lifecycle tests invoke plugin entry points explicitly.
os.environ.setdefault("VLLM_PLUGINS", "__disabled__")

import vllm_hcu as plugin
from vllm_hcu.patch.import_coordinator import IMPORT_COORDINATOR
from vllm_hcu.patch.runtime_state import (
    PATCH_REGISTRY,
    LatchedPatchError,
    ProcessRole,
    set_process_role,
)


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


@pytest.fixture(autouse=True)
def _clean_plugin_registry():
    IMPORT_COORDINATOR.reset_for_tests()
    yield
    IMPORT_COORDINATOR.reset_for_tests()


def _fresh_python(
    code: str,
    *,
    plugins: str = "__disabled__",
    assert_target_first: bool = True,
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["VLLM_PLUGINS"] = plugins
    env["VLLM_V0251_SOURCE_ROOT"] = str(TARGET_VLLM_ROOT)
    env["PYTHONPATH"] = os.pathsep.join((str(TARGET_VLLM_ROOT), str(REPO)))
    child_code = (
        _TARGET_SOURCE_ASSERTION + code
        if assert_target_first
        else code + _TARGET_SOURCE_ASSERTION
    )
    return subprocess.run(
        [sys.executable, "-c", child_code],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )


@contextmanager
def _cpu_safe_hcu_worker_module() -> Iterator[ModuleType]:
    """Import the HCU Worker without loading the upstream GPU runtime.

    These lifecycle tests exercise ordering around the parent Worker methods,
    not GPU kernels.  Importing the real upstream ``gpu_worker`` pulls in
    flash-attn at module import time, which aborts a CPU-only pytest process.
    The fake parent is scoped to this context and every module/package binding
    is restored before subsequent tests run.
    """

    upstream_name = "vllm.v1.worker.gpu_worker"
    worker_utils_name = "vllm.v1.worker.utils"
    hcu_name = "vllm_hcu.v1.worker"
    missing = object()
    previous_upstream = sys.modules.get(upstream_name, missing)
    previous_worker_utils = sys.modules.get(worker_utils_name, missing)
    previous_hcu = sys.modules.get(hcu_name, missing)

    import vllm_hcu.v1 as hcu_v1_package

    previous_hcu_attribute = getattr(hcu_v1_package, "worker", missing)

    class Worker:
        def load_model(self, *, load_dummy_weights: bool = False) -> None:
            del load_dummy_weights

    fake_upstream = ModuleType(upstream_name)
    fake_upstream.Worker = Worker
    fake_upstream.init_worker_distributed_environment = lambda *args, **kwargs: None
    fake_worker_utils = ModuleType(worker_utils_name)
    fake_worker_utils.request_memory = lambda *args, **kwargs: 0
    sys.modules[upstream_name] = fake_upstream
    sys.modules[worker_utils_name] = fake_worker_utils
    sys.modules.pop(hcu_name, None)
    hcu_v1_package.__dict__.pop("worker", None)

    try:
        worker_module = importlib.import_module(hcu_name)
        yield worker_module
    finally:
        sys.modules.pop(hcu_name, None)
        if previous_hcu is not missing:
            sys.modules[hcu_name] = previous_hcu
        if previous_hcu_attribute is missing:
            hcu_v1_package.__dict__.pop("worker", None)
        else:
            hcu_v1_package.worker = previous_hcu_attribute
        if previous_upstream is missing:
            sys.modules.pop(upstream_name, None)
        else:
            sys.modules[upstream_name] = previous_upstream
        if previous_worker_utils is missing:
            sys.modules.pop(worker_utils_name, None)
        else:
            sys.modules[worker_utils_name] = previous_worker_utils


@pytest.fixture
def cpu_safe_hcu_worker_module() -> Iterator[ModuleType]:
    with _cpu_safe_hcu_worker_module() as worker_module:
        yield worker_module


def test_plugin_entries_are_interleavable_idempotent_and_registry_backed(monkeypatch):
    from vllm_hcu.patch import worker as worker_dispatcher

    calls: list[str] = []
    monkeypatch.setattr(
        plugin, "_apply_platform_preserving_role", lambda: calls.append("platform")
    )
    monkeypatch.setattr(
        worker_dispatcher,
        "prepare_worker_patches",
        lambda: calls.append("prepare"),
    )

    models = ModuleType("vllm_hcu.models")
    models.register_model = lambda: calls.append("models")
    monkeypatch.setitem(sys.modules, models.__name__, models)

    real_import_module = plugin.importlib.import_module

    def import_module(name: str):
        if name == "vllm_hcu.ops":
            calls.append("ops")
            return ModuleType(name)
        return real_import_module(name)

    monkeypatch.setattr(plugin.importlib, "import_module", import_module)

    assert plugin.hcu_platform_plugin() == "vllm_hcu.platforms.hcu.HCUPlatform"
    plugin.hcu_platform_register_ops()
    plugin.hcu_platform_register_model()
    plugin.hcu_platform_register_model()
    plugin.hcu_platform_register_ops()

    assert calls.count("platform") == 5
    assert calls.count("prepare") == 4
    assert calls.count("models") == 1
    assert calls.count("ops") == 1
    report = PATCH_REGISTRY.report()["patches"]
    assert report[plugin._MODEL_REGISTRY_PATCH_ID]["status"] == "applied"
    assert report[plugin._OPS_REGISTRY_PATCH_ID]["status"] == "applied"


def test_general_plugin_failure_is_latched_and_never_retried(monkeypatch):
    from vllm_hcu.patch import worker as worker_dispatcher

    monkeypatch.setattr(plugin, "_apply_platform_preserving_role", lambda: None)
    monkeypatch.setattr(worker_dispatcher, "prepare_worker_patches", lambda: None)
    calls = []
    models = ModuleType("vllm_hcu.models")

    def fail_registration():
        calls.append("register")
        raise RuntimeError("registry exploded")

    models.register_model = fail_registration
    monkeypatch.setitem(sys.modules, models.__name__, models)

    with pytest.raises(RuntimeError, match="registry exploded"):
        plugin.hcu_platform_register_model()
    with pytest.raises(LatchedPatchError, match="previously failed"):
        plugin.hcu_platform_register_model()
    assert calls == ["register"]
    record = PATCH_REGISTRY.report()["patches"][plugin._MODEL_REGISTRY_PATCH_ID]
    assert record["status"] == "failed"
    assert "registry exploded" in record["failure_reason"]


def test_platform_reentry_preserves_an_authoritative_role(monkeypatch):
    import vllm_hcu.patch as patch_package

    set_process_role(ProcessRole.ENGINE_CORE)

    def imprecise_platform_detection():
        set_process_role(ProcessRole.MAIN)

    monkeypatch.setattr(
        patch_package, "apply_platform_patches", imprecise_platform_detection
    )
    plugin._apply_platform_preserving_role()
    assert PATCH_REGISTRY.process_role() is ProcessRole.ENGINE_CORE


def test_pcp_kv_cache_callbacks_precede_mtp_coordinator_deterministically():
    from vllm_hcu.patch.platform import platform_framework_callback_names

    inventory = platform_framework_callback_names()
    mtp_index = inventory.index(
        (
            "platform.framework_opt.mtp_indexer_kv_cache_coordinator",
            "vllm.v1.core.kv_cache_coordinator",
        )
    )
    assert inventory[mtp_index - 4 : mtp_index] == (
        (
            "platform.framework_opt.pcp_kv_cache_utils",
            "vllm.v1.core.kv_cache_utils",
        ),
        (
            "platform.framework_opt.pcp_kv_cache_interface",
            "vllm.v1.kv_cache_interface",
        ),
        (
            "platform.framework_opt.pcp_single_type_kv_cache_manager",
            "vllm.v1.core.single_type_kv_cache_manager",
        ),
        (
            "platform.framework_opt.pcp_kv_cache_coordinator",
            "vllm.v1.core.kv_cache_coordinator",
        ),
    )


def test_platform_probe_failure_is_exposed_on_vllm_second_invocation(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(plugin, "_PLATFORM_INIT_FAILURE", None)

    def fail_platform() -> None:
        calls.append("apply")
        raise RuntimeError("platform contract mismatch")

    monkeypatch.setattr(plugin, "_apply_platform_preserving_role", fail_platform)
    assert plugin.hcu_platform_plugin() == plugin._PLATFORM_CLASS_PATH
    with pytest.raises(RuntimeError, match="previously failed"):
        plugin.hcu_platform_plugin()
    assert calls == ["apply"]


def test_clean_plugin_import_has_no_legacy_hook_or_eager_runtime_modules():
    result = _fresh_python(
        "import builtins,json,sys; "
        "old=builtins.__import__; "
        "import vllm_hcu; "
        "path=vllm_hcu.hcu_platform_plugin(); "
        "heavy=['torch','vllm','vllm_hcu.platforms.hcu',"
        "'vllm.v1.attention.backends.registry','vllm._aiter_ops','vllm_hcu.ops',"
        "'vllm_hcu.v1.core.sched.scheduler',"
        "'vllm_hcu.v1.executor.multiproc_executor']; "
        "print(json.dumps({'path':path,'builtins_same':builtins.__import__ is old,"
        "'patch_utils':'vllm_hcu.patch_utils' in sys.modules,"
        "'heavy':[name for name in heavy if name in sys.modules]}))",
        assert_target_first=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(
        next(
            line
            for line in reversed(result.stdout.strip().splitlines())
            if line.startswith("{")
        )
    )
    assert payload == {
        "path": "vllm_hcu.platforms.hcu.HCUPlatform",
        "builtins_same": True,
        "patch_utils": False,
        "heavy": [],
    }


@pytest.mark.hcu
def test_engine_core_first_import_does_not_patch_partial_modules_or_fallback():
    result = _fresh_python(
        "import json; "
        "import vllm.v1.engine.core as core; "
        "wrapped_before_reentry=getattr(core.EngineCoreProc.__init__,"
        "'_vllm_hcu_engine_core_init_wrapper',False); "
        "import vllm_hcu; "
        "vllm_hcu.hcu_platform_plugin(); "
        "from vllm.platforms import current_platform; "
        "from vllm_hcu.patch import patch_report; "
        "report=patch_report()['patches']; "
        "print(json.dumps({'platform':type(current_platform).__module__+'.'+"
        "type(current_platform).__name__,"
        "'failed':{k:v['failure_reason'] for k,v in report.items() "
        "if v['status']=='failed'},"
        "'engine_init_wrapped':wrapped_before_reentry,"
        "'engine_core':report['platform.framework_opt.engine_core']['status'],"
        "'parallel_state':report["
        "'platform.framework_opt.group_coordinator_all_to_all']['status']}))",
        plugins="hcu",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload == {
        "platform": "vllm_hcu.platforms.hcu.HCUPlatform",
        "failed": {},
        "engine_init_wrapped": True,
        "engine_core": "applied",
        "parallel_state": "applied",
    }


@pytest.mark.hcu
def test_arg_utils_first_import_applies_sidecar_before_first_construction():
    result = _fresh_python(
        "import dataclasses,json; "
        "import vllm.engine.arg_utils as arg_utils; "
        "from vllm_hcu.patch import patch_report; "
        "from vllm_hcu.patch.config import get_hcu_config; "
        "args=arg_utils.EngineArgs(enable_custom_sp=True,"
        "enable_multi_layers_mtp=True,moe_backend='dpsk_deep_gemm'); "
        "feature=get_hcu_config(args); "
        "record=patch_report()['patches']["
        "'platform.core_fix.hcu_config.engine_args']; "
        "print(json.dumps({'marker':getattr(arg_utils,"
        "'_vllm_hcu_feature_sidecar_patch_applied',False),"
        "'status':record['status'],"
        "'dataclass_restored':arg_utils.dataclass is dataclasses.dataclass,"
        "'upstream_backend':args.moe_backend,"
        "'custom_sp':feature.enable_custom_sp,"
        "'multi_mtp':feature.enable_multi_layers_mtp,"
        "'hcu_backend':feature.moe_backend}))",
        plugins="hcu",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload == {
        "marker": True,
        "status": "applied",
        "dataclass_restored": True,
        "upstream_backend": "auto",
        "custom_sp": True,
        "multi_mtp": True,
        "hcu_backend": "dpsk_deep_gemm",
    }


def test_engine_args_normal_cold_post_import_callback_still_applies():
    result = _fresh_python(
        "import dataclasses,json,vllm_hcu; "
        "vllm_hcu.hcu_platform_plugin(); "
        "import vllm.engine.arg_utils as arg_utils; "
        "from vllm_hcu.patch import patch_report; "
        "args=arg_utils.EngineArgs(enable_custom_sp=True); "
        "record=patch_report()['patches']["
        "'platform.core_fix.hcu_config.engine_args']; "
        "print(json.dumps({'marker':getattr(arg_utils,"
        "'_vllm_hcu_feature_sidecar_patch_applied',False),"
        "'status':record['status'],"
        "'dataclass_restored':arg_utils.dataclass is dataclasses.dataclass,"
        "'sidecar':args.additional_config['hcu']['enable_custom_sp']}))"
    )
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload == {
        "marker": True,
        "status": "applied",
        "dataclass_restored": True,
        "sidecar": True,
    }


def test_worker_applies_before_parent_init_and_validates_after_load(
    monkeypatch,
    cpu_safe_hcu_worker_module,
):
    from vllm_hcu.patch import runtime_state, worker as worker_dispatcher

    worker_module = cpu_safe_hcu_worker_module
    events: list[object] = []
    monkeypatch.setattr(
        runtime_state,
        "set_process_role",
        lambda role: events.append(("role", role)),
    )
    monkeypatch.setattr(
        worker_dispatcher,
        "apply_worker_patches",
        lambda config: events.append(("apply", config)),
    )
    monkeypatch.setattr(
        worker_dispatcher,
        "validate_worker_patches",
        lambda *, require_applied: events.append(("validate", require_applied)),
    )

    def parent_init(self, **kwargs):
        events.append(("parent_init", kwargs))

    def parent_load(self, *, load_dummy_weights=False):
        events.append(("parent_load", load_dummy_weights))

    monkeypatch.setattr(worker_module.Worker, "__init__", parent_init)
    monkeypatch.setattr(worker_module.Worker, "load_model", parent_load)
    config = object()
    worker = object.__new__(worker_module.HcuGPUWorker)
    worker_module.HcuGPUWorker.__init__(
        worker,
        vllm_config=config,
        local_rank=1,
        rank=2,
        distributed_init_method="tcp://test",
        is_driver_worker=True,
    )
    assert events[:3] == [
        ("role", "Worker"),
        ("apply", config),
        (
            "parent_init",
            {
                "vllm_config": config,
                "local_rank": 1,
                "rank": 2,
                "distributed_init_method": "tcp://test",
                "is_driver_worker": True,
            },
        ),
    ]

    worker.load_model(load_dummy_weights=True)
    assert events[-2:] == [("parent_load", True), ("validate", True)]
    assert str(inspect.signature(worker_module.HcuGPUWorker.load_model)) == (
        "(self, *, load_dummy_weights: bool = False) -> None"
    )


@pytest.mark.parametrize(
    ("use_v2", "expected_module", "expected_class"),
    [
        (True, "vllm_hcu.v1.hcu_model_runner_v2", "HcuGPUModelRunnerV2"),
        (False, "vllm_hcu.v1.hcu_model_runner", "GPUModelRunner"),
    ],
)
def test_worker_selects_plugin_owned_model_runner(
    monkeypatch,
    cpu_safe_hcu_worker_module,
    use_v2,
    expected_module,
    expected_class,
):
    events: list[tuple[object, object]] = []
    runner_module = ModuleType(expected_module)

    class Runner:
        def __init__(self, config, device):
            events.append((config, device))

    setattr(runner_module, expected_class, Runner)
    installed: list[object] = []
    if use_v2:
        runner_module.install_fixed_width_pp_sample_broadcast = installed.append
    monkeypatch.setitem(sys.modules, expected_module, runner_module)
    config = object()
    device = object()

    result = cpu_safe_hcu_worker_module._create_model_runner(
        config,
        device,
        use_v2_model_runner=use_v2,
    )

    assert isinstance(result, Runner)
    assert events == [(config, device)]
    assert installed == ([result] if use_v2 else [])


def test_hcu_model_runner_v2_is_thin_upstream_adapter(monkeypatch):
    upstream_name = "vllm.v1.worker.gpu.model_runner"
    adapter_name = "vllm_hcu.v1.hcu_model_runner_v2"
    adapter_attribute = "hcu_model_runner_v2"
    missing = object()
    import vllm_hcu.v1 as hcu_v1_package

    previous_adapter = sys.modules.get(adapter_name, missing)
    previous_adapter_attribute = hcu_v1_package.__dict__.get(
        adapter_attribute, missing
    )
    upstream_module = ModuleType(upstream_name)

    class UpstreamGPUModelRunner:
        def __init__(self, vllm_config, device):
            self.upstream_init = (vllm_config, device)

        def execute_model(self):
            return "upstream"

    upstream_module.GPUModelRunner = UpstreamGPUModelRunner
    monkeypatch.setitem(sys.modules, upstream_name, upstream_module)
    sys.modules.pop(adapter_name, None)
    hcu_v1_package.__dict__.pop(adapter_attribute, None)

    try:
        adapter_module = importlib.import_module(adapter_name)
        adapter = adapter_module.HcuGPUModelRunnerV2
        config = object()
        device = object()
        runner = adapter(config, device)

        assert issubclass(adapter, UpstreamGPUModelRunner)
        assert runner.upstream_init == (config, device)
        assert runner.pcp_manager is None
        assert runner.execute_model() == "upstream"
        assert "execute_model" not in adapter.__dict__
    finally:
        sys.modules.pop(adapter_name, None)
        if previous_adapter is not missing:
            sys.modules[adapter_name] = previous_adapter
        if previous_adapter_attribute is missing:
            hcu_v1_package.__dict__.pop(adapter_attribute, None)
        else:
            hcu_v1_package.__dict__[adapter_attribute] = (
                previous_adapter_attribute
            )


def test_worker_does_not_terminal_validate_after_failed_parent_load(
    monkeypatch,
    cpu_safe_hcu_worker_module,
):
    from vllm_hcu.patch import worker as worker_dispatcher

    worker_module = cpu_safe_hcu_worker_module
    worker = object.__new__(worker_module.HcuGPUWorker)
    monkeypatch.setattr(
        worker_module.Worker,
        "load_model",
        lambda self, **kwargs: (_ for _ in ()).throw(RuntimeError("load failed")),
    )
    monkeypatch.setattr(
        worker_dispatcher,
        "validate_worker_patches",
        lambda **kwargs: pytest.fail("validation ran after failed model load"),
    )
    with pytest.raises(RuntimeError, match="load failed"):
        worker.load_model()


@pytest.mark.parametrize("error_type", [ImportError, RuntimeError])
def test_hcu_management_api_load_is_lazy_and_optional(monkeypatch, error_type):
    import vllm_hcu.platforms.hcu as hcu_module

    hcu_module._load_hcu_management_api.cache_clear()
    monkeypatch.setattr(
        hcu_module.importlib,
        "import_module",
        lambda name: (_ for _ in ()).throw(error_type("private backend detail")),
    )
    try:
        assert hcu_module._load_hcu_management_api() is None
    finally:
        hcu_module._load_hcu_management_api.cache_clear()


def test_hcu_topology_missing_dependency_disables_custom_allreduce(monkeypatch):
    import vllm_hcu.platforms.hcu as hcu_module

    warnings: list[str] = []
    monkeypatch.setattr(
        hcu_module,
        "_load_hcu_management_api",
        lambda: None,
    )
    monkeypatch.setattr(
        hcu_module.logger,
        "warning_once",
        lambda message, *args, **kwargs: warnings.append(message),
    )

    assert hcu_module.HCUPlatform.is_fully_connected([0, 1]) is False
    assert warnings == [
        "HCU management dependency is unavailable; "
        "custom all-reduce will be disabled."
    ]


def test_hcu_topology_query_and_cleanup_fail_closed(monkeypatch):
    import vllm_hcu.platforms.hcu as hcu_module

    events: list[str] = []
    warnings: list[str] = []

    def fail_query():
        events.append("query")
        raise RuntimeError("private backend detail")

    def fail_cleanup():
        events.append("cleanup")
        raise RuntimeError("private cleanup detail")

    api = SimpleNamespace(
        amdsmi_init=lambda: events.append("init"),
        amdsmi_get_processor_handles=fail_query,
        amdsmi_shut_down=fail_cleanup,
    )
    monkeypatch.setattr(
        hcu_module,
        "_load_hcu_management_api",
        lambda: api,
    )
    monkeypatch.setattr(
        hcu_module.logger,
        "warning_once",
        lambda message, *args, **kwargs: warnings.append(message),
    )

    assert hcu_module.HCUPlatform.is_fully_connected([0, 1]) is False
    assert events == ["init", "query", "cleanup"]
    assert warnings == [
        "HCU topology detection failed; custom all-reduce will be disabled.",
        "HCU management cleanup failed.",
    ]


def test_hcu_topology_direct_links_preserve_success(monkeypatch):
    import vllm_hcu.platforms.hcu as hcu_module

    events: list[object] = []
    api = SimpleNamespace(
        amdsmi_init=lambda: events.append("init"),
        amdsmi_get_processor_handles=lambda: ["zero", "one", "two"],
        amdsmi_topo_get_link_type=lambda source, target: (
            events.append((source, target))
            or {"hops": 1, "type": 2}
        ),
        amdsmi_shut_down=lambda: events.append("cleanup"),
    )
    monkeypatch.setattr(
        hcu_module,
        "_load_hcu_management_api",
        lambda: api,
    )

    assert hcu_module.HCUPlatform.is_fully_connected([0, 2]) is True
    assert events == ["init", ("zero", "two"), "cleanup"]


def test_hcu_device_discovery_hides_backend_error(monkeypatch):
    import vllm_hcu.platforms.hcu as hcu_module

    monkeypatch.setattr(hcu_module.torch.cuda, "_is_compiled", lambda: True)
    monkeypatch.setattr(
        hcu_module.torch.cuda,
        "_device_count_amdsmi",
        lambda: (_ for _ in ()).throw(RuntimeError("private backend detail")),
    )
    hcu_module._rocm_device_count_stateless.cache_clear()

    with pytest.raises(RuntimeError, match="^HCU device discovery failed\\.$"):
        hcu_module._rocm_device_count_stateless(None)


def test_platform_defaults_prearm_worker_before_aiter_import(monkeypatch):
    import torch

    # HCUPlatform historically queried the accelerator at import time.  Keep
    # this lifecycle unit test independent of the host GPU inventory.
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(gcnArchName="gfx936"),
    )
    from vllm_hcu.patch import worker as worker_dispatcher
    from vllm_hcu.platforms.hcu import HCUPlatform

    events: list[str] = []
    monkeypatch.setattr(
        worker_dispatcher,
        "prepare_worker_patches",
        lambda: events.append("prepare"),
    )
    rocm_ops = SimpleNamespace(
        is_fused_moe_enabled=lambda: events.append("fused_moe") or False,
        is_linear_fp8_enabled=lambda: events.append("linear") or False,
        is_fusion_moe_shared_experts_enabled=lambda: events.append("shared") or False,
    )
    aiter_module = ModuleType("vllm._aiter_ops")
    aiter_module.rocm_aiter_ops = rocm_ops
    monkeypatch.setitem(sys.modules, aiter_module.__name__, aiter_module)
    config = SimpleNamespace(
        compilation_config=SimpleNamespace(custom_ops=[]),
    )
    HCUPlatform.apply_config_platform_defaults(config)
    assert events == ["prepare", "fused_moe", "linear", "shared"]


def test_platform_check_uses_lazy_scheduler_and_executor_selectors(monkeypatch):
    import torch

    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(gcnArchName="gfx936"),
    )
    from vllm_hcu.patch.platform.core_fix import patch_vllm_config
    from vllm_hcu.patch.platform.framework_opt import (
        patch_multiproc_executor,
        patch_scheduler,
    )
    from vllm_hcu.platforms.hcu import HCUPlatform

    events: list[str] = []
    monkeypatch.setattr(
        patch_vllm_config,
        "validate_and_update_hcu_config",
        lambda config: events.append("validate"),
    )
    monkeypatch.setattr(
        patch_scheduler,
        "select_hcu_scheduler",
        lambda config: events.append("scheduler") or False,
    )
    monkeypatch.setattr(
        patch_multiproc_executor,
        "select_hcu_multiproc_executor",
        lambda config: events.append("executor") or False,
    )
    config = SimpleNamespace(
        model_config=SimpleNamespace(use_mla=False),
        cache_config=None,
        compilation_config=SimpleNamespace(
            cudagraph_mode=SimpleNamespace(has_full_cudagraphs=lambda: False)
        ),
        parallel_config=SimpleNamespace(worker_cls="auto"),
    )
    HCUPlatform.check_and_update_config(config)
    assert events == ["validate", "scheduler", "executor"]
    assert config.parallel_config.worker_cls == "vllm_hcu.v1.worker.HcuGPUWorker"


def test_hcu_config_preserves_mla_prefix_caching(monkeypatch):
    import torch

    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(gcnArchName="gfx936"),
    )
    from vllm_hcu.patch.platform.core_fix import patch_vllm_config
    from vllm_hcu.patch.platform.framework_opt import (
        patch_multiproc_executor,
        patch_scheduler,
    )
    from vllm_hcu.platforms.hcu import HCUPlatform

    events: list[tuple[str, bool]] = []
    monkeypatch.setattr(
        patch_vllm_config,
        "validate_and_update_hcu_config",
        lambda config: events.append(
            ("validate", config.cache_config.enable_prefix_caching)
        ),
    )
    monkeypatch.setattr(
        patch_scheduler,
        "select_hcu_scheduler",
        lambda config: events.append(
            ("scheduler", config.cache_config.enable_prefix_caching)
        )
        or False,
    )
    monkeypatch.setattr(
        patch_multiproc_executor,
        "select_hcu_multiproc_executor",
        lambda config: events.append(
            ("executor", config.cache_config.enable_prefix_caching)
        )
        or False,
    )
    config = SimpleNamespace(
        model_config=SimpleNamespace(use_mla=True),
        cache_config=SimpleNamespace(
            enable_prefix_caching=True,
            user_specified_block_size=True,
        ),
        compilation_config=SimpleNamespace(
            cudagraph_mode=SimpleNamespace(has_full_cudagraphs=lambda: False)
        ),
        parallel_config=SimpleNamespace(worker_cls="auto"),
    )

    HCUPlatform.check_and_update_config(config)

    assert events == [
        ("validate", True),
        ("scheduler", True),
        ("executor", True),
    ]
    assert config.cache_config.enable_prefix_caching is True


def test_scheduler_selector_matrix_is_lazy_and_conflict_safe(monkeypatch):
    from vllm_hcu.patch.platform.framework_opt import patch_scheduler

    monkeypatch.setattr(patch_scheduler.henvs, "VLLM_HCU_USE_PD_SPLIT", False)
    config = SimpleNamespace(
        additional_config={"hcu": {}},
        cache_config=SimpleNamespace(enable_prefix_caching=False),
        scheduler_config=SimpleNamespace(
            scheduler_cls="custom.Scheduler",
            async_scheduling=False,
        ),
    )
    assert patch_scheduler.select_hcu_scheduler(config) is False
    assert config.scheduler_config.scheduler_cls == "custom.Scheduler"

    monkeypatch.setattr(patch_scheduler.henvs, "VLLM_HCU_USE_PD_SPLIT", True)
    monkeypatch.setattr(patch_scheduler.henvs, "VLLM_HCU_USE_CUSTOM_OPS", True)
    for initial in (None, patch_scheduler.UPSTREAM_SCHEDULER_PATH):
        config.scheduler_config.scheduler_cls = initial
        assert patch_scheduler.select_hcu_scheduler(config) is True
        assert config.scheduler_config.scheduler_cls == patch_scheduler.HCU_SCHEDULER_PATH
        assert patch_scheduler.select_hcu_scheduler(config) is False

    config.scheduler_config.scheduler_cls = "custom.Scheduler"
    with pytest.raises(RuntimeError, match="another scheduler_cls"):
        patch_scheduler.select_hcu_scheduler(config)

    config.scheduler_config.scheduler_cls = None
    monkeypatch.setattr(patch_scheduler.henvs, "VLLM_HCU_USE_CUSTOM_OPS", False)
    with pytest.raises(RuntimeError, match="requires VLLM_HCU_USE_CUSTOM_OPS"):
        patch_scheduler.select_hcu_scheduler(config)

    monkeypatch.setattr(patch_scheduler.henvs, "VLLM_HCU_USE_CUSTOM_OPS", True)
    config.scheduler_config.async_scheduling = True
    with pytest.raises(RuntimeError, match="--no-async-scheduling"):
        patch_scheduler.select_hcu_scheduler(config)


@pytest.mark.parametrize(
    ("initial", "selected"),
    [
        ("mp", True),
        (
            "vllm.v1.executor.multiproc_executor.MultiprocExecutor",
            True,
        ),
        ("vllm_hcu.v1.executor.multiproc_executor.HcuMultiprocExecutor", False),
        ("uni", False),
        ("ray", False),
        ("custom.Executor", False),
    ],
)
def test_multiproc_selector_matrix_is_lazy(initial, selected):
    from vllm_hcu.patch.platform.framework_opt import patch_multiproc_executor

    config = SimpleNamespace(
        parallel_config=SimpleNamespace(distributed_executor_backend=initial)
    )
    assert patch_multiproc_executor.select_hcu_multiproc_executor(config) is selected
    expected = (
        patch_multiproc_executor.HCU_MULTIPROC_EXECUTOR_PATH if selected else initial
    )
    assert config.parallel_config.distributed_executor_backend == expected


def _fake_engine_module(events: list[object]) -> ModuleType:
    class EngineCore:
        def __init__(self):
            events.append("inproc")

        def post_step(self, model_executed):
            return model_executed

    class EngineCoreProc:
        def __init__(self, vllm_config):
            events.append(("parent", PATCH_REGISTRY.process_role()))
            # Simulate a defensive plugin callback doing imprecise process-name
            # detection inside the upstream constructor.
            set_process_role(ProcessRole.MAIN)

        def _handle_client_request(self, request_type, request):
            return request

    module = ModuleType("vllm.v1.engine.core")
    module.EngineCore = EngineCore
    module.EngineCoreProc = EngineCoreProc
    module.EngineCoreRequestType = SimpleNamespace(ADD="add")
    return module


@pytest.mark.parametrize("launch_style", ["mp", "ray-actor"])
def test_engine_core_proc_sets_role_before_prepare_and_after_parent(
    monkeypatch,
    cpu_safe_hcu_worker_module,
    launch_style,
):
    del cpu_safe_hcu_worker_module
    from vllm_hcu.patch import worker as worker_dispatcher
    from vllm_hcu.patch.platform.framework_opt import patch_engine_core

    events: list[object] = []
    module = _fake_engine_module(events)
    patch_engine_core.apply_to_module(module)
    monkeypatch.setattr(
        worker_dispatcher,
        "prepare_worker_patches",
        lambda: events.append(("prepare", PATCH_REGISTRY.process_role())),
    )
    set_process_role(ProcessRole.MAIN)
    proc_type = module.EngineCoreProc
    if launch_style == "ray-actor":
        proc_type = type("EngineCoreActor", (proc_type,), {})
    proc_type(object())
    assert events == [
        ("prepare", ProcessRole.ENGINE_CORE),
        ("parent", ProcessRole.ENGINE_CORE),
    ]
    assert PATCH_REGISTRY.process_role() is ProcessRole.ENGINE_CORE


def test_inproc_engine_keeps_main_role_and_explicit_override_wins(monkeypatch):
    from vllm_hcu.patch.platform.framework_opt import patch_engine_core

    events: list[object] = []
    module = _fake_engine_module(events)
    patch_engine_core.apply_to_module(module)
    set_process_role(ProcessRole.MAIN)
    module.EngineCore()
    assert PATCH_REGISTRY.process_role() is ProcessRole.MAIN

    monkeypatch.setenv("VLLM_HCU_PROCESS_ROLE", "Main")
    set_process_role(ProcessRole.WORKER)
    patch_engine_core._set_engine_core_process_role()
    assert PATCH_REGISTRY.process_role() is ProcessRole.MAIN


def test_plugin_source_has_no_builtins_import_override():
    assert builtins.__import__ is not None
    source = inspect.getsource(plugin)
    assert "patch_utils" not in source
    assert "builtins.__import__" not in source
