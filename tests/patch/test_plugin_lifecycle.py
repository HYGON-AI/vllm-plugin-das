# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import builtins
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

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
CLEAN_VLLM = Path(
    os.environ.get("VLLM_V021_SOURCE_ROOT", REPO.parent / "vllm_dcu_v0.21")
)


@pytest.fixture(autouse=True)
def _clean_plugin_registry():
    IMPORT_COORDINATOR.reset_for_tests()
    yield
    IMPORT_COORDINATOR.reset_for_tests()


def _fresh_python(
    code: str,
    *,
    plugins: str = "__disabled__",
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["VLLM_PLUGINS"] = plugins
    env["PYTHONPATH"] = os.pathsep.join((str(REPO), str(CLEAN_VLLM)))
    return subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )


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
        "'heavy':[name for name in heavy if name in sys.modules]}))"
    )
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload == {
        "path": "vllm_hcu.platforms.hcu.HCUPlatform",
        "builtins_same": True,
        "patch_utils": False,
        "heavy": [],
    }


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


def test_worker_applies_before_parent_init_and_validates_after_load(monkeypatch):
    from vllm_hcu.patch import runtime_state, worker as worker_dispatcher
    from vllm_hcu.v1 import worker as worker_module

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


def test_worker_does_not_terminal_validate_after_failed_parent_load(monkeypatch):
    from vllm_hcu.patch import worker as worker_dispatcher
    from vllm_hcu.v1 import worker as worker_module

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
        cache_config=None,
        compilation_config=SimpleNamespace(
            cudagraph_mode=SimpleNamespace(has_full_cudagraphs=lambda: False)
        ),
        parallel_config=SimpleNamespace(worker_cls="auto"),
    )
    HCUPlatform.check_and_update_config(config)
    assert events == ["validate", "scheduler", "executor"]
    assert config.parallel_config.worker_cls == "vllm_hcu.v1.worker.HcuGPUWorker"


def test_scheduler_selector_matrix_is_lazy_and_conflict_safe(monkeypatch):
    from vllm_hcu.patch.platform.framework_opt import patch_scheduler

    monkeypatch.setattr(patch_scheduler.henvs, "VLLM_HCU_USE_PD_SPLIT", False)
    config = SimpleNamespace(
        additional_config={"hcu": {}},
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
    monkeypatch, launch_style
):
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
