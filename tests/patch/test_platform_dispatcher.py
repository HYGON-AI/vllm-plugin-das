# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest

import vllm_hcu.patch.platform as platform_dispatcher
from vllm_hcu.patch.platform import platform_framework_callback_names
from vllm_hcu.patch.platform.core_fix import platform_core_callback_names


REPO = Path(__file__).resolve().parents[2]
TARGET_VLLM_ROOT = Path(
    os.environ.get("VLLM_V025_SOURCE_ROOT", REPO.parent / "vllm_025")
).resolve()
if not (TARGET_VLLM_ROOT / "vllm" / "__init__.py").is_file():
    raise RuntimeError(
        f"VLLM_V025_SOURCE_ROOT does not contain vllm: {TARGET_VLLM_ROOT}"
    )

_TARGET_SOURCE_ASSERTION = r'''
import os as _vllm_hcu_os
from pathlib import Path as _VllmHcuPath
import vllm as _vllm_hcu_target
_vllm_hcu_root = _VllmHcuPath(
    _vllm_hcu_os.environ["VLLM_V025_SOURCE_ROOT"]
).resolve()
_vllm_hcu_file = _VllmHcuPath(_vllm_hcu_target.__file__).resolve()
assert _vllm_hcu_file.is_relative_to(_vllm_hcu_root), (
    f"vllm resolved outside target root: {_vllm_hcu_file} not under {_vllm_hcu_root}"
)
'''


def _run_fresh(
    code: str, *, timeout: int = 120
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["VLLM_PLUGINS"] = "__disabled__"
    env["VLLM_V025_SOURCE_ROOT"] = str(TARGET_VLLM_ROOT)
    env["PYTHONPATH"] = os.pathsep.join((str(TARGET_VLLM_ROOT), str(REPO)))
    return subprocess.run(
        [sys.executable, "-c", _TARGET_SOURCE_ASSERTION + code],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        timeout=timeout,
    )


def test_platform_core_inventory_is_explicit_and_ordered():
    assert platform_core_callback_names() == (
        ("platform.core_fix.envs", "vllm.envs"),
        ("platform.core_fix.import_utils.deep_gemm", "vllm.utils.import_utils"),
        ("platform.core_fix.hcu_config.engine_args", "vllm.engine.arg_utils"),
        ("platform.core_fix.hcu_config.compilation_custom_sp", "vllm.config.compilation"),
        ("platform.core_fix.hcu_config.vllm", "vllm.config.vllm"),
        (
            "platform.core_fix.hcu_config.slimquant_registry",
            "vllm.model_executor.layers.quantization",
        ),
        (
            "platform.core_fix.hy_v3_reasoning_parser",
            "vllm.reasoning.hy_v3_reasoning_parser",
        ),
        (
            "platform.core_fix.hy_v3_tool_parser",
            "vllm.tool_parsers.hy_v3_tool_parser",
        ),
    )


def test_platform_framework_inventory_is_explicit_and_dependency_ordered():
    assert platform_framework_callback_names() == (
        (
            "platform.framework_opt.hcu_mooncake_connector",
            "vllm_hcu.distributed.kv_transfer.kv_connector.v1.mooncake."
            "mooncake_connector",
        ),
        (
            "platform.framework_opt.kv_connector_factory",
            "vllm.distributed.kv_transfer.kv_connector.factory",
        ),
        (
            "platform.framework_opt.group_coordinator_all_to_all",
            "vllm.distributed.parallel_state",
        ),
        (
            "platform.framework_opt.hybrid_kv_page_size",
            "vllm.v1.core.kv_cache_utils",
        ),
        (
            "platform.framework_opt.hcu_scheduler",
            "vllm.v1.core.sched.scheduler",
        ),
        ("platform.framework_opt.engine_core", "vllm.v1.engine.core"),
        (
            "platform.framework_opt.output_processor_ttft",
            "vllm.v1.engine.output_processor",
        ),
        (
            "platform.framework_opt.hcu_multiproc_executor",
            "vllm.v1.executor.multiproc_executor",
        ),
        ("platform.framework_opt.outputs_draft_token_ids", "vllm.v1.outputs"),
    )


def test_slimquant_adapter_import_does_not_preload_target_package():
    result = _run_fresh(
        "import sys; "
        "from vllm_hcu.patch.platform.core_fix import patch_slimquant_registry; "
        "print(patch_slimquant_registry.TARGET_MODULE in sys.modules)"
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("False")


def test_platform_publishes_worker_and_platform_replacements_in_one_batch(
    monkeypatch: pytest.MonkeyPatch,
):
    import vllm_hcu.patch.worker as worker_dispatcher

    events: list[str] = []

    class FakeCoordinator:
        in_batch = False

        @contextmanager
        def registration_batch(self):
            assert not self.in_batch
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

        def drain_ready_callbacks(self):
            assert not self.in_batch
            events.append("drain")

    coordinator = FakeCoordinator()

    def cold(value):
        assert value is coordinator and coordinator.in_batch
        events.append("worker-cold")

    def exchanges(value):
        assert value is coordinator and coordinator.in_batch
        events.append("platform-exchanges")

    def after_batch(name):
        def register(value=None):
            assert not coordinator.in_batch
            events.append(name)

        return register

    monkeypatch.setattr(platform_dispatcher, "IMPORT_COORDINATOR", coordinator)
    monkeypatch.setattr(platform_dispatcher, "set_process_role", lambda role: None)
    monkeypatch.setattr(platform_dispatcher, "detect_process_role", lambda: "Main")
    monkeypatch.setattr(
        worker_dispatcher,
        "_register_worker_cold_replacements",
        cold,
    )
    monkeypatch.setattr(
        platform_dispatcher,
        "register_all_module_exchanges",
        exchanges,
    )
    monkeypatch.setattr(
        platform_dispatcher,
        "register_platform_core_callbacks",
        after_batch("platform-core"),
    )
    monkeypatch.setattr(
        platform_dispatcher,
        "register_tokenizer_callbacks",
        after_batch("tokenizer"),
    )
    monkeypatch.setattr(
        platform_dispatcher,
        "register_runtime_method_callbacks",
        after_batch("runtime"),
    )
    monkeypatch.setattr(
        platform_dispatcher,
        "_register_platform_framework_callbacks",
        after_batch("platform-framework"),
    )

    platform_dispatcher.apply_platform_patches()
    assert events == [
        "batch-enter",
        "install",
        "worker-cold",
        "platform-exchanges",
        "batch-exit",
        "platform-core",
        "tokenizer",
        "runtime",
        "platform-framework",
        "drain",
    ]


def test_apply_platform_patches_is_idempotent_narrow_and_reported():
    result = _run_fresh(
        "import builtins,json; "
        "old=builtins.__import__; "
        "from vllm_hcu.patch import "
        "apply_platform_patches,IMPORT_COORDINATOR,patch_report; "
        "apply_platform_patches(); apply_platform_patches(); "
        "regs=IMPORT_COORDINATOR.registrations(); "
        "print(json.dumps({'count':len(regs),"
        "'replacements':sum(x.action.value=='replacement' for x in regs),"
        "'callbacks':sum(x.action.value=='callback' for x in regs),"
        "'failed':[x.patch_id for x in regs if x.status=='failed'],"
        "'builtins_same':builtins.__import__ is old,"
        "'role':patch_report()['process_role']}))"
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload == {
        "count": 36,
        "replacements": 11,
        "callbacks": 25,
        "failed": [],
        "builtins_same": True,
        "role": "Main",
    }


def test_platform_framework_registration_is_lazy_on_clean_vllm():
    result = _run_fresh(
        "import builtins,json,sys; "
        "old=builtins.__import__; "
        "from vllm_hcu.patch import apply_platform_patches,IMPORT_COORDINATOR; "
        "from vllm_hcu.patch.platform import platform_framework_callback_names; "
        "targets={name for _,name in platform_framework_callback_names()}; "
        "cold_adapters={"
        "'vllm_hcu.patch.worker.op_opt.moe.patch_shared_experts',"
        "'vllm_hcu.patch.worker.op_opt.moe.patch_moe_runner',"
        "'vllm_hcu.patch.worker.op_opt.patch_aiter_ops',"
        "'vllm_hcu.patch.worker.framework_opt.patch_cuda_communicator'}; "
        "assert not (targets & sys.modules.keys()); "
        "assert not (cold_adapters & sys.modules.keys()); "
        "apply_platform_patches(); apply_platform_patches(); "
        "assert not (targets & sys.modules.keys()); "
        "cold_adapters_loaded=sorted(cold_adapters & sys.modules.keys()); "
        "from vllm_hcu.patch.worker import (worker_callback_names,"
        "worker_module_exchange_names); "
        "worker_replacements=worker_module_exchange_names(); "
        "replacement_modules={replacement for _,_,replacement in "
        "worker_replacements}; "
        "assert not (replacement_modules & sys.modules.keys()); "
        "assert not ({name for _,name in worker_callback_names()} & "
        "sys.modules.keys()); "
        "import vllm.v1.outputs; "
        "regs={r.patch_id:r for r in IMPORT_COORDINATOR.registrations()}; "
        "print(json.dumps({'output_status':regs["
        "'platform.framework_opt.outputs_draft_token_ids'].status,"
        "'cold_statuses':{patch_id:regs[patch_id].status for "
        "patch_id,_,_ in worker_replacements},"
        "'worker_callbacks_registered':any(patch_id in regs for "
        "patch_id,_ in worker_callback_names()),"
        "'cold_adapters_loaded':cold_adapters_loaded,"
        "'mooncake_loaded':"
        "'vllm_hcu.distributed.kv_transfer.kv_connector.v1.mooncake."
        "mooncake_connector' in sys.modules,"
        "'builtins_same':builtins.__import__ is old}))",
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload == {
        "output_status": "applied",
        "cold_statuses": {
            "worker.op_opt.moe.runner.shared_experts": "armed",
            "worker.op_opt.moe.runner": "armed",
            "worker.op_opt.aiter_ops.hcu_runtime": "armed",
            "worker.framework_opt.communicator.hcu_custom_allreduce_exchange": (
                "armed"
            ),
        },
        "worker_callbacks_registered": False,
        "cold_adapters_loaded": [],
        "mooncake_loaded": False,
        "builtins_same": True,
    }
