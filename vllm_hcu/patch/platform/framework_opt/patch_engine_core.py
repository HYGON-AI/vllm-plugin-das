# SPDX-License-Identifier: Apache-2.0
"""EngineCore adapters for process setup and DP connector registration."""

from __future__ import annotations

import functools
import importlib
import os
from types import ModuleType

from vllm_hcu.patch.runtime_state import (
    ProcessRole,
    detect_process_role,
    set_process_role,
)
from vllm_hcu.platforms import envs as henvs

from ._common import (
    PatchCompatibilityError,
    already_applied,
    load_exact_module,
    require_callable,
    require_class,
    require_signature_prefix,
)

TARGET_MODULE = "vllm.v1.engine.core"
PATCH_ID = "platform.framework_opt.engine_core"
TARGETS = (
    f"{TARGET_MODULE}.EngineCoreProc.__init__",
    f"{TARGET_MODULE}.EngineCoreProc._handle_client_request",
    f"{TARGET_MODULE}.EngineCore.post_step",
)
_MARKER = "_vllm_hcu_engine_core_applied"
_INIT_WRAPPER = "_vllm_hcu_engine_core_init_wrapper"
_REQUEST_WRAPPER = "_vllm_hcu_dp_request_wrapper"


def _set_engine_core_process_role() -> None:
    """Record an independent EngineCore without relabelling inproc Main.

    Only ``EngineCoreProc`` is wrapped.  vLLM's ``InprocClient`` constructs
    ``EngineCore`` directly and therefore retains the Main role in its shared
    PID.  An explicit environment override remains authoritative for process
    launchers with their own role contract.
    """

    if os.getenv("VLLM_HCU_PROCESS_ROLE"):
        set_process_role(detect_process_role())
    else:
        set_process_role(ProcessRole.ENGINE_CORE)


def _prepare_engine_core_runtime() -> None:
    _set_engine_core_process_role()
    # EngineCoreProc is the first cycle-free boundary after vllm.config and
    # vllm.v1.engine.core are fully initialized, but before general plugins,
    # model registration, or accelerator initialization.  Resolve the HCU
    # Worker here so ROCm-sensitive imports have a deterministic pre-device
    # order.  Importing it from the platform plugin itself is too early and
    # resolving worker_cls later is observably too late on DeepSeek-V2 MoE.
    importlib.import_module("vllm_hcu.v1.worker")
    from vllm_hcu.patch.worker import prepare_worker_patches

    prepare_worker_patches()


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    core = require_class(target, "EngineCore", f"{TARGET_MODULE}.EngineCore")
    proc = require_class(target, "EngineCoreProc", f"{TARGET_MODULE}.EngineCoreProc")
    request_type = getattr(target, "EngineCoreRequestType", None)
    if request_type is None or not hasattr(request_type, "ADD"):
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGET_MODULE}.EngineCoreRequestType.ADD is missing"
        )
    if already_applied(
        target,
        _MARKER,
        (
            (proc, "__init__", _INIT_WRAPPER),
            (proc, "_handle_client_request", _REQUEST_WRAPPER),
        ),
    ):
        return False
    proc_init = require_callable(proc, "__init__", TARGETS[0])
    handle = require_callable(proc, "_handle_client_request", TARGETS[1])
    post_step = require_callable(core, "post_step", TARGETS[2])
    require_signature_prefix(proc_init, TARGETS[0], ("self", "vllm_config"))
    require_signature_prefix(handle, TARGETS[1], ("self", "request_type", "request"))
    require_signature_prefix(post_step, TARGETS[2], ("self", "model_executed"))

    @functools.wraps(proc_init)
    def hcu_engine_core_proc_init(self, *args, **kwargs):
        _prepare_engine_core_runtime()
        try:
            return proc_init(self, *args, **kwargs)
        finally:
            # EngineCore.__init__ loads general plugins.  Reassert the exact
            # role after those defensive, best-effort platform callbacks.
            _set_engine_core_process_role()

    @functools.wraps(handle)
    def hcu_handle_client_request(self, request_type, request):
        _set_engine_core_process_role()
        if (
            henvs.VLLM_HCU_USE_DP_CONNECTOR
            and request_type == target.EngineCoreRequestType.ADD
        ):
            if not isinstance(request, tuple) or not request:
                raise RuntimeError(
                    "DP connector ADD request must be a non-empty tuple"
                )
            request_id = getattr(request[0], "request_id", None)
            if request_id is None:
                raise RuntimeError(
                    "DP connector ADD request is missing request_id"
                )
            connector = getattr(getattr(self, "scheduler", None), "connector", None)
            if connector is None:
                raise RuntimeError(
                    "VLLM_HCU_USE_DP_CONNECTOR is enabled but EngineCore has no connector"
                )
            register_req = getattr(connector, "register_req", None)
            if not callable(register_req):
                raise RuntimeError(
                    "selected DP connector does not implement required register_req"
                )
            register_req(request_id)
        return handle(self, request_type, request)

    setattr(hcu_engine_core_proc_init, _INIT_WRAPPER, True)
    setattr(hcu_handle_client_request, _REQUEST_WRAPPER, True)
    proc._vllm_hcu_original_init = proc_init
    proc._vllm_hcu_original_handle_client_request = handle
    proc.__init__ = hcu_engine_core_proc_init
    proc._handle_client_request = hcu_handle_client_request
    # ``post_step`` is deliberately validated but left untouched.  Clean
    # vLLM retrieves DraftTokenIds through model_executor.take_draft_token_ids
    # here and forwards them to scheduler.update_draft_token_ids.  Bypassing
    # this method for PP + multi-layer MTP would sever the supported IPC path.
    assert core.post_step is post_step
    setattr(target, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = [
    "PATCH_ID",
    "TARGET_MODULE",
    "TARGETS",
    "apply",
    "apply_to_module",
]
