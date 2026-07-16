# SPDX-License-Identifier: Apache-2.0
"""Capability-gated RCCL ``ncclAllToAll`` binding for vLLM's ctypes wrapper."""

from __future__ import annotations

import ctypes
from types import ModuleType

from ._common import (
    PatchCompatibilityError,
    load_exact_module,
    require_callable,
    require_class,
    require_exact_signature,
)

TARGET_MODULE = "vllm.distributed.device_communicators.pynccl_wrapper"
PATCH_ID = "worker.framework_opt.communicator.pynccl_wrapper_all_to_all"
TARGETS = (
    f"{TARGET_MODULE}.NCCLLibrary.ncclAllToAll",
    f"{TARGET_MODULE}.NCCLLibrary.exported_functions[ncclAllToAll]",
)
_MARKER = "_vllm_hcu_pynccl_all_to_all_applied"
_PROBE_MARKER = "_vllm_hcu_pynccl_all_to_all_probe"
_WRAPPER = "_vllm_hcu_pynccl_all_to_all_wrapper"


def _unavailable(module: ModuleType, reason: str, required: bool) -> bool:
    setattr(module, _PROBE_MARKER, reason)
    if required:
        raise RuntimeError(
            "PyNccl all-to-all was explicitly requested but is unavailable: " + reason
        )
    return False


def _probe_rccl_symbol(module: ModuleType) -> tuple[bool, str]:
    platform = getattr(module, "current_platform", None)
    if platform is None or not callable(getattr(platform, "is_rocm", None)):
        return False, "vLLM platform does not expose the audited ROCm capability"
    if not platform.is_rocm():
        return False, "ncclAllToAll is an HCU/RCCL-only optional capability"
    try:
        so_file = module.find_nccl_library()
        library = ctypes.CDLL(so_file)
    except Exception as exc:
        return False, f"RCCL library could not be loaded ({type(exc).__name__}: {exc})"
    try:
        getattr(library, "ncclAllToAll")
    except AttributeError:
        return False, f"library {so_file!r} does not export ncclAllToAll"
    return True, str(so_file)


def apply_to_module(module: ModuleType, *, required: bool = False) -> bool:
    wrapper = load_exact_module(TARGET_MODULE, module)
    library_class = require_class(
        wrapper, "NCCLLibrary", f"{TARGET_MODULE}.NCCLLibrary"
    )
    if getattr(wrapper, _MARKER, False):
        method = require_callable(library_class, "ncclAllToAll", TARGETS[0])
        if not getattr(method, _WRAPPER, False):
            raise PatchCompatibilityError(
                f"required HCU patch marker for {TARGETS[0]} is stale"
            )
        return False
    previous_probe = getattr(wrapper, _PROBE_MARKER, None)
    if previous_probe is not None:
        return _unavailable(wrapper, str(previous_probe), required)

    init = require_callable(library_class, "__init__", f"{TARGET_MODULE}.NCCLLibrary.__init__")
    require_exact_signature(
        init,
        f"{TARGET_MODULE}.NCCLLibrary.__init__",
        positional=("self", "so_file"),
        defaults={"so_file": None},
    )
    send = require_callable(library_class, "ncclSend", f"{TARGET_MODULE}.NCCLLibrary.ncclSend")
    require_exact_signature(
        send,
        f"{TARGET_MODULE}.NCCLLibrary.ncclSend",
        positional=("self", "sendbuff", "count", "datatype", "dest", "comm", "stream"),
    )
    if "ncclAllToAll" in vars(library_class):
        raise PatchCompatibilityError(f"audited v0.21 target {TARGETS[0]} unexpectedly exists")
    exported = getattr(library_class, "exported_functions", None)
    if not isinstance(exported, list):
        raise PatchCompatibilityError(f"required HCU patch target {TARGETS[1]} is missing")
    if any(getattr(function, "name", None) == "ncclAllToAll" for function in exported):
        raise PatchCompatibilityError(f"audited v0.21 target {TARGETS[1]} unexpectedly exists")

    # Extending exported_functions after a path has been cached would leave an
    # NCCLLibrary instance whose _funcs dictionary cannot contain the symbol.
    if library_class.path_to_library_cache or library_class.path_to_dict_mapping:
        return _unavailable(
            wrapper,
            "NCCLLibrary cache was created before HCU capability registration",
            required,
        )
    available, detail = _probe_rccl_symbol(wrapper)
    if not available:
        return _unavailable(wrapper, detail, required)

    function_type = require_class(wrapper, "Function", f"{TARGET_MODULE}.Function")
    exported.append(
        function_type(
            "ncclAllToAll",
            wrapper.ncclResult_t,
            [
                wrapper.buffer_type,
                wrapper.buffer_type,
                ctypes.c_size_t,
                wrapper.ncclDataType_t,
                wrapper.ncclComm_t,
                wrapper.cudaStream_t,
            ],
        )
    )

    def nccl_all_to_all(self, sendbuff, recvbuff, count, datatype, comm, stream):
        self.NCCL_CHECK(
            self._funcs["ncclAllToAll"](
                sendbuff, recvbuff, count, datatype, comm, stream
            )
        )

    setattr(nccl_all_to_all, _WRAPPER, True)
    setattr(library_class, "ncclAllToAll", nccl_all_to_all)
    setattr(wrapper, _PROBE_MARKER, f"available:{detail}")
    setattr(wrapper, _MARKER, True)
    return True


def apply(module: ModuleType | None = None, *, required: bool = False) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module), required=required)


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
