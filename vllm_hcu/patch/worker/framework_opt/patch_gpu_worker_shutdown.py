# SPDX-License-Identifier: Apache-2.0
"""Keep the v0.25.1 GPU-worker shutdown contract usable on DCU.

vLLM v0.25.1 imports ``vllm.device_allocator.cumem`` at the end of every
CUDA-like worker shutdown in order to release an existing allocator singleton.
On DCU, an ordinary (non-sleep-mode) worker never imports that module, and the
shutdown-only probe can fail while constructing ``CudaRTLibrary`` because the
process does not have an upstream-named ``libamdhip64`` mapping.  When the
module was not loaded before shutdown, no ``CuMemAllocator`` singleton can
exist, so only that exact failed tail probe is an HCU no-op.

The target method remains responsible for every other lifecycle step.  This
adapter deliberately does not catch missing-runtime failures when cumem was
already loaded, failures from model-runner teardown, or any traceback shape
other than the audited v0.25.1 import probe.
"""

from __future__ import annotations

import functools
import sys
from types import ModuleType, TracebackType

from ._common import (
    PatchCompatibilityError,
    already_applied,
    load_exact_module,
    require_callable,
    require_class,
    require_exact_signature,
)

TARGET_MODULE = "vllm.v1.worker.gpu_worker"
PATCH_ID = "worker.framework_opt.lifecycle.dcu_gpu_worker_shutdown"
TARGETS = (f"{TARGET_MODULE}.Worker.shutdown",)

_CUMEM_MODULE = "vllm.device_allocator.cumem"
_CUDA_WRAPPER_MODULE = (
    "vllm.distributed.device_communicators.cuda_wrapper"
)
_MISSING_CUDART_MESSAGE = (
    "libcudart is not loaded in the current process, "
    "try setting VLLM_CUDART_SO_PATH"
)
_MARKER = "_vllm_hcu_dcu_shutdown_applied"
_WRAPPER = "_vllm_hcu_dcu_shutdown_wrapper"


def _next_frame(traceback: TracebackType | None) -> TracebackType | None:
    return None if traceback is None else traceback.tb_next


def _frame_matches(
    traceback: TracebackType | None,
    *,
    module_name: str,
    function_name: str,
) -> bool:
    if traceback is None:
        return False
    frame = traceback.tb_frame
    return (
        frame.f_globals.get("__name__") == module_name
        and frame.f_code.co_name == function_name
    )


def _is_unused_cumem_probe_failure(
    error: AssertionError,
    *,
    original_shutdown: object,
    current_platform: object,
    cumem_loaded_before_shutdown: bool,
) -> bool:
    """Recognize only the audited v0.25.1 shutdown-tail import failure."""

    if cumem_loaded_before_shutdown or error.args != (_MISSING_CUDART_MESSAGE,):
        return False

    is_rocm = getattr(current_platform, "is_rocm", None)
    if not callable(is_rocm) or not is_rocm():
        return False

    original_code = getattr(original_shutdown, "__code__", None)
    traceback = error.__traceback__
    while traceback is not None and traceback.tb_frame.f_code is not original_code:
        traceback = traceback.tb_next
    if traceback is None:
        return False

    # A failure raised by model_runner.shutdown() would have an intervening
    # frame here.  Requiring the direct module-execution frame proves this is
    # the final ``from ...cumem import CuMemAllocator`` in Worker.shutdown.
    cumem_frame = _next_frame(traceback)
    if not _frame_matches(
        cumem_frame, module_name=_CUMEM_MODULE, function_name="<module>"
    ):
        return False
    cuda_wrapper_frame = _next_frame(cumem_frame)
    if not _frame_matches(
        cuda_wrapper_frame,
        module_name=_CUDA_WRAPPER_MODULE,
        function_name="__init__",
    ):
        return False
    return _next_frame(cuda_wrapper_frame) is None


def apply_to_module(module: ModuleType) -> bool:
    gpu_worker = load_exact_module(TARGET_MODULE, module)
    worker = require_class(gpu_worker, "Worker", f"{TARGET_MODULE}.Worker")
    wrapped = ((worker, "shutdown", TARGETS[0], _WRAPPER),)
    if already_applied(gpu_worker, _MARKER, wrapped):
        return False

    original_shutdown = require_callable(worker, "shutdown", TARGETS[0])
    require_exact_signature(
        original_shutdown,
        TARGETS[0],
        positional=("self",),
    )
    original_code = getattr(original_shutdown, "__code__", None)
    required_names = {
        "current_platform",
        "is_cuda_alike",
        _CUMEM_MODULE,
        "CuMemAllocator",
        "instance",
        "release_pools",
    }
    if original_code is None or not required_names.issubset(original_code.co_names):
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGETS[0]} no longer contains the "
            "audited v0.25.1 cumem shutdown probe"
        )
    current_platform = getattr(gpu_worker, "current_platform", None)
    if current_platform is None:
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGET_MODULE}.current_platform is missing"
        )

    @functools.wraps(original_shutdown)
    def hcu_shutdown(self):
        cumem_loaded = _CUMEM_MODULE in sys.modules
        try:
            return original_shutdown(self)
        except AssertionError as error:
            if _is_unused_cumem_probe_failure(
                error,
                original_shutdown=original_shutdown,
                current_platform=current_platform,
                cumem_loaded_before_shutdown=cumem_loaded,
            ):
                return None
            raise

    setattr(hcu_shutdown, _WRAPPER, True)
    setattr(worker, "_vllm_hcu_original_shutdown", original_shutdown)
    setattr(worker, "shutdown", hcu_shutdown)
    setattr(gpu_worker, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
