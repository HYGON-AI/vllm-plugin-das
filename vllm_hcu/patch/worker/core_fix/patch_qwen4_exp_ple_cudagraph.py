# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Join Qwen4Exp PLE prefetch work before piecewise graph capture."""

from __future__ import annotations

import functools
import os
from types import ModuleType

from ._common import (
    PatchCompatibilityError,
    load_exact_module,
    require_callable,
    require_class,
)

TARGET_MODULE = "vllm.compilation.cuda_graph"
PATCH_ID = "worker.core_fix.qwen4_exp.ple_prefetch_cudagraph"
TARGETS = (f"{TARGET_MODULE}.CUDAGraphWrapper.__call__",)
_MARKER = "_vllm_hcu_qwen4_exp_ple_cudagraph_applied"
_WRAPPER = "_vllm_hcu_qwen4_exp_ple_cudagraph_wrapper"


def _requested() -> bool:
    return os.environ.get("VLLM_HCU_PLE_PREFETCH_STREAM", "False").lower() in (
        "true",
        "1",
    )


def _will_capture(cudagraph_module: ModuleType, wrapper: object) -> bool:
    if not cudagraph_module.is_forward_context_available():
        return False
    forward_context = cudagraph_module.get_forward_context()
    runtime_mode = forward_context.cudagraph_runtime_mode
    if (
        runtime_mode == cudagraph_module.CUDAGraphMode.NONE
        or runtime_mode != wrapper.runtime_mode
    ):
        return False
    batch_descriptor = forward_context.batch_descriptor
    if batch_descriptor is None:
        return False
    entry = wrapper.concrete_cudagraph_entries.get(batch_descriptor)
    return entry is None or entry.cudagraph is None


def _validate_call(call: object) -> None:
    call_code = getattr(call, "__code__", None)
    required_names = {
        "is_forward_context_available",
        "get_forward_context",
        "concrete_cudagraph_entries",
        "CUDAGraphEntry",
        "cudagraph",
    }
    if call_code is None or not required_names.issubset(call_code.co_names):
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGETS[0]} no longer contains the "
            "audited piecewise graph capture path"
        )


def apply_to_module(module: ModuleType) -> bool:
    cudagraph_module = load_exact_module(TARGET_MODULE, module)
    if not _requested():
        return False
    wrapper_class = require_class(
        cudagraph_module,
        "CUDAGraphWrapper",
        f"{TARGET_MODULE}.CUDAGraphWrapper",
    )
    call = require_callable(wrapper_class, "__call__", TARGETS[0])

    if getattr(cudagraph_module, _MARKER, False):
        if not getattr(call, _WRAPPER, False):
            raise PatchCompatibilityError(
                f"required HCU patch marker for {TARGETS[0]} is stale"
            )
        return False
    if getattr(call, _WRAPPER, False):
        raise PatchCompatibilityError(
            f"refusing a partial Qwen4Exp PLE graph-boundary patch for {TARGETS[0]}"
        )

    _validate_call(call)

    @functools.wraps(call)
    def hcu_call(self, *args, **kwargs):
        if not _will_capture(cudagraph_module, self):
            return call(self, *args, **kwargs)

        from .patch_qwen4_exp_ple_prefetch import join_pending_prefetches

        # Drain work launched before this wrapper starts capture.  This is
        # necessary for an outer model capture that fired PLE prefetch before
        # entering an inner compiled model wrapper.
        join_pending_prefetches()

        # CUDAGraphWrapper owns the ``torch.cuda.graph`` context internally.
        # Temporarily proxy its runnable so that the side stream is joined
        # after the runnable returns but before the wrapper exits capture.
        # A pre-capture wait alone cannot close work forked by hcu_forward
        # during the active graph, which ROCm reports as
        # hipErrorStreamCaptureUnjoined at capture_end().
        runnable = self.runnable

        @functools.wraps(runnable)
        def runnable_with_ple_join(*runnable_args, **runnable_kwargs):
            output = runnable(*runnable_args, **runnable_kwargs)
            join_pending_prefetches()
            return output

        self.runnable = runnable_with_ple_join
        try:
            return call(self, *args, **kwargs)
        finally:
            self.runnable = runnable

    setattr(hcu_call, _WRAPPER, True)
    wrapper_class.__call__ = hcu_call
    setattr(cudagraph_module, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
