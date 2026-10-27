# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Call ModelState.capture_inner_cudagraphs after decoder CUDA-graph capture.

Mirrors upstream GPUModelRunner.capture_model (#59532): after
``cudagraph_manager.capture(...)`` succeeds, capture YOCO decoder-replay
inner graphs before speculative / adaptive verification captures.
"""

from __future__ import annotations

import functools
from types import ModuleType

from ._common import (
    PatchCompatibilityError,
    load_exact_module,
    require_callable,
    require_class,
    require_exact_signature,
)

TARGET_MODULE = "vllm.v1.worker.gpu.model_runner"
PATCH_ID = "worker.framework_opt.gpu_model_runner.capture_inner_cudagraphs"
TARGETS = (f"{TARGET_MODULE}.GPUModelRunner.capture_model",)
_MARKER = "_vllm_hcu_gpu_model_runner_capture_inner_applied"
_WRAPPER = "_vllm_hcu_gpu_model_runner_capture_inner_wrapper"


def apply_to_module(module: ModuleType) -> bool:
    mod = load_exact_module(TARGET_MODULE, module)
    cls = require_class(mod, "GPUModelRunner", f"{TARGET_MODULE}.GPUModelRunner")
    if getattr(cls, _MARKER, False):
        capture = require_callable(cls, "capture_model", TARGETS[0])
        if not getattr(capture, _WRAPPER, False):
            raise PatchCompatibilityError(f"stale marker for {PATCH_ID}")
        return False

    original = require_callable(cls, "capture_model", TARGETS[0])
    require_exact_signature(
        original,
        TARGETS[0],
        positional=("self",),
        keyword_only=("profile_only",),
        defaults={"profile_only": False},
    )

    @functools.wraps(original)
    def capture_model(self, *, profile_only: bool = False) -> int:
        manager = self.cudagraph_manager
        if manager is None:
            return original(self, profile_only=profile_only)

        original_manager_capture = manager.capture

        @functools.wraps(original_manager_capture)
        def capture_then_inner(*args, **kwargs):
            result = original_manager_capture(*args, **kwargs)
            # ModelCudaGraphManager.capture(model, model_state, input_buffers,
            # intermediate_tensors, block_tables, attn_groups, kv_cache_config, ...)
            input_buffers = args[2] if len(args) > 2 else kwargs["input_buffers"]
            block_tables = args[4] if len(args) > 4 else kwargs["block_tables"]
            attn_groups = args[5] if len(args) > 5 else kwargs["attn_groups"]
            kv_cache_config = (
                args[6] if len(args) > 6 else kwargs["kv_cache_config"]
            )
            self.model_state.capture_inner_cudagraphs(
                input_buffers,
                block_tables,
                attn_groups,
                kv_cache_config,
            )
            return result

        manager.capture = capture_then_inner
        try:
            return original(self, profile_only=profile_only)
        finally:
            manager.capture = original_manager_capture

    setattr(capture_model, _WRAPPER, True)
    cls.capture_model = capture_model
    setattr(cls, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
