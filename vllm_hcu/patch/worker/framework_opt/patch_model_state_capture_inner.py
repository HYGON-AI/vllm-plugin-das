# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Add ModelState.capture_inner_cudagraphs noop when missing (upstream #59532)."""

from __future__ import annotations

from types import ModuleType
from typing import Any

from ._common import (
    PatchCompatibilityError,
    load_exact_module,
    require_class,
)

TARGET_MODULE = "vllm.v1.worker.gpu.model_states.interface"
PATCH_ID = "worker.framework_opt.model_state.capture_inner_cudagraphs"
TARGETS = (f"{TARGET_MODULE}.ModelState.capture_inner_cudagraphs",)
_MARKER = "_vllm_hcu_model_state_capture_inner_applied"


def apply_to_module(module: ModuleType) -> bool:
    mod = load_exact_module(TARGET_MODULE, module)
    cls = require_class(mod, "ModelState", f"{TARGET_MODULE}.ModelState")
    if getattr(cls, _MARKER, False):
        return False

    existing = getattr(cls, "capture_inner_cudagraphs", None)
    if callable(existing):
        setattr(cls, _MARKER, True)
        return False

    def capture_inner_cudagraphs(
        self,
        input_buffers: Any,
        block_tables: Any,
        attn_groups: Any,
        kv_cache_config: Any,
    ) -> None:
        """Capture the CUDA graphs this state runs inside the model's forward."""
        return None

    cls.capture_inner_cudagraphs = capture_inner_cudagraphs
    setattr(cls, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
