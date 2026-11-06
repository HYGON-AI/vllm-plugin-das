# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Allow torch FP8 scaled-mm on gfx938 without mutating vLLM sources."""

from __future__ import annotations

import functools
from types import ModuleType

from ._common import (
    already_applied,
    load_exact_module,
    require_callable,
    require_exact_signature,
)

TARGET_MODULE = "vllm.model_executor.kernels.linear.scaled_mm.pytorch"
PATCH_ID = "worker.op_opt.scaled_mm.rocm_gfx938_torch_fp8"
TARGETS = (f"{TARGET_MODULE}._rocm_torch_fp8_scaled_mm_supported",)
_MODULE_MARKER = "_vllm_hcu_rocm_gfx938_torch_fp8_applied"
_WRAPPER_MARKER = "_vllm_hcu_rocm_gfx938_torch_fp8_wrapper"


def apply_to_module(module: ModuleType) -> bool:
    pytorch_scaled_mm = load_exact_module(TARGET_MODULE, module)
    if already_applied(
        pytorch_scaled_mm,
        _MODULE_MARKER,
        (
            (
                pytorch_scaled_mm,
                "_rocm_torch_fp8_scaled_mm_supported",
                TARGETS[0],
                _WRAPPER_MARKER,
            ),
        ),
    ):
        return False

    original = require_callable(
        pytorch_scaled_mm,
        "_rocm_torch_fp8_scaled_mm_supported",
        TARGETS[0],
    )
    require_exact_signature(original, TARGETS[0])

    @functools.wraps(original)
    def hcu_rocm_torch_fp8_scaled_mm_supported() -> bool:
        if original():
            return True
        try:
            from vllm.platforms.rocm import _GCN_ARCH
        except (AttributeError, ImportError):
            return False

        return "gfx938" in _GCN_ARCH

    setattr(hcu_rocm_torch_fp8_scaled_mm_supported, _WRAPPER_MARKER, True)
    setattr(
        pytorch_scaled_mm,
        "_vllm_hcu_original_rocm_torch_fp8_scaled_mm_supported",
        original,
    )
    setattr(
        pytorch_scaled_mm,
        "_rocm_torch_fp8_scaled_mm_supported",
        hcu_rocm_torch_fp8_scaled_mm_supported,
    )
    setattr(pytorch_scaled_mm, _MODULE_MARKER, True)
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
