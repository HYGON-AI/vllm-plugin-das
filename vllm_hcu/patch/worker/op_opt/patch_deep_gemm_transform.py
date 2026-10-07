# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Provide fallback for transform_sf_into_required_layout without DeepGEMM."""

from __future__ import annotations

import functools
from types import ModuleType

from ._common import (
    already_applied,
    load_exact_module,
    require_callable,
)

TARGET_MODULE = "vllm.utils.deep_gemm"
PATCH_ID = "worker.op_opt.deep_gemm.transform_sf_fallback"
TARGETS = (f"{TARGET_MODULE}.transform_sf_into_required_layout",)
_MODULE_MARKER = "_vllm_hcu_transform_sf_fallback_applied"
_WRAPPER_MARKER = "_vllm_hcu_transform_sf_fallback_wrapper"


def apply_to_module(module: ModuleType) -> bool:
    deep_gemm = load_exact_module(TARGET_MODULE, module)
    if already_applied(
        deep_gemm,
        _MODULE_MARKER,
        ((deep_gemm, "transform_sf_into_required_layout", TARGETS[0], _WRAPPER_MARKER),),
    ):
        return False

    original = require_callable(deep_gemm, "transform_sf_into_required_layout", TARGETS[0])
    lazy_init = require_callable(deep_gemm, "_lazy_init", f"{TARGET_MODULE}._lazy_init")

    @functools.wraps(original)
    def hcu_transform_sf_into_required_layout(*args, **kwargs):
        lazy_init()
        if getattr(deep_gemm, "_transform_sf_into_required_layout_impl", None) is not None:
            return original(*args, **kwargs)

        # Fallback implementation for HCU: return the scale tensor as-is
        # since DeepGEMM-specific layout transformation is not available
        sf = kwargs.get('sf', args[0] if args else None)
        if sf is None:
            raise ValueError("transform_sf_into_required_layout requires 'sf' parameter")

        # Simply return the input scale tensor without transformation
        # This works for HCU custom kernels that don't require DeepGEMM layout
        return sf

    setattr(hcu_transform_sf_into_required_layout, _WRAPPER_MARKER, True)
    setattr(deep_gemm, "_vllm_hcu_original_transform_sf_into_required_layout", original)
    setattr(deep_gemm, "transform_sf_into_required_layout", hcu_transform_sf_into_required_layout)
    setattr(deep_gemm, _MODULE_MARKER, True)
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
