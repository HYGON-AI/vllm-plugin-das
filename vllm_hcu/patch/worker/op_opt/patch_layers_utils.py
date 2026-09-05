# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Use the HCU-owned linear layer's default GEMM dispatch on ROCm."""

from __future__ import annotations

import functools
from types import ModuleType

from ._common import (
    already_applied,
    load_exact_module,
    require_callable,
    require_exact_signature,
)

TARGET_MODULE = "vllm.model_executor.layers.utils"
PATCH_ID = "worker.op_opt.gemm.default_unquantized_on_hcu"
TARGETS = (f"{TARGET_MODULE}.dispatch_unquantized_gemm",)
_MODULE_MARKER = "_vllm_hcu_default_gemm_dispatch_applied"
_WRAPPER_MARKER = "_vllm_hcu_default_gemm_dispatch_wrapper"


def apply_to_module(module: ModuleType) -> bool:
    layer_utils = load_exact_module(TARGET_MODULE, module)
    if already_applied(
        layer_utils,
        _MODULE_MARKER,
        ((layer_utils, "dispatch_unquantized_gemm", TARGETS[0], _WRAPPER_MARKER),),
    ):
        return False

    original = require_callable(layer_utils, "dispatch_unquantized_gemm", TARGETS[0])
    require_exact_signature(
        original,
        TARGETS[0],
        positional=("linear_backend",),
        defaults={"linear_backend": "auto"},
    )
    default_gemm = require_callable(
        layer_utils,
        "default_unquantized_gemm",
        f"{TARGET_MODULE}.default_unquantized_gemm",
    )

    @functools.wraps(original)
    def hcu_dispatch_unquantized_gemm(linear_backend="auto"):
        if layer_utils.current_platform.is_rocm():
            return default_gemm
        return original(linear_backend)

    setattr(hcu_dispatch_unquantized_gemm, _WRAPPER_MARKER, True)
    setattr(layer_utils, "_vllm_hcu_original_dispatch_unquantized_gemm", original)
    setattr(layer_utils, "dispatch_unquantized_gemm", hcu_dispatch_unquantized_gemm)
    setattr(layer_utils, _MODULE_MARKER, True)
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
