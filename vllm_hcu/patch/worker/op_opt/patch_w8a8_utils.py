# SPDX-License-Identifier: Apache-2.0
"""Register HCU's INT8 Marlin2 layout transform as a normal function."""

from __future__ import annotations

from types import ModuleType

from vllm_hcu.model_executor.layers.quantization.int8_runtime import (
    weight8bit_nt_kpack2_marlin2,
)

from ._common import (
    PatchCompatibilityError,
    already_applied,
    load_exact_module,
)

TARGET_MODULE = "vllm.model_executor.layers.quantization.utils.w8a8_utils"
PATCH_ID = "worker.op_opt.quantization.weight8bit_marlin2_layout"
TARGETS = (f"{TARGET_MODULE}.weight8bit_nt_kpack2_marlin2",)
_MODULE_MARKER = "_vllm_hcu_weight8bit_marlin2_applied"
_FUNCTION_MARKER = "_vllm_hcu_weight8bit_marlin2_function"


def apply_to_module(module: ModuleType) -> bool:
    w8a8_utils = load_exact_module(TARGET_MODULE, module)
    if already_applied(
        w8a8_utils,
        _MODULE_MARKER,
        ((w8a8_utils, "weight8bit_nt_kpack2_marlin2", TARGETS[0], _FUNCTION_MARKER),),
    ):
        return False
    if hasattr(w8a8_utils, "weight8bit_nt_kpack2_marlin2"):
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGETS[0]} unexpectedly already exists"
        )

    setattr(weight8bit_nt_kpack2_marlin2, _FUNCTION_MARKER, True)
    setattr(w8a8_utils, "weight8bit_nt_kpack2_marlin2", weight8bit_nt_kpack2_marlin2)
    setattr(w8a8_utils, _MODULE_MARKER, True)
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
