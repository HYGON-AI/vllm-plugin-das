# SPDX-License-Identifier: Apache-2.0
"""Register HCU's INT8 Marlin2 layout transform as a normal function."""

from __future__ import annotations

from types import ModuleType

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

    def weight8bit_nt_kpack2_marlin2(
        weight,
        k_tile=16,
        k_tile1=4,
        n_tile=16,
    ):
        if getattr(weight, "element_size", lambda: None)() != 1:
            raise ValueError("weight8bit_nt_kpack2_marlin2 requires an 8-bit tensor")
        if any(
            not isinstance(value, int) or value <= 0
            for value in (k_tile, k_tile1, n_tile)
        ):
            raise ValueError("k_tile, k_tile1, and n_tile must be positive integers")
        if n_tile != k_tile:
            raise ValueError(
                "the audited Marlin2 layout requires n_tile == k_tile"
            )
        if weight.dim() not in (2, 3):
            raise ValueError("weight8bit_nt_kpack2_marlin2 supports rank 2 or 3")

        size_n, size_k = weight.shape[-2:]
        if size_n % n_tile != 0 or size_k % (k_tile * k_tile1) != 0:
            raise ValueError(
                "weight dimensions must be divisible by n_tile and "
                "k_tile * k_tile1"
            )
        if weight.dim() == 2:
            packed = weight.reshape(
                size_n // n_tile,
                n_tile,
                size_k // (k_tile * k_tile1),
                k_tile1,
                k_tile,
            )
            packed = packed.permute(2, 0, 3, 1, 4).contiguous()
            return packed.reshape(size_n // k_tile, size_k * k_tile)

        experts = weight.shape[0]
        packed = weight.reshape(
            experts,
            size_n // n_tile,
            n_tile,
            size_k // (k_tile * k_tile1),
            k_tile1,
            k_tile,
        )
        packed = packed.permute(0, 3, 1, 4, 2, 5).contiguous()
        return packed.reshape(experts, size_n // k_tile, size_k * k_tile)

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
