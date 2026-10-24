# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Runtime-only HCU lowering replacement for the DeepSeek FP4 packer.

The upstream ``_fp32x2_to_fp4x2`` is a Triton JIT function whose body emits
``cvt.rn.satfinite.e2m1x2.f32`` inline assembly.  HCU clang cannot lower that
body.  This adapter rebinds the caller-module globals to an arithmetic Triton
JIT function, without editing the external vLLM source tree.
"""

from __future__ import annotations

import inspect
import os
from types import ModuleType

from vllm.triton_utils import tl, triton

from vllm_hcu.platforms import envs as henvs
from ._common import PatchCompatibilityError, load_exact_module


TARGET_MODULE = "vllm.models.deepseek_v41.common.ops.indexer_k_store"
PATCH_ID = "worker.core_fix.fp4_packer_lowering"
TARGETS = (f"{TARGET_MODULE}._fp32x2_to_fp4x2",)
_MARKER = "_vllm_hcu_fp4_packer_lowering_applied"
_STATUS = "_vllm_hcu_fp4_packer_lowering_status"
_ORIGINAL = "_vllm_hcu_fp4_packer_lowering_original"
_INLINE_ASM_SOURCE = (
    "inline_asm_elementwise",
    "cvt.rn.satfinite.e2m1x2.f32",
)


def e2m1_code_arithmetic_reference(x):
    """CPU reference for the replacement's threshold/sign semantics."""
    import torch

    ax = torch.minimum(torch.abs(x), torch.tensor(6.0, dtype=x.dtype))
    idx = torch.zeros_like(ax, dtype=torch.uint8)
    for boundary in (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0):
        idx += (ax >= boundary).to(torch.uint8)
    is_boundary = torch.zeros_like(ax, dtype=torch.bool)
    for boundary in (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0):
        is_boundary |= ax == boundary
    idx = torch.where(is_boundary & ((idx & 1) == 1), idx - 1, idx)
    sign = ((x.view(torch.int32) >> 31) & 1).to(torch.uint8)
    return idx | (sign << 3)


def e2m1_payload_reference(x_lo, x_hi):
    """CPU reference for low/high nibble order."""
    return e2m1_code_arithmetic_reference(x_lo) | (
        e2m1_code_arithmetic_reference(x_hi) << 4
    )


def _enabled() -> bool:
    # Read the environment at callback time.  Do not let lazy env attribute
    # materialization freeze the parent/child policy before worker config.
    child = os.environ.get("VLLM_HCU_USE_FP4_PACKER_LOWERING", "True")
    return henvs.optional_custom_op_enabled(child.lower() in ("true", "1"))


def _source_of(value: object) -> str:
    try:
        return inspect.getsource(value)
    except (OSError, TypeError):
        fn = getattr(value, "fn", None)
        if fn is None:
            return ""
        try:
            return inspect.getsource(fn)
        except (OSError, TypeError):
            return ""


def _is_audited_inline_asm(value: object) -> bool:
    source = _source_of(value)
    return callable(value) and all(marker in source for marker in _INLINE_ASM_SOURCE)


@triton.jit
def _e2m1_code_arithmetic(x):
    ax = tl.minimum(tl.abs(x), 6.0)
    idx = (ax >= 0.25).to(tl.uint8)
    idx += (ax >= 0.75).to(tl.uint8)
    idx += (ax >= 1.25).to(tl.uint8)
    idx += (ax >= 1.75).to(tl.uint8)
    idx += (ax >= 2.5).to(tl.uint8)
    idx += (ax >= 3.5).to(tl.uint8)
    idx += (ax >= 5.0).to(tl.uint8)
    boundary = (
        (ax == 0.25)
        | (ax == 0.75)
        | (ax == 1.25)
        | (ax == 1.75)
        | (ax == 2.5)
        | (ax == 3.5)
        | (ax == 5.0)
    )
    idx = tl.where(boundary & ((idx & 1) == 1), idx - 1, idx)
    # The inline conversion keeps negative zero.  Preserve the IEEE sign bit
    # for ordinary negatives, negative zero, and signed non-finite inputs.
    sign_bit = ((x.to(tl.int32, bitcast=True) >> 31) & 1).to(tl.uint8)
    sign = sign_bit
    return idx | (sign << 3)


@triton.jit
def _fp32x2_to_fp4x2_arithmetic(x_lo, x_hi):
    """Pure arithmetic equivalent of e2m1 RNE + satfinite conversion."""

    low = _e2m1_code_arithmetic(x_lo)
    high = _e2m1_code_arithmetic(x_hi)
    return low | (high << 4)


def _rebind(module: ModuleType, replacement: object) -> None:
    current = getattr(module, "_fp32x2_to_fp4x2", None)
    if current is None:
        raise PatchCompatibilityError(
            f"{module.__name__} has no _fp32x2_to_fp4x2 binding"
        )
    if not _is_audited_inline_asm(current):
        raise PatchCompatibilityError(
            f"{module.__name__} _fp32x2_to_fp4x2 is not the audited inline-asm JIT"
        )
    setattr(module, _ORIGINAL, current)
    setattr(module, "_fp32x2_to_fp4x2", replacement)


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    if getattr(target, _MARKER, False):
        return False
    if not _enabled():
        setattr(target, _STATUS, "OFF")
        return False

    original = getattr(target, "_fp32x2_to_fp4x2", None)
    if not _is_audited_inline_asm(original):
        setattr(target, _STATUS, "UPSTREAM_CHANGED")
        return False

    import importlib

    definition = importlib.import_module(
        "vllm.models.deepseek_v4.common.ops.fused_indexer_q"
    )
    v41_compress = importlib.import_module(
        "vllm.models.deepseek_v41.common.ops.fused_compress_quant_cache"
    )
    # Validate every binding before mutating any one of them: fail closed with
    # no partial rebinding if an upstream namespace has drifted.
    for owner in (definition, target, v41_compress):
        current = getattr(owner, "_fp32x2_to_fp4x2", None)
        if not _is_audited_inline_asm(current):
            setattr(target, _STATUS, "UPSTREAM_CHANGED")
            return False
    for owner in (definition, target, v41_compress):
        _rebind(owner, _fp32x2_to_fp4x2_arithmetic)
    setattr(target, _STATUS, "APPLIED")
    setattr(target, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = [
    "PATCH_ID",
    "TARGET_MODULE",
    "TARGETS",
    "apply",
    "apply_to_module",
    "_fp32x2_to_fp4x2_arithmetic",
]
