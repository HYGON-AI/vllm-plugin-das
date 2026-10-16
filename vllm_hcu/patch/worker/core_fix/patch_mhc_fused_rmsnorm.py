# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Optionally fuse the mHC trailing RMSNorm into one aten kernel.

``_apply_mhc_norm`` routes through ``vllm.ir.ops.rms_norm``; on HCU no fused
IR provider is registered, so the native implementation decomposes into 8
elementwise/reduce kernels (to/pow/mean/add/rsqrt/mul/to/mul) after every mHC
pre / fused-post-pre kernel. With VLLM_HCU_MHC_FUSED_RMSNORM=1 this patch
replaces it with a single ``torch.nn.functional.rms_norm`` call.

Scope: module-level ``_apply_mhc_norm`` is shared by every model using
MHCPreOp / MHCFusedPostPreOp on the native mHC fallback path (GLM5Next,
DeepSeek V4, ...); enabling the switch affects all of them. The AITER and
TileLang mHC paths never call ``_apply_mhc_norm`` and are unaffected.

Numerics: fp32 normalization matches the native path; the weight multiply is
rounded once instead of twice, so outputs differ by <= 1 ulp of the input
dtype. When ``norm_weight.dtype != layer_input.dtype`` the wrapper falls back
to the original implementation, because aten's rms_norm would promote the
output dtype in that case.

The switch is enabled by default and can be disabled by setting it to 0. It is
consulted once, at patch-apply time; when off the module is left untouched.
"""

from __future__ import annotations

from types import ModuleType

from ._common import (
    PatchCompatibilityError,
    load_exact_module,
    require_callable,
)

TARGET_MODULE = "vllm.model_executor.layers.mhc"
PATCH_ID = "worker.core_fix.mhc_fused_rmsnorm"
TARGET_SYMBOL = f"{TARGET_MODULE}._apply_mhc_norm"
_MARKER = "_vllm_hcu_mhc_fused_rmsnorm_applied"
_WRAPPER = "_vllm_hcu_mhc_fused_rmsnorm_wrapper"


def apply_to_module(module: ModuleType) -> bool:
    mhc = load_exact_module(TARGET_MODULE, module)
    original = require_callable(mhc, "_apply_mhc_norm", TARGET_SYMBOL)
    if getattr(mhc, _MARKER, False):
        if not getattr(original, _WRAPPER, False):
            raise PatchCompatibilityError(
                f"required HCU patch marker for {TARGET_SYMBOL} is stale"
            )
        return False

    import vllm_hcu.platforms.envs as henvs

    if not henvs.VLLM_HCU_MHC_FUSED_RMSNORM or not henvs.VLLM_HCU_USE_CUSTOM_OPS:
        # Leave the module untouched when the feature or its master switch is off.
        return False

    import torch

    def _fused_apply_mhc_norm(layer_input, norm_weight, norm_eps):
        if norm_weight is None:
            return layer_input
        if norm_weight.dtype != layer_input.dtype:
            # aten rms_norm would promote the output dtype; keep the
            # original two-step rounding semantics instead.
            return original(layer_input, norm_weight, norm_eps)
        return torch.nn.functional.rms_norm(
            layer_input, (layer_input.shape[-1],), norm_weight, norm_eps
        )

    setattr(_fused_apply_mhc_norm, _WRAPPER, True)
    mhc._vllm_hcu_original_apply_mhc_norm = original
    mhc._apply_mhc_norm = _fused_apply_mhc_norm
    setattr(mhc, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGET_SYMBOL", "apply", "apply_to_module"]
