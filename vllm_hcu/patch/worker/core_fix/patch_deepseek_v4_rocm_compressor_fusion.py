# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Keep DeepSeek-V4 compressor GEMMs separate for HCU NN weights."""

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

TARGET_MODULE = "vllm.models.deepseek_v4.amd.rocm"
PATCH_ID = "worker.core_fix.deepseek_v4_rocm.compressor_nn_weight_layout"
TARGET_SYMBOL = (
    f"{TARGET_MODULE}.DeepseekV4ROCMAiterMLAAttention."
    "prepare_compressor_gemm_fusion"
)
_MARKER = "_vllm_hcu_compressor_nn_weight_layout_applied"
_WRAPPER_MARKER = "_vllm_hcu_compressor_nn_weight_layout_wrapper"


def apply_to_module(module: ModuleType) -> bool:
    rocm = load_exact_module(TARGET_MODULE, module)
    attention_cls = require_class(
        rocm,
        "DeepseekV4ROCMAiterMLAAttention",
        f"{TARGET_MODULE}.DeepseekV4ROCMAiterMLAAttention",
    )
    original = require_callable(
        attention_cls,
        "prepare_compressor_gemm_fusion",
        TARGET_SYMBOL,
    )
    if getattr(rocm, _MARKER, False):
        if not getattr(original, _WRAPPER_MARKER, False):
            raise PatchCompatibilityError(
                f"required HCU patch marker for {TARGET_SYMBOL} is stale"
            )
        return False
    require_exact_signature(original, TARGET_SYMBOL, positional=("self",))

    @functools.wraps(original)
    def hcu_prepare_compressor_gemm_fusion(self) -> bool:
        if self._fused_compressor_weight is not None:
            return original(self)

        compressor = self.compressor
        indexer = self.indexer
        if compressor is None or indexer is None:
            return original(self)

        main_weight = compressor.fused_wkv_wgate.weight
        indexer_weight = indexer.compressor.fused_wkv_wgate.weight

        # Upstream fuses checkpoint-style NT [N, K] weights along N.  HCU's
        # unquantized NN loader stores both projections as [K, N], so fusing
        # them would require concatenating a different dimension and would no
        # longer match the upstream fused execution path.  Keep the two GEMMs
        # separate; the HCU DeepSeek-V4 projection patch handles this layout.
        hcu_nn_layout = (
            main_weight.ndim == 2
            and indexer_weight.ndim == 2
            and main_weight.shape[0] == indexer_weight.shape[0]
            and main_weight.shape[1] != indexer_weight.shape[1]
            and main_weight.dtype == indexer_weight.dtype
            and main_weight.device == indexer_weight.device
        )
        if hcu_nn_layout:
            return False
        return original(self)

    setattr(hcu_prepare_compressor_gemm_fusion, _WRAPPER_MARKER, True)
    setattr(
        attention_cls,
        "_vllm_hcu_original_prepare_compressor_gemm_fusion",
        original,
    )
    setattr(
        attention_cls,
        "prepare_compressor_gemm_fusion",
        hcu_prepare_compressor_gemm_fusion,
    )
    setattr(rocm, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "apply", "apply_to_module"]
