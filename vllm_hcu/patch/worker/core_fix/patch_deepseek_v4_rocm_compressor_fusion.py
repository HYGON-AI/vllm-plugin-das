# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Keep optional DeepSeek-V4 ROCm compressor fusion shape-safe."""

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
PATCH_ID = "worker.core_fix.deepseek_v4_rocm.compressor_fusion_shape"
TARGET_SYMBOL = (
    f"{TARGET_MODULE}.DeepseekV4ROCMAiterMLAAttention."
    "prepare_compressor_gemm_fusion"
)
_CLASS_MARKER = "_vllm_hcu_compressor_fusion_shape_applied"
_WRAPPER_MARKER = "_vllm_hcu_compressor_fusion_shape_wrapper"


def apply_to_module(module: ModuleType) -> bool:
    rocm = load_exact_module(TARGET_MODULE, module)
    cls = require_class(
        rocm,
        "DeepseekV4ROCMAiterMLAAttention",
        TARGET_SYMBOL,
    )
    original = require_callable(
        cls,
        "prepare_compressor_gemm_fusion",
        TARGET_SYMBOL,
    )
    if getattr(cls, _CLASS_MARKER, False):
        if not getattr(original, _WRAPPER_MARKER, False):
            raise PatchCompatibilityError(
                f"required HCU patch marker for {TARGET_SYMBOL} is stale"
            )
        return False
    require_exact_signature(original, TARGET_SYMBOL, positional=("self",))

    @functools.wraps(original)
    def hcu_prepare_compressor_gemm_fusion(self) -> bool:
        compressor = self.compressor
        indexer = self.indexer
        if compressor is not None and indexer is not None:
            main_weight = compressor.fused_wkv_wgate.weight
            indexer_weight = indexer.compressor.fused_wkv_wgate.weight
            if (
                main_weight.ndim == 2
                and indexer_weight.ndim == 2
                and main_weight.shape[1] != indexer_weight.shape[1]
            ):
                rocm.logger.warning_once(
                    "Skipping optional DeepSeek V4 compressor GEMM fusion "
                    "because main and indexer K differ (%d vs %d); this is "
                    "expected for heterogeneous MTP attention inputs.",
                    main_weight.shape[1],
                    indexer_weight.shape[1],
                )
                return False
        return original(self)

    setattr(hcu_prepare_compressor_gemm_fusion, _WRAPPER_MARKER, True)
    setattr(
        cls,
        "_vllm_hcu_original_prepare_compressor_gemm_fusion",
        original,
    )
    setattr(
        cls,
        "prepare_compressor_gemm_fusion",
        hcu_prepare_compressor_gemm_fusion,
    )
    setattr(cls, _CLASS_MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "apply", "apply_to_module"]
