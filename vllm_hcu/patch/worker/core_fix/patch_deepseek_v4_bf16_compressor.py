# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Keep DeepSeek V4 compressor writes consistent with plain BF16 caches."""

from __future__ import annotations

import functools
from types import ModuleType

import torch

from vllm_hcu.v1.attention.ops.deepseek_v4_bf16_compressor import (
    compress_norm_rope_store_bf16,
)

from ._common import (
    PatchCompatibilityError,
    load_exact_module,
    require_callable,
    require_exact_signature,
)

TARGET_MODULE = "vllm.models.deepseek_v4.compressor"
PATCH_ID = "worker.core_fix.deepseek_v4.bf16_compressor_store"
TARGET_SYMBOL = f"{TARGET_MODULE}.compress_norm_rope_store_triton"
_MARKER = "_vllm_hcu_bf16_compressor_store_applied"
_WRAPPER_MARKER = "_vllm_hcu_bf16_compressor_store_wrapper"
_TWO_STAGE_WRAPPER_MARKER = "_vllm_hcu_bf16_compressor_two_stage_wrapper"

_COMMON_POSITIONAL = (
    "state_cache",
    "num_actual",
    "token_to_req_indices",
    "positions",
    "slot_mapping",
    "block_table",
    "block_size",
    "state_width",
    "cos_sin_cache",
    "kv_cache",
    "k_cache_metadata",
    "pdl_kwargs",
    "head_dim",
    "rope_head_dim",
    "compress_ratio",
    "overlap",
    "use_fp4_cache",
    "rms_norm_weight",
    "rms_norm_eps",
    "quant_block",
    "token_stride",
    "scale_dim",
)


def apply_to_module(module: ModuleType) -> bool:
    compressor = load_exact_module(TARGET_MODULE, module)
    original = require_callable(
        compressor,
        "compress_norm_rope_store_triton",
        TARGET_SYMBOL,
    )
    original_two_stage = require_callable(
        compressor,
        "compress_norm_rope_store_two_stage_triton",
        f"{TARGET_MODULE}.compress_norm_rope_store_two_stage_triton",
    )
    if getattr(compressor, _MARKER, False):
        if not getattr(original, _WRAPPER_MARKER, False) or not getattr(
            original_two_stage, _TWO_STAGE_WRAPPER_MARKER, False
        ):
            raise PatchCompatibilityError(
                f"required HCU patch marker for {TARGET_SYMBOL} is stale"
            )
        return False
    require_exact_signature(original, TARGET_SYMBOL, positional=_COMMON_POSITIONAL)
    require_exact_signature(
        original_two_stage,
        f"{TARGET_MODULE}.compress_norm_rope_store_two_stage_triton",
        positional=(*_COMMON_POSITIONAL, "num_decode_tokens", "compress_scratch"),
    )

    @functools.wraps(original)
    def hcu_compress_norm_rope_store_triton(
        state_cache,
        num_actual,
        token_to_req_indices,
        positions,
        slot_mapping,
        block_table,
        block_size,
        state_width,
        cos_sin_cache,
        kv_cache,
        k_cache_metadata,
        pdl_kwargs,
        head_dim,
        rope_head_dim,
        compress_ratio,
        overlap,
        use_fp4_cache,
        rms_norm_weight,
        rms_norm_eps,
        quant_block,
        token_stride,
        scale_dim,
    ):
        kwargs = {
            name: value
            for name, value in zip(
                _COMMON_POSITIONAL,
                (
                    state_cache,
                    num_actual,
                    token_to_req_indices,
                    positions,
                    slot_mapping,
                    block_table,
                    block_size,
                    state_width,
                    cos_sin_cache,
                    kv_cache,
                    k_cache_metadata,
                    pdl_kwargs,
                    head_dim,
                    rope_head_dim,
                    compress_ratio,
                    overlap,
                    use_fp4_cache,
                    rms_norm_weight,
                    rms_norm_eps,
                    quant_block,
                    token_stride,
                    scale_dim,
                ),
                strict=True,
            )
        }
        if kv_cache.dtype == torch.bfloat16:
            return compress_norm_rope_store_bf16(**kwargs)
        return original(**kwargs)

    @functools.wraps(original_two_stage)
    def hcu_compress_norm_rope_store_two_stage_triton(
        state_cache,
        num_actual,
        token_to_req_indices,
        positions,
        slot_mapping,
        block_table,
        block_size,
        state_width,
        cos_sin_cache,
        kv_cache,
        k_cache_metadata,
        pdl_kwargs,
        head_dim,
        rope_head_dim,
        compress_ratio,
        overlap,
        use_fp4_cache,
        rms_norm_weight,
        rms_norm_eps,
        quant_block,
        token_stride,
        scale_dim,
        num_decode_tokens,
        compress_scratch,
    ):
        kwargs = {
            name: value
            for name, value in zip(
                _COMMON_POSITIONAL,
                (
                    state_cache,
                    num_actual,
                    token_to_req_indices,
                    positions,
                    slot_mapping,
                    block_table,
                    block_size,
                    state_width,
                    cos_sin_cache,
                    kv_cache,
                    k_cache_metadata,
                    pdl_kwargs,
                    head_dim,
                    rope_head_dim,
                    compress_ratio,
                    overlap,
                    use_fp4_cache,
                    rms_norm_weight,
                    rms_norm_eps,
                    quant_block,
                    token_stride,
                    scale_dim,
                ),
                strict=True,
            )
        }
        if kv_cache.dtype == torch.bfloat16:
            return compress_norm_rope_store_bf16(**kwargs)
        return original_two_stage(
            **kwargs,
            num_decode_tokens=num_decode_tokens,
            compress_scratch=compress_scratch,
        )

    setattr(hcu_compress_norm_rope_store_triton, _WRAPPER_MARKER, True)
    setattr(
        hcu_compress_norm_rope_store_two_stage_triton,
        _TWO_STAGE_WRAPPER_MARKER,
        True,
    )
    setattr(
        compressor,
        "_vllm_hcu_original_compress_norm_rope_store_triton",
        original,
    )
    setattr(
        compressor,
        "_vllm_hcu_original_compress_norm_rope_store_two_stage_triton",
        original_two_stage,
    )
    setattr(
        compressor,
        "compress_norm_rope_store_triton",
        hcu_compress_norm_rope_store_triton,
    )
    setattr(
        compressor,
        "compress_norm_rope_store_two_stage_triton",
        hcu_compress_norm_rope_store_two_stage_triton,
    )
    setattr(compressor, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "apply", "apply_to_module"]
