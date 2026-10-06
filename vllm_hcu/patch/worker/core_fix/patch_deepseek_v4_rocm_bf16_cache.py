# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Route DeepSeek V4 plain BF16 caches around packed-FP8 kernels."""

from __future__ import annotations

import functools
from types import ModuleType

import torch

from vllm_hcu.v1.attention.ops.deepseek_v4_bf16_cache import (
    bf16_sparse_attn_decode,
    gather_bf16_k_cache,
)

from ._common import (
    PatchCompatibilityError,
    load_exact_module,
    require_callable,
    require_exact_signature,
)

TARGET_MODULE = "vllm.models.deepseek_v4.amd.rocm"
PATCH_ID = "worker.core_fix.deepseek_v4_rocm.bf16_cache_gather"
TARGET_SYMBOL = f"{TARGET_MODULE}.dequantize_and_gather_k_cache"
_MARKER = "_vllm_hcu_bf16_cache_gather_applied"
_WRAPPER_MARKER = "_vllm_hcu_bf16_cache_gather_wrapper"
_DECODE_WRAPPER_MARKER = "_vllm_hcu_bf16_cache_decode_wrapper"


def apply_to_module(module: ModuleType) -> bool:
    rocm = load_exact_module(TARGET_MODULE, module)
    original = require_callable(
        rocm,
        "dequantize_and_gather_k_cache",
        TARGET_SYMBOL,
    )
    if getattr(rocm, _MARKER, False):
        decode = getattr(rocm, "rocm_sparse_attn_decode", None)
        if not getattr(original, _WRAPPER_MARKER, False) or not getattr(
            decode, _DECODE_WRAPPER_MARKER, False
        ):
            raise PatchCompatibilityError(
                f"required HCU patch marker for {TARGET_SYMBOL} is stale"
            )
        return False
    require_exact_signature(
        original,
        TARGET_SYMBOL,
        positional=(
            "out",
            "k_cache",
            "seq_lens",
            "gather_lens",
            "block_table",
            "block_size",
            "offset",
            "use_fnuz",
        ),
        defaults={"use_fnuz": False},
    )
    original_decode = require_callable(
        rocm,
        "rocm_sparse_attn_decode",
        f"{TARGET_MODULE}.rocm_sparse_attn_decode",
    )
    require_exact_signature(
        original_decode,
        f"{TARGET_MODULE}.rocm_sparse_attn_decode",
        positional=(
            "q",
            "kv_cache",
            "swa_k_cache",
            "swa_only",
            "topk_indices",
            "topk_lens",
            "swa_indices",
            "swa_lens",
            "swa_ragged_indices",
            "swa_ragged_indptr",
            "topk_ragged_indices",
            "topk_ragged_indptr",
            "attn_sink",
            "scale",
            "head_dim",
            "nope_head_dim",
            "rope_head_dim",
            "output",
            "extra_cache_nan_free",
            "adaptive_splits",
        ),
        defaults={
            "extra_cache_nan_free": False,
            "adaptive_splits": False,
        },
    )

    @functools.wraps(original)
    def hcu_dequantize_and_gather_k_cache(
        out,
        k_cache,
        seq_lens,
        gather_lens,
        block_table,
        block_size,
        offset,
        use_fnuz=False,
    ):
        if k_cache.dtype == torch.bfloat16:
            return gather_bf16_k_cache(
                out,
                k_cache,
                seq_lens,
                gather_lens,
                block_table,
                block_size,
                offset,
            )
        return original(
            out,
            k_cache,
            seq_lens,
            gather_lens,
            block_table,
            block_size,
            offset,
            use_fnuz,
        )

    setattr(hcu_dequantize_and_gather_k_cache, _WRAPPER_MARKER, True)

    @functools.wraps(original_decode)
    def hcu_rocm_sparse_attn_decode(
        q,
        kv_cache,
        swa_k_cache,
        swa_only,
        topk_indices,
        topk_lens,
        swa_indices,
        swa_lens,
        swa_ragged_indices,
        swa_ragged_indptr,
        topk_ragged_indices,
        topk_ragged_indptr,
        attn_sink,
        scale,
        head_dim,
        nope_head_dim,
        rope_head_dim,
        output,
        extra_cache_nan_free=False,
        adaptive_splits=False,
    ):
        target = (
            bf16_sparse_attn_decode
            if swa_k_cache.dtype == torch.bfloat16
            else original_decode
        )
        return target(
            q=q,
            kv_cache=kv_cache,
            swa_k_cache=swa_k_cache,
            swa_only=swa_only,
            topk_indices=topk_indices,
            topk_lens=topk_lens,
            swa_indices=swa_indices,
            swa_lens=swa_lens,
            swa_ragged_indices=swa_ragged_indices,
            swa_ragged_indptr=swa_ragged_indptr,
            topk_ragged_indices=topk_ragged_indices,
            topk_ragged_indptr=topk_ragged_indptr,
            attn_sink=attn_sink,
            scale=scale,
            head_dim=head_dim,
            nope_head_dim=nope_head_dim,
            rope_head_dim=rope_head_dim,
            output=output,
            extra_cache_nan_free=extra_cache_nan_free,
            adaptive_splits=adaptive_splits,
        )

    setattr(hcu_rocm_sparse_attn_decode, _DECODE_WRAPPER_MARKER, True)
    setattr(rocm, "_vllm_hcu_original_dequantize_and_gather_k_cache", original)
    setattr(rocm, "_vllm_hcu_original_rocm_sparse_attn_decode", original_decode)
    setattr(rocm, "dequantize_and_gather_k_cache", hcu_dequantize_and_gather_k_cache)
    setattr(rocm, "rocm_sparse_attn_decode", hcu_rocm_sparse_attn_decode)
    setattr(rocm, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "apply", "apply_to_module"]
