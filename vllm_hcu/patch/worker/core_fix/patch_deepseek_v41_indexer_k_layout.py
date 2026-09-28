# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Write V4.1 indexer keys in the row-major layout used by LightOp paged MQA."""

from __future__ import annotations

import functools
import importlib
import os
from types import ModuleType

import torch

from ._common import load_exact_module, require_callable

TARGET_MODULE = "vllm.models.deepseek_v41.attention"
PATCH_ID = "worker.core_fix.deepseek_v41.indexer_k_lightop_layout"
_MARKER = "_vllm_hcu_indexer_k_lightop_layout_applied"
_FLAG = "VLLM_HCU_DSV41_LIGHTOP_PAGED_NORMAL"


def use_normal_indexer_k_layout() -> bool:
    return os.environ.get(_FLAG, "0").lower() in ("1", "true")


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    ops_module = importlib.import_module("vllm.models.deepseek_v41.common.ops")
    store_module = importlib.import_module(
        "vllm.models.deepseek_v41.common.ops.indexer_k_store"
    )
    original = require_callable(
        target,
        "indexer_k_norm_rope_store",
        f"{TARGET_MODULE}.indexer_k_norm_rope_store",
    )
    if getattr(target, _MARKER, False):
        return False

    @functools.wraps(original)
    def store(k_pre, positions, cos_sin_cache, rms_norm_weight, rms_norm_eps,
              k_cache, kv_slot_mapping, compress_ratio, use_fp4_cache):
        if not use_normal_indexer_k_layout():
            return original(
                k_pre, positions, cos_sin_cache, rms_norm_weight, rms_norm_eps,
                k_cache, kv_slot_mapping, compress_ratio, use_fp4_cache,
            )
        if use_fp4_cache:
            raise ValueError("LightOp paged MQA NORMAL layout requires FP8 indexer K")

        # Reuse the upstream fused norm/RoPE/FP8 store kernel. Only its existing
        # SHUFFLE constexpr changes; the page and scale addressing stay identical.
        num_tokens = kv_slot_mapping.numel()
        assert k_pre.ndim == 2 and k_pre.shape[1] == 128
        assert k_pre.dtype == torch.bfloat16 and k_pre.stride(1) == 1
        assert num_tokens <= k_pre.shape[0] and num_tokens <= positions.numel()
        assert compress_ratio in (1, 2)
        if not num_tokens:
            return
        kernel = store_module._indexer_k_norm_rope_quant_store_kernel
        kernel[(num_tokens,)](
            k_pre, k_pre.stride(0), positions, rms_norm_weight, rms_norm_eps,
            cos_sin_cache, cos_sin_cache.stride(0), k_cache, kv_slot_mapping,
            k_cache.shape[1], HEAD_SIZE=128, ROPE_HEAD_DIM=64,
            COMPRESS_RATIO=compress_ratio, TOKEN_STRIDE=128, SCALE_DIM=4,
            KV_BLOCK_STRIDE=k_cache.stride(0), FP8_MAX=448.0,
            USE_FP4=False, SHUFFLE=False, BLOCK_TILE_SIZE=16,
            HEAD_TILE_SIZE=16, num_warps=1,
        )

    target.indexer_k_norm_rope_store = store
    ops_module.indexer_k_norm_rope_store = store
    store_module.indexer_k_norm_rope_store = store
    setattr(target, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "apply", "apply_to_module",
           "use_normal_indexer_k_layout"]
