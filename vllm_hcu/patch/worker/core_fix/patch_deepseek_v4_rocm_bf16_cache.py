# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Route DeepSeek V4 plain BF16 caches around the packed-FP8 gather."""

from __future__ import annotations

import functools
from types import ModuleType

import torch

from vllm_hcu.v1.attention.ops.deepseek_v4_bf16_cache import (
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


def apply_to_module(module: ModuleType) -> bool:
    rocm = load_exact_module(TARGET_MODULE, module)
    original = require_callable(
        rocm,
        "dequantize_and_gather_k_cache",
        TARGET_SYMBOL,
    )
    if getattr(rocm, _MARKER, False):
        if not getattr(original, _WRAPPER_MARKER, False):
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
    setattr(rocm, "_vllm_hcu_original_dequantize_and_gather_k_cache", original)
    setattr(rocm, "dequantize_and_gather_k_cache", hcu_dequantize_and_gather_k_cache)
    setattr(rocm, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "apply", "apply_to_module"]
