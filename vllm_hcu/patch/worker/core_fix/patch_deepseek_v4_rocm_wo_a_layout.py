# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Adapt DeepSeek-V4 ROCm ``wo_a`` caching to HCU's NN weight layout."""

from __future__ import annotations

import functools
from types import ModuleType, SimpleNamespace

from ._common import (
    PatchCompatibilityError,
    load_exact_module,
    require_callable,
    require_exact_signature,
)

TARGET_MODULE = "vllm.v1.attention.ops.rocm_aiter_mla_sparse"
PATCH_ID = "worker.core_fix.deepseek_v4_rocm.wo_a_nn_weight_layout"
TARGET_SYMBOL = f"{TARGET_MODULE}._get_cached_wo_a_bf16"
_MARKER = "_vllm_hcu_wo_a_nn_weight_layout_applied"
_WRAPPER_MARKER = "_vllm_hcu_wo_a_nn_weight_layout_wrapper"


def apply_to_module(module: ModuleType) -> bool:
    rocm_ops = load_exact_module(TARGET_MODULE, module)
    original = require_callable(
        rocm_ops,
        "_get_cached_wo_a_bf16",
        TARGET_SYMBOL,
    )
    if getattr(rocm_ops, _MARKER, False):
        if not getattr(original, _WRAPPER_MARKER, False):
            raise PatchCompatibilityError(
                f"required HCU patch marker for {TARGET_SYMBOL} is stale"
            )
        return False
    require_exact_signature(
        original,
        TARGET_SYMBOL,
        positional=(
            "wo_a",
            "n_local_groups",
            "o_lora_rank",
            "hidden_dim",
        ),
    )

    @functools.wraps(original)
    def hcu_get_cached_wo_a_bf16(
        wo_a,
        n_local_groups,
        o_lora_rank,
        hidden_dim,
    ):
        weight = wo_a.weight
        expected_rows = n_local_groups * o_lora_rank
        expected_shape = (expected_rows, hidden_dim)
        nn_shape = (hidden_dim, expected_rows)

        # Quantized wo_a retains upstream storage.  Only the unquantized BF16
        # parameter is transposed by HCU's VLLM_USE_NN linear loader.
        if hasattr(wo_a, "weight_scale_inv") or tuple(weight.shape) == expected_shape:
            return original(wo_a, n_local_groups, o_lora_rank, hidden_dim)
        if tuple(weight.shape) != nn_shape:
            raise RuntimeError(
                "DeepSeek-V4 wo_a has an unsupported physical shape: "
                f"expected {expected_shape} or HCU NN {nn_shape}, "
                f"got {tuple(weight.shape)}"
            )

        cached = getattr(wo_a, "_dsv4_wo_a_bf16", None)
        if cached is not None:
            return cached
        logical_wo_a = SimpleNamespace(weight=weight.T.contiguous())
        cached = original(
            logical_wo_a,
            n_local_groups,
            o_lora_rank,
            hidden_dim,
        )
        wo_a._dsv4_wo_a_bf16 = cached
        return cached

    setattr(hcu_get_cached_wo_a_bf16, _WRAPPER_MARKER, True)
    setattr(rocm_ops, "_vllm_hcu_original_get_cached_wo_a_bf16", original)
    setattr(rocm_ops, "_get_cached_wo_a_bf16", hcu_get_cached_wo_a_bf16)
    setattr(rocm_ops, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "apply", "apply_to_module"]
