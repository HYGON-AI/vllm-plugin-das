# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Normalize DeepSeek-v4 hash routing indices for the HCU custom op."""

from __future__ import annotations

import functools
from types import ModuleType

from ._common import load_exact_module, require_callable, require_parameter_names

TARGET_MODULE = "vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router"
PATCH_ID = "worker.op_opt.moe.router.fused_topk_bias"
TARGETS = (f"{TARGET_MODULE}.vllm_topk_softplus_sqrt",)
_MARKER = "_vllm_hcu_hash_router_dtype_applied"


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    if getattr(target, _MARKER, False):
        return False
    original = require_callable(target, "vllm_topk_softplus_sqrt", TARGETS[0])
    require_parameter_names(
        original,
        TARGETS[0],
        (
            "topk_weights",
            "topk_indices",
            "token_expert_indices",
            "gating_output",
            "renormalize",
            "e_score_correction_bias",
            "input_tokens",
            "hash_indices_table",
            "routed_scaling_factor",
        ),
    )

    @functools.wraps(original)
    def hcu_topk_softplus_sqrt(
        topk_weights,
        topk_indices,
        token_expert_indices,
        gating_output,
        renormalize=False,
        e_score_correction_bias=None,
        input_tokens=None,
        hash_indices_table=None,
        routed_scaling_factor=1.0,
    ):
        if hash_indices_table is not None:
            if hash_indices_table.dtype != topk_indices.dtype:
                hash_indices_table = hash_indices_table.to(dtype=topk_indices.dtype)
            if input_tokens is not None and input_tokens.dtype != topk_indices.dtype:
                input_tokens = input_tokens.to(dtype=topk_indices.dtype)
        return original(
            topk_weights,
            topk_indices,
            token_expert_indices,
            gating_output,
            renormalize,
            e_score_correction_bias,
            input_tokens,
            hash_indices_table,
            routed_scaling_factor,
        )

    target._vllm_hcu_original_vllm_topk_softplus_sqrt = original
    target.vllm_topk_softplus_sqrt = hcu_topk_softplus_sqrt
    setattr(target, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
