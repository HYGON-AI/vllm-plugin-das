# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Normalize DeepSeek-v4 hash routing indices for the HCU custom op."""

from __future__ import annotations

import functools
from types import ModuleType

from ._common import (
    check_module_marker,
    load_exact_module,
    require_callable,
    require_parameter_names,
)

TARGET_MODULE = "vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router"
PATCH_ID = "worker.op_opt.moe.router.fused_topk_bias"
TARGETS = (f"{TARGET_MODULE}.vllm_topk_softplus_sqrt",)
_MARKER = "_vllm_hcu_hash_router_dtype_applied"
_WRAPPER_MARKER = "_vllm_hcu_hash_router_dtype_wrapper"


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    if check_module_marker(
        target,
        _MARKER,
        ((target, "vllm_topk_softplus_sqrt", _WRAPPER_MARKER),),
    ):
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
            "bias_vl",
            "image_sentinel_lo",
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
        bias_vl=None,
        image_sentinel_lo=0,
    ):
        from vllm_hcu.model_executor.layers.fused_moe.sqrtsoftplus_routing import (
            can_use_lightop_sqrtsoftplus,
            is_lightop_sqrtsoftplus_available,
            run_lightop_sqrtsoftplus,
        )
        from vllm_hcu.platforms import envs as henvs

        use_lightop = bool(
            henvs.VLLM_HCU_USE_CUSTOM_OPS
            and henvs.VLLM_HCU_USE_LIGHTOP_SQRTSOFTPLUS_GATE
            and can_use_lightop_sqrtsoftplus(
                gating_output,
                e_score_correction_bias,
                topk=topk_indices.shape[-1],
                input_tokens=input_tokens,
                hash_indices_table=hash_indices_table,
                bias_vl=bias_vl,
                image_sentinel_lo=image_sentinel_lo,
            )
            and is_lightop_sqrtsoftplus_available()
        )
        if use_lightop:
            return run_lightop_sqrtsoftplus(
                gating_output,
                e_score_correction_bias,
                topk=topk_indices.shape[-1],
                renormalize=renormalize,
                routed_scaling_factor=routed_scaling_factor,
                indices_dtype=topk_indices.dtype,
            )
        if hash_indices_table is not None:
            if hash_indices_table.dtype != topk_indices.dtype:
                hash_indices_table = hash_indices_table.to(dtype=topk_indices.dtype)
            if input_tokens is not None and input_tokens.dtype != topk_indices.dtype:
                input_tokens = input_tokens.to(dtype=topk_indices.dtype)
        native_fallback = getattr(target, "_topk_softplus_sqrt_torch", None)
        torch_module = getattr(target, "torch", None)
        torch_ops = getattr(torch_module, "ops", None)
        moe_ops = getattr(torch_ops, "_moe_C", None)
        compiled_op = getattr(moe_ops, "topk_softplus_sqrt", None)
        if callable(native_fallback) and not callable(compiled_op):
            return native_fallback(
                topk_weights,
                topk_indices,
                token_expert_indices,
                gating_output,
                renormalize,
                e_score_correction_bias,
                input_tokens,
                hash_indices_table,
                routed_scaling_factor,
                bias_vl,
                image_sentinel_lo,
            )
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
            bias_vl,
            image_sentinel_lo,
        )

    setattr(hcu_topk_softplus_sqrt, _WRAPPER_MARKER, True)
    target._vllm_hcu_original_vllm_topk_softplus_sqrt = original
    target.vllm_topk_softplus_sqrt = hcu_topk_softplus_sqrt
    setattr(target, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
