# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Optional HCU AITER W4A16 fused-expert dispatch."""

from __future__ import annotations

import functools
from types import ModuleType

from ._common import load_exact_module, require_callable, require_parameter_names

TARGET_MODULE = "vllm.model_executor.layers.fused_moe.fused_moe"
PATCH_ID = "worker.op_opt.moe.fused_moe.aiter_w4a16"
TARGETS = (f"{TARGET_MODULE}.fused_experts_impl",)
_MARKER = "_vllm_hcu_fused_moe_w4a16_applied"


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    if getattr(target, _MARKER, False):
        return False
    original = require_callable(target, "fused_experts_impl", TARGETS[0])
    require_parameter_names(
        original,
        TARGETS[0],
        (
            "hidden_states", "w1", "w2", "topk_weights", "topk_ids",
            "activation", "apply_router_weight_on_input", "use_fp8_w8a8",
            "use_int8_w8a8", "use_int8_w8a16", "use_int4_w4a16",
            "ocp_mx_scheme", "per_channel_quant", "global_num_experts",
            "expert_map", "w1_scale", "w2_scale", "w1_zp", "w2_zp",
            "a1_scale", "a2_scale", "block_shape", "w1_bias", "w2_bias",
        ),
    )

    @functools.wraps(original)
    def hcu_fused_experts_impl(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        activation="silu",
        apply_router_weight_on_input=False,
        use_fp8_w8a8=False,
        use_int8_w8a8=False,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        ocp_mx_scheme=None,
        per_channel_quant=False,
        global_num_experts=-1,
        expert_map=None,
        w1_scale=None,
        w2_scale=None,
        w1_zp=None,
        w2_zp=None,
        a1_scale=None,
        a2_scale=None,
        block_shape=None,
        w1_bias=None,
        w2_bias=None,
    ):
        from vllm_hcu.platforms import envs as henvs
        from vllm_hcu.model_executor.layers.fused_moe.aiter_runtime import (
            is_aiter_moe_explicitly_disabled,
            is_aiter_moe_requested,
        )

        enabled = bool(
            henvs.VLLM_HCU_USE_CUSTOM_OPS
            and not is_aiter_moe_explicitly_disabled()
            and (
                henvs.VLLM_HCU_USE_AITER_W4A16_MOE
                or is_aiter_moe_requested()
            )
            and use_int4_w4a16
            and hidden_states.dtype == target.torch.bfloat16
        )
        if enabled:
            if not block_shape or len(block_shape) < 2:
                raise ValueError("HCU AITER W4A16 MoE requires a two-dimensional block_shape")
            from aiter.moe import MoeQuantType
            from vllm_hcu.model_executor.layers.fused_moe.aiter_moe_dispatch import (
                AiterMoeProblem,
                aiter_expert_map_for_solution,
                execute_aiter_moe,
                prepare_aiter_moe_scales,
                prepare_aiter_moe_weights,
                resolve_aiter_expert_maps,
                select_aiter_moe_config,
            )

            _, n1, _ = w1.shape
            _, n2, _ = w2.shape
            problem = AiterMoeProblem(
                M=hidden_states.shape[0],
                E=w1.shape[0],
                N1=n1,
                N2=n2,
                K=hidden_states.shape[1],
                top_k=topk_ids.size(1),
                block_size=block_shape[1],
                dtype=hidden_states.dtype,
                device=hidden_states.device,
                quant_type=MoeQuantType.W4A16,
                activation=activation,
                use_shuffle=bool(henvs.VLLM_HCU_USE_AITER_MOE_SHUFFLE),
            )
            moe_config = select_aiter_moe_config(problem, cache_owner=w1)
            native_expert_map, expert_mask = resolve_aiter_expert_maps(
                expert_map,
                int(global_num_experts),
            )
            if moe_config is not None:
                prepared_w1, prepared_w2 = prepare_aiter_moe_weights(
                    w1,
                    w2,
                    moe_config,
                    cache_owner=w1,
                    block_shape=block_shape,
                )
                prepared_w1_scale, prepared_w2_scale = prepare_aiter_moe_scales(
                    w1_scale,
                    w2_scale,
                    moe_config,
                    cache_owner=w1_scale if w1_scale is not None else w1,
                )
                aiter_expert_map = aiter_expert_map_for_solution(
                    native_expert_map,
                    moe_config,
                    int(global_num_experts),
                    expert_mask=expert_mask,
                )
                return execute_aiter_moe(
                    moe_config,
                    hidden_states=hidden_states,
                    w1=prepared_w1,
                    w2=prepared_w2,
                    topk_weights=topk_weights,
                    topk_ids=topk_ids,
                    inplace=False,
                    activation=activation,
                    w1_scale=prepared_w1_scale,
                    w2_scale=prepared_w2_scale,
                    w1_zp=w1_zp,
                    w2_zp=w2_zp,
                    a1_scale=a1_scale,
                    a2_scale=a2_scale,
                    block_shape=block_shape,
                    global_num_experts=global_num_experts,
                    expert_map=aiter_expert_map,
                    use_weight_shuffle=bool(
                        getattr(moe_config, "need_shuffle", False)
                    ),
                    output_dtype=hidden_states.dtype,
                )
            expert_map = native_expert_map
        return original(
            hidden_states, w1, w2, topk_weights, topk_ids, activation,
            apply_router_weight_on_input, use_fp8_w8a8, use_int8_w8a8,
            use_int8_w8a16, use_int4_w4a16, ocp_mx_scheme, per_channel_quant,
            global_num_experts, expert_map, w1_scale, w2_scale, w1_zp, w2_zp,
            a1_scale, a2_scale, block_shape, w1_bias, w2_bias,
        )

    target._vllm_hcu_original_fused_experts_impl = original
    target.fused_experts_impl = hcu_fused_experts_impl
    setattr(target, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = [
    "PATCH_ID",
    "TARGET_MODULE",
    "TARGETS",
    "apply",
    "apply_to_module",
]
