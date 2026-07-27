# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Bind the feature-gated HCU grouped-top-k router in the official factory."""

from __future__ import annotations

from types import ModuleType

from ._common import load_exact_module, require_callable, require_class, require_parameter_names

TARGET_MODULE = "vllm.model_executor.layers.fused_moe.router.router_factory"
PATCH_ID = "worker.op_opt.moe.router.factory"
TARGETS = (
    f"{TARGET_MODULE}.GroupedTopKRouter",
    f"{TARGET_MODULE}.create_fused_moe_router",
)
_MARKER = "_vllm_hcu_grouped_router_factory_applied"


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    if getattr(target, _MARKER, False):
        return False
    from vllm_hcu.model_executor.layers.fused_moe.router_runtime import (
        make_hcu_grouped_topk_router,
    )

    base = require_class(target, "GroupedTopKRouter", TARGETS[0])
    factory = require_callable(target, "create_fused_moe_router", TARGETS[1])
    require_parameter_names(
        factory,
        TARGETS[1],
        (
            "top_k", "global_num_experts", "renormalize",
            "use_grouped_topk", "num_expert_group", "topk_group", "scoring_func",
            "num_fused_shared_experts", "shared_expert_weight",
            "routed_scaling_factor", "e_score_correction_bias",
            "custom_routing_function",
            "eplb_state", "zero_expert_type", "num_logical_experts",
            "hash_indices_table",
        ),
    )
    hcu_class = make_hcu_grouped_topk_router(base)
    target._vllm_hcu_original_grouped_topk_router = base
    target.GroupedTopKRouter = hcu_class
    setattr(target, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
