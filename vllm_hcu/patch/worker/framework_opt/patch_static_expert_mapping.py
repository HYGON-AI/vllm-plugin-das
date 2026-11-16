# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Keep legacy inline expert mappings in logical checkpoint ID space."""

from __future__ import annotations

import functools
from types import ModuleType

from ._common import (
    PatchCompatibilityError,
    load_exact_module,
    require_class,
    require_exact_signature,
)

TARGET_MODULE = "vllm.model_executor.layers.fused_moe.routed_experts"
PATCH_ID = "worker.framework_opt.model_loader.static_expert_mapping"
TARGETS = (f"{TARGET_MODULE}.RoutedExperts.make_expert_params_mapping",)
_MARKER = "_vllm_hcu_static_expert_mapping_wrapper"


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    routed_experts_cls = require_class(
        target,
        "RoutedExperts",
        f"{TARGET_MODULE}.RoutedExperts",
    )
    descriptor = vars(routed_experts_cls).get("make_expert_params_mapping")
    if not isinstance(descriptor, staticmethod):
        raise PatchCompatibilityError(
            "Static expert mapping requires the current staticmethod"
        )
    current = descriptor.__func__
    installed = getattr(target, _MARKER, None)
    if installed is not None:
        if installed is not current:
            raise PatchCompatibilityError(
                "Static expert mapping wrapper is stale; restart the process"
            )
        return False

    require_exact_signature(
        current,
        TARGETS[0],
        positional=(
            "model",
            "ckpt_gate_proj_name",
            "ckpt_down_proj_name",
            "ckpt_up_proj_name",
            "num_experts",
            "num_redundant_experts",
            "routed_experts_prefix",
        ),
        defaults={
            "num_redundant_experts": 0,
            "routed_experts_prefix": "routed_experts",
        },
    )

    @functools.wraps(current)
    def make_expert_params_mapping(
        model,
        ckpt_gate_proj_name,
        ckpt_down_proj_name,
        ckpt_up_proj_name,
        num_experts,
        num_redundant_experts=0,
        routed_experts_prefix="routed_experts",
    ):
        from vllm_hcu.model_executor.layers.fused_moe.static_eplb import (
            StaticEplbPlan,
        )

        plan = getattr(model, "_vllm_hcu_static_eplb_plan", None)
        if plan is not None:
            if not isinstance(plan, StaticEplbPlan):
                raise ValueError(
                    "Inline expert mapping requires a validated static plan"
                )
            experts = [
                owner
                for owner in model.modules()
                if isinstance(owner, routed_experts_cls)
            ]
            if len(experts) != len(plan._map_values) or any(
                getattr(owner, "_vllm_hcu_static_eplb_row", None)
                != plan.layer_map(index)
                for index, owner in enumerate(experts)
            ):
                raise ValueError(
                    "Inline expert mapping does not match bound static rows"
                )
            fused_shared_experts = (
                experts[0].expert_map_manager.num_fused_shared_experts
            )
            if (
                num_experts
                != plan.num_logical_experts + fused_shared_experts
                or num_redundant_experts
                not in (0, plan.num_redundant_experts)
            ):
                raise ValueError(
                    "Inline expert mapping counts do not match the static plan"
                )
            num_redundant_experts = 0
        return current(
            model,
            ckpt_gate_proj_name,
            ckpt_down_proj_name,
            ckpt_up_proj_name,
            num_experts,
            num_redundant_experts,
            routed_experts_prefix,
        )

    routed_experts_cls.make_expert_params_mapping = staticmethod(
        make_expert_params_mapping
    )
    setattr(target, _MARKER, make_expert_params_mapping)
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
