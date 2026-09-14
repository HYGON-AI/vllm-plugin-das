# SPDX-License-Identifier: Apache-2.0
"""Keep inline legacy expert mappings in logical checkpoint ID space."""
from __future__ import annotations

import functools

from ._common import load_exact_module, require_exact_signature, PatchCompatibilityError

TARGET_MODULE = "vllm.model_executor.layers.fused_moe.routed_experts"
PATCH_ID = "worker.framework_opt.model_loader.static_expert_mapping"
TARGETS = (f"{TARGET_MODULE}.RoutedExperts.make_expert_params_mapping",)
_MARKER = "_vllm_hcu_static_expert_mapping"


def apply_to_module(module):
    target = load_exact_module(TARGET_MODULE, module)
    cls = target.RoutedExperts
    descriptor = vars(cls).get("make_expert_params_mapping")
    if not isinstance(descriptor, staticmethod):
        raise PatchCompatibilityError("Static expert mapping requires the current staticmethod")
    original = descriptor.__func__
    installed = getattr(target, _MARKER, None)
    if installed is not None:
        if installed is not original:
            raise PatchCompatibilityError("Static expert mapping wrapper is stale")
        return False
    require_exact_signature(original, TARGETS[0], positional=("model",
        "ckpt_gate_proj_name", "ckpt_down_proj_name", "ckpt_up_proj_name",
        "num_experts", "num_redundant_experts", "routed_experts_prefix"),
        defaults=dict(num_redundant_experts=0, routed_experts_prefix="routed_experts"))

    @functools.wraps(original)
    def mapping(model, ckpt_gate_proj_name, ckpt_down_proj_name, ckpt_up_proj_name,
                num_experts, num_redundant_experts=0, routed_experts_prefix="routed_experts"):
        from vllm_hcu.model_executor.layers.fused_moe.static_eplb import StaticEplbPlan
        plan = getattr(model, "_vllm_hcu_static_eplb_plan", None)
        if plan is not None:
            if not isinstance(plan, StaticEplbPlan):
                raise ValueError("Inline expert mapping requires a validated static plan")
            experts = [owner for owner in model.modules() if isinstance(owner, cls)]
            if len(experts) != len(plan._map_values) or any(
                getattr(owner, "_vllm_hcu_static_eplb_row", None) != plan.layer_map(index)
                for index, owner in enumerate(experts)
            ):
                raise ValueError("Inline expert mapping does not match bound static rows")
            shared = experts[0].expert_map_manager.num_fused_shared_experts
            if (num_experts != plan.num_logical_experts + shared
                    or num_redundant_experts not in (0, plan.num_redundant_experts)):
                raise ValueError("Inline expert mapping counts do not match the static plan")
            num_redundant_experts = 0
        return original(model, ckpt_gate_proj_name, ckpt_down_proj_name,
                        ckpt_up_proj_name, num_experts, num_redundant_experts,
                        routed_experts_prefix)

    cls.make_expert_params_mapping = staticmethod(mapping)
    setattr(target, _MARKER, mapping)
    return True


def apply(module=None):
    return apply_to_module(load_exact_module(TARGET_MODULE, module))
