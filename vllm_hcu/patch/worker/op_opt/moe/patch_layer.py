# SPDX-License-Identifier: Apache-2.0
"""Wire HCU-owned MoE method/runner capabilities into ``FusedMoE``."""

from __future__ import annotations

import functools
from types import ModuleType

from ._common import load_exact_module, require_callable, require_class, require_parameter_names

TARGET_MODULE = "vllm.model_executor.layers.fused_moe.layer"
PATCH_ID = "worker.op_opt.moe.layer"
TARGETS = (
    f"{TARGET_MODULE}.FusedMoE.__init__",
    f"{TARGET_MODULE}.FusedMoE.forward",
    f"{TARGET_MODULE}.FusedMoE.get_expert_weights",
)
_MARKER = "_vllm_hcu_moe_layer_applied"


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    if getattr(target, _MARKER, False):
        return False
    cls = require_class(target, "FusedMoE", TARGETS[0].rsplit(".", 1)[0])
    init = require_callable(cls, "__init__", TARGETS[0])
    forward = require_callable(cls, "forward", TARGETS[1])
    get_weights = require_callable(cls, "get_expert_weights", TARGETS[2])
    require_parameter_names(
        init,
        TARGETS[0],
        (
            "self", "num_experts", "top_k", "hidden_size", "intermediate_size",
            "params_dtype", "renormalize", "use_grouped_topk", "num_expert_group",
            "topk_group", "quant_config", "tp_size", "ep_size", "dp_size",
            "pcp_size", "prefix", "custom_routing_function", "scoring_func",
            "routed_scaling_factor", "swiglu_limit", "e_score_correction_bias",
            "apply_router_weight_on_input", "activation", "is_act_and_mul",
            "enable_eplb", "num_redundant_experts", "has_bias",
            "is_sequence_parallel", "expert_mapping", "n_shared_experts",
            "router_logits_dtype", "gate", "shared_experts", "shared_expert_gate",
            "routed_input_transform", "routed_output_transform",
            "apply_routed_scale_to_output", "zero_expert_type", "hash_indices_table",
        ),
    )
    require_parameter_names(
        forward,
        TARGETS[1],
        ("self", "hidden_states", "router_logits", "input_ids"),
    )
    require_parameter_names(get_weights, TARGETS[2], ("self",))

    @functools.wraps(init)
    def hcu_init(self, *args, **kwargs):
        init(self, *args, **kwargs)
        official_cls = target.UnquantizedFusedMoEMethod
        if type(self.quant_method) is not official_cls:
            return
        from vllm_hcu.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
            HcuUnquantizedFusedMoEMethod,
        )

        old_method = self.quant_method
        hcu_method = HcuUnquantizedFusedMoEMethod(self.moe_config)
        hcu_method.moe_quant_config = getattr(old_method, "moe_quant_config", None)
        self.quant_method = hcu_method
        self.base_quant_method = hcu_method
        self.runner._replace_quant_method(hcu_method)

    @functools.wraps(forward)
    def hcu_forward(
        self,
        hidden_states,
        router_logits,
        input_ids=None,
        quanted_hidden_states=None,
        scale=None,
        topk_weights=None,
        topk_ids=None,
    ):
        return self.runner.forward(
            hidden_states,
            router_logits,
            input_ids,
            quanted_hidden_states=quanted_hidden_states,
            scale=scale,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
        )

    del hcu_forward.__wrapped__

    @functools.wraps(get_weights)
    def hcu_get_expert_weights(self):
        if getattr(self, "_dsv4_channel_fp8_deepgemm_repacked", False):
            names = ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale")
            weights = [getattr(self, name, None) for name in names]
            if all(weight is not None for weight in weights):
                return [weight.view(self.local_num_experts, -1) for weight in weights]
            missing = [name for name, value in zip(names, weights) if value is None]
            raise RuntimeError(
                "DPSK DeepGEMM repacked layer is missing expert weights: "
                + ", ".join(missing)
            )
        return get_weights(self)

    cls._vllm_hcu_original_init = init
    cls.__init__ = hcu_init
    cls._vllm_hcu_original_forward = forward
    cls.forward = hcu_forward
    cls._vllm_hcu_original_get_expert_weights = get_weights
    cls.get_expert_weights = hcu_get_expert_weights
    setattr(target, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
