# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Wire HCU-owned MoE capabilities into the v0.25.1 factory pipeline."""

from __future__ import annotations

import functools
import sys
from types import ModuleType

from ._common import (
    PatchCompatibilityError,
    load_exact_module,
    require_callable,
    require_class,
    require_parameter_names,
)

TARGET_MODULE = "vllm.model_executor.layers.fused_moe"
LAYER_MODULE = f"{TARGET_MODULE}.layer"
PATCH_ID = "worker.op_opt.moe.layer"
TARGETS = (
    f"{TARGET_MODULE}.FusedMoE",
    f"{TARGET_MODULE}.RoutedExperts.get_expert_weights",
    f"{TARGET_MODULE}.RoutedExperts.load_weights",
    f"{TARGET_MODULE}.RoutedExperts.expert_map",
)
_MARKER = "_vllm_hcu_moe_layer_applied"


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    if getattr(target, _MARKER, False):
        return False
    factory = require_callable(target, "FusedMoE", TARGETS[0])
    layer_module = load_exact_module(
        LAYER_MODULE,
        sys.modules.get(LAYER_MODULE),
    )
    layer_factory = require_callable(
        layer_module,
        "FusedMoE",
        f"{LAYER_MODULE}.FusedMoE",
    )
    if layer_factory is not factory:
        raise PatchCompatibilityError(
            f"{TARGETS[0]} does not reference the required v0.25.1 "
            f"{LAYER_MODULE}.FusedMoE factory"
        )
    routed_experts_cls = require_class(
        target, "RoutedExperts", f"{TARGET_MODULE}.RoutedExperts"
    )
    layer_routed_experts_cls = require_class(
        layer_module,
        "RoutedExperts",
        f"{LAYER_MODULE}.RoutedExperts",
    )
    if layer_routed_experts_cls is not routed_experts_cls:
        raise PatchCompatibilityError(
            f"{TARGET_MODULE}.RoutedExperts does not reference the required "
            f"v0.25.1 {LAYER_MODULE}.RoutedExperts class"
        )
    expert_map_property = vars(routed_experts_cls).get("expert_map")
    if not isinstance(expert_map_property, property) or not callable(
        expert_map_property.fget
    ):
        raise PatchCompatibilityError(
            f"{TARGETS[3]} must remain a property"
        )
    get_weights = require_callable(
        routed_experts_cls, "get_expert_weights", TARGETS[1]
    )
    load_weights = require_callable(routed_experts_cls, "load_weights", TARGETS[2])
    require_parameter_names(
        factory,
        TARGETS[0],
        (
            "num_experts", "top_k", "hidden_size", "intermediate_size",
            "intermediate_pad", "params_dtype", "renormalize", "use_grouped_topk",
            "num_expert_group", "topk_group", "quant_config", "tp_size", "dp_size",
            "pcp_size", "prefix", "custom_routing_function", "router",
            "scoring_func", "routed_scaling_factor", "swiglu_limit",
            "swiglu_alpha", "swiglu_beta", "e_score_correction_bias",
            "apply_router_weight_on_input", "activation", "enable_eplb",
            "num_redundant_experts", "has_bias", "is_sequence_parallel",
            "reduce_results", "ckpt_names", "n_shared_experts", "router_logits_dtype",
            "gate", "shared_experts", "shared_expert_gate", "routed_input_transform",
            "routed_output_transform", "apply_routed_scale_to_output",
            "zero_expert_type", "hash_indices_table", "runner_cls", "runner_args",
            "routed_experts_cls", "routed_experts_args",
        ),
    )
    require_parameter_names(get_weights, TARGETS[1], ("self",))
    require_parameter_names(load_weights, TARGETS[2], ("self", "weights"))

    @functools.wraps(expert_map_property.fget)
    def hcu_expert_map(self):
        value = expert_map_property.fget(self)
        native_map = getattr(self, "_expert_map", None)
        expert_mask = getattr(self, "expert_mask", None)
        if (
            value is expert_mask
            and expert_mask is not None
            and native_map is not None
        ):
            expert_mask._vllm_hcu_native_expert_map = native_map
        return value

    routed_experts_cls.expert_map = property(
        hcu_expert_map,
        expert_map_property.fset,
        expert_map_property.fdel,
        expert_map_property.__doc__,
    )

    @functools.wraps(factory)
    def hcu_factory(*args, **kwargs):
        runner = factory(*args, **kwargs)
        experts = runner.routed_experts
        official_cls = getattr(target, "UnquantizedFusedMoEMethod", None)
        if official_cls is None:
            from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
                UnquantizedFusedMoEMethod as official_cls,
            )
        if type(experts.quant_method) is not official_cls:
            return runner
        from vllm_hcu.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
            HcuUnquantizedFusedMoEMethod,
        )

        old_method = experts.quant_method
        hcu_method = HcuUnquantizedFusedMoEMethod(experts.moe_config)
        hcu_method.moe_quant_config = getattr(old_method, "moe_quant_config", None)
        experts._replace_quant_method(hcu_method)
        runner._replace_quant_method(hcu_method)
        return runner

    @functools.wraps(get_weights)
    def hcu_get_expert_weights(self):
        if getattr(self, "_dsv4_channel_deepgemm_repacked", False):
            names = ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale")
            weights = [getattr(self, name, None) for name in names]
            if all(weight is not None for weight in weights):
                return [weight.view(self.local_num_experts, -1) for weight in weights]
            missing = [name for name, value in zip(names, weights) if value is None]
            raise RuntimeError(
                "HCU DeepGEMM repacked layer is missing expert weights: "
                + ", ".join(missing)
            )
        return get_weights(self)

    def _load_fused_channel_scale(self, expert_name, loaded_weight):
        """Load fused [E, N, 1] channel scales without weight transposes."""
        if loaded_weight.ndim != 3 or "scale" not in expert_name:
            return None

        qual_name = f"{self.layer_name}.{expert_name}"
        matches = []
        matched = False
        for param_name, checkpoint_name, shard_index, shard_id in (
            self.get_expert_mapping(include_fused=True)
        ):
            if checkpoint_name not in qual_name:
                if matched:
                    break
                continue
            matched = True
            mapped_name = qual_name.replace(checkpoint_name, param_name)
            local_name = mapped_name.removeprefix(f"{self.layer_name}.")
            param = getattr(self, local_name)
            if getattr(param, "quant_method", None) != "channel":
                return None
            matches.append((local_name, mapped_name, param, shard_index, shard_id))

        if not matches:
            return None
        if any(
            shard_id in {"w1", "w3"}
            and (loaded_weight.shape[1] % 2 or shard_index not in (0, 1))
            for _, _, _, shard_index, shard_id in matches
        ):
            return None

        loaded_names = []
        for local_name, mapped_name, param, shard_index, shard_id in matches:
            if shard_id in {"w1", "w3"}:
                scale_shards = loaded_weight.chunk(2, dim=1)
                experts_shard = scale_shards[shard_index]
            else:
                experts_shard = loaded_weight

            for expert_id, loaded_expert in enumerate(experts_shard.unbind()):
                success = param.weight_loader(
                    param=param,
                    loaded_weight=loaded_expert,
                    weight_name=mapped_name,
                    shard_id=shard_id,
                    expert_id=expert_id,
                    return_success=True,
                )
                if success:
                    loaded_names.append(local_name)
        return loaded_names

    @functools.wraps(load_weights)
    def hcu_load_weights(self, weights):
        for expert_name, loaded_weight in weights:
            loaded_names = _load_fused_channel_scale(
                self, expert_name, loaded_weight
            )
            if loaded_names is None:
                yield from load_weights(self, ((expert_name, loaded_weight),))
            else:
                yield from loaded_names

    target._vllm_hcu_original_fused_moe_factory = factory
    target.FusedMoE = hcu_factory
    layer_module._vllm_hcu_original_fused_moe_factory = factory
    layer_module.FusedMoE = hcu_factory
    routed_experts_cls._vllm_hcu_original_get_expert_weights = get_weights
    routed_experts_cls.get_expert_weights = hcu_get_expert_weights
    routed_experts_cls._vllm_hcu_original_load_weights = load_weights
    routed_experts_cls.load_weights = hcu_load_weights
    setattr(target, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
