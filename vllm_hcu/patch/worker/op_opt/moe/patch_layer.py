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
    f"{TARGET_MODULE}.RoutedExperts.weight_loader",
    f"{TARGET_MODULE}.RoutedExperts.get_expert_mapping",
    f"{TARGET_MODULE}.RoutedExperts.make_expert_params_mapping",
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
    get_weights = require_callable(
        routed_experts_cls, "get_expert_weights", TARGETS[1]
    )
    load_weights = require_callable(routed_experts_cls, "load_weights", TARGETS[2])
    weight_loader = require_callable(
        routed_experts_cls, "weight_loader", TARGETS[3]
    )
    get_expert_mapping = require_callable(
        routed_experts_cls, "get_expert_mapping", TARGETS[4]
    )
    make_expert_params_mapping = require_callable(
        routed_experts_cls, "make_expert_params_mapping", TARGETS[5]
    )
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
    require_parameter_names(
        weight_loader,
        TARGETS[3],
        (
            "self",
            "param",
            "loaded_weight",
            "weight_name",
            "shard_id",
            "expert_id",
            "return_success",
        ),
    )
    require_parameter_names(
        get_expert_mapping,
        TARGETS[4],
        (
            "self",
            "ckpt_gate_proj_name",
            "ckpt_down_proj_name",
            "ckpt_up_proj_name",
            "include_fused",
        ),
    )
    require_parameter_names(
        make_expert_params_mapping,
        TARGETS[5],
        (
            "model",
            "ckpt_gate_proj_name",
            "ckpt_down_proj_name",
            "ckpt_up_proj_name",
            "num_experts",
            "num_redundant_experts",
            "routed_experts_prefix",
        ),
    )

    @functools.wraps(weight_loader)
    def hcu_weight_loader(
        self,
        param,
        loaded_weight,
        weight_name,
        shard_id,
        expert_id,
        return_success=False,
    ):
        if not hasattr(self, "_vllm_hcu_static_eplb_row"):
            return weight_loader(
                self,
                param,
                loaded_weight,
                weight_name,
                shard_id,
                expert_id,
                return_success=return_success,
            )

        row = self._vllm_hcu_static_eplb_row
        num_logical_experts = self.moe_config.num_logical_experts
        num_fused_shared_experts = (
            self.expert_map_manager.num_fused_shared_experts
        )
        if (
            not isinstance(expert_id, bool)
            and isinstance(expert_id, int)
            and num_logical_experts
            <= expert_id
            < num_logical_experts + num_fused_shared_experts
        ):
            # Shared experts are appended after all routed physical slots and
            # remain fixed while EPLB remaps only routed logical experts.
            shared_physical_expert_id = len(row) + (
                expert_id - num_logical_experts
            )
            return weight_loader(
                self,
                param,
                loaded_weight,
                weight_name,
                shard_id,
                shared_physical_expert_id,
                return_success=return_success,
            )

        from vllm_hcu.model_executor.layers.fused_moe.static_eplb import (
            load_static_logical_expert,
        )

        return load_static_logical_expert(
            self,
            weight_loader,
            param=param,
            loaded_weight=loaded_weight,
            weight_name=weight_name,
            shard_id=shard_id,
            logical_expert_id=expert_id,
            return_success=return_success,
        )

    @functools.wraps(get_expert_mapping)
    def hcu_get_expert_mapping(
        self,
        ckpt_gate_proj_name=None,
        ckpt_down_proj_name=None,
        ckpt_up_proj_name=None,
        include_fused=False,
    ):
        if not hasattr(self, "_vllm_hcu_static_eplb_row"):
            return get_expert_mapping(
                self,
                ckpt_gate_proj_name,
                ckpt_down_proj_name,
                ckpt_up_proj_name,
                include_fused,
            )

        moe_config = self.moe_config
        num_fused_shared_experts = (
            self.expert_map_manager.num_fused_shared_experts
        )
        return self.build_expert_params_mapping(
            ckpt_gate_proj_name or self.ckpt_gate_proj_name,
            ckpt_down_proj_name or self.ckpt_down_proj_name,
            ckpt_up_proj_name or self.ckpt_up_proj_name,
            num_experts=(
                moe_config.num_logical_experts + num_fused_shared_experts
            ),
            num_redundant_experts=0,
            routed_experts_prefix="",
            lora_base_layer_prefix=self.lora_base_layer_prefix,
            include_fused=include_fused,
        )

    def _model_has_static_eplb_binding(model) -> bool:
        if hasattr(model, "_vllm_hcu_static_eplb_plan"):
            return True
        modules = getattr(model, "modules", None)
        return callable(modules) and any(
            hasattr(module, "_vllm_hcu_static_eplb_row")
            for module in modules()
        )

    @functools.wraps(make_expert_params_mapping)
    def hcu_make_expert_params_mapping(
        model,
        ckpt_gate_proj_name,
        ckpt_down_proj_name,
        ckpt_up_proj_name,
        num_experts,
        num_redundant_experts=0,
        routed_experts_prefix="routed_experts",
    ):
        if _model_has_static_eplb_binding(model):
            num_redundant_experts = 0
        return make_expert_params_mapping(
            model,
            ckpt_gate_proj_name,
            ckpt_down_proj_name,
            ckpt_up_proj_name,
            num_experts,
            num_redundant_experts,
            routed_experts_prefix,
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

    routed_experts_cls._vllm_hcu_original_weight_loader = weight_loader
    routed_experts_cls.weight_loader = hcu_weight_loader
    routed_experts_cls._vllm_hcu_original_get_expert_mapping = get_expert_mapping
    routed_experts_cls.get_expert_mapping = hcu_get_expert_mapping
    routed_experts_cls._vllm_hcu_original_make_expert_params_mapping = (
        make_expert_params_mapping
    )
    routed_experts_cls.make_expert_params_mapping = staticmethod(
        hcu_make_expert_params_mapping
    )
    routed_experts_cls._vllm_hcu_original_get_expert_weights = get_weights
    routed_experts_cls.get_expert_weights = hcu_get_expert_weights
    routed_experts_cls._vllm_hcu_original_load_weights = load_weights
    routed_experts_cls.load_weights = hcu_load_weights
    # Install RoutedExperts wrappers before exposing the factory: quantization
    # create_weights captures a bound weight_loader while instances are built.
    target._vllm_hcu_original_fused_moe_factory = factory
    target.FusedMoE = hcu_factory
    layer_module._vllm_hcu_original_fused_moe_factory = factory
    layer_module.FusedMoE = hcu_factory
    setattr(target, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
