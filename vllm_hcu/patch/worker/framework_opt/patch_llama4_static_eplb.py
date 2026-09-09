# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Direct-load Llama4 all-expert tensors through the static EPLB row."""

from __future__ import annotations

import functools
from types import ModuleType

from ._common import (
    PatchCompatibilityError,
    load_exact_module,
    require_callable,
    require_class,
    require_exact_signature,
)


TARGET_MODULE = "vllm.model_executor.models.llama4"
PATCH_ID = "worker.framework_opt.model_loader.llama4_static_eplb_fused"
TARGET = f"{TARGET_MODULE}.Llama4Model.load_moe_expert_weights"
_MODULE_MARKER = "_vllm_hcu_llama4_static_eplb_fused_applied"
_WRAPPER_MARKER = "_vllm_hcu_llama4_static_eplb_fused_wrapper"


def apply_to_module(module: ModuleType) -> bool:
    """Wrap only the audited Llama4 all-expert checkpoint helper."""

    llama4 = load_exact_module(TARGET_MODULE, module)
    model_class = require_class(llama4, "Llama4Model", f"{TARGET_MODULE}.Llama4Model")
    current = require_callable(model_class, "load_moe_expert_weights", TARGET)
    if getattr(llama4, _MODULE_MARKER, False):
        if not getattr(current, _WRAPPER_MARKER, False):
            raise PatchCompatibilityError(
                f"required HCU patch marker for {TARGET} is stale"
            )
        return False
    require_exact_signature(
        current,
        TARGET,
        positional=(
            "self",
            "name",
            "loaded_weight",
            "params_dict",
            "loaded_params",
            "expert_params_mapping",
            "fused",
        ),
        defaults={"fused": True},
    )
    is_pp_missing_parameter = require_callable(
        llama4,
        "is_pp_missing_parameter",
        f"{TARGET_MODULE}.is_pp_missing_parameter",
    )
    extract_layer_index = require_callable(
        llama4,
        "extract_layer_index",
        f"{TARGET_MODULE}.extract_layer_index",
    )
    original = current

    @functools.wraps(original)
    def hcu_load_moe_expert_weights(
        self,
        name,
        loaded_weight,
        params_dict,
        loaded_params,
        expert_params_mapping,
        fused=True,
    ):
        if not fused:
            return original(
                self,
                name,
                loaded_weight,
                params_dict,
                loaded_params,
                expert_params_mapping,
                fused=fused,
            )

        fused_checkpoint_names: list[str] = []
        for _, weight_name, _, _ in expert_params_mapping:
            try:
                expert_prefix, _, projection, _ = weight_name.split(".")
            except ValueError:
                # Preserve the audited upstream error for an incompatible
                # mapping whenever static EPLB is not known to be active.
                return original(
                    self,
                    name,
                    loaded_weight,
                    params_dict,
                    loaded_params,
                    expert_params_mapping,
                    fused=fused,
                )
            fused_checkpoint_names.append(f"{expert_prefix}.{projection}")
        if not any(checkpoint_name in name for checkpoint_name in fused_checkpoint_names):
            return original(
                self,
                name,
                loaded_weight,
                params_dict,
                loaded_params,
                expert_params_mapping,
                fused=fused,
            )

        layer_idx = extract_layer_index(name)
        try:
            runner = self.layers[layer_idx].feed_forward.experts
            routed_experts = runner.routed_experts
        except (AttributeError, IndexError, TypeError):
            return original(
                self,
                name,
                loaded_weight,
                params_dict,
                loaded_params,
                expert_params_mapping,
                fused=fused,
            )
        row = getattr(routed_experts, "_vllm_hcu_static_eplb_row", None)
        if row is None:
            return original(
                self,
                name,
                loaded_weight,
                params_dict,
                loaded_params,
                expert_params_mapping,
                fused=fused,
            )

        # Llama4's fused checkpoint stores all logical experts in dimension
        # zero.  Preserve its audited transpose/split rules, then feed each
        # logical tensor to the standard RoutedExperts loader.  The bound
        # common loader fans that tensor out to the configured physical slots.
        if getattr(loaded_weight, "ndim", None) == 3:
            loaded_weight = loaded_weight.transpose(-1, -2)
            if "experts.gate_up_proj" in name:
                loaded_weight = loaded_weight.chunk(2, dim=-2)

        expert_param_loaded = False
        for param_name, weight_name, _expert_id, shard_id in expert_params_mapping:
            new_loaded_weight = loaded_weight
            try:
                expert_prefix, _, projection, _ = weight_name.split(".")
            except ValueError as exc:
                raise PatchCompatibilityError(
                    f"required HCU patch target {TARGET} received an "
                    f"incompatible fused expert mapping {weight_name!r}"
                ) from exc
            checkpoint_name = f"{expert_prefix}.{projection}"
            parameter_name = f"{param_name}weight"
            if checkpoint_name not in name:
                continue

            full_param_name = name.replace(checkpoint_name, parameter_name)
            if is_pp_missing_parameter(name, self):
                continue
            if (
                name.endswith(".bias") or name.endswith("_bias")
            ) and name not in params_dict:
                continue

            try:
                param = params_dict[full_param_name]
                weight_loader = param.weight_loader
            except (AttributeError, KeyError) as exc:
                raise PatchCompatibilityError(
                    "Static EPLB Llama4 fused loading could not resolve "
                    f"the audited parameter {full_param_name!r}."
                ) from exc

            if "w13" in full_param_name:
                if shard_id not in ("w1", "w3"):
                    raise PatchCompatibilityError(
                        "Static EPLB Llama4 fused w13 loading received "
                        f"invalid shard {shard_id!r}."
                    )
                if not isinstance(new_loaded_weight, tuple) or len(new_loaded_weight) != 2:
                    raise ValueError(
                        "Static EPLB Llama4 fused gate/up checkpoint must "
                        "split into exactly two expert tensors."
                    )
                new_loaded_weight = new_loaded_weight[0 if shard_id == "w1" else 1]

            num_logical_experts = max(row, default=-1) + 1
            if (
                isinstance(num_logical_experts, bool)
                or not isinstance(num_logical_experts, int)
                or num_logical_experts <= 0
                or set(row) != set(range(num_logical_experts))
            ):
                raise ValueError(
                    "Static EPLB Llama4 fused loading received an invalid "
                    "bound logical-expert row."
                )
            if (
                getattr(new_loaded_weight, "ndim", None) is None
                or new_loaded_weight.ndim < 1
                or new_loaded_weight.shape[0] != num_logical_experts
            ):
                raise ValueError(
                    "Static EPLB Llama4 fused checkpoint has logical-expert "
                    f"dimension {getattr(new_loaded_weight, 'shape', None)!r}; "
                    f"expected {num_logical_experts} rows."
                )

            for logical_expert_id, logical_weight in enumerate(
                new_loaded_weight.unbind(0)
            ):
                weight_loader(
                    param=param,
                    loaded_weight=logical_weight,
                    weight_name=full_param_name,
                    shard_id=shard_id,
                    expert_id=logical_expert_id,
                    return_success=True,
                )
            loaded_params.add(full_param_name)
            expert_param_loaded = True

        return expert_param_loaded

    setattr(hcu_load_moe_expert_weights, _WRAPPER_MARKER, True)
    setattr(llama4, "_vllm_hcu_original_load_moe_expert_weights", original)
    setattr(model_class, "load_moe_expert_weights", hcu_load_moe_expert_weights)
    setattr(llama4, _MODULE_MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET", "TARGET_MODULE", "apply", "apply_to_module"]
