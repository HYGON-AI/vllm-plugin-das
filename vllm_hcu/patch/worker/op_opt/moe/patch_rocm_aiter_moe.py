# SPDX-License-Identifier: Apache-2.0
"""Add GELU-tanh support to the ROCm AITER expert implementation."""

from __future__ import annotations

import functools
import types
from types import ModuleType, SimpleNamespace

from ._common import (
    PatchCompatibilityError,
    load_exact_module,
    require_callable,
    require_class,
    require_parameter_names,
)

TARGET_MODULE = "vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe"
PATCH_ID = "worker.op_opt.moe.experts.rocm_aiter"
TARGETS = (
    f"{TARGET_MODULE}.ActivationMethod",
    f"{TARGET_MODULE}.rocm_aiter_fused_experts",
    f"{TARGET_MODULE}.AiterExperts._supports_activation",
)
_MARKER = "_vllm_hcu_aiter_gelu_tanh_applied"


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    if getattr(target, _MARKER, False):
        return False
    activation_method = require_class(target, "ActivationMethod", TARGETS[0])
    fused_experts = require_callable(target, "rocm_aiter_fused_experts", TARGETS[1])
    experts_class = require_class(target, "AiterExperts", TARGETS[2].rsplit(".", 1)[0])
    supports = require_callable(experts_class, "_supports_activation", TARGETS[2])
    require_parameter_names(
        fused_experts,
        TARGETS[1],
        (
            "hidden_states", "w1", "w2", "topk_weights", "topk_ids",
            "moe_config", "activation", "apply_router_weight_on_input",
            "expert_map", "quant_config", "a1q_scale", "num_local_tokens",
            "output_dtype",
        ),
    )
    require_parameter_names(supports, TARGETS[2], ("activation",))
    values = {member.name: member.value for member in activation_method}
    if values != {"SILU": 0, "GELU": 1}:
        raise PatchCompatibilityError(
            f"required HCU target {TARGETS[0]} has unexpected values {values}"
        )
    hcu_activation_method = target.IntEnum(
        "ActivationMethod",
        {"SILU": 0, "GELU": 1, "GELU_TANH": 3},
        module=target.__name__,
    )
    moe_activation = target.MoEActivation
    gelu_tanh = getattr(moe_activation, "GELU_TANH", None)
    if gelu_tanh is None:
        raise PatchCompatibilityError("MoEActivation.GELU_TANH is missing")

    # Execute the audited upstream function body with two local enum proxies
    # only for GELU_TANH.  This retains all current kernel/quant branches and
    # avoids process-global enum swapping under concurrent calls.
    special_globals = dict(fused_experts.__globals__)
    special_globals["MoEActivation"] = SimpleNamespace(
        SILU=moe_activation.SILU,
        GELU=gelu_tanh,
        SWIGLUOAI=moe_activation.SWIGLUOAI,
    )
    special_globals["ActivationMethod"] = SimpleNamespace(
        SILU=hcu_activation_method.SILU,
        GELU=hcu_activation_method.GELU_TANH,
    )
    special_impl = types.FunctionType(
        fused_experts.__code__,
        special_globals,
        fused_experts.__name__,
        fused_experts.__defaults__,
        fused_experts.__closure__,
    )
    special_impl.__kwdefaults__ = fused_experts.__kwdefaults__

    @functools.wraps(fused_experts)
    def hcu_fused_experts(*args, **kwargs):
        activation = kwargs.get("activation")
        if activation is None and len(args) > 6:
            activation = args[6]
        if activation == gelu_tanh:
            return special_impl(*args, **kwargs)
        return fused_experts(*args, **kwargs)

    @functools.wraps(supports)
    def hcu_supports_activation(activation):
        return activation == gelu_tanh or supports(activation)

    target._vllm_hcu_original_activation_method = activation_method
    target.ActivationMethod = hcu_activation_method
    target._vllm_hcu_original_rocm_aiter_fused_experts = fused_experts
    target.rocm_aiter_fused_experts = hcu_fused_experts
    experts_class._vllm_hcu_original_supports_activation = supports
    experts_class._supports_activation = staticmethod(hcu_supports_activation)
    setattr(target, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
