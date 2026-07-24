# SPDX-License-Identifier: Apache-2.0
"""Pass HCU kernel dimensions and pre-quantized inputs through modular MoE."""

from __future__ import annotations

import functools
from types import ModuleType

from ._common import load_exact_module, require_callable, require_class, require_parameter_names

TARGET_MODULE = "vllm.model_executor.layers.fused_moe.fused_moe_modular_method"
PATCH_ID = "worker.op_opt.moe.fused_moe_modular_method"
TARGETS = (
    f"{TARGET_MODULE}.FusedMoEModularMethod.make",
    f"{TARGET_MODULE}.FusedMoEModularMethod.apply",
)
_MARKER = "_vllm_hcu_modular_method_applied"


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    if getattr(target, _MARKER, False):
        return False
    cls = require_class(target, "FusedMoEModularMethod", TARGETS[0].rsplit(".", 1)[0])
    make = require_callable(cls, "make", TARGETS[0])
    method_apply = require_callable(cls, "apply", TARGETS[1])
    require_parameter_names(
        make,
        TARGETS[0],
        (
            "routed_experts",
            "old_quant_method",
            "prepare_finalize",
        ),
    )
    require_parameter_names(
        method_apply,
        TARGETS[1],
        (
            "self", "layer", "x", "topk_weights", "topk_ids",
            "shared_experts", "shared_experts_input",
        ),
    )

    @functools.wraps(make)
    def hcu_make(routed_experts, old_quant_method, prepare_finalize):
        kernel = target.FusedMoEKernel(
            prepare_finalize,
            old_quant_method.select_gemm_impl(prepare_finalize, routed_experts),
            N=getattr(old_quant_method, "N", -1),
            K=getattr(old_quant_method, "K", -1),
        )
        return cls(old_quant_method, kernel)

    @functools.wraps(method_apply)
    def hcu_apply(
        self,
        layer,
        x,
        topk_weights,
        topk_ids,
        shared_experts,
        shared_experts_input,
        use_nn_moe=False,
        i_q=None,
        i_s=None,
    ):
        if use_nn_moe:
            raise RuntimeError("HCU v0.25.1 modular MoE does not support use_nn_moe=True")
        if (i_q is None) != (i_s is None):
            raise ValueError("HCU modular MoE requires i_q and i_s together")
        if self.moe_kernel is None:
            raise RuntimeError("HCU modular MoE kernel was not initialized")
        return self.moe_kernel.apply(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            expert_map=layer.expert_map,
            shared_experts=shared_experts,
            shared_experts_input=shared_experts_input,
            quanted_hidden_states=i_q,
            scale=i_s,
        )

    del hcu_apply.__wrapped__

    cls._vllm_hcu_original_make = make
    cls.make = staticmethod(hcu_make)
    cls._vllm_hcu_original_apply = method_apply
    cls.apply = hcu_apply
    setattr(target, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
