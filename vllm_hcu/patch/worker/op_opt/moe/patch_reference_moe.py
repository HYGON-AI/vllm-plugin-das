# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Use the plain-matmul reference expert kernel when requested."""

from __future__ import annotations

from types import ModuleType

from vllm.logger import init_logger

from vllm_hcu.model_executor.layers.fused_moe.reference_moe import (
    apply_reference_moe,
)
from vllm_hcu.platforms import envs as henvs

from ._common import (
    check_module_marker,
    load_exact_module,
    marked_wrapper,
    require_callable,
    require_class,
    require_parameter_names,
)

TARGET_MODULE = "vllm.model_executor.layers.fused_moe.experts.triton_moe"
PATCH_ID = "worker.op_opt.moe.experts.reference_unquantized"
TARGETS = (f"{TARGET_MODULE}.TritonExperts.apply",)
logger = init_logger(__name__)
_MARKER = "_vllm_hcu_reference_moe_applied"
_WRAPPER_MARKER = "_vllm_hcu_reference_moe_wrapper"
_PARAMETERS = (
    "self",
    "output",
    "hidden_states",
    "w1",
    "w2",
    "topk_weights",
    "topk_ids",
    "activation",
    "global_num_experts",
    "expert_map",
    "a1q_scale",
    "a2_scale",
    "workspace13",
    "workspace2",
    "expert_tokens_meta",
    "apply_router_weight_on_input",
)


def _reject(reason: str) -> None:
    raise RuntimeError(
        f"VLLM_HCU_USE_REFERENCE_MOE is enabled but {reason}; disable it or "
        "use a supported unquantized Triton MoE configuration"
    )


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    experts_cls = require_class(
        target,
        "TritonExperts",
        TARGETS[0].rsplit(".", 1)[0],
    )
    if check_module_marker(
        target,
        _MARKER,
        ((experts_cls, "apply", _WRAPPER_MARKER),),
    ):
        return False

    original = require_callable(experts_cls, "apply", TARGETS[0])
    require_parameter_names(original, TARGETS[0], _PARAMETERS)

    def hcu_reference_apply(
        self,
        output,
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        activation,
        global_num_experts,
        expert_map,
        a1q_scale,
        a2_scale,
        workspace13,
        workspace2,
        expert_tokens_meta,
        apply_router_weight_on_input,
    ):
        if not henvs.VLLM_HCU_USE_REFERENCE_MOE:
            return original(
                self,
                output,
                hidden_states,
                w1,
                w2,
                topk_weights,
                topk_ids,
                activation,
                global_num_experts,
                expert_map,
                a1q_scale,
                a2_scale,
                workspace13,
                workspace2,
                expert_tokens_meta,
                apply_router_weight_on_input,
            )

        if getattr(self, "_lora_context", None) is not None:
            _reject("MoE LoRA is active")
        if a1q_scale is not None or a2_scale is not None:
            _reject("the layer is quantized")
        if not hidden_states.is_floating_point() or not w1.is_floating_point():
            _reject("the layer is not unquantized fp16/bf16/fp32")
        if expert_tokens_meta is not None:
            _reject("the layer uses batched expert token metadata")

        logger.warning_once(
            "VLLM_HCU_USE_REFERENCE_MOE is enabled: routed experts use the "
            "plain-matmul reference kernel instead of the fused MoE kernel"
        )
        apply_reference_moe(
            output=output,
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=activation,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            apply_router_weight_on_input=apply_router_weight_on_input,
            # Both are consumed by the fused kernel but are independent of
            # quantization, so an unquantized layer can legitimately have them.
            w1_bias=self.w1_bias,
            w2_bias=self.w2_bias,
            # The fused SILU path clamps when the layer sets a limit; passing it
            # through keeps the reference activation the same function.
            clamp_limit=self.activation_config.clamp_limit,
        )
        return None

    setattr(experts_cls, "_vllm_hcu_original_apply", original)
    setattr(
        experts_cls,
        "apply",
        marked_wrapper(original, _WRAPPER_MARKER, hcu_reference_apply),
    )
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
