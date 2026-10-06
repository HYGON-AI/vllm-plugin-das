# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Advertise the audited Triton INT8 MoE path on ROCm/HCU."""

from __future__ import annotations

import functools
import importlib
from types import ModuleType

from ._common import load_exact_module, require_callable, require_class, require_parameter_names
from . import patch_moe_align_block_size

TARGET_MODULE = "vllm.model_executor.layers.fused_moe.experts.triton_moe"
PATCH_ID = "worker.op_opt.moe.experts.triton_int8"
TARGETS = (
    f"{TARGET_MODULE}.TritonExperts._supports_quant_scheme",
    f"{TARGET_MODULE}.moe_align_block_size",
    f"{TARGET_MODULE}.TritonWNA16Experts.apply",
)
_MARKER = "_vllm_hcu_triton_int8_applied"


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    if getattr(target, _MARKER, False):
        return False
    cls = require_class(target, "TritonExperts", TARGETS[0].rsplit(".", 1)[0])
    wna16_cls = require_class(
        target,
        "TritonWNA16Experts",
        TARGETS[2].rsplit(".", 1)[0],
    )
    original = require_callable(cls, "_supports_quant_scheme", TARGETS[0])
    original_wna16_apply = require_callable(wna16_cls, "apply", TARGETS[2])
    require_parameter_names(original, TARGETS[0], ("weight_key", "activation_key"))
    require_parameter_names(
        original_wna16_apply,
        TARGETS[2],
        (
            "self", "output", "hidden_states", "w1", "w2", "topk_weights",
            "topk_ids", "activation", "global_num_experts", "expert_map",
            "a1q_scale", "a2_scale", "workspace13", "workspace2",
            "expert_tokens_meta", "apply_router_weight_on_input",
        ),
    )
    align_module = importlib.import_module(patch_moe_align_block_size.TARGET_MODULE)
    patch_moe_align_block_size.apply_to_module(align_module)
    hcu_moe_align_block_size = require_callable(
        align_module,
        "moe_align_block_size",
        TARGETS[1],
    )

    @functools.wraps(original)
    def hcu_supports_quant_scheme(weight_key, activation_key):
        if target.current_platform.is_rocm() and (
            weight_key,
            activation_key,
        ) == (target.kInt8StaticChannelSym, target.kInt8DynamicTokenSym):
            return True
        return original(weight_key, activation_key)

    @functools.wraps(original_wna16_apply)
    def hcu_wna16_apply(
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
        quant_config = getattr(self, "quant_config", None)
        moe_config = getattr(self, "moe_config", None)
        is_channel_aiter = bool(
            getattr(quant_config, "use_int4_w4a16", False)
            and getattr(quant_config, "_hcu_channel_w4a16", False)
            and getattr(quant_config, "_hcu_aiter_w4a16_available", False)
            and getattr(moe_config, "moe_backend", None) == "aiter"
        )
        if is_channel_aiter:
            from vllm_hcu.model_executor.layers.quantization import (
                compressed_tensors_moe_runtime as hcu_runtime,
            )

            result = hcu_runtime.apply_aiter_w4a16_moe(
                hidden_states=hidden_states,
                w1=w1,
                w2=w2,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                activation=activation,
                global_num_experts=global_num_experts,
                expert_map=expert_map,
                quant_config=quant_config,
                vllm_moe_config=moe_config,
                apply_router_weight_on_input=apply_router_weight_on_input,
            )
            if result is not None:
                output.copy_(result)
                return None

        return original_wna16_apply(
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

    cls._vllm_hcu_original_supports_quant_scheme = original
    cls._supports_quant_scheme = staticmethod(hcu_supports_quant_scheme)
    wna16_cls._vllm_hcu_original_apply = original_wna16_apply
    wna16_cls.apply = hcu_wna16_apply
    target.moe_align_block_size = hcu_moe_align_block_size
    setattr(target, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
