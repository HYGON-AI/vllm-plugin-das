# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""DeepEP high-throughput per-token quantization and alignment."""

from __future__ import annotations

import functools
from types import ModuleType

from ._common import load_exact_module, require_callable, require_class, require_parameter_names

TARGET_MODULE = "vllm.model_executor.layers.fused_moe.prepare_finalize.deepep_ht"
PATCH_ID = "worker.op_opt.moe.prepare_finalize.deepep_ht"
TARGETS = (
    f"{TARGET_MODULE}.DeepEPHTPrepareAndFinalize._do_dispatch",
    f"{TARGET_MODULE}.DeepEPHTPrepareAndFinalize._receiver",
    f"{TARGET_MODULE}.DeepEPHTPrepareAndFinalize.prepare_async",
)
_MARKER = "_vllm_hcu_deepep_ht_applied"


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    if getattr(target, _MARKER, False):
        return False
    from vllm_hcu.model_executor.layers.fused_moe import deepep_runtime

    cls = require_class(target, "DeepEPHTPrepareAndFinalize", TARGETS[0].rsplit(".", 1)[0])
    do_dispatch = require_callable(cls, "_do_dispatch", TARGETS[0])
    receiver = require_callable(cls, "_receiver", TARGETS[1])
    prepare = require_callable(cls, "prepare_async", TARGETS[2])
    require_parameter_names(
        do_dispatch,
        TARGETS[0],
        (
            "self",
            "tokens",
            "token_scales",
            "rank_topk_ids",
            "rank_topk_weights",
            "num_experts",
            "a1_scale",
            "quant_config",
            "defer_input_quant",
        ),
    )
    require_parameter_names(
        receiver,
        TARGETS[1],
        (
            "self",
            "event",
            "has_scales",
            "token_data",
            "expert_topk_ids",
            "num_experts",
            "expert_num_tokens_per_expert_list",
            "expert_topk_weights",
            "a1_scale",
            "quant_config",
            "defer_input_quant",
        ),
    )
    require_parameter_names(
        prepare,
        TARGETS[2],
        (
            "self",
            "a1",
            "topk_weights",
            "topk_ids",
            "num_experts",
            "expert_map",
            "apply_router_weight_on_input",
            "quant_config",
            "defer_input_quant",
        ),
    )

    @functools.wraps(do_dispatch)
    def hcu_do_dispatch(
        self,
        tokens,
        token_scales,
        rank_topk_ids,
        rank_topk_weights,
        num_experts,
        a1_scale,
        quant_config,
        defer_input_quant,
    ):
        return deepep_runtime.ht_do_dispatch(
            target,
            self,
            tokens,
            token_scales,
            rank_topk_ids,
            rank_topk_weights,
            num_experts,
            a1_scale,
            quant_config,
            defer_input_quant,
        )

    @functools.wraps(receiver)
    def hcu_receiver(
        self,
        event,
        has_scales,
        token_data,
        expert_topk_ids,
        num_experts,
        expert_num_tokens_per_expert_list,
        expert_topk_weights,
        a1_scale,
        quant_config,
        defer_input_quant,
    ):
        return deepep_runtime.ht_receiver(
            target,
            self,
            event,
            has_scales,
            token_data,
            expert_topk_ids,
            num_experts,
            expert_num_tokens_per_expert_list,
            expert_topk_weights,
            a1_scale,
            quant_config,
            defer_input_quant,
        )

    @functools.wraps(prepare)
    def hcu_prepare_async(
        self,
        a1,
        topk_weights,
        topk_ids,
        num_experts,
        expert_map,
        apply_router_weight_on_input,
        quant_config,
        defer_input_quant=False,
    ):
        return deepep_runtime.ht_prepare_async(
            target,
            self,
            a1,
            topk_weights,
            topk_ids,
            num_experts,
            expert_map,
            apply_router_weight_on_input,
            quant_config,
            defer_input_quant,
        )

    cls._vllm_hcu_original_do_dispatch = do_dispatch
    cls._do_dispatch = hcu_do_dispatch
    cls._vllm_hcu_original_receiver = receiver
    cls._receiver = hcu_receiver
    cls._vllm_hcu_original_prepare_async = prepare
    cls.prepare_async = hcu_prepare_async
    setattr(target, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
