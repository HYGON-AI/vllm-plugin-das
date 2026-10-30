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
    f"{TARGET_MODULE}.DeepEPHTPrepareAndFinalize._get_dispatch_config",
    f"{TARGET_MODULE}.DeepEPHTPrepareAndFinalize._get_combine_config",
    f"{TARGET_MODULE}.DeepEPHTPrepareAndFinalize._finalize",
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
    dispatch_config = require_callable(cls, "_get_dispatch_config", TARGETS[3])
    combine_config = require_callable(cls, "_get_combine_config", TARGETS[4])
    finalize = require_callable(cls, "_finalize", TARGETS[5])
    require_parameter_names(dispatch_config, TARGETS[3], ("self",))
    require_parameter_names(combine_config, TARGETS[4], ("self",))
    require_parameter_names(finalize, TARGETS[5], (
        "self", "output", "fused_expert_output", "topk_weights", "topk_ids",
        "apply_router_weight_on_input", "weight_and_reduce_impl", "do_async",
    ))
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

    @functools.wraps(dispatch_config)
    def hcu_dispatch_config(self):
        return deepep_runtime.step5_ht_launch_config(self, dispatch_config(self))

    @functools.wraps(combine_config)
    def hcu_combine_config(self):
        return deepep_runtime.step5_ht_launch_config(self, combine_config(self), combine=True)

    @functools.wraps(finalize)
    def hcu_finalize(self, output, fused_expert_output, topk_weights, topk_ids,
                     apply_router_weight_on_input, weight_and_reduce_impl, do_async):
        return deepep_runtime.ht_finalize_with_buffer_pair(
            target, self, output, fused_expert_output, topk_weights, topk_ids,
            apply_router_weight_on_input, weight_and_reduce_impl, do_async, finalize,
        )

    cls._get_dispatch_config = hcu_dispatch_config
    cls._get_combine_config = hcu_combine_config
    cls._finalize = hcu_finalize
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
