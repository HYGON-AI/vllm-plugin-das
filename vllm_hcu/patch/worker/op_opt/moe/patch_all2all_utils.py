# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""DeepEP LL FP8/INT8 dispatch selection without source mutation."""

from __future__ import annotations

import functools
from types import ModuleType

from ._common import (
    check_module_marker,
    load_exact_module,
    require_callable,
    require_parameter_names,
)

TARGET_MODULE = "vllm.model_executor.layers.fused_moe.all2all_utils"
PATCH_ID = "worker.op_opt.moe.all2all_utils"
TARGETS = (
    f"{TARGET_MODULE}.maybe_make_prepare_finalize",
    f"{TARGET_MODULE}.maybe_roundup_layer_hidden_size",
)
_MARKER = "_vllm_hcu_all2all_dispatch_applied"
_WRAPPER = "_vllm_hcu_all2all_dispatch_wrapper"


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    wrapped = (
        (target, "maybe_make_prepare_finalize", _WRAPPER),
        (target, "maybe_roundup_layer_hidden_size", _WRAPPER),
    )
    if check_module_marker(target, _MARKER, wrapped):
        return False
    original = require_callable(target, "maybe_make_prepare_finalize", TARGETS[0])
    require_parameter_names(
        original,
        TARGETS[0],
        (
            "moe",
            "quant_config",
            "routing_tables",
            "allow_new_interface",
            "use_monolithic",
            "eep_stage",
        ),
    )
    roundup = require_callable(target, "maybe_roundup_layer_hidden_size", TARGETS[1])
    require_parameter_names(
        roundup,
        TARGETS[1],
        ("hidden_size", "act_dtype", "moe_parallel_config"),
    )

    @functools.wraps(original)
    def hcu_maybe_make_prepare_finalize(
        moe,
        quant_config,
        routing_tables=None,
        allow_new_interface=False,
        use_monolithic=False,
        eep_stage=False,
    ):
        if getattr(
            moe.moe_parallel_config, "use_deepep_auto_kernels", False
        ):
            if quant_config is None:
                raise RuntimeError("DeepEP auto requires a FusedMoEQuantConfig")
            all2all_manager = target.get_ep_all2all_manager(eep_stage)
            if not getattr(all2all_manager, "is_deepep_auto_manager", False):
                raise RuntimeError(
                    "DeepEP auto requires DeepEPAutoAll2AllManager"
                )
            assert moe.dp_size == all2all_manager.dp_world_size
            global_to_physical = None
            physical_to_global = None
            local_expert_global_ids = None
            if routing_tables is not None:
                (
                    global_to_physical,
                    physical_to_global,
                    local_expert_global_ids,
                ) = routing_tables

            vllm_config = target.get_current_vllm_config()
            max_tokens = vllm_config.scheduler_config.max_num_seqs
            speculative_config = vllm_config.speculative_config
            num_speculative_tokens = getattr(
                speculative_config, "num_speculative_tokens", 0
            )
            max_tokens *= 1 + int(num_speculative_tokens or 0)
            handle = all2all_manager.get_handle(
                {
                    "max_num_tokens_per_dp_rank": max_tokens,
                    "token_hidden_size": moe.hidden_dim,
                    "num_ep_ranks": all2all_manager.world_size,
                    "num_global_experts": moe.num_experts,
                    "num_local_experts": (
                        moe.num_experts // all2all_manager.world_size
                    ),
                }
            )
            use_fp8_dispatch = (
                quant_config.quant_dtype == target.current_platform.fp8_dtype()
            )
            ht_prepare_finalize = target.DeepEPHTPrepareAndFinalize(
                handle,
                num_dispatchers=all2all_manager.world_size,
                dp_size=all2all_manager.dp_world_size,
                rank_expert_offset=(
                    all2all_manager.rank * moe.num_local_experts
                ),
            )
            ll_prepare_finalize = target.DeepEPLLPrepareAndFinalize(
                handle,
                max_tokens_per_rank=max_tokens,
                num_dispatchers=all2all_manager.world_size,
                use_fp8_dispatch=use_fp8_dispatch,
                global_to_physical=global_to_physical,
                physical_to_global=physical_to_global,
                local_expert_global_ids=local_expert_global_ids,
            )
            from vllm_hcu.model_executor.layers.fused_moe.prepare_finalize.deepep_auto import (
                DeepEPAutoPrepareAndFinalize,
            )

            return DeepEPAutoPrepareAndFinalize(
                ht_prepare_finalize,
                ll_prepare_finalize,
            )

        prepare_finalize = original(
            moe,
            quant_config,
            routing_tables,
            allow_new_interface,
            use_monolithic,
            eep_stage,
        )
        ll_class = getattr(target, "DeepEPLLPrepareAndFinalize", None)
        if ll_class is None or not isinstance(prepare_finalize, ll_class):
            return prepare_finalize
        if quant_config is None:
            raise RuntimeError("DeepEP LL requires a FusedMoEQuantConfig")
        use_fp8 = quant_config.quant_dtype == target.current_platform.fp8_dtype()
        use_int8 = quant_config.quant_dtype == target.torch.int8
        if use_fp8 and use_int8:
            raise RuntimeError("DeepEP LL cannot enable FP8 and INT8 dispatch together")
        prepare_finalize.use_fp8_dispatch = use_fp8
        prepare_finalize.use_int8_dispatch = use_int8
        return prepare_finalize

    @functools.wraps(roundup)
    def hcu_roundup(hidden_size, act_dtype, moe_parallel_config):
        if not getattr(
            moe_parallel_config, "use_deepep_auto_kernels", False
        ):
            return roundup(hidden_size, act_dtype, moe_parallel_config)
        hidden_size = target.DeepEPHTPrepareAndFinalize.maybe_roundup_layer_hidden_size(
            hidden_size, act_dtype
        )
        return target.DeepEPLLPrepareAndFinalize.maybe_roundup_layer_hidden_size(
            hidden_size
        )

    setattr(hcu_maybe_make_prepare_finalize, _WRAPPER, True)
    setattr(hcu_roundup, _WRAPPER, True)
    target._vllm_hcu_original_maybe_make_prepare_finalize = original
    target._vllm_hcu_original_maybe_roundup_layer_hidden_size = roundup
    target.maybe_make_prepare_finalize = hcu_maybe_make_prepare_finalize
    target.maybe_roundup_layer_hidden_size = hcu_roundup
    setattr(target, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
