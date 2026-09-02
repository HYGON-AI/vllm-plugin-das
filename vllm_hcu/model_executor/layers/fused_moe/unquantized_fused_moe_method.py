# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# Modified by Hygon Information Technology Co., Ltd., 2026.

"""HCU unquantized MoE integration."""

from __future__ import annotations

import torch

from vllm.model_executor.layers.fused_moe.oracle.unquantized import (
    UnquantizedMoeBackend,
    make_unquantized_moe_kernel,
)
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod as _Original,
)
from vllm.model_executor.utils import replace_parameter
from vllm_hcu.model_executor.layers.fused_moe.aiter_moe_dispatch import (
    AiterMoeProblem,
    HcuAiterMoeDispatchError,
    aiter_moe_weight_layout_signature,
    prepare_aiter_moe_weights,
    select_aiter_moe_config,
)
from vllm_hcu.platforms import envs as henvs


def _is_hcu_aiter_moe_requested(method: object | None = None) -> bool:
    from vllm_hcu.model_executor.layers.fused_moe.aiter_runtime import (
        is_aiter_moe_requested,
    )

    return is_aiter_moe_requested(getattr(method, "moe", None))


def _expert_routing_tables(
    layer: torch.nn.Module,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    routing_tables = getattr(layer, "_expert_routing_tables", None)
    if callable(routing_tables):
        return routing_tables()

    legacy_routing_tables = getattr(layer, "_maybe_init_expert_routing_tables", None)
    if callable(legacy_routing_tables):
        return legacy_routing_tables()

    return None


class HcuUnquantizedFusedMoEMethod(_Original):
    """Install the AITER ASM layout once instead of caching a second copy."""

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if (
            not _is_hcu_aiter_moe_requested(self)
            or getattr(self, "unquantized_backend", None)
            != UnquantizedMoeBackend.AITER
        ):
            return super().process_weights_after_loading(layer)

        if getattr(layer, "_hcu_aiter_moe_initialized", False):
            return

        original_w13 = layer.w13_weight
        original_w2 = layer.w2_weight
        activation = getattr(self.moe.activation, "value", self.moe.activation)
        problem = AiterMoeProblem(
            M=1,
            E=int(self.moe.num_experts),
            N1=int(original_w13.shape[1]),
            N2=int(original_w2.shape[1]),
            K=int(original_w13.shape[2]),
            top_k=int(self.moe.experts_per_token),
            block_size=0,
            dtype=self.moe.in_dtype,
            device=original_w13.device,
            quant_type="w16a16",
            activation=str(activation),
            use_shuffle=bool(henvs.VLLM_HCU_USE_AITER_MOE_SHUFFLE),
        )
        config = select_aiter_moe_config(
            problem,
            cache_owner=original_w13,
            solution_type="asm" if problem.use_shuffle else None,
        )
        if problem.use_shuffle:
            if config is None or not bool(getattr(config, "need_shuffle", False)):
                raise HcuAiterMoeDispatchError(
                    "HCU AITER BF16 MoE requires an ASM shuffle solution before "
                    "installing the runtime weight layout; " + problem.describe()
                )
            shuffled_w13, shuffled_w2 = prepare_aiter_moe_weights(
                original_w13,
                original_w2,
                config,
                cache_owner=object(),
            )
            replace_parameter(layer, "w13_weight", shuffled_w13)
            replace_parameter(layer, "w2_weight", shuffled_w2)
        elif config is not None and bool(getattr(config, "need_shuffle", False)):
            raise HcuAiterMoeDispatchError(
                "HCU AITER selected a shuffle solution for canonical BF16 "
                "weights; " + problem.describe()
            )

        if config is not None:
            solution = getattr(config, "solution_type", None)
            solution = getattr(solution, "value", solution)
            if solution is None:
                raise HcuAiterMoeDispatchError(
                    "HCU AITER selected a BF16 MoE config without a solution type"
                )
            solution = str(solution).rsplit(".", 1)[-1].lower()
            if not solution:
                raise HcuAiterMoeDispatchError(
                    "HCU AITER selected a BF16 MoE config without a solution type"
                )
            layout = aiter_moe_weight_layout_signature(config)
        else:
            solution = "native"
            layout = None
        for weight in (layer.w13_weight, layer.w2_weight):
            weight.is_shuffled = bool(problem.use_shuffle)
            weight._hcu_aiter_moe_solution_type = solution
            if layout is not None:
                weight._hcu_aiter_moe_weight_layout = layout

        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        self.moe_kernel = make_unquantized_moe_kernel(
            quant_config=self.moe_quant_config,
            moe_config=self.moe,
            backend=self.unquantized_backend,
            experts_cls=self.experts_cls,
            routing_tables=_expert_routing_tables(layer),
        )
        layer._hcu_aiter_moe_initialized = True
