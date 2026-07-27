# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Per-forward DeepEP HT/LL prepare/finalize selection for HCU."""

from __future__ import annotations

from collections.abc import Callable

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEPrepareAndFinalize,
)


def _forward_uses_low_latency() -> bool:
    from vllm.forward_context import get_forward_context

    try:
        context = get_forward_context()
    except AssertionError:
        return False
    return bool(getattr(context, "deepep_auto_use_low_latency", False))


class DeepEPAutoPrepareAndFinalize(mk.FusedMoEPrepareAndFinalizeModular):
    """Route one forward consistently through DeepEP HT or LL."""

    def __init__(
        self,
        ht_prepare_finalize: FusedMoEPrepareAndFinalize,
        ll_prepare_finalize: FusedMoEPrepareAndFinalize,
    ):
        super().__init__()
        self.ht_prepare_finalize = ht_prepare_finalize
        self.ll_prepare_finalize = ll_prepare_finalize
        self._use_low_latency_snapshot = False
        self._auto_experts: mk.FusedMoEExperts | None = None

    def _snapshot_forward_mode(self) -> None:
        self._use_low_latency_snapshot = _forward_uses_low_latency()
        if self._auto_experts is not None:
            self._auto_experts.set_deepep_auto_use_low_latency(
                self._use_low_latency_snapshot
            )

    def _current(self) -> FusedMoEPrepareAndFinalize:
        return (
            self.ll_prepare_finalize
            if self._use_low_latency_snapshot
            else self.ht_prepare_finalize
        )

    def post_init_setup(self, fused_experts: mk.FusedMoEExperts):
        self._auto_experts = fused_experts
        self.ht_prepare_finalize.post_init_setup(
            getattr(fused_experts, "ht_experts", fused_experts)
        )
        self.ll_prepare_finalize.post_init_setup(
            getattr(fused_experts, "ll_experts", fused_experts)
        )

    def num_dispatchers(self) -> int:
        return self._current().num_dispatchers()

    def output_is_reduced(self) -> bool:
        return self._current().output_is_reduced()

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return self._current().activation_format

    def max_num_tokens_per_rank(self) -> int | None:
        return self._current().max_num_tokens_per_rank()

    def topk_indices_dtype(self) -> torch.dtype | None:
        return self._current().topk_indices_dtype()

    def supports_async(self) -> bool:
        return (
            self.ht_prepare_finalize.supports_async()
            and self.ll_prepare_finalize.supports_async()
        )

    def prepare_async(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool = False,
    ) -> Callable | tuple[Callable, Callable]:
        self._snapshot_forward_mode()
        return self._current().prepare_async(
            a1,
            topk_weights,
            topk_ids,
            num_experts,
            expert_map,
            apply_router_weight_on_input,
            quant_config,
            defer_input_quant,
        )

    def prepare(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool = False,
    ) -> mk.PrepareResultType:
        self._snapshot_forward_mode()
        return self._current().prepare(
            a1,
            topk_weights,
            topk_ids,
            num_experts,
            expert_map,
            apply_router_weight_on_input,
            quant_config,
            defer_input_quant,
        )

    def finalize_async(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> Callable | tuple[Callable, Callable]:
        return self._current().finalize_async(
            output,
            fused_expert_output,
            topk_weights,
            topk_ids,
            apply_router_weight_on_input,
            weight_and_reduce_impl,
        )

    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> None:
        return self._current().finalize(
            output,
            fused_expert_output,
            topk_weights,
            topk_ids,
            apply_router_weight_on_input,
            weight_and_reduce_impl,
        )


__all__ = ["DeepEPAutoPrepareAndFinalize"]
