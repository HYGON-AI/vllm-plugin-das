# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Per-forward DeepEP HT/LL prepare/finalize selection for HCU."""

from __future__ import annotations

from collections.abc import Callable

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEPrepareAndFinalize,
)

# Keep per-forward HT/LL selection visible in the normal vLLM worker log.
logger = init_logger("vllm.hcu.deepseek_v4_deepep_auto")


def dspark_mooncake_pd_use_low_latency(vllm_config: object) -> bool | None:
    """Return the fixed DeepEP layout for a supported Mooncake P/D role."""

    speculative_config = getattr(vllm_config, "speculative_config", None)
    if getattr(speculative_config, "method", None) != "dspark":
        return None
    kv_transfer_config = getattr(vllm_config, "kv_transfer_config", None)
    if getattr(kv_transfer_config, "kv_connector", None) != "MooncakeConnector":
        return None

    model_config = getattr(vllm_config, "model_config", None)
    architectures = getattr(model_config, "architectures", None)
    if architectures is None:
        hf_config = getattr(model_config, "hf_config", None)
        architectures = getattr(hf_config, "architectures", ())
    if "DeepseekV4ForCausalLM" not in (architectures or ()):
        return None

    role = getattr(kv_transfer_config, "kv_role", None)
    if role == "kv_producer":
        return False
    if role == "kv_consumer":
        return True
    return None


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
        fixed_use_low_latency: bool | None = None,
    ):
        super().__init__()
        self.ht_prepare_finalize = ht_prepare_finalize
        self.ll_prepare_finalize = ll_prepare_finalize
        self._fixed_use_low_latency = fixed_use_low_latency
        self._use_low_latency_snapshot = False
        self._auto_experts: mk.FusedMoEExperts | None = None

    def _snapshot_forward_mode(self) -> None:
        if self._fixed_use_low_latency is None:
            self._use_low_latency_snapshot = _forward_uses_low_latency()
        else:
            self._use_low_latency_snapshot = self._fixed_use_low_latency
        if self._use_low_latency_snapshot:
            logger.info_once(
                "DeepEP auto selected masked low-latency experts for this forward."
            )
        else:
            logger.info_once(
                "DeepEP auto selected contiguous high-throughput experts for this "
                "forward."
            )
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
        if self._fixed_use_low_latency is not None:
            self._use_low_latency_snapshot = self._fixed_use_low_latency
            self._current().post_init_setup(
                getattr(
                    fused_experts,
                    "ll_experts" if self._fixed_use_low_latency else "ht_experts",
                    fused_experts,
                )
            )
            return
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
        if self._fixed_use_low_latency is not None:
            return self._current().supports_async()
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


__all__ = [
    "DeepEPAutoPrepareAndFinalize",
    "dspark_mooncake_pd_use_low_latency",
]
