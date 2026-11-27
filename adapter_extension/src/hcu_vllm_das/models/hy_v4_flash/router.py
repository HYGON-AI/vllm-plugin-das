# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""HY4 Flash's single-group sigmoid routing with a stable reduction order.

HCU grouped-topk may emit the same experts in different slot orders. This
changes normalization/expert-sum rounding even for identical router logits.
Select a deterministic score order *before* normalizing, not by sorting the
already-normalized output. No global routing or Torch function is replaced.
"""

import torch

from vllm import envs
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import (
    GroupedTopKRouter,
)


def stable_sigmoid_topk(
    logits: torch.Tensor,
    bias: torch.Tensor,
    top_k: int,
    renormalize: bool,
    routed_scaling_factor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bias selects experts; original FP32 sigmoid scores provide their weights."""
    if logits.ndim != 2 or logits.dtype != torch.float32:
        raise ValueError("Flash routing requires a matrix of FP32 router logits")
    if (
        bias.ndim != 1
        or bias.shape[0] != logits.shape[1]
        or bias.dtype != torch.float32
        or bias.device != logits.device
    ):
        raise ValueError("Flash routing requires matching FP32 expert bias")
    if type(top_k) is not int or not 0 < top_k <= logits.shape[1]:
        raise ValueError("Flash routing requires 0 < top_k <= number of experts")
    scores = logits.sigmoid()
    # Stable ties keep ascending logical expert ID. The same slot order is
    # used by both the normalization reduction and the downstream expert sum.
    ids = torch.argsort(scores + bias, dim=-1, descending=True, stable=True)[:, :top_k]
    weights = scores.gather(1, ids)
    if renormalize:
        weights = weights / weights.sum(dim=-1, keepdim=True)
    if routed_scaling_factor != 1.0:
        weights = weights * routed_scaling_factor
    return weights, ids.to(torch.int32)


class HYV4FlashStableRouter(GroupedTopKRouter):
    """Retain Core's padding/capture/EPLB lifecycle, specialize only Flash math."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if (
            self.num_expert_group != 1
            or self.topk_group != 1
            or self.scoring_func != "sigmoid"
            or self.e_score_correction_bias is None
            or self.num_fused_shared_experts != 0
        ):
            raise ValueError(
                "Stable Flash routing requires biased single-group sigmoid"
            )

    @property
    def supports_monolithic_kernels(self) -> bool:
        # Internal routing would bypass this class's stable score/slot order.
        return False

    @classmethod
    def from_router(cls, router: GroupedTopKRouter):
        if not isinstance(router, GroupedTopKRouter):
            raise TypeError("Flash requires the Core grouped-topk router contract")
        result = cls(
            top_k=router.top_k,
            global_num_experts=router.global_num_experts,
            num_expert_group=router.num_expert_group,
            topk_group=router.topk_group,
            renormalize=router.renormalize,
            scoring_func=router.scoring_func,
            routed_scaling_factor=router.routed_scaling_factor,
            e_score_correction_bias=router.e_score_correction_bias,
            num_fused_shared_experts=router.num_fused_shared_experts,
            eplb_state=router.eplb_state,
            skip_padding=router.skip_padding,
        )
        result.set_capture_fn(router.capture_fn)
        return result

    def _compute_routing(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        indices_type: torch.dtype | None,
        *,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            hidden_states.shape[0] != router_logits.shape[0]
            or router_logits.shape[-1] != self.global_num_experts
        ):
            raise ValueError("Flash routing token/expert dimensions do not match")
        weights, ids = stable_sigmoid_topk(
            router_logits,
            self.e_score_correction_bias,
            self.top_k,
            self.renormalize,
            self.routed_scaling_factor,
        )
        if (
            self.skip_padding
            and envs.VLLM_MOE_SKIP_PADDING
            and is_forward_context_available()
        ):
            is_padding = get_forward_context().is_padding
            if is_padding is not None:
                padding = is_padding[: ids.shape[0]].unsqueeze(1)
                weights = weights.masked_fill(padding, 0)
                ids = ids.masked_fill(padding, -1)
        # BaseRouter applies capture -> EPLB -> requested index dtype conversion.
        return weights, ids
