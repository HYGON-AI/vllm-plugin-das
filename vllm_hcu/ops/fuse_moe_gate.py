# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import GroupedTopKRouter
import os
import torch
import lightop.op as op
import vllm_hcu.platforms.envs as henvs
from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import grouped_topk

class HcuGroupedTopKRouter(GroupedTopKRouter):
    def _valid_grouping(self, router_logits: torch.Tensor) -> bool:
        """Mirror GroupedTopKRouter._compute_routing.<locals>.valid_grouping (not accessible from outside)."""
        num_experts = router_logits.shape[-1]
        if num_experts <= self.num_expert_group:
            return False
        return num_experts % self.num_expert_group == 0

    def _compute_routing(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        indices_type: torch.dtype | None,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        condition = self._valid_grouping(router_logits) and self.e_score_correction_bias is not None and henvs.VLLM_HCU_USE_FUSE_MOE_GATE and henvs.VLLM_HCU_USE_CUSTOM_OPS
        enable_shared_experts_fusion = False
        if condition:
            if self.scoring_func != "sigmoid" or not self.renormalize:
                raise ValueError(
                    "HCU LightOp moe_fused_gate supports only sigmoid scoring "
                    "with renormalize=True; got "
                    f"scoring_func={self.scoring_func!r}, "
                    f"renormalize={self.renormalize!r}. "
                    "Set VLLM_HCU_USE_FUSE_MOE_GATE=0 to use the standard router."
                )
            topk_weights, topk_ids = op.moe_fused_gate(
                router_logits,
                self.e_score_correction_bias,
                self.num_expert_group,
                self.topk_group,
                self.top_k,
                self.num_fused_shared_experts if enable_shared_experts_fusion else 0,
                self.routed_scaling_factor,
                True,  # Apply the router scale for every expert count.
            )       
            return topk_weights, topk_ids
        else:
            return super()._compute_routing(hidden_states, router_logits, indices_type, input_ids=input_ids)
