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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        condition = self._valid_grouping(router_logits) and self.e_score_correction_bias is not None and henvs.VLLM_HCU_USE_FUSE_MOE_GATE and henvs.VLLM_HCU_USE_CUSTOM_OPS
        enable_shared_experts_fusion = False
        if condition:
            topk_weights, topk_ids = op.moe_fused_gate(
                router_logits,
                self.e_score_correction_bias,
                self.num_expert_group,
                self.topk_group,
                self.top_k,
                self.num_fused_shared_experts if enable_shared_experts_fusion else 0,
                self.routed_scaling_factor,
            )       
            return topk_weights, topk_ids
        else:
            return super()._compute_routing(hidden_states, router_logits, indices_type)