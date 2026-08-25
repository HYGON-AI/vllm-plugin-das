# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

import torch
import vllm_hcu.platforms.envs as henvs
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import GroupedTopKRouter


logger = init_logger(__name__)

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
            try:
                from lightop import moe as lightop_moe
            except (ImportError, AttributeError):
                from lightop import op as lightop_moe

                logger.warning_once(
                    "Using deprecated lightop.op MoE APIs because lightop.moe is "
                    "unavailable; upgrade LightOp."
                )
            topk_weights, topk_ids = lightop_moe.moe_fused_gate(
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
            return super()._compute_routing(hidden_states, router_logits, indices_type, input_ids=input_ids)
