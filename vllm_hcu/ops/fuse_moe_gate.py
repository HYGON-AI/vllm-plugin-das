# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

import torch
import vllm_hcu.platforms.envs as henvs
from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import GroupedTopKRouter

from vllm_hcu.model_executor.layers.fused_moe.lightop_routing import (
    lightop_moe_gate_kwargs,
)

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
        condition = (
            self._valid_grouping(router_logits)
            and self.e_score_correction_bias is not None
            and henvs.VLLM_HCU_USE_FUSE_MOE_GATE
            and henvs.VLLM_HCU_USE_CUSTOM_OPS
        )
        enable_shared_experts_fusion = False
        if condition:
            # Keep this entry point in lock-step with router_runtime.py.  The
            # current LightOp ABI is limited to sigmoid + renormalize; a
            # future backend can opt into more modes through the capability
            # hook without requiring another vLLM condition change.
            scoring_func = getattr(self, "scoring_func", None)
            renormalize = getattr(self, "renormalize", None)
            try:
                import lightop.moe as lightop_moe
            except ImportError:
                if scoring_func != "sigmoid" or not bool(renormalize):
                    return super()._compute_routing(
                        hidden_states,
                        router_logits,
                        indices_type,
                        input_ids=input_ids,
                    )
                raise
            gate_kwargs = lightop_moe_gate_kwargs(
                lightop_moe,
                scoring_func,
                renormalize,
            )
            if gate_kwargs is None:
                return super()._compute_routing(
                    hidden_states,
                    router_logits,
                    indices_type,
                    input_ids=input_ids,
                )
            from lightop.moe import moe_fused_gate as lightop_moe_fused_gate

            topk_weights, topk_ids = lightop_moe_fused_gate(
                router_logits,
                self.e_score_correction_bias,
                self.num_expert_group,
                self.topk_group,
                self.top_k,
                self.num_fused_shared_experts if enable_shared_experts_fusion else 0,
                self.routed_scaling_factor,
                # FusedMoE gives the router 1.0 when MoERunner owns output
                # scaling; otherwise LightOp must scale the routing weights.
                self.routed_scaling_factor != 1.0,
                **gate_kwargs,
            )
            return topk_weights, topk_ids
        else:
            return super()._compute_routing(hidden_states, router_logits, indices_type, input_ids=input_ids)
