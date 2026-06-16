# SPDX-License-Identifier: Apache-2.0

"""
vllm.model_executor.layers.fused_moe.fused_moe_modular_method: N/K for kernel ctor, use_nn_moe on apply
"""

PATCHES = [
(
"""
                inplace=inplace,
""",
"""
                inplace=inplace,
                N=old_quant_method.N if hasattr(old_quant_method, "N") else -1,
                K=old_quant_method.K if hasattr(old_quant_method, "K") else -1,
""",
),
(
"""
    ) -> torch.Tensor:
""",
"""
        use_nn_moe: bool | None = False,
        i_q: torch.Tensor | None = None,
        i_s: torch.Tensor | None = None,
    ) -> torch.Tensor:
""",
),
(
"""
        return self.moe_kernel.apply(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            expert_map=None if self.disable_expert_map else layer.expert_map,
            shared_experts_input=shared_experts_input,
        )
""",
"""
        return self.moe_kernel.apply(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            expert_map=None if self.disable_expert_map else layer.expert_map,
            shared_experts_input=shared_experts_input,
            quanted_hidden_states=i_q,
            scale=i_s,
        )
""",
),
]
