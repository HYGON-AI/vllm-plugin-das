# SPDX-License-Identifier: Apache-2.0

"""
vllm.model_executor.layers.fused_moe.layer
"""

PATCHES = [
(
"""
    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.runner.forward(
            hidden_states,
            router_logits,
            input_ids,
        )
""",
"""
    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor | None,
        input_ids: torch.Tensor | None = None,
        quanted_hidden_states: torch.Tensor | None = None,
        scale: torch.Tensor | None = None,
        topk_weights: torch.Tensor | None = None,
        topk_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.runner.forward(
            hidden_states,
            router_logits,
            input_ids,
            quanted_hidden_states=quanted_hidden_states,
            scale=scale,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
        )
""",
),
]
