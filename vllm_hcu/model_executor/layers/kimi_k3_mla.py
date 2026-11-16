# SPDX-License-Identifier: Apache-2.0
"""Fused MLA projection preserving the audited Kimi reference execution order."""
import torch
from vllm.models.kimi_k3.amd.mla import KimiK3MultiHeadLatentAttentionWrapper


class HcuKimiK3MergedMLAWrapper(KimiK3MultiHeadLatentAttentionWrapper):
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not getattr(self, "_hcu_fused_output_gate", False):
            return super().forward(positions, hidden_states, llama_4_scaling)
        q_c = None
        gate_logits = None
        if self.q_lora_rank is not None:
            assert self.fused_qkv_a_proj is not None
            assert self.q_a_layernorm is not None
            assert self.q_b_proj is not None
            qkv_lora = self.fused_qkv_a_proj(hidden_states)[0]
            q_c, kv_lora, gate_logits = qkv_lora.split(
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim,
                 self.num_heads * self.v_head_dim],
                dim=-1,
            )
            q_c = self.q_a_layernorm(q_c)
            q = self.q_b_proj(q_c)[0]
        else:
            assert self.kv_a_proj_with_mqa is not None
            assert self.q_proj is not None
            kv_lora = self.kv_a_proj_with_mqa(hidden_states)[0]
            q = self.q_proj(hidden_states)[0]

        kv_c, k_pe = kv_lora.split(
            [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        kv_c_normed = self.kv_a_layernorm(kv_c)
        q = q.view(-1, self.num_heads, self.qk_head_dim)
        k_pe = k_pe.unsqueeze(1)
        if self.rotary_emb is not None:
            q[..., self.qk_nope_head_dim :], k_pe = self.rotary_emb(
                positions, q[..., self.qk_nope_head_dim :], k_pe
            )
        if self.indexer and self.is_sparse and not self.skip_topk:
            self.indexer(hidden_states, q_c, positions, self.indexer_rope_emb)
        if llama_4_scaling is not None:
            q *= llama_4_scaling

        attn_out = self.mla_attn(
            q,
            kv_c_normed,
            k_pe,
            output_shape=(hidden_states.shape[0], self.num_heads * self.v_head_dim),
        )
        assert gate_logits is not None
        attn_out = attn_out * gate_logits.sigmoid()
        return self.o_proj(attn_out)[0]
