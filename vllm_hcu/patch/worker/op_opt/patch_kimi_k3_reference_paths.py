# SPDX-License-Identifier: Apache-2.0
"""Preserve the audited v0.25.1 Kimi-K3 kernel and TP reduction paths."""
from __future__ import annotations
from types import ModuleType
import functools
import torch
from vllm_hcu.platforms import envs as henvs
from ._common import load_exact_module, require_class

TARGET_MODULE = "vllm.models.kimi_k3.amd.linear"
PATCH_ID = "worker.op_opt.kimi_k3.reference_paths"
TARGETS = (f"{TARGET_MODULE}.KimiDecoderLayer.forward_attn_residual",)
_MARKER = "_vllm_hcu_kimi_reference_paths_applied"

def _apply_attn_res(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    proj: ReplicatedLinear,
    norm: RMSNorm,
    num_valid_blocks: int,
) -> torch.Tensor:
    if num_valid_blocks <= 0:
        return prefix_sum

    from boltops.generic import attn_res
    return attn_res(
        prefix_sum,
        block_residual,
        norm.weight,
        proj.weight.squeeze(0),
        num_valid_blocks,
        norm.variance_epsilon,
    )

def forward_attn_residual(
    self,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    block_residual: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    prefix_sum = hidden_states
    hidden_states = _apply_attn_res(
        prefix_sum,
        block_residual,
        self.self_attention_res_proj,
        self.self_attention_res_norm,
        self.prev_valid_blocks,
    )

    if self.is_block_write_layer:
        block_residual[:, self.block_write_idx, :].copy_(prefix_sum)
        prefix_sum = None

    hidden_states = self.input_layernorm(hidden_states)
    hidden_states = self._run_self_attn(positions, hidden_states)
    if prefix_sum is not None:
        prefix_sum = prefix_sum + hidden_states
    else:
        prefix_sum = hidden_states

    mlp_valid_blocks = self.prev_valid_blocks + (
        1 if self.is_block_write_layer else 0
    )
    hidden_states = _apply_attn_res(
        prefix_sum,
        block_residual,
        self.mlp_res_proj,
        self.mlp_res_norm,
        mlp_valid_blocks,
    )

    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    prefix_sum = prefix_sum + hidden_states
    return prefix_sum, block_residual

def _reference_mla_forward(
    self,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    llama_4_scaling: torch.Tensor | None = None,
) -> torch.Tensor:
    q_c = None
    if self.q_lora_rank is not None:
        assert self.fused_qkv_a_proj is not None
        assert self.q_a_layernorm is not None
        assert self.q_b_proj is not None
        qkv_lora = self.fused_qkv_a_proj(hidden_states)[0]
        q_c, kv_lora = qkv_lora.split(
            [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
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
    g_proj = getattr(self, "g_proj", None)
    if g_proj is not None:
        attn_out = attn_out * g_proj(hidden_states)[0].sigmoid()
    return self.o_proj(attn_out)[0]


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    if getattr(target, _MARKER, False):
        return False
    decoder = require_class(target, "KimiDecoderLayer", f"{TARGET_MODULE}.KimiDecoderLayer")
    transform = require_class(target, "KimiRoutedOutputTransform", f"{TARGET_MODULE}.KimiRoutedOutputTransform")
    from vllm_hcu.model_executor.layers.fused_moe.moe_runner import MoERunner
    from vllm_hcu.model_executor.layers.quantization.kimi_k3_situ import SituAndMul
    target.SituAndMul = SituAndMul
    mla_wrapper = target.KimiK3MultiHeadLatentAttentionWrapper
    original_mla_forward = mla_wrapper.forward

    @functools.wraps(original_mla_forward)
    def hcu_mla_forward(self, positions, hidden_states, llama_4_scaling=None):
        if getattr(self, "dcp_q_replicate", False):
            return original_mla_forward(self, positions, hidden_states, llama_4_scaling)
        return _reference_mla_forward(self, positions, hidden_states, llama_4_scaling)

    mla_wrapper.forward = hcu_mla_forward
    # The 028 sharded latent-tail runner uses a different collective/GEMM
    # sequence. The HCU runner fuses latent+shared partials before RMSNorm.
    target.ROCmLatentMoERunner = MoERunner
    transform._hcu_fuse_shared_and_routed_tp_all_reduce = True
    original_residual_forward = decoder.forward_attn_residual

    @functools.wraps(original_residual_forward)
    def hcu_residual_forward(self, *args, **kwargs):
        if not henvs.custom_ops_enabled():
            return original_residual_forward(self, *args, **kwargs)
        return forward_attn_residual(self, *args, **kwargs)

    # The upstream helper accepts fused norm/delta arguments that the BoltOps
    # helper does not. Restore the complete upstream forward when disabled.
    decoder.forward_attn_residual = hcu_residual_forward
    moe = require_class(target, "KimiMoE", f"{TARGET_MODULE}.KimiMoE")
    original_init = moe.__init__

    @functools.wraps(original_init)
    def hcu_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        # 0251 uses the BF16 projection followed by an FP32 cast on ROCm.
        # 028 enables BF16->FP32 hipBLASLt GEMM for all ROCm routers.
        self.gate.allow_cublas_router_gemm = False
        # Keep the same checkpoint rounding and LightOp gate specialization
        # as 0251. Preserve the Parameter object referenced by the router.
        self.gate.e_score_correction_bias.data = (
            self.gate.e_score_correction_bias.data.to(self.gate.weight.dtype)
        )
        shared = getattr(self.experts, "_shared_experts", None)
        if shared is not None:
            # 0251's TP path is single-stream. Do not introduce auxiliary
            # stream graph capture or reorder shared-expert GEMMs.
            shared._stream = None

    moe.__init__ = hcu_init
    setattr(target, _MARKER, True)
    return True

def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))
