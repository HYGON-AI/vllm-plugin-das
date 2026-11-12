# SPDX-License-Identifier: Apache-2.0
"""Eager HCU V41 PCP attention using native replicated-cache preparation.

Only owned classes advertise PCP. Common V41 produces rank-major cache payloads
and local queries; AMD owns attention and inverse RoPE. No Core symbols change.
"""
from __future__ import annotations

import torch
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer
from vllm.models.deepseek_v41.attention import DeepseekV4Indexer
from vllm.models.deepseek_v41.amd.rocm import (
    DeepseekV41ROCMAiterMLAAttention,
    DeepseekV41ROCMAiterMLASparseBackend,
    DeepseekV41ROCMAiterSparseSWABackend,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from . import v41_indexer as reference
from .quantization.dsv41_channel_fp8 import _is_v41


def validate_config(cfg):
    p = cfg.parallel_config
    if (not _is_v41(cfg.model_config.hf_config)
            or (p.tensor_parallel_size, p.data_parallel_size, p.pipeline_parallel_size,
                p.prefill_context_parallel_size, p.decode_context_parallel_size) != (1, 1, 1, 8, 1)
            or not p.enable_expert_parallel):
        raise ValueError("V4.1 PCP MegaMoE requires TP1/DP1/PP1/PCP8/DCP1/EP8")
    if cfg.kernel_config.moe_backend not in ("deep_gemm_mega_moe", "deep_gemm"):
        raise ValueError("V4.1 PCP requires deep_gemm_mega_moe or deep_gemm in this adapter")
    if not cfg.model_config.enforce_eager or not cfg.use_v2_model_runner:
        raise ValueError("V4.1 PCP MegaMoE requires eager model runner V2")
    if (cfg.speculative_config is not None or p.enable_eplb or p.enable_dbo
            or cfg.lora_config is not None or cfg.kv_transfer_config is not None
            or cfg.kernel_config.monokernel or cfg.cache_config.swa_bounded_replay):
        raise ValueError("V4.1 PCP MegaMoE does not support speculation, EPLB, DBO, LoRA, KV transfer or replay")
    if (cfg.attention_config.resolve_indexer_kv_dtype("fp8") != "fp8"
            or cfg.attention_config.indexer_sparse_logits
            or cfg.attention_config.backend not in (None,
                AttentionBackendEnum.ROCM_FLASHMLA_SPARSE_DSV4,
                AttentionBackendEnum.ROCM_FLASHMLA_SPARSE_DSV41)):
        raise ValueError("V4.1 PCP MegaMoE requires the native ROCm FP8 attention layout")
    return True


class HcuV41PcpBackend(DeepseekV41ROCMAiterMLASparseBackend):
    @classmethod
    def supports_pcp(cls) -> bool:
        return True


class HcuV41PcpSwaBackend(DeepseekV41ROCMAiterSparseSWABackend):
    @classmethod
    def supports_pcp(cls) -> bool:
        return True


class HcuV41PcpSparseIndexer(SparseAttnIndexer):
    """Local scoring over the full cache replicated by common V41 sources.

    This is an owned layer selected by an owned model, independent of the
    optional PCP1 OOT scorer. Never falsify the native parallel configuration.
    """
    def forward_hip(self, hidden_states, q_quant, k, weights):
        return reference.indexer(self, hidden_states, q_quant, k, weights)

    # CustomOp disables hardware dispatch when the user excludes its op name.
    # The owned model still needs the same scorer in that configuration.
    forward_native = forward_hip


class HcuV41PcpIndexer(DeepseekV4Indexer):
    attn_cls = HcuV41PcpSparseIndexer


class HcuV41PcpAttention(DeepseekV41ROCMAiterMLAAttention):
    backend_cls = HcuV41PcpBackend
    swa_backend_cls = HcuV41PcpSwaBackend

    def __init__(self, *args, **kwargs):
        # select_attention() validated the configuration for this layer.
        super().__init__(*args, **kwargs)
        # Common preparation serializes PCP gathers. AMD's fast compressor
        # chain publishes local latent against global slots and cannot run here.
        self.aux_stream_list = None

    def _indexer_cls(self, k_cache):
        if k_cache is not None and k_cache.rocm_mxfp4:
            raise ValueError('PCP adapter requires the FP8 SHUFFLE index K layout')
        return HcuV41PcpIndexer

    def forward_mqa(self, q, kv, positions, output):
        if not isinstance(output, torch.Tensor):
            raise TypeError('HCU PCP attention requires its BF16 output allocation')
        metadata = get_forward_context().attn_metadata
        if metadata is None:
            return super().forward_mqa(q, kv, positions, output)
        if not isinstance(metadata, dict) or self.swa_cache_layer.prefix not in metadata:
            raise RuntimeError('HCU PCP attention requires native local SWA metadata')
        swa = metadata[self.swa_cache_layer.prefix]
        count = swa.num_decode_tokens + swa.num_prefill_tokens
        if count < 0 or count > q.shape[0] or q.shape[0] != output.shape[0]:
            raise ValueError('PCP local query/output extent mismatch')
        # Collectives already ran during preparation, including on empty ranks.
        # AMD inverse RoPE sees only initialized, actual local query rows.
        output[count:].zero_()
        if count:
            return super().forward_mqa(q[:count], kv[:count], positions[:count], output[:count])

    def _forward_decode(self, q, *args, **kwargs):
        # A native PCP placeholder may retain a request with zero actual tokens.
        if q.shape[0] == 0:
            return 0
        return super()._forward_decode(q, *args, **kwargs)
