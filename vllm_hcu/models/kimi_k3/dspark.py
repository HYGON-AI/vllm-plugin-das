# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# Modified by Hygon Information Technology Co., Ltd., 2026.
"""HCU implementation of the Kimi-K3 DSpark draft architecture.

The v0.25.1 runtime owns DSpark scheduling and sampling. This module supplies
the K3 draft model with HCU MLA/linear layers and the model hooks consumed by
that runtime.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn

import vllm._custom_ops as ops
from vllm.config import VllmConfig
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.models.qwen3_dspark import DSparkMarkovHead
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    get_draft_quant_config,
    maybe_prefix,
)

from vllm_hcu.models.deepseek_v2 import DeepseekV2MLAAttention
from vllm_hcu.models.kimi_k3.amd.linear import KimiMLP


class K3DSparkDecoderLayer(nn.Module):
    """Dense K3 draft layer using the plugin's HCU MLA implementation."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        config,
        layer_idx: int,
        start_layer_id: int,
        prefix: str,
        quant_config,
    ) -> None:
        super().__init__()
        layer_prefix = maybe_prefix(prefix, f"layers.{start_layer_id + layer_idx}")
        self.self_attn = DeepseekV2MLAAttention(
            vllm_config=vllm_config,
            config=config,
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            q_lora_rank=config.q_lora_rank,
            kv_lora_rank=config.kv_lora_rank,
            max_position_embeddings=config.max_position_embeddings,
            cache_config=vllm_config.cache_config,
            quant_config=quant_config,
            prefix=maybe_prefix(layer_prefix, "self_attn"),
        )
        self.mlp = KimiMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=maybe_prefix(layer_prefix, "mlp"),
        )
        self.input_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(positions, hidden_states, None)
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual
        )
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class K3DSparkModel(nn.Module):
    """K3 DSpark backbone with the interfaces expected by v0.25.1 DSpark."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        start_layer_id: int,
        prefix: str,
    ) -> None:
        super().__init__()
        speculative_config = vllm_config.speculative_config
        if speculative_config is None or speculative_config.draft_model_config is None:
            raise ValueError("K3 DSpark requires a configured draft model")
        self.config = speculative_config.draft_model_config.hf_config
        quant_config = get_draft_quant_config(vllm_config)

        # DSpark aliases these frozen target weights after loading the draft.
        self.embed_tokens: nn.Module | None = None
        self.context_proj = ReplicatedLinear(
            self.config.target_hidden_size * self.config.num_target_layers,
            self.config.hidden_size,
            bias=False,
            return_bias=False,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "context_proj"),
        )
        self.context_norm = RMSNorm(
            self.config.hidden_size, eps=self.config.rms_norm_eps
        )
        self.layers = nn.ModuleList(
            [
                K3DSparkDecoderLayer(
                    vllm_config=vllm_config,
                    config=self.config,
                    layer_idx=layer_idx,
                    start_layer_id=start_layer_id,
                    prefix=prefix,
                    quant_config=quant_config,
                )
                for layer_idx in range(self.config.num_hidden_layers)
            ]
        )
        self.final_norm = RMSNorm(
            self.config.hidden_size, eps=self.config.rms_norm_eps
        )
        self.markov_head = DSparkMarkovHead(
            self.config.vocab_size,
            self.config.draft_vocab_size,
            self.config.markov_rank,
            prefix=maybe_prefix(prefix, "markov_head"),
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.embed_tokens is None:
            raise RuntimeError("DSpark target embedding has not been attached")
        return self.embed_tokens(input_ids)

    def combine_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.context_norm(self.context_proj(hidden_states))

    @torch.inference_mode()
    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor | list[torch.Tensor | None] | None = None,
    ) -> None:
        """Project target context into each draft layer's latent KV cache."""
        for layer_idx, layer in enumerate(self.layers):
            attn = layer.self_attn
            qkv_lora = attn.fused_qkv_a_proj(context_states)[0]
            kv_lora = qkv_lora[..., attn.q_lora_rank :]
            kv_c, k_pe = kv_lora.split(
                [attn.kv_lora_rank, attn.qk_rope_head_dim], dim=-1
            )
            kv_c = attn.kv_a_layernorm(kv_c)
            k_pe = k_pe.unsqueeze(1)
            rotary_emb = attn.rotary_emb
            ops.rotary_embedding(
                context_positions,
                k_pe,
                None,
                rotary_emb.head_size,
                rotary_emb.cos_sin_cache,
                rotary_emb.is_neox_style,
            )

            if isinstance(context_slot_mapping, (list, tuple)):
                slot_mapping = context_slot_mapping[layer_idx]
            else:
                slot_mapping = context_slot_mapping
            if slot_mapping is None:
                continue

            cache_layer = attn.mla_attn.mla_attn
            cache_layer.impl.do_kv_cache_update(
                kv_c,
                k_pe,
                cache_layer.kv_cache,
                slot_mapping,
                cache_layer.kv_cache_dtype,
                cache_layer._k_scale,
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)

        hidden_states = inputs_embeds
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
            )
        hidden_states, _ = self.final_norm(hidden_states, residual)
        return hidden_states


class K3DSparkForCausalLM(nn.Module):
    """K3 DSpark draft model registered by the HCU plugin."""

    has_own_embed_tokens = False
    has_own_lm_head = False
    draft_id_to_target_id = None
    checkpoint_skip_substrs = ("confidence_head", "embed_tokens", "lm_head")

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={"": "model."},
        orig_to_new_stacked={
            ".gate_proj": (".gate_up_proj", 0),
            ".up_proj": (".gate_up_proj", 1),
            ".q_a_proj": (".fused_qkv_a_proj", 0),
            ".kv_a_proj_with_mqa": (".fused_qkv_a_proj", 1),
        },
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        speculative_config = vllm_config.speculative_config
        if speculative_config is None or speculative_config.draft_model_config is None:
            raise ValueError("K3 DSpark requires a configured draft model")
        self.draft_model_config = speculative_config.draft_model_config
        self.config = self.draft_model_config.hf_config
        target_layer_num = vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config
        )
        self.model = K3DSparkModel(
            vllm_config=vllm_config,
            start_layer_id=target_layer_num,
            prefix=maybe_prefix(prefix, "model"),
        )
        self.lm_head: nn.Module | None = None
        self.logits_processor = LogitsProcessor(
            self.config.draft_vocab_size,
            scale=getattr(self.config, "logit_scale", 1.0),
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def combine_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.model.combine_hidden_states(hidden_states)

    def get_draft_kv_cache_layer_names(self) -> list[str]:
        return [
            layer.self_attn.mla_attn.mla_attn.layer_name
            for layer in self.model.layers
        ]

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor | list[torch.Tensor | None] | None = None,
    ) -> None:
        self.model.precompute_and_store_context_kv(
            context_states, context_positions, context_slot_mapping
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.lm_head is None:
            raise RuntimeError("DSpark target LM head has not been attached")
        return self.logits_processor(self.lm_head, hidden_states)

    def compute_draft_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.compute_logits(hidden_states)

    def map_draft_to_target(self, draft_ids: torch.Tensor) -> torch.Tensor:
        return draft_ids

    def markov_embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.model.markov_head.embed(token_ids)

    def markov_bias(self, markov_embed: torch.Tensor) -> torch.Tensor:
        return self.model.markov_head.bias(markov_embed, self.logits_processor)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_substrs=list(self.checkpoint_skip_substrs),
        )
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
