# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# Modified by Hygon Information Technology Co., Ltd., 2026.
"""HCU adapter for the upstream DeepSeek-V4 DSpark draft model."""

from __future__ import annotations

import torch

from vllm.model_executor.layers.mhc import HCHeadOp
from vllm.models.deepseek_v4.nvidia import dspark as _dspark


class DSparkDeepseekV4Model(_dspark.DSparkDeepseekV4Model):
    """Use the HCU decoder, mHC operators, and cache insertion kernel."""

    def __init__(self, *, vllm_config, prefix: str = "") -> None:
        # Registry inspection happens before HCU module exchange, so HCU-only
        # imports remain local to worker-side model construction.
        from vllm.models.deepseek_v4.amd.model import (
            DeepseekV4DecoderLayer as AMDDeepseekV4DecoderLayer,
        )
        from vllm.platforms import current_platform

        class HCUDeepseekV4DecoderLayer(AMDDeepseekV4DecoderLayer):
            def __init__(self, vllm_config, prefix):
                config = vllm_config.model_config.hf_config
                topk_indices_buffer = torch.empty(
                    vllm_config.scheduler_config.max_num_batched_tokens,
                    config.index_topk,
                    dtype=torch.int32,
                    device=current_platform.device_type,
                )
                super().__init__(
                    vllm_config,
                    prefix,
                    topk_indices_buffer=topk_indices_buffer,
                    aux_stream_list=None,
                )
                # AMD/HCU uses regular FusedMoE rather than NVIDIA MegaMoE.
                self.ffn.use_mega_moe = False

        # Reproduce the small upstream constructor with the HCU decoder class
        # selected locally. Mutating the upstream module globals here would
        # make later NVIDIA DSpark constructions order-dependent.
        torch.nn.Module.__init__(self)
        assert vllm_config.speculative_config is not None
        config = vllm_config.speculative_config.draft_model_config.hf_config
        self.config = config
        self.hidden_size = config.hidden_size
        self.hc_mult = config.hc_mult
        self.hc_eps = config.hc_eps
        self.rms_norm_eps = config.rms_norm_eps
        self.num_hidden_layers = config.num_hidden_layers
        self.target_layer_ids = tuple(config.dspark_target_layer_ids)
        self.num_dspark_layers = getattr(config, "n_mtp_layers", None) or 3

        self.embed_tokens = _dspark.VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=_dspark.maybe_prefix(prefix, "embed_tokens"),
        )
        self.main_proj = _dspark.ReplicatedLinear(
            config.hidden_size * len(self.target_layer_ids),
            config.hidden_size,
            bias=False,
            return_bias=False,
            quant_config=vllm_config.quant_config,
            prefix=_dspark.maybe_prefix(prefix, "main_proj"),
        )
        self.main_norm = _dspark.RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        current_vllm_config = _dspark.get_current_vllm_config()
        self.layers = torch.nn.ModuleList(
            [
                HCUDeepseekV4DecoderLayer(
                    current_vllm_config,
                    prefix=_dspark.maybe_prefix(
                        prefix,
                        f"layers.{self.num_hidden_layers + index}",
                    ),
                )
                for index in range(self.num_dspark_layers)
            ]
        )
        self.norm = _dspark.RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        hc_dim = self.hc_mult * config.hidden_size
        self.hc_head_fn = torch.nn.Parameter(
            torch.empty(self.hc_mult, hc_dim, dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_head_base = torch.nn.Parameter(
            torch.empty(self.hc_mult, dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_head_scale = torch.nn.Parameter(
            torch.empty(1, dtype=torch.float32),
            requires_grad=False,
        )
        draft_vocab_size = (
            getattr(config, "draft_vocab_size", None) or config.vocab_size
        )
        self.markov_head = _dspark.DSparkMarkovHead(
            config.vocab_size,
            draft_vocab_size,
            config.dspark_markov_rank,
            prefix=_dspark.maybe_prefix(prefix, "markov_head"),
        )
        self.hc_head_op = HCHeadOp()

    @torch.inference_mode()
    def precompute_and_store_context_kv(
        self,
        main_x,
        context_positions,
        context_slot_mappings=None,
    ) -> None:
        for index, layer in enumerate(self.layers):
            slot_mapping = (
                None
                if context_slot_mappings is None
                else context_slot_mappings[index]
            )
            attn = layer.attn
            qr_kv, _ = attn.fused_wqa_wkv(main_x)
            kv = attn.kv_norm(qr_kv[..., attn.q_lora_rank :])
            if slot_mapping is not None:
                _insert_context_kv(
                    attn,
                    kv,
                    context_positions,
                    slot_mapping,
                )

    def forward(self, input_ids, positions, inputs_embeds=None):
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)
        hidden_states = inputs_embeds.unsqueeze(-2).repeat(1, self.hc_mult, 1)

        residual = post_mix = res_mix = None
        layer = None
        for layer in self.layers:
            hidden_states, residual, post_mix, res_mix = layer(
                hidden_states,
                positions,
                input_ids,
                post_mix,
                res_mix,
                residual,
            )
        assert layer is not None
        if layer.use_fused_mhc:
            hidden_states = layer.hc_post(hidden_states, residual, post_mix, res_mix)
        return self.hc_head_op(
            hidden_states,
            self.hc_head_fn,
            self.hc_head_scale,
            self.hc_head_base,
            self.rms_norm_eps,
            self.hc_eps,
        )


def _insert_context_kv(attn, kv, positions, slot_mapping) -> None:
    """Insert DSpark context KV using the non-PCP HCU fused cache op."""

    swa_cache = attn.swa_cache_layer.kv_cache
    if swa_cache.dtype != torch.uint8:
        return _dspark._insert_context_kv(
            attn,
            kv,
            positions,
            slot_mapping,
        )

    import lightop

    num_tokens = kv.shape[0]
    dummy_q = torch.zeros(
        (num_tokens, attn.n_local_heads, attn.head_dim),
        dtype=kv.dtype,
        device=kv.device,
    )
    lightop.op.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
        dummy_q,
        kv,
        swa_cache.view(swa_cache.shape[0], -1),
        slot_mapping,
        positions.to(torch.int64),
        attn.rotary_emb.cos_sin_cache,
        attn.eps,
        attn.swa_cache_layer.block_size,
    )


class DSparkDeepseekV4ForCausalLM(_dspark.DSparkDeepseekV4ForCausalLM):
    """Upstream DSpark semantics backed by HCU model components."""

    def __init__(self, *, vllm_config, prefix: str = "") -> None:
        torch.nn.Module.__init__(self)
        assert vllm_config.speculative_config is not None
        self.draft_model_config = (
            vllm_config.speculative_config.draft_model_config
        )
        self.config = self.draft_model_config.hf_config
        self.model = DSparkDeepseekV4Model(
            vllm_config=vllm_config,
            prefix=_dspark.maybe_prefix(prefix, "model"),
        )
        self.lm_head = _dspark.ParallelLMHead(
            self.config.vocab_size,
            self.config.hidden_size,
            prefix=_dspark.maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = _dspark.LogitsProcessor(
            self.config.vocab_size
        )

    def _finalize_moe(self) -> None:
        for layer in self.model.layers:
            ffn = layer.ffn
            if getattr(ffn, "use_mega_moe", False):
                ffn.finalize_mega_moe_weights()

    def named_parameters(
        self,
        prefix: str = "",
        recurse: bool = True,
        remove_duplicate: bool = True,
    ):
        """Expose Channel-FP8 scale aliases only during DSpark loading."""

        parameters = super().named_parameters(
            prefix=prefix,
            recurse=recurse,
            remove_duplicate=remove_duplicate,
        )
        for name, parameter in parameters:
            yield name, parameter
            if not getattr(self, "_loading_dspark_weights", False):
                continue
            if name.endswith(".weight_scale") or name.endswith("_weight_scale"):
                yield name + "_inv", parameter

    def load_weights(self, weights):
        self._loading_dspark_weights = True
        try:
            return super().load_weights(weights)
        finally:
            self._loading_dspark_weights = False
