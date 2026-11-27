# SPDX-License-Identifier: Apache-2.0
"""HCU V4.1 draft construction and context KV insertion.

The native loader and the compressed-tensors scale-alias load view belong to the
adapter (`hcu_adapter.dsv41_draft_weight_alias`): they are the naming bridge the
distinct channel-FP8 route creates, which is an adapter route this plugin does
not own. This plugin owns the AG-RS construction config and the draft compute
path. Enumeration and loading stay inherited.
"""
from __future__ import annotations

import torch

from ..adapter_api import adapter_module as _adapter_module

HcuDSV41DraftWeightAliasModel = _adapter_module('dsv41_draft_weight_alias').HcuDSV41DraftWeightAliasModel


class HcuDSparkDeepseekV41ForCausalLM(HcuDSV41DraftWeightAliasModel):
    """Keep native draft construction/loading; own CT DP communication policy."""

    def __init__(self, *, vllm_config, prefix=""):
        from vllm.config import set_current_vllm_config
        from hcu_vllm_das.dspark_agrs import _owned_quant_kind, draft_config
        from hcu_vllm_das.dspark_backend import native_draft_ll_requested
        from hcu_vllm_das.dspark_post import requested, initialize_owned

        native_ll = native_draft_ll_requested(vllm_config)
        constructor_config = vllm_config
        constructor = super().__init__
        if requested(vllm_config):
            constructor = lambda **kwargs: initialize_owned(self, **kwargs)

        target_parallel = getattr(vllm_config.speculative_config, 'target_parallel_config', None)

        # CT W4A8 and Channel-FP8 drafts both take a private AG-RS construction
        # config so they do not allocate a second DeepEP/RocSHMEM LL buffer on
        # top of the target's already-resident weights.
        if (
            _owned_quant_kind(vllm_config) is not None
            and vllm_config.parallel_config.data_parallel_size > 1
            and not native_ll
        ):
            local = draft_config(vllm_config)
            constructor_config = local
            # Native draft decoder construction reads current_config as well as
            # its explicit argument. Keep both references on the same private
            # copy and retain native compiler live tables for layer discovery.
            with set_current_vllm_config(local):
                constructor(vllm_config=local, prefix=prefix)
        elif (
            _owned_quant_kind(vllm_config) is not None
            and getattr(target_parallel, 'prefill_context_parallel_size', 1) > 1
        ):
            from vllm.logger import init_logger
            from ..pcp_dspark_draft import draft_config as pcp_draft_config, moe_pcp_size

            local = pcp_draft_config(vllm_config)
            constructor_config = local
            with set_current_vllm_config(local):
                constructor(vllm_config=local, prefix=prefix)
            init_logger('vllm.hcu_dspark_pcp').info(
                'HCU PCP DSpark drafter ready: attention_PCP=%d expert_PCP=%d speculative_tokens=%d',
                local.parallel_config.prefill_context_parallel_size, moe_pcp_size(local),
                local.speculative_config.num_speculative_tokens,
            )
        else:
            constructor(vllm_config=vllm_config, prefix=prefix)

        # Draft decoder layers use the native constructor, so the target's
        # HcuV41Fp4DecoderLayer activation binding never visits them. Reuse the
        # same guarded consumer contract with the actual construction config;
        # keep CT INT8, heterogeneous/mega MoE and unsupported topology native.
        quant_kind = _owned_quant_kind(constructor_config)
        if quant_kind == "channel_fp8":
            from ..moe.shared_act_quant import bind_dspark_shared_experts_fp8

            bind_dspark_shared_experts_fp8(constructor_config, self.model)
            from ..qkv_split import bind_dspark_qkv

            bind_dspark_qkv(self.model)
        self._hcu_int8_markov = False
        if quant_kind == "ct_w4a8":
            from ..w4a8_projections import bind_vocab_head, bind_wo_a

            bind_wo_a(self.model.layers)
            self._hcu_int8_markov = bind_vocab_head(
                self.model.markov_head.markov_w2, constructor_config.quant_config,
                "model.markov_head.markov_w2",
            )
            if self._hcu_int8_markov and getattr(self.config, "dspark_draft_topk", None):
                raise ValueError("DSV4.1 INT8 Markov Head requires full-vocabulary drafting")
        if native_ll:
            from hcu_vllm_das.dspark_ll import install_model_entry_reset
            install_model_entry_reset(self, vllm_config, 128)
        self._hcu_markov_pointwise = native_ll and not self._hcu_int8_markov

    @torch.inference_mode()
    def precompute_and_store_context_kv(
        self, context_states, context_positions, context_slot_mappings=None,
    ) -> None:
        """Write the draft's sliding-window context cache without a dummy query."""
        if context_slot_mappings is None:
            # Core profiles the projection and normalization without writing KV.
            return super().precompute_and_store_context_kv(
                context_states, context_positions, context_slot_mappings,
            )

        insert = getattr(torch.ops._C, "fused_deepseek_v4_kv_rope_insert", None)
        if insert is None:
            return super().precompute_and_store_context_kv(
                context_states, context_positions, context_slot_mappings,
            )

        for index, layer in enumerate(self.model.layers):
            slots = context_slot_mappings[index]
            attn = layer.attn
            qr_kv, _ = attn.fused_wqa_wkv(context_states)
            kv = attn.kv_norm(qr_kv[..., attn.q_lora_rank :])
            if slots is None:
                continue

            swa = attn.swa_cache_layer
            cache = swa.kv_cache
            fp8_scale = (
                attn._flashinfer_fp8_kv_scale
                if cache.dtype == torch.float8_e4m3fn else None
            )
            # The same KV RoPE, quantization and paged write as the fused Q/KV op.
            # Context tokens have no query, so schedule only the KV branch.
            insert(
                kv.contiguous(),
                cache,
                slots,
                context_positions,
                attn.rotary_emb.cos_sin_cache,
                swa.block_size,
                fp8_scale,
                attn.kv_mxfp8,
            )

    def sample_sequential(self, speculator, num_reqs, head_hidden):
        from hcu_vllm_das.dspark_greedy_admission import sample_sequential
        if sample_sequential(self, speculator, num_reqs, head_hidden):
            return True
        return super().sample_sequential(speculator, num_reqs, head_hidden)
