# SPDX-License-Identifier: Apache-2.0
"""MiniMax DSpark checkpoint on Core's DFlash/DSpark execution interfaces.

The backbone retains trained Gemma normalization, OAI SwiGLU and partial
RoPE. Core owns proposal scheduling, KV allocation and verification.
"""
from __future__ import annotations
from copy import copy
import torch
from torch import nn
from vllm import envs
from vllm.model_executor.models.qwen3_dflash import (
    DFlashQwen3Attention,DFlashQwen3DecoderLayer,DFlashQwen3ForCausalLM,
    _resolve_layer_attention)
from vllm.model_executor.models.qwen3_dspark import Qwen3DSparkModel,Qwen3DSparkForCausalLM
from vllm.models.minimax_m3.amd.model import MiniMAXGemmaRMSNorm,MiniMaxM3MLP


class MiniMaxDraftLayer(DFlashQwen3DecoderLayer):
    def __init__(self,vllm_config,*,config,layer_idx,cache_config=None,quant_config=None,prefix=""):
        nn.Module.__init__(self)
        self.hidden_size=config.hidden_size
        sliding_window,causal=_resolve_layer_attention(config,layer_idx)
        self.self_attn=DFlashQwen3Attention(
            hidden_size=config.hidden_size,num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,head_dim=config.head_dim,
            max_position=config.max_position_embeddings,rope_parameters=config.rope_parameters,
            rms_norm_eps=config.rms_norm_eps,sliding_window=sliding_window,causal=causal,
            cache_config=cache_config,quant_config=quant_config,prefix=prefix+'.self_attn')
        self.self_attn.q_norm=MiniMAXGemmaRMSNorm(config.head_dim,eps=config.rms_norm_eps)
        self.self_attn.k_norm=MiniMAXGemmaRMSNorm(config.head_dim,eps=config.rms_norm_eps)
        self.input_layernorm=MiniMAXGemmaRMSNorm(config.hidden_size,eps=config.rms_norm_eps)
        self.post_attention_layernorm=MiniMAXGemmaRMSNorm(config.hidden_size,eps=config.rms_norm_eps)
        mlp_config=copy(config)
        # The export stores alpha/limit; its OAI activation has the fixed +1
        # up-branch offset. Core's text config normally supplies this default.
        mlp_config.swiglu_beta=getattr(config,'swiglu_beta',1.0)
        if mlp_config.swiglu_beta!=1.0:raise ValueError('unsupported draft OAI offset')
        self.mlp=MiniMaxM3MLP(config=mlp_config,quant_config=quant_config,prefix=prefix+'.mlp',
            intermediate_size=config.dspark_ffn_hidden_size,reduce_results=False)


class MiniMaxDSparkBackbone(Qwen3DSparkModel):
    decoder_layer_cls=MiniMaxDraftLayer

    def __init__(self,*,vllm_config,start_layer_id=0,prefix=""):
        # Checkpoint precision policies use the original exported prefix even
        # though Python registration is model.layers.<local index>.
        super().__init__(vllm_config=vllm_config,start_layer_id=start_layer_id,
                         prefix='language_model.model.dspark')
        c=self.config
        if not c.use_gemma_norm or c.hidden_act!='swigluoai':
            raise ValueError("MiniMax DSpark requires Gemma norm and trained OAI activation")
        self.hidden_norm=MiniMAXGemmaRMSNorm(c.hidden_size,eps=c.rms_norm_eps)
        self.norm=MiniMAXGemmaRMSNorm(c.hidden_size,eps=c.rms_norm_eps)
        if getattr(c,'has_own_embed_tokens',False):
            raise ValueError('this MiniMax draft requires the target embedding')
        # No checkpoint embedding exists. Publish no uninitialized duplicate;
        # Core's public DSpark loader binds the actual target module before use.
        self.embed_tokens=None

    def _build_fused_kv_buffers(self):
        # The generic fused context projection implements RMS rather than
        # Gemma norm. This path deliberately uses the actual norm modules.
        self._minimax_context_kv_ready=True

    def precompute_and_store_context_kv(self,context_states,context_positions,context_slot_mapping=None):
        normed=self.hidden_norm(context_states)
        per_layer=isinstance(context_slot_mapping,(list,tuple))
        for index,layer in enumerate(self.layers):
            a=layer.self_attn
            qkv,_=a.qkv_proj(normed)
            k,v=qkv[:,a.q_size:].split(a.kv_size,dim=-1)
            k=a.k_norm(k.reshape(-1,a.num_kv_heads,a.head_dim)).reshape(-1,a.kv_size)
            # Public rotary accepts no key: rotate K in its query operand.
            k,_=a.rotary_emb(context_positions,k,None)
            slots=context_slot_mapping[index] if per_layer else context_slot_mapping
            if slots is not None:
                a.attn.impl.do_kv_cache_update(a.attn,k.view(-1,a.num_kv_heads,a.head_dim),
                    v.view(-1,a.num_kv_heads,a.head_dim),a.attn.kv_cache,slots)


class DSparkMiniMaxDraftModel(Qwen3DSparkForCausalLM):
    model_cls=MiniMaxDSparkBackbone

    def __init__(self,*,vllm_config,prefix=""):
        # Reuse the generic constructor that honors model_cls. The Qwen DSpark
        # subclass otherwise hard-codes its own numerical backbone.
        DFlashQwen3ForCausalLM.__init__(self,vllm_config=vllm_config,prefix=prefix)
        self.target_vocab_size=vllm_config.model_config.get_vocab_size()
        if getattr(self.config,'has_own_lm_head',False):
            raise ValueError('this MiniMax draft requires the target output head')
        self.lm_head=None

    def sample_sequential(self, speculator, num_reqs, head_hidden):
        from ..parallel.minimax31_markov import sample_online_markov

        if envs.VLLM_BATCH_INVARIANT:
            return False
        return sample_online_markov(self, speculator, num_reqs, head_hidden)

    def load_weights(self,weights):
        def normalized():
            for name,tensor in weights:
                root='language_model.model.dspark.'
                if not name.startswith(root):
                    raise ValueError('unexpected MiniMax DSpark checkpoint key: '+name)
                name=name[len(root):].replace('.decoder_layer.','.')
                if name.startswith(('embed_tokens.','lm_head.')):
                    raise ValueError('unexpected independently owned draft embedding/head')
                if name.startswith('final_norm.'):
                    name='norm.'+name[len('final_norm.'):]
                yield name,tensor
        return super().load_weights(normalized())
