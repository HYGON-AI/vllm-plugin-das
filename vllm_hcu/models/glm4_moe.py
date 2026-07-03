# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2025 The ZhipuAI Team.
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Verson 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only GLM-4.5, GLM-4.6, GLM-4.7 model
compatible with HuggingFace weights."""

import typing
from collections.abc import Callable, Iterable
from itertools import islice

import torch
from torch import nn
from transformers.models.glm4_moe import Glm4MoeConfig

import vllm_hcu.platforms.envs as henvs
from vllm_hcu.model_executor.layers.sp_utils import (
    configure_hcu_runtime_sp,
    finalize_attention_output_for_sp,
    finalize_mlp_output_for_sp,
    finalize_moe_output_for_sp,
    gather_tokens_for_sp,
    hcu_runtime_sp_enabled,
    prepare_attention_inputs_for_sp,
    prepare_mlp_inputs_for_sp,
    prepare_moe_inputs_for_sp,
    split_positions_for_sp,
    split_tokens_for_sp,
    sp_mlp_down_proj_reduce_results,
    use_sp_mlp_token_gather,
)
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.distributed import (
    get_ep_group,
    get_pp_group,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention import Attention, FusedQkvSplitRmsNormRopeAttention
from vllm.model_executor.layers.fused_moe import (
    FusedMoE,
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.sequence import IntermediateTensors

from vllm.model_executor.models.interfaces import MixtureOfExperts, SupportsLoRA, SupportsPP
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from vllm_hcu.ops.fuse_silu_mul_quant import FusedSiluAndMulAndQuant
from vllm_hcu.ops.fuse_rms_norm_quant import FusedRMSNormAndQuant

logger = init_logger(__name__)


class Glm4MoeMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        disable_tp: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.use_sp_token_gather = use_sp_mlp_token_gather(
            reduce_results and ".shared_experts" not in prefix
        )
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            disable_tp=disable_tp,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=sp_mlp_down_proj_reduce_results(
                reduce_results and ".shared_experts" not in prefix
            ),
            disable_tp=disable_tp,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported for now."
            )

        self.enable_fuse_silu_mul_quant = (
            henvs.VLLM_HCU_USE_FUSED_SILU_MUL_QUANT
            and henvs.VLLM_HCU_USE_CUSTOM_OPS
            and quant_config is not None 
        )
        weight = getattr(self.gate_up_proj, "weight", None)
        self.quant_dtype = weight.dtype if weight is not None else None

        if self.enable_fuse_silu_mul_quant:
            self.act_fn = FusedSiluAndMulAndQuant()
        else:
            self.act_fn = SiluAndMul()

    def forward(self, x,
                x_and_scale_quanted: tuple[torch.Tensor, torch.Tensor] | None = None):
        x, x_and_scale_quanted = prepare_mlp_inputs_for_sp(
            x, x_and_scale_quanted, self.use_sp_token_gather
        )

        if self.enable_fuse_silu_mul_quant:
            gate_up, _ = self.gate_up_proj(x, x_and_scale_quanted=x_and_scale_quanted)
            xq, xs = self.act_fn(gate_up, quant_dtype=self.quant_dtype)
            x, _ = self.down_proj(gate_up, x_and_scale_quanted=(xq, xs))
        else:
            gate_up, _ = self.gate_up_proj(x)
            x = self.act_fn(gate_up)
            x, _ = self.down_proj(x)

        return finalize_mlp_output_for_sp(x, self.use_sp_token_gather)


class Glm4MoE(nn.Module):
    def __init__(
        self,
        config: Glm4MoeConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        enable_eplb: bool = False,
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.routed_scaling_factor = config.routed_scaling_factor

        self.ep_group = get_ep_group().device_group
        self.ep_rank = get_ep_group().rank_in_group
        self.ep_size = self.ep_group.size()
        self.n_routed_experts: int = config.n_routed_experts
        self.n_shared_experts: int = config.n_shared_experts

        if config.hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {config.hidden_act}. "
                "Only silu is supported for now."
            )
        # NOTE In the transformers implementation, the gate isn't an nn.Linear,
        # so we cannot use ReplicatedLinear here.
        # See: https://github.com/huggingface/transformers/blob/v4.55.1/src/transformers/models/glm4_moe/modeling_glm4_moe.py#L260
        self.gate = nn.Linear(
            config.hidden_size,
            config.n_routed_experts,
            bias=False,
            dtype=torch.float32,
        )
        self.gate.e_score_correction_bias = nn.Parameter(
            torch.empty(config.n_routed_experts, dtype=torch.float32)
        )

        # Load balancing settings.
        vllm_config = get_current_vllm_config()
        eplb_config = vllm_config.parallel_config.eplb_config
        self.enable_eplb = enable_eplb

        self.n_redundant_experts = eplb_config.num_redundant_experts
        self.n_logical_experts = self.n_routed_experts
        self.n_physical_experts = self.n_logical_experts + self.n_redundant_experts
        self.n_local_physical_experts = self.n_physical_experts // self.ep_size

        self.physical_expert_start = self.ep_rank * self.n_local_physical_experts
        self.physical_expert_end = (
            self.physical_expert_start + self.n_local_physical_experts
        )

        self.runtime_sp = hcu_runtime_sp_enabled()
        disable_shared_mlp_tp = (
            self.runtime_sp and vllm_config.parallel_config.enable_expert_parallel
        )

        if config.n_shared_experts is not None:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = Glm4MoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
                disable_tp=disable_shared_mlp_tp,
                prefix=f"{prefix}.shared_experts",
            )
        else:
            self.shared_experts = None

        self.experts = FusedMoE(
            shared_experts=self.shared_experts,
            num_experts=config.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            use_grouped_topk=True,
            num_expert_group=config.n_group,
            topk_group=config.topk_group,
            prefix=f"{prefix}.experts",
            scoring_func="sigmoid",
            routed_scaling_factor=self.routed_scaling_factor,
            #apply_routed_scale_to_output=True,
            e_score_correction_bias=self.gate.e_score_correction_bias,
            enable_eplb=self.enable_eplb,
            num_redundant_experts=self.n_redundant_experts,
            router_logits_dtype=torch.float32,
            is_sequence_parallel=self.runtime_sp
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (num_tokens, n_experts)
        router_logits = self.gate(hidden_states.to(dtype=torch.float32))

        (
            hidden_states,
            router_logits,
            gathered_quanted_hidden_states,
            hidden_states_scale,
            topk_weights,
            topk_ids,
        ) = prepare_moe_inputs_for_sp(hidden_states, router_logits, self.experts)
        final_hidden_states = self.experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            quanted_hidden_states=gathered_quanted_hidden_states,
            scale=hidden_states_scale,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
        )
        final_hidden_states = finalize_moe_output_for_sp(
            final_hidden_states, self.experts
        )
        return final_hidden_states.view(num_tokens, hidden_dim)


class Glm4MoeAttention(nn.Module):
    def __init__(
        self,
        config: Glm4MoeConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position_embeddings: int = 131072,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-05,
        qkv_bias: bool = False,
        use_qk_norm: bool = False,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.runtime_sp = hcu_runtime_sp_enabled()
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim or (hidden_size // self.total_num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.max_position_embeddings = max_position_embeddings
        self.use_qk_norm = use_qk_norm

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
            disable_tp=self.runtime_sp,
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
            disable_tp=self.runtime_sp,
        )

        config.rope_parameters.setdefault("partial_rotary_factor", 0.5)
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=max_position_embeddings,
            rope_parameters=config.rope_parameters,
        )

        self.enable_fused_qkv_split_rms_rope_kvstore = henvs.VLLM_HCU_USE_FUSED_QKV_SPLIT_RMS_ROPE_KVSTORE \
            and henvs.VLLM_HCU_USE_CUSTOM_OPS
        weight = getattr(self.qkv_proj, "weight", None)
        self.quant_dtype = weight.dtype if weight is not None else None

        if self.enable_fused_qkv_split_rms_rope_kvstore:
            self.attn = FusedQkvSplitRmsNormRopeAttention(
                self.num_heads,
                self.head_dim,
                self.scaling,
                num_kv_heads=self.num_kv_heads,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.attn",
            )
        else:
            self.attn = Attention(
                self.num_heads,
                self.head_dim,
                self.scaling,
                num_kv_heads=self.num_kv_heads,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.attn",
            )

        if self.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        x_and_scale_quanted: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states, x_and_scale_quanted=x_and_scale_quanted)

        if self.enable_fused_qkv_split_rms_rope_kvstore and self.use_qk_norm:
            if self.runtime_sp:
                raise NotImplementedError(
                    "Runtime SP is not supported with fused qkv/rms/rope/kvstore."
                )
            cos_sin_cache = self.rotary_emb.cos_sin_cache
            if (cos_sin_cache.device != qkv.device
                    or cos_sin_cache.dtype != qkv.dtype):
                cos_sin_cache = cos_sin_cache.to(qkv.device,
                                                dtype=qkv.dtype,
                                                non_blocking=True)
                # Persist the converted cache so we don't re-copy/re-allocate
                # on every forward when the original buffer starts on CPU.
                self.rotary_emb.cos_sin_cache = cos_sin_cache
            attn_output = self.attn(qkv,
                                    positions,
                                    cos_sin_cache,
                                    self.q_norm.weight,
                                    self.k_norm.weight,
                                    self.q_norm.variance_epsilon,
                                    is_neox=self.rotary_emb.is_neox_style)
        else:
            if self.runtime_sp:
                q_size = self.total_num_heads * self.head_dim
                kv_size = self.total_num_kv_heads * self.head_dim
            else:
                q_size = self.q_size
                kv_size = self.kv_size
            q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
            if self.use_qk_norm:
                q_heads = self.total_num_heads if self.runtime_sp else self.num_heads
                kv_heads = (
                    self.total_num_kv_heads if self.runtime_sp else self.num_kv_heads
                )
                q_by_head = q.view(
                    *q.shape[:-1], q_heads, self.head_dim
                )
                q_by_head = self.q_norm(q_by_head)
                q = q_by_head.view(q.shape)

                k_by_head = k.view(
                    *k.shape[:-1], kv_heads, self.head_dim
                )
                k_by_head = self.k_norm(k_by_head)
                k = k_by_head.view(k.shape)

            q, k = self.rotary_emb(positions, q, k)
            if self.runtime_sp:
                q, k, v = prepare_attention_inputs_for_sp(
                    q,
                    k,
                    v,
                    self.total_num_heads,
                    self.total_num_kv_heads,
                    self.head_dim,
                )
            attn_output = self.attn(q, k, v)
            attn_output = attn_output.view(q.shape[0], -1)
            if self.runtime_sp:
                attn_output = finalize_attention_output_for_sp(
                    attn_output, self.total_num_heads, self.head_dim
                )
        output, _ = self.o_proj(attn_output)
        return output


class Glm4MoeDecoderLayer(nn.Module):
    def __init__(
        self,
        config: Glm4MoeConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        enable_eplb: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        max_position_embeddings = getattr(config, "max_position_embeddings", 131072)
        # DecoderLayers are created with `make_layers` which passes the prefix
        # with the layer's index.
        layer_idx = int(prefix.split(sep=".")[-1])
        self.layer_idx = layer_idx

        self.self_attn = Glm4MoeAttention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position_embeddings=max_position_embeddings,
            head_dim=config.head_dim,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=config.attention_bias,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
            use_qk_norm=config.use_qk_norm,
        )

        if (
            config.n_routed_experts is not None
            and layer_idx >= config.first_k_dense_replace
        ):
            self.mlp = Glm4MoE(
                config=config,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
                enable_eplb=enable_eplb,
            )
        else:
            self.mlp = Glm4MoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )

        self.runtime_sp = hcu_runtime_sp_enabled()
        self.enable_fuse_rmsnorm_quant = (
            henvs.VLLM_HCU_USE_FUSED_RMS_QUANT
            and henvs.VLLM_HCU_USE_CUSTOM_OPS
            and not self.runtime_sp
        )
        self.quant_dtype = self.self_attn.quant_dtype
        if self.enable_fuse_rmsnorm_quant:
            self.input_layernorm = FusedRMSNormAndQuant(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        if self.enable_fuse_rmsnorm_quant and isinstance(self.mlp, Glm4MoeMLP):
            self.post_attention_layernorm = FusedRMSNormAndQuant(
                config.hidden_size, eps=config.rms_norm_eps
            )
        else:
            self.post_attention_layernorm = RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
        self.routed_scaling_factor = config.routed_scaling_factor

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.enable_fuse_rmsnorm_quant:
            if residual is None:
                residual = hidden_states
                hidden_states = self.input_layernorm(hidden_states)
            else:
                hidden_states, residual = self.input_layernorm(hidden_states, residual)
            hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
        else:
            if residual is None:
                residual = hidden_states.clone()

                hidden_states_q, hidden_states_scale, _ = self.input_layernorm(x=hidden_states,
                                                                               residual=None,
                                                                               quant_dtype=self.quant_dtype,
                                                                               update_input=False)
            else:
                hidden_states_q, hidden_states_scale, residual = self.input_layernorm(x=hidden_states,
                                                                                        residual=residual,
                                                                                        quant_dtype=self.quant_dtype,
                                                                                        update_input=False)

            hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states,
                                           x_and_scale_quanted=(hidden_states_q, hidden_states_scale))
        if self.enable_fuse_rmsnorm_quant and isinstance(self.mlp, Glm4MoeMLP):
            hidden_states_q, hidden_states_scale, residual = self.post_attention_layernorm(x=hidden_states,
                                                                                           residual=residual,
                                                                                           quant_dtype=self.quant_dtype,
                                                                                           update_input=False)
            hidden_states = self.mlp(hidden_states, x_and_scale_quanted=(hidden_states_q, hidden_states_scale))
        else:
            hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
            hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    }
)
class Glm4MoeModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        configure_hcu_runtime_sp(
            getattr(vllm_config.parallel_config, "enable_custom_sp", False)
        )
        enable_eplb = vllm_config.parallel_config.enable_eplb
        self.config = config

        self.vocab_size = config.vocab_size

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size, config.hidden_size, prefix=f"{prefix}.embed_tokens"
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: Glm4MoeDecoderLayer(
                config=config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=prefix,
                enable_eplb=enable_eplb,
            ),
            prefix=f"{prefix}.layers",
        )

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            hidden_states = split_tokens_for_sp(hidden_states)
            positions = split_positions_for_sp(positions)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
            positions = split_positions_for_sp(positions)

        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, residual = layer(positions, hidden_states, residual)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        hidden_states, _ = self.norm(hidden_states, residual)
        hidden_states = gather_tokens_for_sp(hidden_states)
        return hidden_states

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        return fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        expert_params_mapping = self.get_expert_mapping()
        for name, loaded_weight in weights:
            spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)
            if spec_layer is not None:
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                # Skip non-stacked layers and experts (experts handled below).
                if weight_name not in name:
                    continue
                # We have mlp.experts[0].gate_proj in the checkpoint.
                # Since we handle the experts below in expert_params_mapping,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to mlp.experts[0].gate_up_proj, which
                # will then be updated below in expert_params_mapping
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                if ("mlp.experts." in name) and name not in params_dict:
                    continue

                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                if weight_loader == default_weight_loader:
                    weight_loader(param, loaded_weight)
                else:
                    weight_loader(param, loaded_weight, shard_id)
                break
            else:
                is_expert_weight = False
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue

                    # Anyway, this is an expert weight and should not be
                    # attempted to load as other weights later
                    is_expert_weight = True

                    # Do not modify `name` since the loop may continue here
                    # Instead, create a new variable
                    name_mapped = name.replace(weight_name, param_name)

                    if is_pp_missing_parameter(name_mapped, self):
                        continue

                    param = params_dict[name_mapped]
                    # We should ask the weight loader to return success or not
                    # here since otherwise we may skip experts with other
                    # available replicas.
                    weight_loader = typing.cast(
                        Callable[..., bool], param.weight_loader
                    )
                    success = weight_loader(
                        param,
                        loaded_weight,
                        name_mapped,
                        shard_id=shard_id,
                        expert_id=expert_id,
                        return_success=True,
                    )
                    if success:
                        name = name_mapped
                        break
                else:
                    if is_expert_weight:
                        # We've checked that this is an expert weight
                        # However it's not mapped locally to this rank
                        # So we simply skip it
                        continue

                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue

                    # Remapping the name of FP8 kv-scale.
                    name = maybe_remap_kv_scale_name(name, params_dict)
                    if name is None:
                        continue

                    if is_pp_missing_parameter(name, self):
                        continue

                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)

        return loaded_params


class Glm4MixtureOfExperts(MixtureOfExperts):
    def extract_moe_parameters(self, example_moe: Glm4MoE | None) -> None:
        if example_moe is None:
            raise RuntimeError("No Glm4MoE layer found in model.layers.")
        else:
            self.num_logical_experts = example_moe.n_logical_experts
            self.num_physical_experts = example_moe.n_physical_experts
            self.num_local_physical_experts = example_moe.n_local_physical_experts
            self.num_routed_experts = example_moe.n_routed_experts
            self.num_shared_experts = example_moe.n_shared_experts
            self.num_redundant_experts = example_moe.n_redundant_experts

    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        assert self.num_local_physical_experts == num_local_physical_experts
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts
        for moe in self.moe_mlp_layers:
            moe.n_local_physical_experts = num_local_physical_experts
            moe.n_physical_experts = num_physical_experts
            moe.n_redundant_experts = self.num_redundant_experts
            moe.experts.update_expert_map()


class Glm4MoeForCausalLM(nn.Module, SupportsPP, SupportsLoRA, Glm4MixtureOfExperts):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    fall_back_to_pt_during_load = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        self.model = Glm4MoeModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )
        self.expert_weights = []

        # Set MoE hyperparameters
        self.num_moe_layers = config.num_hidden_layers - config.first_k_dense_replace
        self.num_expert_groups = config.n_group

        self.moe_layers = []
        self.moe_mlp_layers: list[Glm4MoE] = []

        example_moe = None
        for layer in self.model.layers:
            if isinstance(layer, PPMissingLayer):
                continue

            assert isinstance(layer, Glm4MoeDecoderLayer)
            if isinstance(layer.mlp, Glm4MoE):
                # Pick last one layer since the first ones may be dense layers.
                example_moe = layer.mlp
                self.moe_mlp_layers.append(layer.mlp)
                self.moe_layers.append(layer.mlp.experts)

        self.extract_moe_parameters(example_moe)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()


def get_spec_layer_idx_from_weight_name(
    config: Glm4MoeConfig, weight_name: str
) -> int | None:
    if hasattr(config, "num_nextn_predict_layers") and (
        config.num_nextn_predict_layers > 0
    ):
        layer_idx = config.num_hidden_layers
        for i in range(config.num_nextn_predict_layers):
            if f"layers.{layer_idx + i}." in weight_name:
                return layer_idx + i
    return None
