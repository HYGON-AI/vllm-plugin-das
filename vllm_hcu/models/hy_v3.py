# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# Modified by Hygon Information Technology Co., Ltd., 2026.

# coding=utf-8
# Copyright 2026 The HY team.
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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
"""Inference-only HY model compatible with HuggingFace weights."""

import typing
from collections.abc import Callable, Iterable
from itertools import islice
from typing import Any

import torch
from torch import nn
from transformers import PretrainedConfig

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
    GateLinear,
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
from vllm.transformers_utils.configs.hy_v3 import HYV3Config
from vllm.v1.attention.selector import get_attn_backend

from vllm.model_executor.models.interfaces import SupportsLoRA, SupportsPP
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
import vllm_hcu.platforms.envs as henvs
from vllm_hcu.patch.config import get_hcu_config
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
from vllm_hcu.ops.fuse_silu_mul_quant import FusedSiluAndMulAndQuant

logger = init_logger(__name__)


def fused_qkv_cache_layout_supported(attn_backend: type) -> bool:
    return attn_backend.get_name() != "TRITON_ATTN"


class HYV3FeedForward(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        expert_gate: torch.nn.Linear | None = None,
        disable_tp: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.use_sp_token_gather = use_sp_mlp_token_gather(reduce_results)
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
            reduce_results=sp_mlp_down_proj_reduce_results(reduce_results),
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

        if self.enable_fuse_silu_mul_quant and self.quant_dtype is not None:
            self.act_fn = FusedSiluAndMulAndQuant()
        else:
            self.act_fn = SiluAndMul()

    def forward(
        self,
        x,
        x_and_scale_quanted: tuple[torch.Tensor, torch.Tensor] | None = None,
    ):
        x, x_and_scale_quanted = prepare_mlp_inputs_for_sp(
            x, x_and_scale_quanted, self.use_sp_token_gather
        )

        if self.enable_fuse_silu_mul_quant and self.quant_dtype is not None:
            gate_up, _ = self.gate_up_proj._forward_with_hcu_quanted(
                x, x_and_scale_quanted
            )
            xq, xs = self.act_fn(gate_up, quant_dtype=self.quant_dtype)
            x, _ = self.down_proj._forward_with_hcu_quanted(xq, (xq, xs))
        else:
            gate_up, _ = self.gate_up_proj(x)
            x = self.act_fn(gate_up)
            x, _ = self.down_proj(x)

        return finalize_mlp_output_for_sp(x, self.use_sp_token_gather)


class HYV3MoEFused(nn.Module):
    def __init__(
        self,
        config: HYV3Config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        enable_eplb: bool = False,
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.ep_group = get_ep_group().device_group
        self.ep_rank = get_ep_group().rank_in_group
        self.ep_size = self.ep_group.size()
        self.n_routed_experts = config.num_experts
        if self.tp_size > config.num_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.num_experts}."
            )
        top_k = config.num_experts_per_tok
        intermediate_size = config.expert_hidden_dim
        router_scaling_factor = getattr(config, "router_scaling_factor", 1.0)
        vllm_config = get_current_vllm_config()
        eplb_config = vllm_config.parallel_config.eplb_config
        self.enable_eplb = enable_eplb

        self.n_logical_experts = self.n_routed_experts
        self.n_redundant_experts = eplb_config.num_redundant_experts
        self.n_physical_experts = self.n_logical_experts + self.n_redundant_experts
        self.n_local_physical_experts = self.n_physical_experts // self.ep_size
        self.physical_expert_start = self.ep_rank * self.n_local_physical_experts
        self.physical_expert_end = (
            self.physical_expert_start + self.n_local_physical_experts
        )
        self.gate = GateLinear(
            config.hidden_size,
            config.num_experts,
            bias=False,
            out_dtype=torch.float32,
            params_dtype=torch.float32,
            prefix=f"{prefix}.gate",
        )

        self.runtime_sp = hcu_runtime_sp_enabled()
        disable_shared_mlp_tp = (
            self.runtime_sp and vllm_config.parallel_config.enable_expert_parallel
        )

        if config.num_shared_experts > 0:
            self.shared_mlp = HYV3FeedForward(
                hidden_size=config.hidden_size,
                intermediate_size=config.expert_hidden_dim * config.num_shared_experts,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}",
                reduce_results=False,
                disable_tp=disable_shared_mlp_tp,
            )
        else:
            self.shared_mlp = None

        self.expert_bias = nn.Parameter(torch.empty(config.num_experts))
        scoring_func = "sigmoid"
        e_score_correction_bias = self.expert_bias

        self.experts = FusedMoE(
            num_experts=self.n_routed_experts,
            top_k=top_k,
            hidden_size=config.hidden_size,
            intermediate_size=intermediate_size,
            renormalize=config.route_norm,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            enable_eplb=self.enable_eplb,
            num_redundant_experts=self.n_redundant_experts,
            scoring_func=scoring_func,
            use_grouped_topk=True,
            num_expert_group=1,
            topk_group=1,
            routed_scaling_factor=router_scaling_factor,
            e_score_correction_bias=e_score_correction_bias,
            n_shared_experts=config.num_shared_experts,
            shared_experts=self.shared_mlp,
            is_sequence_parallel=self.runtime_sp
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        orig_shape = hidden_states.shape
        hidden_dim = hidden_states.shape[-1]
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (num_tokens, n_experts)
        router_logits, _ = self.gate(hidden_states)
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
        return final_hidden_states.view(orig_shape)


class HYV3Attention(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_parameters: dict[str, Any],
        max_position_embeddings: int = 8192,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-5,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        dual_chunk_attention_config: dict[str, Any] | None = None,
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

        if hasattr(config, "head_dim") and config.head_dim:
            self.head_dim = config.head_dim
        else:
            self.head_dim = head_dim or (hidden_size // self.total_num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.use_qk_norm = getattr(config, "qk_norm", False)
        self.max_position_embeddings = max_position_embeddings

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            quant_config=quant_config,
            bias=None,
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
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=max_position_embeddings,
            rope_parameters=rope_parameters,
            is_neox_style=True,
        )
        self.enable_fused_qkv_split_rms_rope_kvstore = henvs.VLLM_HCU_USE_FUSED_QKV_SPLIT_RMS_ROPE_KVSTORE \
            and henvs.VLLM_HCU_USE_CUSTOM_OPS
        if self.enable_fused_qkv_split_rms_rope_kvstore:
            kv_cache_dtype = cache_config.cache_dtype if cache_config is not None else "auto"
            attn_backend = get_attn_backend(
                self.head_dim,
                torch.get_default_dtype(),
                kv_cache_dtype,
            )
            if not fused_qkv_cache_layout_supported(attn_backend):
                self.enable_fused_qkv_split_rms_rope_kvstore = False
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
            self.q_norm = RMSNorm(self.head_dim, rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        x_and_scale_quanted: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj._forward_with_hcu_quanted(
            hidden_states, x_and_scale_quanted
        )

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
            output_shape = None
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
            attn_output = self.attn(q, k, v, output_shape)
            attn_output = attn_output.view(q.shape[0], -1)
            if self.runtime_sp:
                attn_output = finalize_attention_output_for_sp(
                    attn_output, self.total_num_heads, self.head_dim
                )
        output, _ = self.o_proj(attn_output)
        return output


class HYV3DecoderLayer(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        layer_idx = int(prefix.split(".")[-1])
        parallel_config = get_current_vllm_config().parallel_config
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        self.self_attn = HYV3Attention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            rope_parameters=config.rope_parameters,
            max_position_embeddings=max_position_embeddings,
            head_dim=config.head_dim,
            rms_norm_eps=config.rms_norm_eps,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        if not hasattr(config, "first_k_dense_replace"):
            raise ValueError("first_k_dense_replace not exist,please check config")
        if layer_idx < config.first_k_dense_replace:
            self.mlp = HYV3FeedForward(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
            self.block_type = "feedforward"
        else:
            self.mlp = HYV3MoEFused(
                config=config,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
                enable_eplb=parallel_config.enable_eplb,
            )
            self.block_type = "moe"

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        idx: int = -1,
    ) -> torch.Tensor:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


@support_torch_compile
class HYV3Model(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        parallel_config = vllm_config.parallel_config
        configure_hcu_runtime_sp(get_hcu_config(vllm_config).enable_custom_sp)
        eplb_config = parallel_config.eplb_config
        self.num_redundant_experts = eplb_config.num_redundant_experts

        self.vocab_size = config.vocab_size
        self.config = config
        self.quant_config = quant_config

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: HYV3DecoderLayer(
                config=config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=prefix,
            ),
            prefix=f"{prefix}.layers",
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

        # Set MoE hyperparameters
        self.expert_weights = []
        self.num_expert_groups = 1
        self.moe_layers = []
        example_layer = None
        for layer in self.layers:
            if isinstance(layer, PPMissingLayer):
                continue

            assert isinstance(layer, HYV3DecoderLayer)
            if layer.block_type == "moe":
                example_layer = layer.mlp
                self.moe_layers.append(layer.mlp.experts)

        if example_layer is None:
            self.num_moe_layers = 0
            raise RuntimeError("No MoE layer found in model.layers.")

        self.num_moe_layers = len(self.moe_layers)
        self.num_logical_experts = getattr(example_layer, "n_logical_experts", None)
        self.num_physical_experts = getattr(example_layer, "n_physical_experts", None)
        self.num_local_physical_experts = getattr(
            example_layer, "n_local_physical_experts", None
        )
        self.num_routed_experts = getattr(example_layer, "n_routed_experts", None)
        self.num_shared_experts = config.num_shared_experts
        self.num_redundant_experts = getattr(example_layer, "n_redundant_experts", None)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        assert self.num_local_physical_experts == num_local_physical_experts
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts
        for layer in self.layers:
            if isinstance(layer.mlp, HYV3MoEFused):
                moe = layer.mlp
                moe.n_local_physical_experts = num_local_physical_experts
                moe.n_physical_experts = num_physical_experts
                moe.n_redundant_experts = self.num_redundant_experts
                moe.experts.update_expert_map()

    def set_eplb_state(
        self,
        expert_load_view: torch.Tensor,
        logical_to_physical_map: torch.Tensor,
        logical_replica_count: torch.Tensor,
    ) -> None:
        self.expert_weights.clear()
        for layer_idx, layer in enumerate(self.moe_layers):
            self.expert_weights.append(layer.get_expert_weights())
            layer.set_eplb_state(
                moe_layer_idx=layer_idx,
                expert_load_view=expert_load_view,
                logical_to_physical_map=logical_to_physical_map,
                logical_replica_count=logical_replica_count,
            )

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        return fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts,
            num_redundant_experts=self.num_redundant_experts,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
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

        for idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer)
        ):
            hidden_states, residual = layer(positions, hidden_states, residual, idx=idx)
        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        hidden_states, _ = self.norm(hidden_states, residual)
        hidden_states = gather_tokens_for_sp(hidden_states)

        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        expert_params_mapping = self.get_expert_mapping()
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if self.config.tie_word_embeddings and "lm_head.weight" in name:
                continue
            if "scale" in name:
                # Remapping the name of FP8 kv-scale.
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
            is_found = False
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if "mlp.experts" in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                # Skip layers on other devices.
                if is_pp_missing_parameter(name, self):
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(name)
                is_found = True
                break
            if is_found:
                continue

            if name.endswith(".bias") and name not in params_dict:
                continue
            is_expert_weight = False
            for mapping in expert_params_mapping:
                param_name, weight_name, expert_id, shard_id = mapping
                if weight_name not in name:
                    continue
                is_expert_weight = True
                name_mapped = name.replace(weight_name, param_name)
                # Skip layers on other devices.
                if is_pp_missing_parameter(name_mapped, self):
                    continue

                param = params_dict[name_mapped]
                weight_loader = typing.cast(Callable[..., bool], param.weight_loader)
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
                if name is None:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                if "router.gate." in name:
                    name = name.replace("router.", "")

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)

        return loaded_params


def get_spec_layer_idx_from_weight_name(
    config: PretrainedConfig, weight_name: str
) -> int | None:
    # HYV3MTP is enabled only when num_nextn_predict_layers is greater than 1
    if (
        hasattr(config, "num_nextn_predict_layers")
        and config.num_nextn_predict_layers > 0
    ):
        layer_idx = config.num_hidden_layers
        for i in range(config.num_nextn_predict_layers):
            if weight_name.startswith(f"model.layers.{layer_idx + i}."):
                return layer_idx + i
    return None


class HYV3ForCausalLM(nn.Module, SupportsPP, SupportsLoRA):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config

        parallel_config = vllm_config.parallel_config
        eplb_config = parallel_config.eplb_config
        self.num_redundant_experts = eplb_config.num_redundant_experts

        self.model = HYV3Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )
        self.expert_weights = self.model.expert_weights
        self.num_moe_layers = self.model.num_moe_layers
        self.num_expert_groups = self.model.num_expert_groups
        self.num_logical_experts = self.model.num_logical_experts
        self.num_physical_experts = self.model.num_physical_experts
        self.num_local_physical_experts = self.model.num_local_physical_experts
        self.num_routed_experts = self.model.num_routed_experts
        self.num_shared_experts = self.model.num_shared_experts
        self.num_redundant_experts = self.model.num_redundant_experts
        self.moe_layers = self.model.moe_layers

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def set_eplb_state(
        self,
        expert_load_view: torch.Tensor,
        logical_to_physical_map: torch.Tensor,
        logical_replica_count: torch.Tensor,
    ) -> None:
        self.model.set_eplb_state(
            expert_load_view=expert_load_view,
            logical_to_physical_map=logical_to_physical_map,
            logical_replica_count=logical_replica_count,
        )

    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        self.model.update_physical_experts_metadata(
            num_physical_experts=num_physical_experts,
            num_local_physical_experts=num_local_physical_experts,
        )
        self.num_physical_experts = self.model.num_physical_experts
        self.num_local_physical_experts = self.model.num_local_physical_experts
        self.num_redundant_experts = self.model.num_redundant_experts

    def forward(
        self,
        input_ids: torch.Tensor,
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
        def _filter_weights(weights):
            for name, weight in weights:
                spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)
                if spec_layer is not None:
                    continue
                yield name, weight

        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(_filter_weights(weights))

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()
