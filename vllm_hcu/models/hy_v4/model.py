# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from
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
"""Inference-only HY V4 model for HCU with strict checkpoint loading."""

import typing
from collections.abc import Callable, Iterable, MutableSequence, Sequence
from itertools import islice

import regex as re
import torch
from torch import nn
from transformers import PretrainedConfig

from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import fused_moe_make_expert_params_mapping
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.deepseek_v2 import (
    get_spec_layer_idx_from_weight_name,
)
from vllm.model_executor.models.interfaces import SupportsLoRA, SupportsPP
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    get_pp_missing_layer_names,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors

from .attention import (
    HYV4MLAAttention,
    compute_skip_topk_layers,
    is_skip_topk_indexer_weight,
    require_local_indexer_producer,
)
from .hc import HYV4HCHeadLayer, HYV4HCLayer
from .moe import HYV4FeedForward, HYV4MoEFused

logger = init_logger(__name__)

HYV4_PACKED_MODULES_MAPPING = {
    "gate_up_proj": ["gate_proj", "up_proj"],
}


def _rewrite_hyv4_weight_name(name: str) -> str:
    """Map HY V4 checkpoint-only names to runtime parameter names."""
    name = name.replace("gate.e_score_correction_bias", "expert_bias")
    name = name.replace("router.gate.", "gate.")
    if name.endswith(".hc_head_fn"):
        name += ".weight"
    elif name.endswith(".hc_fn"):
        name += ".weight"
    return name


def _slice_sink_for_tp(
    loaded_weight: torch.Tensor,
    *,
    num_heads: int,
    tp_size: int,
    tp_rank: int,
) -> torch.Tensor:
    """Return the contiguous learnable-sink shard owned by one TP rank."""
    if tp_size <= 0:
        raise ValueError(f"TP size must be positive, got {tp_size}.")
    if not 0 <= tp_rank < tp_size:
        raise ValueError(f"TP rank {tp_rank} is outside [0, {tp_size}).")
    if num_heads % tp_size != 0:
        raise ValueError(
            f"Attention heads ({num_heads}) must be divisible by TP size ({tp_size})."
        )
    if loaded_weight.ndim == 0 or loaded_weight.shape[0] != num_heads:
        shape = tuple(loaded_weight.shape)
        raise ValueError(
            f"Learnable sink must contain {num_heads} attention heads; got {shape}."
        )
    local_heads = num_heads // tp_size
    return loaded_weight.narrow(0, tp_rank * local_heads, local_heads)


def _dequantize_indexer_channel_fp8(
    weight: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Dequantize an indexer projection exported with channel-wise FP8."""
    expected_scale_shape = (weight.shape[0], 1)
    if weight.ndim != 2 or tuple(scale.shape) != expected_scale_shape:
        raise ValueError(
            "HY V4 indexer FP8 scale must be per-output-channel with shape "
            f"{expected_scale_shape}; got weight={tuple(weight.shape)}, "
            f"scale={tuple(scale.shape)}."
        )
    return (weight.float() * scale.float()).to(torch.bfloat16)


def _try_load_fp8_indexer_projection(
    name: str,
    tensor: torch.Tensor,
    buffer: dict[str, dict[str, torch.Tensor]],
    params_dict: dict[str, torch.nn.Parameter],
    loaded_params: set[str],
    pp_missing_layer_names: set[str],
) -> bool:
    """Dequantize and load fused ``wk``/``weights_proj`` FP8 shards."""
    marker = ".indexer."
    if marker not in name:
        return False
    layer_prefix, suffix = name.rsplit(marker, 1)
    projection = suffix.split(".", 1)[0]
    if projection not in {"wk", "weights_proj"}:
        return False

    is_weight = suffix == f"{projection}.weight" and tensor.dtype in {
        torch.float8_e4m3fn,
        getattr(torch, "float8_e4m3fnuz", torch.float8_e4m3fn),
    }
    is_scale = suffix in {
        f"{projection}.weight_scale",
        f"{projection}.weight_scale_inv",
    }
    if not is_weight and not is_scale:
        return False
    if any(name.startswith(missing) for missing in pp_missing_layer_names):
        return True

    indexer_prefix = f"{layer_prefix}{marker[:-1]}"
    fused_name = f"{indexer_prefix}.wk_weights_proj.weight"
    key_prefix = f"{indexer_prefix}.{projection}"
    entry = buffer.setdefault(key_prefix, {})
    entry["weight" if is_weight else "scale"] = tensor
    if entry.keys() < {"weight", "scale"}:
        return True

    if fused_name not in params_dict:
        raise RuntimeError(
            f"Unknown HY V4 fused indexer parameter: {fused_name}"
        )
    dequantized = _dequantize_indexer_channel_fp8(
        entry["weight"],
        entry["scale"],
    )
    shard_id = 0 if projection == "wk" else 1
    param = params_dict[fused_name]
    param.weight_loader(param, dequantized, shard_id)
    loaded_params.add(fused_name)
    del buffer[key_prefix]
    return True


def _try_load_fp8_router_gate(
    name: str,
    tensor: torch.Tensor,
    buffer: dict[str, dict[str, torch.Tensor]],
    params_dict: dict[str, torch.nn.Parameter],
    loaded_params: set[str],
    pp_missing_layer_names: set[str],
) -> bool:
    """Dequantize a channel-wise FP8 MoE router into its FP32 gate."""
    mapped_name = _rewrite_hyv4_weight_name(name)
    if not mapped_name.endswith((".mlp.gate.weight", ".mlp.gate.weight_scale")):
        return False

    is_weight = mapped_name.endswith(".weight") and tensor.dtype in {
        torch.float8_e4m3fn,
        getattr(torch, "float8_e4m3fnuz", torch.float8_e4m3fn),
    }
    is_scale = mapped_name.endswith(".weight_scale")
    if not is_weight and not is_scale:
        return False
    if any(name.startswith(missing) for missing in pp_missing_layer_names):
        return True

    parameter_name = (
        mapped_name.removesuffix("_scale") if is_scale else mapped_name
    )
    entry = buffer.setdefault(parameter_name, {})
    entry["weight" if is_weight else "scale"] = tensor
    if entry.keys() < {"weight", "scale"}:
        return True
    if parameter_name not in params_dict:
        raise RuntimeError(
            f"Unknown HY V4 FP8 router parameter: {parameter_name}"
        )

    weight = entry["weight"]
    scale = entry["scale"]
    expected_scale_shape = (weight.shape[0], 1)
    if weight.ndim != 2 or tuple(scale.shape) != expected_scale_shape:
        raise ValueError(
            "HY V4 router FP8 scale must be per-output-channel with shape "
            f"{expected_scale_shape}; got weight={tuple(weight.shape)}, "
            f"scale={tuple(scale.shape)}."
        )
    dequantized = weight.float() * scale.float()
    param = params_dict[parameter_name]
    weight_loader = getattr(param, "weight_loader", default_weight_loader)
    weight_loader(param, dequantized)
    loaded_params.add(parameter_name)
    del buffer[parameter_name]
    return True


def _normalize_hyv4_config(config: PretrainedConfig) -> PretrainedConfig:
    """Populate the aliases consumed by the shared MoE implementation."""
    config.router_scaling_factor = config.routed_scaling_factor
    config.num_experts = config.n_routed_experts
    config.expert_hidden_dim = config.moe_intermediate_size
    config.num_shared_experts = config.n_shared_experts
    config.route_norm = config.norm_topk_prob
    return config


class HYV4DecoderLayer(nn.Module):
    """One HY V4 decoder layer: MLA attention plus a dense or MoE MLP.

    When``config.enable_ihc`` is set the layer runs on ``hc_mult`` residual
    channels and each sub-block is wrapped by an `HYV4HCLayer` boundary;
    otherwise it uses the standard single-stream residual.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        vllm_config: VllmConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        topk_indices_buffer: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.config = config
        self.enable_ihc = getattr(config, "enable_ihc", False)
        layer_idx = int(prefix.split(".")[-1])
        self.layer_idx = layer_idx

        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        self.self_attn = HYV4MLAAttention(
            vllm_config=vllm_config,
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            q_lora_rank=config.q_lora_rank,
            kv_lora_rank=config.kv_lora_rank,
            max_position_embeddings=max_position_embeddings,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
            layer_idx=layer_idx,
            topk_indices_buffer=topk_indices_buffer,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        if config.mlp_layer_types[layer_idx] == "dense":
            self.mlp = HYV4FeedForward(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
            self.block_type = "feedforward"
        else:
            self.mlp = HYV4MoEFused(
                config=config,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
                enable_eplb=vllm_config.parallel_config.enable_eplb,
                vllm_config=vllm_config,
            )
            self.block_type = "moe"
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.hc_attn_layer = HYV4HCLayer(config, layer_idx)
        self.hc_mlp_layer = HYV4HCLayer(config, layer_idx)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.enable_ihc:
            return self._forward_ihc(positions, hidden_states)
        return self._forward_normal(positions, hidden_states, residual)

    def _forward_normal(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Standard single-residual-stream forward (iHC disabled)."""
        if residual is not None:
            hidden_states = hidden_states + residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        hidden_states = hidden_states + residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual

    def _forward_ihc(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        """iHC forward: each sub-block reduces and re-scatters the channels."""
        hidden_states = self.hc_attn_layer.prepare_input(hidden_states)
        hidden_states, post_gates, residual = self.hc_attn_layer.pre(hidden_states)
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )
        hidden_states = self.hc_attn_layer.post(hidden_states, residual, post_gates)

        hidden_states = self.hc_mlp_layer.prepare_input(hidden_states)
        hidden_states, post_gates, residual = self.hc_mlp_layer.pre(hidden_states)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.hc_mlp_layer.post(hidden_states, residual, post_gates)

        # Under iHC the residual is carried inside hidden_states.
        return hidden_states, None


class HYV4Model(nn.Module):
    """HY V4 backbone."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = _normalize_hyv4_config(vllm_config.model_config.hf_config)
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config
        eplb_config = parallel_config.eplb_config
        self.num_redundant_experts = eplb_config.num_redundant_experts
        self.device = current_platform.device_type
        self.vocab_size = config.vocab_size
        self.is_sparse = hasattr(config, "index_topk")
        if self.is_sparse:
            self.topk_indices_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                config.index_topk,
                dtype=torch.int32,
                device=self.device,
            )
        else:
            self.topk_indices_buffer = None
        self.config = config
        self.quant_config = quant_config
        self.enable_ihc = getattr(config, "enable_ihc", False)

        if get_pp_group().is_first_rank or (
            self.config.tie_word_embeddings and get_pp_group().is_last_rank
        ):
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: HYV4DecoderLayer(
                config=config,
                vllm_config=vllm_config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=prefix,
                topk_indices_buffer=self.topk_indices_buffer,
            ),
            prefix=f"{prefix}.layers",
        )
        require_local_indexer_producer(
            config,
            start_layer=self.start_layer,
            end_layer=self.end_layer,
        )
        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()
        if self.enable_ihc:
            # iHC head layer: merge the residual channels into one stream.
            if get_pp_group().is_last_rank:
                self.hc_head = HYV4HCHeadLayer(
                    config,
                    hidden_size=config.hidden_size,
                    hc_mult=config.hc_mult,
                    hc_eps=config.hc_eps,
                )
            else:
                self.hc_head = PPMissingLayer()
            self.make_empty_intermediate_tensors = (
                make_empty_intermediate_tensors_factory(
                    ["hidden_states"], config.hc_mult * config.hidden_size
                )
            )
        else:
            self.make_empty_intermediate_tensors = (
                make_empty_intermediate_tensors_factory(
                    ["hidden_states", "residual"], config.hidden_size
                )
            )

        # MoE hyperparameters (consumed by EPLB).
        self.expert_weights: MutableSequence[Sequence[torch.Tensor]] = []
        self.num_expert_groups = 1
        self.moe_layers: list[nn.Module] = []
        example_layer: HYV4MoEFused | None = None
        for layer in self.layers:
            if isinstance(layer, PPMissingLayer):
                continue

            assert isinstance(layer, HYV4DecoderLayer)
            if layer.block_type == "moe":
                assert isinstance(layer.mlp, HYV4MoEFused)
                example_layer = layer.mlp
                self.moe_layers.append(layer.mlp.experts)

        if example_layer is None:
            self.num_moe_layers = 0
            self.num_logical_experts = 0
            self.num_physical_experts = 0
            self.num_local_physical_experts = 0
            self.num_routed_experts = 0
            self.num_redundant_experts = 0
            return

        self.num_moe_layers = len(self.moe_layers)
        self.num_logical_experts = example_layer.n_logical_experts
        self.num_physical_experts = example_layer.n_physical_experts
        self.num_local_physical_experts = example_layer.n_local_physical_experts
        self.num_routed_experts = example_layer.n_routed_experts
        self.num_redundant_experts = example_layer.n_redundant_experts

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
            if isinstance(layer, PPMissingLayer):
                continue
            if isinstance(layer.mlp, HYV4MoEFused):
                moe = layer.mlp
                moe.n_local_physical_experts = num_local_physical_experts
                moe.n_physical_experts = num_physical_experts
                moe.n_redundant_experts = self.num_redundant_experts
                moe.experts.update_expert_map()

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        # (param_name, weight_name, expert_id, shard_id) for weights, fp8
        # weight scales and fp8 activation scales.
        if not hasattr(self, "_cached_expert_params_mapping"):
            self._cached_expert_params_mapping = fused_moe_make_expert_params_mapping(
                self,
                ckpt_gate_proj_name="gate_proj",
                ckpt_down_proj_name="down_proj",
                ckpt_up_proj_name="up_proj",
                num_experts=self.config.num_experts,
                num_redundant_experts=self.num_redundant_experts,
            )
        return self._cached_expert_params_mapping

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
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            # In iHC mode the flattened [num_tokens, hc*h] tensor from the
            # previous PP stage is reshaped back to 3D by the first layer's
            # prepare_input, and residual is unused.
            residual = None if self.enable_ihc else intermediate_tensors["residual"]

        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, residual = layer(positions, hidden_states, residual)

        if not get_pp_group().is_last_rank:
            if self.enable_ihc:
                # hidden_states is [num_tokens, hc, h]; flatten the channel dim
                # for PP transfer (matches the 2D receive buffer).
                return IntermediateTensors({"hidden_states": hidden_states.flatten(1)})
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        if self.enable_ihc:
            hidden_states = self.hc_head(hidden_states)
        else:
            hidden_states = hidden_states + residual

        return self.norm(hidden_states)

    def load_fused_expert_weights(
        self,
        name: str,
        params_dict: dict,
        loaded_weight: torch.Tensor,
        shard_id: str,
        num_experts: int,
    ) -> bool:
        param = params_dict[name]
        weight_loader = typing.cast(Callable[..., bool], param.weight_loader)
        loaded_local_expert = False
        for expert_id in range(num_experts):
            curr_expert_weight = loaded_weight[expert_id]
            success = weight_loader(
                param,
                curr_expert_weight,
                name,
                shard_id,
                expert_id,
                return_success=True,
            )
            if success:
                loaded_local_expert = True

        return loaded_local_expert

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
            # The indexer fuses wk and weights_proj into one GEMM.
            ("wk_weights_proj", "wk", 0),
            ("wk_weights_proj", "weights_proj", 1),
        ]
        # FP8 indexer projection buffers (weights/scales can be in different
        # checkpoint shards and therefore arrive in either order).
        pending_indexer_fp8: dict[str, dict[str, torch.Tensor]] = {}
        pending_router_fp8: dict[str, dict[str, torch.Tensor]] = {}
        pp_missing_layer_names = get_pp_missing_layer_names(self)
        skip_topk_layers = compute_skip_topk_layers(self.config)

        # Must not be cached on `self`: `process_weights_after_loading` swaps in
        # kernel-specific expert weights via `replace_parameter`, and a cache
        # outliving this call would pin the pre-shuffle storage (OOM on large
        # MoE) and make a later reload target orphaned tensors.
        params_dict = dict(self.named_parameters())
        # Split per-expert mapping (V3 style): experts.0.gate_proj.weight
        split_expert_params_mapping = self.get_expert_mapping()
        loaded_params: set[str] = set()

        # Sink weights are sharded like the q/k/v linears.
        sink_tp_size = get_tensor_model_parallel_world_size()
        sink_tp_rank = get_tensor_model_parallel_rank()
        base_layer = (
            "base_layer." if any(".base_layer." in name for name in params_dict) else ""
        )
        # Fused expert mapping: experts.gate_up_proj (all experts in one tensor).
        # The packed weights are owned by the RoutedExperts submodule, so the
        # targets are experts.routed_experts.[base_layer.]w{13,2}_weight.
        fused_expert_prefix = f".experts.routed_experts.{base_layer}"
        fused_expert_params_mapping = [
            (f"{fused_expert_prefix}w13_weight", ".experts.gate_up_proj", 0, "w1"),
            (f"{fused_expert_prefix}w2_weight", ".experts.down_proj", 0, "w2"),
        ]
        num_experts = getattr(self.config, "num_experts", 0)

        def _should_skip_missing_param(param_name: str) -> bool:
            # Shared-indexer entries are filtered before dispatch. Anything
            # else missing indicates an adapter/checkpoint mismatch.
            return False

        def _is_split_expert_weight(weight_name: str) -> bool:
            """Whether this weight is in split (per-expert) format."""
            # Split format: mlp.experts.<id>.gate_proj.weight
            return bool(re.search(r"\.experts\.\d+\.", weight_name))

        def _is_fused_expert_weight(weight_name: str) -> bool:
            """Whether this weight is in fused (all-experts-packed) format."""
            return ".experts.gate_up_proj" in weight_name or (
                ".experts.down_proj" in weight_name
                and not _is_split_expert_weight(weight_name)
            )

        for name, loaded_weight in weights:
            if is_skip_topk_indexer_weight(name, skip_topk_layers):
                continue
            mapped_name = _rewrite_hyv4_weight_name(name)
            if mapped_name != name and mapped_name.endswith(".expert_bias"):
                if is_pp_missing_parameter(mapped_name, self):
                    continue
                if mapped_name not in params_dict:
                    raise RuntimeError(
                        "Unknown HY V4 expert correction bias after mapping: "
                        f"{mapped_name}"
                    )
                param = params_dict[mapped_name]
                default_weight_loader(param, loaded_weight)
                loaded_params.add(mapped_name)
                continue
            if _try_load_fp8_indexer_projection(
                name,
                loaded_weight,
                pending_indexer_fp8,
                params_dict,
                loaded_params,
                pp_missing_layer_names,
            ):
                continue
            if _try_load_fp8_router_gate(
                name,
                loaded_weight,
                pending_router_fp8,
                params_dict,
                loaded_params,
                pp_missing_layer_names,
            ):
                continue
            if self.config.tie_word_embeddings and "lm_head.weight" in name:
                continue
            # KV-cache scale names are normalized upstream: AutoWeightsLoader
            # applies quant_config.get_cache_scale_mapper() before dispatching
            # here, so the scales arrive under their vLLM parameter names and
            # are handled by the generic branch at the end of this loop.
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

                if is_pp_missing_parameter(name, self):
                    continue

                if name not in params_dict:
                    if _should_skip_missing_param(name):
                        continue
                    raise RuntimeError(
                        f"Unknown HY V4 checkpoint weight after mapping: {name}"
                    )

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

            # Determine per-weight whether this is fused or split format.
            is_fused_expert = _is_fused_expert_weight(name)
            expert_params_mapping = (
                fused_expert_params_mapping
                if is_fused_expert
                else split_expert_params_mapping
            )

            is_expert_weight = False
            loaded_expert_param_names: set[str] = set()
            for mapping in expert_params_mapping:
                param_name, weight_name, expert_id, shard_id = mapping
                if weight_name not in name:
                    continue

                # This is an expert weight and must not be attempted as any
                # other kind of weight later on.
                is_expert_weight = True

                # Do not modify `name`: the loop may continue past this point.
                name_mapped = name.replace(weight_name, param_name)
                if is_pp_missing_parameter(name_mapped, self):
                    continue
                if is_fused_expert:
                    if "experts.gate_up_proj" in name:
                        chunks = loaded_weight.chunk(2, dim=-2)
                        success_w1 = self.load_fused_expert_weights(
                            name_mapped, params_dict, chunks[0], "w1", num_experts
                        )
                        success_w3 = self.load_fused_expert_weights(
                            name_mapped, params_dict, chunks[1], "w3", num_experts
                        )
                        success = success_w1 and success_w3
                    else:
                        success = self.load_fused_expert_weights(
                            name_mapped,
                            params_dict,
                            loaded_weight,
                            shard_id,
                            num_experts,
                        )
                    if success:
                        name = name_mapped
                        break
                else:
                    # Split per-expert format (V3 style).
                    if name_mapped not in params_dict:
                        if _should_skip_missing_param(name_mapped):
                            continue
                        raise RuntimeError(
                            "Unknown HY V4 expert checkpoint weight after mapping: "
                            f"{name_mapped}"
                        )
                    param = params_dict[name_mapped]
                    # Ask the weight loader whether it succeeded, otherwise we
                    # may skip experts that have other available replicas.
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
                    if not is_fused_expert:
                        loaded_expert_param_names.add(name_mapped)
                        continue
                    name = name_mapped
                    break
            else:
                if loaded_expert_param_names:
                    loaded_params.update(loaded_expert_param_names)
                    continue
                if "learnable_sink_param" in name:
                    if is_pp_missing_parameter(name, self):
                        continue
                    if name not in params_dict:
                        raise RuntimeError(
                            f"Unknown HY V4 learnable sink parameter: {name}"
                        )
                    narrow_weight = _slice_sink_for_tp(
                        loaded_weight,
                        num_heads=self.config.num_attention_heads,
                        tp_size=sink_tp_size,
                        tp_rank=sink_tp_rank,
                    )
                    param = params_dict[name]
                    if tuple(param.shape) != tuple(narrow_weight.shape):
                        raise ValueError(
                            f"Learnable sink shard shape mismatch for {name}: "
                            f"parameter={tuple(param.shape)}, "
                            f"checkpoint={tuple(narrow_weight.shape)}"
                        )
                    with torch.no_grad():
                        param.copy_(narrow_weight)
                else:
                    if is_expert_weight:
                        # An expert weight that is not mapped to this rank.
                        continue
                    name = _rewrite_hyv4_weight_name(name)

                    if is_pp_missing_parameter(name, self):
                        continue

                    if name not in params_dict:
                        if _should_skip_missing_param(name):
                            continue
                        raise RuntimeError(
                            f"Unknown HY V4 checkpoint weight after mapping: {name}"
                        )

                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)

        if pending_indexer_fp8:
            missing_parts = {
                key: sorted({"weight", "scale"} - set(parts))
                for key, parts in pending_indexer_fp8.items()
            }
            raise RuntimeError(
                "Incomplete HY V4 FP8 indexer projection pairs: "
                f"{missing_parts}"
            )
        if pending_router_fp8:
            missing_parts = {
                key: sorted({"weight", "scale"} - set(parts))
                for key, parts in pending_router_fp8.items()
            }
            raise RuntimeError(
                f"Incomplete HY V4 FP8 router pairs: {missing_parts}"
            )
        return loaded_params


class HYV4ForCausalLM(nn.Module, SupportsPP, SupportsLoRA):
    packed_modules_mapping = HYV4_PACKED_MODULES_MAPPING

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        if quant_config is not None:
            quant_config.packed_modules_mapping = self.packed_modules_mapping

        parallel_config = vllm_config.parallel_config
        eplb_config = parallel_config.eplb_config
        self.num_redundant_experts = eplb_config.num_redundant_experts

        self.model = HYV4Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.enable_lm_head_fp32 = getattr(self.config, "enable_lm_head_fp32", False)
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                params_dtype=torch.float32 if self.enable_lm_head_fp32 else None,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            # With tie_word_embeddings, embed_tokens is kept on the last rank
            # (see HYV4Model.__init__) so the weight can be shared here.
            if self.config.tie_word_embeddings:
                self.lm_head.weight = self.model.embed_tokens.weight
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        if self.enable_lm_head_fp32:
            # Keep the whole projection in fp32 so the head matches training.
            with torch.autocast(device_type="cuda", enabled=False):
                logits = self.logits_processor(
                    self.lm_head, hidden_states.to(torch.float32)
                )
        else:
            logits = self.logits_processor(self.lm_head, hidden_states)

        if getattr(self.config, "soft_logits_capping", False):
            soft_cap = self.config.soft_logits_capping_logits
            logits = soft_cap * torch.nn.functional.tanh(logits / soft_cap)

        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        def _filter_weights(weights):
            for name, weight in weights:
                # Exclude both model.layers.<N>.* and model.mtp_layers.<i>.*
                # from target loading when speculative MTP is enabled.
                if name.startswith("model.mtp_layers."):
                    continue
                if get_spec_layer_idx_from_weight_name(self.config, name) is not None:
                    continue
                yield name, weight

        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None),
        )
        loaded_params = loader.load_weights(_filter_weights(weights))
        required_params = {name for name, _ in self.named_parameters()}
        missing_params = required_params - loaded_params
        if missing_params:
            raise RuntimeError(
                "Missing HY V4 checkpoint parameters: "
                f"{sorted(missing_params)}"
            )
        return loaded_params

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()
