# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

"""Checkpoint-native HYV4 MTP on the current draft speculator contract."""

import copy
from collections.abc import Iterable

import torch
from torch import nn
from transformers import PretrainedConfig

from vllm.config import CacheConfig, ModelConfig, VllmConfig
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.interfaces import MixtureOfExperts, SupportsPP
from vllm.model_executor.models.utils import maybe_prefix
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler

from .model import (
    HYV4DecoderLayer,
    HYV4Model,
    _is_modelopt_layer_excluded,
    _normalize_hyv4_config,
    _rewrite_hyv4_weight_name,
)


def _make_mtp_layer_config(config, layer_idx: int):
    """Extend the checkpoint's backbone layer lists for its single MTP block."""
    if layer_idx != config.num_hidden_layers:
        raise ValueError("HYV4 MTP supports exactly one layer after the backbone")
    result = copy.deepcopy(config)
    result.enable_ihc = False
    for attr, fallback in (
        ("layer_types", "full_attention"),
        ("mlp_layer_types", "sparse"),
    ):
        values = getattr(result, attr, None)
        values = list(values) if values is not None else [fallback] * layer_idx
        if len(values) == layer_idx:
            values.append(values[-1] if values else fallback)
        if len(values) != layer_idx + 1:
            raise ValueError(f"HYV4 MTP {attr} must extend exactly one layer")
        setattr(result, attr, values)
    # indexer_types remains a backbone-only list. The current attention owner
    # builds a full indexer for layer_idx >= num_hidden_layers.
    return result


def _remap_mtp_quant_exclusions(quant_config, mtp_start_layer_idx, num_mtp_layers):
    """Preserve the selected quantization owner and translate MTP exclusions."""
    if quant_config is None:
        return None
    result = copy.copy(quant_config)
    for attr in ("ignore", "ignored_layers", "exclude_modules"):
        patterns = getattr(quant_config, attr, None)
        if not patterns:
            continue
        extra = []
        for offset in range(num_mtp_layers):
            source = f"model.mtp_layers.{offset}."
            destination = f"model.layers.{mtp_start_layer_idx + offset}."
            extra.extend(
                destination + pattern[len(source):]
                for pattern in patterns if pattern.startswith(source)
            )
        setattr(result, attr, list(patterns) + extra)
    return result


def _create_mtp_quant_config(
    hf_config: PretrainedConfig,
    backbone_quant_config: QuantizationConfig | None = None,
) -> QuantizationConfig | None:
    algo = getattr(hf_config, "mtp_quant_algo", None)
    algo = algo.upper() if algo is not None else "NONE"
    if algo in ("BF16", "FP16"):
        return None
    if algo == "NONE":
        return backbone_quant_config
    if algo != "FP8":
        raise ValueError(f"Unsupported HYV4 mtp_quant_algo: {algo}")
    # Channel FP8 is owned by the checkpoint's current quantization config;
    # constructing a native Fp8Config here would silently change its scheme.
    if backbone_quant_config is not None:
        return backbone_quant_config

    from vllm.model_executor.layers.quantization.fp8 import Fp8Config

    quant = getattr(hf_config, "quantization_config", None) or {}
    result = Fp8Config(
        is_checkpoint_fp8_serialized=True,
        activation_scheme=quant.get("activation_scheme", "dynamic"),
        ignored_layers=(quant.get("ignored_layers")
                        or quant.get("modules_to_not_convert") or []),
        weight_block_size=quant.get("weight_block_size"),
    )
    if quant.get("scale_fmt") == "ue8m0":
        result.is_scale_e8m0 = True
    return result


def _prepare_mtp_fp8_expert_scale(
    quant_config: QuantizationConfig | None,
    name: str,
    loaded_weight: torch.Tensor,
) -> tuple[str, torch.Tensor]:
    """Decode the checkpoint's legacy raw block FP8 expert scale spelling."""
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config

    if (
        isinstance(quant_config, Fp8Config)
        and quant_config.weight_block_size is not None
        and ".mlp.experts." in name
        and name.endswith((".scale", "_scale", "_scale_inv"))
    ):
        if name.endswith(".scale"):
            name = name[:-len(".scale")] + ".weight_scale_inv"
        if (getattr(quant_config, "is_scale_e8m0", False)
                and loaded_weight.dtype == torch.uint8):
            loaded_weight = loaded_weight.view(torch.float8_e8m0fnu)
    return name, loaded_weight


def _normalize_mtp_fused_expert_name(
    name: str, params_dict: dict[str, nn.Parameter],
) -> str:
    """Translate serialized fused suffixes to the current expert parameter ABI."""
    for projection, tag in (
        ("gate_up_proj", "w13_weight"), ("down_proj", "w2_weight"),
    ):
        marker = ".experts." + projection
        if marker not in name:
            continue
        base, suffix = name.split(marker, 1)
        suffixes = {
            "": ("",), ".weight": ("",),
            "_scale": ("_scale", "_scale_inv"),
            ".weight_scale": ("_scale", "_scale_inv"),
            "_scale_inv": ("_scale_inv", "_scale"),
            ".weight_scale_inv": ("_scale_inv", "_scale"),
        }.get(suffix, (suffix,))
        for candidate in suffixes:
            if any(
                param.startswith(base + ".experts.")
                and param.endswith("." + tag + candidate)
                for param in params_dict
            ):
                return base + marker + candidate
    return name


def _rewrite_mtp_weight_name(name: str, mtp_start: int) -> str | None:
    shared = {
        "model.embed_tokens.weight": "model.embed_tokens.weight",
        "lm_head.weight": f"model.layers.{mtp_start}.shared_head.head.weight",
    }
    if name in shared:
        return shared[name]
    if name.startswith("model.mtp_layers."):
        parts = name.split(".", 3)
        if len(parts) != 4 or parts[2] != "0":
            raise ValueError(f"HYV4 requires exactly one checkpoint MTP layer: {name}")
        suffix = parts[3]
    elif name.startswith("model.layers."):
        parts = name.split(".", 3)
        if len(parts) != 4 or not parts[2].isdigit():
            raise ValueError(f"Malformed HYV4 checkpoint layer: {name}")
        layer = int(parts[2])
        if layer < mtp_start:
            return None
        if layer != mtp_start:
            raise ValueError(f"HYV4 requires exactly one checkpoint MTP layer: {name}")
        suffix = parts[3]
    else:
        return None
    if suffix.split(".", 1)[0] not in (
        "enorm", "hnorm", "eh_proj", "final_layernorm", "shared_head", "embed_tokens",
    ):
        suffix = "mtp_block." + suffix
    return _rewrite_hyv4_weight_name(f"model.layers.{mtp_start}.{suffix}")


class HYV4SharedHead(nn.Module):
    """Keep the checkpoint head until the current Eagle loader shares it."""

    def __init__(
        self, config: PretrainedConfig,
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__()
        if _is_modelopt_layer_excluded(quant_config, "lm_head"):
            quant_config = None
        self.head = ParallelLMHead(
            config.vocab_size, config.hidden_size,
            quant_config=quant_config, prefix="lm_head",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states


class HYV4MultiTokenPredictorLayer(nn.Module):
    """One checkpoint-native normalization, fusion and decoder block."""

    def __init__(
        self,
        config: PretrainedConfig,
        prefix: str,
        vllm_config: VllmConfig,
        model_config: ModelConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        topk_indices_buffer: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
        self.shared_head = HYV4SharedHead(config, quant_config)
        mtp_config = _make_mtp_layer_config(config, int(prefix.rsplit(".", 1)[-1]))
        self.mtp_block = HYV4DecoderLayer(
            config=mtp_config, vllm_config=vllm_config, cache_config=cache_config,
            quant_config=quant_config, prefix=prefix,
            topk_indices_buffer=topk_indices_buffer,
        )
        self.final_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self, input_ids: torch.Tensor, positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert inputs_embeds is not None
        hidden_states = self.eh_proj(torch.cat(
            [self.enorm(inputs_embeds), self.hnorm(previous_hidden_states)], dim=-1,
        ))
        hidden_states, residual = self.mtp_block(
            positions=positions, hidden_states=hidden_states, residual=None,
        )
        hidden_states, _ = self.final_layernorm(hidden_states, residual)
        return hidden_states


class HYV4MultiTokenPredictor(nn.Module, MixtureOfExperts):
    """Own one reusable draft block and the current shared-buffer boundary."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        speculative = vllm_config.speculative_config
        if speculative is None or speculative.draft_model_config is None:
            raise RuntimeError("HYV4MTP requires a speculative draft model config")
        draft_config = speculative.draft_model_config
        config = _normalize_hyv4_config(copy.deepcopy(draft_config.hf_config))
        if getattr(config, "num_nextn_predict_layers", 1) != 1:
            raise ValueError("HYV4 MTP requires exactly one checkpoint draft layer")
        self.config = config
        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = 1
        self.quant_config = _remap_mtp_quant_exclusions(
            _create_mtp_quant_config(config, vllm_config.quant_config),
            self.mtp_start_layer_idx, 1,
        )
        if self.quant_config is not None:
            self.quant_config.packed_modules_mapping = HYV4MTP.packed_modules_mapping
        self._topk_indices_buffer = None
        if hasattr(config, "index_topk"):
            self._topk_indices_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens, config.index_topk,
                dtype=torch.int32, device=current_platform.device_type,
            )
        # Keep the embedding canonical in named_parameters when the head is
        # tied, matching the target's strict checkpoint accounting.
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleDict({
            str(self.mtp_start_layer_idx): HYV4MultiTokenPredictorLayer(
                config, f"{prefix}.layers.{self.mtp_start_layer_idx}",
                vllm_config=vllm_config, model_config=draft_config,
                cache_config=vllm_config.cache_config, quant_config=self.quant_config,
                topk_indices_buffer=self.topk_indices_buffer,
            ),
        })
        if config.tie_word_embeddings:
            self.layers[str(self.mtp_start_layer_idx)].shared_head.head.weight = (
                self.embed_tokens.weight
            )
        self.requires_topk_indices_buffer = any(
            layer.mtp_block.self_attn.is_sparse for layer in self.layers.values()
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.spec_step_idx = 0
        self.expert_weights = []
        self.num_expert_groups = 1
        self.moe_layers = [
            layer.mtp_block.mlp.experts for layer in self.layers.values()
            if layer.mtp_block.block_type == "moe"
        ]
        self.num_moe_layers = len(self.moe_layers)
        moe = next((
            layer.mtp_block.mlp for layer in self.layers.values()
            if layer.mtp_block.block_type == "moe"
        ), None)
        for public, internal in (
            ("num_logical_experts", "n_logical_experts"),
            ("num_physical_experts", "n_physical_experts"),
            ("num_local_physical_experts", "n_local_physical_experts"),
            ("num_routed_experts", "n_routed_experts"),
            ("num_redundant_experts", "n_redundant_experts"),
        ):
            setattr(self, public, getattr(moe, internal, 0))
        self.num_shared_experts = config.num_shared_experts if moe is not None else 0

    @property
    def topk_indices_buffer(self) -> torch.Tensor | None:
        return self._topk_indices_buffer

    @topk_indices_buffer.setter
    def topk_indices_buffer(self, buffer: torch.Tensor | None) -> None:
        # load_eagle_model assigns this attribute before graph capture. Its
        # named_modules walk cannot reach non-Module attention implementations.
        self._topk_indices_buffer = buffer
        for layer in self.layers.values():
            attn = layer.mtp_block.self_attn
            if not attn.is_sparse:
                continue
            attn.topk_indices_buffer = buffer
            attn.indexer.topk_indices_buffer = buffer
            attn.indexer.indexer_op.topk_indices_buffer = buffer
            attn.mla_attn.impl.topk_indices_buffer = buffer

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def set_skip_topk(self, skip: bool) -> None:
        """Model-local control; the current speculator does not invoke it."""
        for layer in self.layers.values():
            layer.mtp_block.self_attn.skip_topk = skip

    def compact_topk_indices(self, slot_ids: torch.Tensor) -> None:
        """Reorder selected rows without replacing shared graph storage."""
        if self.topk_indices_buffer is None or slot_ids.numel() == 0:
            return
        self.topk_indices_buffer[:slot_ids.numel()].copy_(
            self.topk_indices_buffer[slot_ids]
        )

    def forward(
        self, input_ids: torch.Tensor, positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)
        inputs_embeds = torch.where((positions == 0).unsqueeze(-1), 0, inputs_embeds)
        return self.layers[str(self.mtp_start_layer_idx)](
            input_ids, positions, previous_hidden_states, inputs_embeds,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        shared_head = self.layers[str(self.mtp_start_layer_idx)].shared_head
        return self.logits_processor(shared_head.head, shared_head(hidden_states))

    set_eplb_state = HYV4Model.set_eplb_state
    update_physical_experts_metadata = HYV4Model.update_physical_experts_metadata


class HYV4MTP(nn.Module, MixtureOfExperts, SupportsPP):
    """Native HYV4 draft, loaded and sampled through the pinned owners."""

    packed_modules_mapping = {
        "gate_up_proj": ["gate_proj", "up_proj"],
        "wk_weights_proj": ["wk", "weights_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.model = HYV4MultiTokenPredictor(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"),
        )
        self.config = self.model.config
        self.quant_config = self.model.quant_config
        self.sampler = Sampler()
        for name in (
            "expert_weights", "num_moe_layers", "num_expert_groups",
            "num_logical_experts", "num_physical_experts", "num_local_physical_experts",
            "num_routed_experts", "num_shared_experts", "num_redundant_experts", "moe_layers",
        ):
            setattr(self, name, getattr(self.model, name))

    set_eplb_state = HYV4Model.set_eplb_state
    update_physical_experts_metadata = HYV4Model.update_physical_experts_metadata
    get_expert_mapping = HYV4Model.get_expert_mapping
    load_fused_expert_weights = HYV4Model.load_fused_expert_weights

    def set_topk_indices_buffer(self, topk_indices_buffer: torch.Tensor) -> None:
        self.model.topk_indices_buffer = topk_indices_buffer

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self, input_ids: torch.Tensor, positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None, spec_step_idx: int = 0,
    ) -> torch.Tensor:
        if (self.model.requires_topk_indices_buffer
                and self.model.topk_indices_buffer is None):
            raise RuntimeError("HYV4 sparse MTP requires a top-k indices buffer")
        self.model.spec_step_idx = spec_step_idx
        return self.model(input_ids, positions, hidden_states, inputs_embeds)

    def compute_logits(
        self, hidden_states: torch.Tensor, spec_step_idx: int = 0,
    ) -> torch.Tensor | None:
        return self.model.compute_logits(hidden_states)

    def sample(
        self, logits: torch.Tensor, sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput:
        return self.sampler(logits, sampling_metadata)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters())
        tied_head_name = f"model.layers.{self.config.num_hidden_layers}.shared_head.head.weight"

        def draft_weights():
            for name, value in weights:
                name = _rewrite_mtp_weight_name(name, self.config.num_hidden_layers)
                if name is None:
                    continue
                # Like the target, a tied checkpoint loads the canonical
                # embedding and ignores optional serialized head aliases.
                if self.config.tie_word_embeddings and name == tied_head_name:
                    continue
                name, value = _prepare_mtp_fp8_expert_scale(self.quant_config, name, value)
                name = _normalize_mtp_fused_expert_name(name, params_dict)
                yield name, value

        # Use the same local FP8 pairing, duplicate/shard ledger, PP missing
        # parameter handling, and current expert loaders as the target.
        mapped_weights = draft_weights()
        if self.quant_config is not None:
            mapper = self.quant_config.get_cache_scale_mapper()
            mapped_weights = mapper.apply(mapped_weights)
        return HYV4Model.load_weights(self, mapped_weights)


__all__ = [
    "HYV4MTP", "HYV4SharedHead", "HYV4MultiTokenPredictor", "HYV4MultiTokenPredictorLayer",
]
