# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

"""Native multi-token prediction head for HY V4 on HCU."""

import copy
import typing
from collections.abc import Callable, Iterable

import torch
from torch import nn
from transformers import PretrainedConfig

from vllm.config import CacheConfig, ModelConfig, VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import fused_moe_make_expert_params_mapping
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.kv_cache import KVCacheScaleParameter
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.utils import maybe_prefix
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.utils import (
    get_pp_missing_layer_names,
    is_pp_missing_parameter,
)
from vllm.sequence import IntermediateTensors
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler

from .model import (
    HYV4DecoderLayer,
    _normalize_hyv4_config,
    _slice_sink_for_tp,
    _try_load_fp8_indexer_projection,
    _try_load_fp8_router_gate,
)

logger = init_logger(__name__)


_MTP_QUANT_EXCLUSION_ATTRS = ("ignored_layers", "exclude_modules")
_MTP_WRAPPER_WEIGHTS = ("enorm", "hnorm", "eh_proj", "final_layernorm")


def _extend_layer_types(
    layer_types: list[str] | None,
    layer_idx: int,
    fallback: str,
) -> list[str] | None:
    if layer_types is None or len(layer_types) > layer_idx:
        return layer_types
    fill = layer_types[-1] if layer_types else fallback
    return list(layer_types) + [fill] * (layer_idx + 1 - len(layer_types))


def _make_mtp_layer_config(config, layer_idx: int):
    """Copy a backbone config into the checkpoint-compatible MTP layout."""
    mtp_config = copy.deepcopy(config)
    mtp_config.enable_ihc = False
    mtp_config.layer_types = _extend_layer_types(
        getattr(mtp_config, "layer_types", None),
        layer_idx,
        "full_attention",
    )
    mtp_config.mlp_layer_types = _extend_layer_types(
        getattr(mtp_config, "mlp_layer_types", None),
        layer_idx,
        "sparse",
    )
    return mtp_config


def _remap_mtp_quant_exclusions(
    quant_config,
    mtp_start_layer_idx: int,
    num_mtp_layers: int,
):
    """Copy and translate checkpoint-side MTP quantization exclusions."""
    if quant_config is None:
        return None

    exclusion_attr = None
    patterns = None
    for attr in _MTP_QUANT_EXCLUSION_ATTRS:
        value = getattr(quant_config, attr, None)
        if value:
            exclusion_attr = attr
            patterns = list(value)
            break
    if exclusion_attr is None or not patterns:
        return quant_config

    extra: list[str] = []
    for offset in range(num_mtp_layers):
        source_prefix = f"model.mtp_layers.{offset}."
        draft_prefix = f"model.layers.{mtp_start_layer_idx + offset}."
        for pattern in patterns:
            if pattern.startswith(source_prefix):
                extra.append(draft_prefix + pattern[len(source_prefix) :])
                continue
            stripped = pattern.rstrip("*")
            if stripped.startswith(source_prefix):
                extra.append(
                    draft_prefix
                    + stripped[len(source_prefix) :]
                    + pattern[len(stripped) :]
                )
    if not extra:
        return quant_config

    result = copy.copy(quant_config)
    setattr(result, exclusion_attr, patterns + extra)
    return result


def _create_mtp_quant_config(
    hf_config: PretrainedConfig,
    backbone_quant_config: QuantizationConfig | None = None,
) -> QuantizationConfig | None:
    """Select the checkpoint-declared MTP quantization or inherit target."""
    mtp_quant_algo = getattr(hf_config, "mtp_quant_algo", None)
    if mtp_quant_algo is None or mtp_quant_algo.upper() == "NONE":
        return backbone_quant_config
    mtp_quant_algo = mtp_quant_algo.upper()
    if mtp_quant_algo in ("BF16", "FP16"):
        return None
    if mtp_quant_algo != "FP8":
        logger.warning(
            "Unknown HYV4 mtp_quant_algo=%s; using target quant config",
            mtp_quant_algo,
        )
        return backbone_quant_config

    from vllm.model_executor.layers.quantization.fp8 import Fp8Config

    hf_quant_config = getattr(hf_config, "quantization_config", None) or {}
    weight_block_size = hf_quant_config.get("weight_block_size")
    activation_scheme = hf_quant_config.get("activation_scheme", "dynamic")
    if weight_block_size is not None:
        activation_scheme = "dynamic"
    ignored_layers = hf_quant_config.get("ignored_layers") or hf_quant_config.get(
        "modules_to_not_convert"
    )
    result = Fp8Config(
        is_checkpoint_fp8_serialized=True,
        activation_scheme=activation_scheme,
        ignored_layers=ignored_layers or [],
        weight_block_size=weight_block_size,
    )
    if hf_quant_config.get("scale_fmt") == "ue8m0":
        result.is_scale_e8m0 = True
    return result


def _prepare_mtp_fp8_expert_scale(
    quant_config: QuantizationConfig | None,
    name: str,
    loaded_weight: torch.Tensor,
) -> tuple[str, torch.Tensor]:
    """Normalize legacy block-wise FP8 expert scale names and dtypes."""
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config

    if (
        isinstance(quant_config, Fp8Config)
        and quant_config.weight_block_size is not None
        and ".mlp.experts." in name
        and name.endswith(".scale")
    ):
        name = name[: -len(".scale")] + ".weight_scale_inv"
        if (
            getattr(quant_config, "is_scale_e8m0", False)
            and loaded_weight.dtype == torch.uint8
        ):
            loaded_weight = loaded_weight.view(torch.float8_e8m0fnu)
    return name, loaded_weight


def _rewrite_mtp_weight_name(name: str, mtp_start_layer_idx: int) -> str | None:
    """Rewrite a checkpoint MTP name to the draft module's parameter name."""
    spec_layer = None
    if name.startswith("model.mtp_layers."):
        parts = name.split(".")
        if len(parts) <= 3 or not parts[2].isdigit():
            return None
        spec_layer = mtp_start_layer_idx + int(parts[2])
        name = name.replace(
            f"model.mtp_layers.{parts[2]}.",
            f"model.layers.{spec_layer}.",
            1,
        )
    elif name.startswith("model.layers."):
        parts = name.split(".")
        if len(parts) <= 3 or not parts[2].isdigit():
            return None
        spec_layer = int(parts[2])
        if spec_layer < mtp_start_layer_idx:
            return None
    else:
        return None

    layer_prefix = f"model.layers.{spec_layer}."
    suffix = name[len(layer_prefix) :]
    if suffix.startswith(("embed_tokens", "shared_head")):
        return None
    if not suffix.startswith(_MTP_WRAPPER_WEIGHTS):
        name = layer_prefix + "mtp_block." + suffix
    return name.replace("gate.e_score_correction_bias", "expert_bias")


def _resolve_fused_expert_param(
    param_base: str,
    checkpoint_suffix: str,
    params_dict: dict,
) -> str | None:
    """Resolve a fused expert weight or scale without dropping its suffix."""
    if not checkpoint_suffix:
        return param_base if param_base in params_dict else None
    suffix = checkpoint_suffix.lstrip(".")
    direct = param_base + suffix if suffix.startswith("_") else f"{param_base}_{suffix}"
    for candidate in (direct, f"{param_base}_scale_inv", f"{param_base}_scale"):
        if candidate in params_dict:
            return candidate
    return None


class HYV4SharedHead(nn.Module):
    """Own the MTP LM-head reference replaced by the Eagle loader."""

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__()
        # ModelOpt's excluded linear method stores the weight as
        # [hidden, vocab], while ParallelLMHead's loader requires the
        # embedding layout [vocab, hidden]. Match the target HYV4 head and
        # let ParallelLMHead choose its unquantized embedding method.
        head_quant_config = quant_config
        if quant_config is not None and quant_config.is_layer_excluded("lm_head"):
            head_quant_config = None
        self.head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=head_quant_config,
            prefix="lm_head",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states


class HYV4MultiTokenPredictorLayer(nn.Module):
    """One checkpoint-native HYV4 MTP block."""

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
        del model_config
        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = nn.Linear(
            config.hidden_size * 2,
            config.hidden_size,
            bias=False,
        )
        self.shared_head = HYV4SharedHead(config, quant_config)

        layer_idx = int(prefix.rsplit(".", 1)[-1])
        mtp_config = _make_mtp_layer_config(config, layer_idx)
        self.mtp_block = HYV4DecoderLayer(
            config=mtp_config,
            vllm_config=vllm_config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=prefix,
            topk_indices_buffer=topk_indices_buffer,
        )
        self.final_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del input_ids
        assert inputs_embeds is not None
        inputs_embeds = self.enorm(inputs_embeds)
        previous_hidden_states = self.hnorm(previous_hidden_states)
        hidden_states = self.eh_proj(
            torch.cat([inputs_embeds, previous_hidden_states], dim=-1)
        )
        hidden_states, residual = self.mtp_block(
            positions=positions,
            hidden_states=hidden_states,
            residual=None,
        )
        hidden_states, _ = self.final_layernorm(hidden_states, residual)
        return hidden_states


class HYV4MultiTokenPredictor(nn.Module):
    """Own the reusable MTP block, embedding, and logits path."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        target_config = vllm_config.model_config.hf_config
        speculative_config = vllm_config.speculative_config
        if speculative_config is None or speculative_config.draft_model_config is None:
            raise RuntimeError("HYV4MTP requires a speculative draft model config")
        draft_model_config = speculative_config.draft_model_config
        config = _normalize_hyv4_config(draft_model_config.hf_config)

        self.mtp_start_layer_idx = target_config.num_hidden_layers
        self.num_mtp_layers = max(
            getattr(target_config, "num_nextn_predict_layers", 1),
            1,
        )
        self.quant_config = _remap_mtp_quant_exclusions(
            _create_mtp_quant_config(config, vllm_config.quant_config),
            self.mtp_start_layer_idx,
            self.num_mtp_layers,
        )
        if self.quant_config is not None:
            self.quant_config.packed_modules_mapping = HYV4MTP.packed_modules_mapping

        self.topk_indices_buffer: torch.Tensor | None = None
        if hasattr(config, "index_topk"):
            from vllm.platforms import current_platform

            self.topk_indices_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                config.index_topk,
                dtype=torch.int32,
                device=current_platform.device_type,
            )

        self.layers = nn.ModuleDict(
            {
                str(idx): HYV4MultiTokenPredictorLayer(
                    config,
                    f"{prefix}.layers.{idx}",
                    vllm_config=vllm_config,
                    model_config=draft_model_config,
                    cache_config=vllm_config.cache_config,
                    quant_config=self.quant_config,
                    topk_indices_buffer=self.topk_indices_buffer,
                )
                for idx in range(
                    self.mtp_start_layer_idx,
                    self.mtp_start_layer_idx + self.num_mtp_layers,
                )
            }
        )
        self.requires_topk_indices_buffer = any(
            layer.mtp_block.self_attn.is_sparse for layer in self.layers.values()
        )
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.spec_step_idx = 0

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def set_skip_topk(self, skip: bool) -> None:
        for layer in self.layers.values():
            layer.mtp_block.self_attn.skip_topk = skip

    def compact_topk_indices(self, slot_ids: torch.Tensor) -> None:
        num_slots = slot_ids.numel()
        if self.topk_indices_buffer is None or num_slots == 0:
            return
        self.topk_indices_buffer[:num_slots] = self.topk_indices_buffer[slot_ids]

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        inputs_embeds = torch.where(
            (positions == 0).unsqueeze(-1),
            0,
            inputs_embeds,
        )
        current_step_idx = self.spec_step_idx % self.num_mtp_layers
        layer = self.layers[str(self.mtp_start_layer_idx + current_step_idx)]
        return layer(
            input_ids,
            positions,
            previous_hidden_states,
            inputs_embeds,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        current_step_idx = self.spec_step_idx % self.num_mtp_layers
        layer = self.layers[str(self.mtp_start_layer_idx + current_step_idx)]
        lm_head = layer.shared_head.head
        projection_input = layer.shared_head(hidden_states)
        if projection_input.dtype != lm_head.weight.dtype:
            projection_input = projection_input.to(lm_head.weight.dtype)
        return self.logits_processor(lm_head, projection_input)


class HYV4MTP(nn.Module):
    """HY V4 native MTP draft model."""

    packed_modules_mapping = {
        "gate_up_proj": ["gate_proj", "up_proj"],
        "wk_weights_proj": ["wk", "weights_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.model = HYV4MultiTokenPredictor(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )
        self.quant_config = self.model.quant_config
        self.sampler = Sampler()

    def set_topk_indices_buffer(self, topk_indices_buffer) -> None:
        """Share one target-produced top-k buffer with all draft consumers."""
        self.model.topk_indices_buffer = topk_indices_buffer
        layers = self.model.layers.values()
        for layer in layers:
            self_attn = layer.mtp_block.self_attn
            if not self_attn.is_sparse:
                continue
            self_attn.topk_indices_buffer = topk_indices_buffer
            indexer = self_attn.indexer
            if indexer is None:
                raise RuntimeError("Sparse HYV4 MTP attention requires an indexer")
            indexer.topk_indices_buffer = topk_indices_buffer
            indexer.indexer_op.topk_indices_buffer = topk_indices_buffer
            attention_impl = self_attn.mla_attn.impl
            if not hasattr(attention_impl, "topk_indices_buffer"):
                raise RuntimeError(
                    "Sparse HYV4 MTP attention backend requires a top-k buffer"
                )
            attention_impl.topk_indices_buffer = topk_indices_buffer

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        del intermediate_tensors
        if (
            self.model.requires_topk_indices_buffer
            and self.model.topk_indices_buffer is None
        ):
            raise RuntimeError(
                "HYV4 sparse MTP requires the target model's top-k indices buffer"
            )
        self.model.spec_step_idx = spec_step_idx
        return self.model(
            input_ids,
            positions,
            hidden_states,
            inputs_embeds,
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor | None:
        self.model.spec_step_idx = spec_step_idx
        return self.model.compute_logits(hidden_states)

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput | None:
        return self.sampler(logits, sampling_metadata)

    def _load_fused_expert_weights(
        self,
        name: str,
        params_dict: dict[str, nn.Parameter],
        loaded_weight: torch.Tensor,
        shard_id: str,
        num_experts: int,
    ) -> bool:
        if name not in params_dict:
            return False
        param = params_dict[name]
        weight_loader = typing.cast(Callable[..., bool], param.weight_loader)
        loaded_local_expert = False
        for expert_id in range(num_experts):
            success = weight_loader(
                param,
                loaded_weight[expert_id],
                name,
                shard_id,
                expert_id,
                return_success=True,
            )
            if success:
                loaded_local_expert = True
        return loaded_local_expert

    def _load_expert_weight(
        self,
        name: str,
        loaded_weight: torch.Tensor,
        params_dict: dict[str, nn.Parameter],
        loaded_params: set[str],
        split_expert_params_mapping: list[tuple[str, str, int, str]],
        fused_expert_param_names: dict[tuple[str, str], str],
        num_experts: int,
    ) -> bool:
        base = name.split(".experts.")[0]
        for checkpoint_projection, parameter_tag in (
            (".experts.gate_up_proj", "w13_weight"),
            (".experts.down_proj", "w2_weight"),
        ):
            if checkpoint_projection not in name:
                continue
            parameter_base = fused_expert_param_names.get((base, parameter_tag))
            if parameter_base is None:
                return False
            target = _resolve_fused_expert_param(
                parameter_base,
                name.split(checkpoint_projection, 1)[1],
                params_dict,
            )
            if target is None:
                return False
            if parameter_tag == "w13_weight":
                gate_weight, up_weight = loaded_weight.chunk(2, dim=-2)
                loaded = self._load_fused_expert_weights(
                    target,
                    params_dict,
                    gate_weight,
                    "w1",
                    num_experts,
                ) and self._load_fused_expert_weights(
                    target,
                    params_dict,
                    up_weight,
                    "w3",
                    num_experts,
                )
            else:
                loaded = self._load_fused_expert_weights(
                    target,
                    params_dict,
                    loaded_weight,
                    "w2",
                    num_experts,
                )
            if loaded:
                loaded_params.add(target)
            return True

        consumed = False
        for param_name, weight_name, expert_id, shard_id in (
            split_expert_params_mapping
        ):
            if weight_name not in name:
                continue
            consumed = True
            mapped_name = name.replace(weight_name, param_name)
            if mapped_name not in params_dict:
                continue
            param = params_dict[mapped_name]
            weight_loader = typing.cast(Callable[..., bool], param.weight_loader)
            if weight_loader(
                param,
                loaded_weight,
                mapped_name,
                shard_id=shard_id,
                expert_id=expert_id,
                return_success=True,
            ):
                loaded_params.add(mapped_name)
        return consumed

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        params_dict = dict(self.named_parameters())
        pp_missing_layer_names = get_pp_missing_layer_names(self)
        loaded_params: set[str] = set()
        mtp_start = self.config.num_hidden_layers
        shared_weights = {
            "model.embed_tokens.weight": "model.embed_tokens.weight",
            "lm_head.weight": f"model.layers.{mtp_start}.shared_head.head.weight",
        }

        num_experts = getattr(self.config, "n_routed_experts", 0)
        split_expert_params_mapping = fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=num_experts,
        )
        fused_expert_param_names: dict[tuple[str, str], str] = {}
        for param_name in params_dict:
            for tag in ("w13_weight", "w2_weight"):
                if param_name.endswith(tag) and ".experts." in param_name:
                    base = param_name.split(".experts.")[0]
                    fused_expert_param_names[base, tag] = param_name

        stacked_mapping = [
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        indexer_stacked_mapping = [
            (".wk_weights_proj", ".wk", 0),
            (".wk_weights_proj", ".weights_proj", 1),
        ]
        pending_indexer_fp8: dict[str, dict[str, torch.Tensor]] = {}
        pending_router_fp8: dict[str, dict[str, torch.Tensor]] = {}
        sink_tp_size = get_tensor_model_parallel_world_size()
        sink_tp_rank = get_tensor_model_parallel_rank()

        for checkpoint_name, loaded_weight in weights:
            if checkpoint_name in shared_weights:
                target_name = shared_weights[checkpoint_name]
                if target_name in params_dict:
                    param = params_dict[target_name]
                    weight_loader = getattr(
                        param,
                        "weight_loader",
                        default_weight_loader,
                    )
                    weight_loader(param, loaded_weight)
                    loaded_params.add(target_name)
                continue

            name = _rewrite_mtp_weight_name(checkpoint_name, mtp_start)
            if name is None:
                continue
            name, loaded_weight = _prepare_mtp_fp8_expert_scale(
                self.quant_config,
                name,
                loaded_weight,
            )
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

            is_loaded = False
            for param_name, weight_name, shard_id in stacked_mapping:
                if weight_name not in name or ".experts." in name:
                    continue
                mapped_name = name.replace(weight_name, param_name)
                if is_pp_missing_parameter(mapped_name, self):
                    is_loaded = True
                    break
                if mapped_name not in params_dict:
                    break
                param = params_dict[mapped_name]
                param.weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(mapped_name)
                is_loaded = True
                break
            if is_loaded:
                continue

            for param_name, weight_name, shard_id in indexer_stacked_mapping:
                if weight_name not in name or "wk_weights" in name:
                    continue
                mapped_name = name.replace(weight_name, param_name)
                if is_pp_missing_parameter(mapped_name, self):
                    is_loaded = True
                    break
                if mapped_name not in params_dict:
                    break
                param = params_dict[mapped_name]
                param.weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(mapped_name)
                is_loaded = True
                break
            if is_loaded:
                continue

            if ".experts." in name and self._load_expert_weight(
                name,
                loaded_weight,
                params_dict,
                loaded_params,
                split_expert_params_mapping,
                fused_expert_param_names,
                num_experts,
            ):
                continue

            if "learnable_sink_param" in name:
                if is_pp_missing_parameter(name, self):
                    continue
                if name not in params_dict:
                    raise RuntimeError(f"Unknown HYV4 MTP sink parameter: {name}")
                local_sink = _slice_sink_for_tp(
                    loaded_weight,
                    num_heads=self.config.num_attention_heads,
                    tp_size=sink_tp_size,
                    tp_rank=sink_tp_rank,
                )
                param = params_dict[name]
                if tuple(param.shape) != tuple(local_sink.shape):
                    raise ValueError(
                        f"HYV4 MTP sink shape mismatch for {name}: "
                        f"parameter={tuple(param.shape)}, "
                        f"checkpoint={tuple(local_sink.shape)}"
                    )
                with torch.no_grad():
                    param.copy_(local_sink)
                loaded_params.add(name)
                continue

            remapped_name = maybe_remap_kv_scale_name(name, params_dict)
            if remapped_name is None:
                continue
            name = remapped_name
            if is_pp_missing_parameter(name, self):
                continue
            if name not in params_dict:
                raise RuntimeError(f"Unknown HYV4 MTP checkpoint weight: {name}")
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)

        if pending_indexer_fp8:
            missing_parts = {
                key: sorted({"weight", "scale"} - set(parts))
                for key, parts in pending_indexer_fp8.items()
            }
            raise RuntimeError(
                f"Incomplete HYV4 MTP FP8 indexer pairs: {missing_parts}"
            )
        if pending_router_fp8:
            missing_parts = {
                key: sorted({"weight", "scale"} - set(parts))
                for key, parts in pending_router_fp8.items()
            }
            raise RuntimeError(
                f"Incomplete HYV4 MTP FP8 router pairs: {missing_parts}"
            )

        required_params = {
            name
            for name, param in params_dict.items()
            if not isinstance(param, KVCacheScaleParameter)
        }
        missing_params = required_params - loaded_params
        if missing_params:
            raise RuntimeError(
                "Missing HYV4 MTP checkpoint parameters: "
                f"{sorted(missing_params)}"
            )
        logger.info_once("HYV4 MTP draft loaded: %d parameters", len(loaded_params))
        return loaded_params


__all__ = [
    "HYV4MTP",
    "HYV4MultiTokenPredictor",
    "HYV4MultiTokenPredictorLayer",
]
