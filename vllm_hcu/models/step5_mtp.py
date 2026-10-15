# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Multi-token-prediction draft model for Step-5 with MoE support.

Adapted from water18-0910 implementation for vLLM 0.28 compatibility.
"""

from collections.abc import Iterable
from copy import copy

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import MixtureOfExperts
from vllm.model_executor.models.utils import (
    get_spec_layer_idx_from_weight_name,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors

from .step5_text import Step5DecoderLayer

logger = init_logger("vllm.step5.step5_mtp")


# Packed modules mapping for Step5
STEP5_PACKED_MODULES_MAPPING = {
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    "qkvg_proj": ["q_proj", "k_proj", "v_proj", "g_proj"],
    "qkv_indexer_proj": [
        "q_proj",
        "k_proj",
        "v_proj",
        "sparse_indexer_q",
        "sparse_indexer_k",
        "sparse_indexer_z",
        "g_proj",
    ],
    "gate_up_proj": ["gate_proj", "up_proj"],
}


def _get_mtp_config(vllm_config: VllmConfig) -> PretrainedConfig:
    """Extract the HF text config for MTP."""
    hf_config = vllm_config.model_config.hf_config
    # For Step5, extract text_config if it exists
    if hasattr(hf_config, "text_config"):
        return hf_config.text_config
    return hf_config


def _fp32_gemma_rms_norm(hidden_size: int, eps: float) -> GemmaRMSNorm:
    """Create a fp32 GemmaRMSNorm for Step5 MTP."""
    return GemmaRMSNorm(hidden_size, eps)


class Step5MTPSharedHead(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        compute_dtype: torch.dtype,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.fp32_residual_connection = getattr(
            config, "fp32_residual_connection", False
        )
        self.compute_dtype = compute_dtype
        self.norm = _fp32_gemma_rms_norm(config.hidden_size, config.rms_norm_eps)
        self.head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.head",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.fp32_residual_connection:
            hidden_states = hidden_states.to(self.compute_dtype)
        return self.norm(hidden_states)


@support_torch_compile
class Step5MultiTokenPredictorLayer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
    ) -> None:
        super().__init__()
        config = _get_mtp_config(vllm_config)
        self.compute_dtype = vllm_config.model_config.dtype
        self.fp32_residual_connection = getattr(
            config, "fp32_residual_connection", False
        )
        quant_config = vllm_config.quant_config
        self.enorm = _fp32_gemma_rms_norm(config.hidden_size, config.rms_norm_eps)
        self.hnorm = _fp32_gemma_rms_norm(config.hidden_size, config.rms_norm_eps)
        self.eh_proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
        self.shared_head = Step5MTPSharedHead(
            config=config,
            compute_dtype=self.compute_dtype,
            quant_config=quant_config,
            prefix=f"{prefix}.shared_head",
        )

        # Preserve the checkpoint's absolute layer index for all per-layer
        # settings and quantization prefixes. Do not mutate the shared config.
        decoder_config = copy(vllm_config)
        decoder_config.model_config = copy(vllm_config.model_config)
        decoder_config.model_config.hf_config = config
        self.mtp_block = Step5DecoderLayer(
            decoder_config,
            prefix=f"{prefix}.mtp_block",
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_index: int = 0,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            raise ValueError("inputs_embeds is required for Step5 MTP")
        if self.fp32_residual_connection:
            inputs_embeds = inputs_embeds.to(self.compute_dtype)
            previous_hidden_states = previous_hidden_states.to(self.compute_dtype)
        inputs_embeds = self.enorm(inputs_embeds)
        previous_hidden_states = self.hnorm(previous_hidden_states)

        hidden_states = self.eh_proj(
            torch.cat([inputs_embeds, previous_hidden_states], dim=-1)
        )
        if self.fp32_residual_connection:
            hidden_states = hidden_states.float()

        hidden_states = self.mtp_block(positions=positions, hidden_states=hidden_states)
        return hidden_states


class Step5MultiTokenPredictor(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = _get_mtp_config(vllm_config)
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )
        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = getattr(config, "num_nextn_predict_layers", 3)
        if self.num_mtp_layers <= 0:
            raise ValueError(
                "Step5 MTP requires num_nextn_predict_layers > 0, got "
                f"{self.num_mtp_layers}."
            )
        # Map exact layer index from weights
        self.layers = torch.nn.ModuleDict(
            {
                str(idx): Step5MultiTokenPredictorLayer(
                    vllm_config=vllm_config,
                    prefix=f"{prefix}.layers.{idx}",
                )
                for idx in range(
                    self.mtp_start_layer_idx,
                    self.mtp_start_layer_idx + self.num_mtp_layers,
                )
            }
        )

        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.logits_processor._step5_skinny_lm_head = True

    def _layer(self, spec_step_idx: int) -> Step5MultiTokenPredictorLayer:
        idx = spec_step_idx % self.num_mtp_layers
        return self.layers[str(self.mtp_start_layer_idx + idx)]

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        return self._layer(spec_step_idx)(
            input_ids,
            positions,
            previous_hidden_states,
            inputs_embeds,
            spec_step_idx,
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        shared_head = self._layer(spec_step_idx).shared_head
        return self.logits_processor(shared_head.head, shared_head(hidden_states))

    def get_top_tokens(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        shared_head = self._layer(spec_step_idx).shared_head
        return self.logits_processor.get_top_tokens(
            shared_head.head,
            shared_head(hidden_states),
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.get_input_embeddings(input_ids)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)


class Step5MTP(nn.Module, MixtureOfExperts):
    packed_modules_mapping = STEP5_PACKED_MODULES_MAPPING
    _enable_weights_track_by_default = True
    # Each MTP layer ships an independently trained shared_head.output (it
    # differs from the target lm_head); load_eagle_model must not replace it.
    has_own_mtp_heads = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config = _get_mtp_config(vllm_config)
        self.vllm_config = vllm_config
        self.model = Step5MultiTokenPredictor(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        # Initialize MoE metadata (simplified for vLLM 0.28)
        self.num_moe_layers = 0
        self.num_expert_groups = 1
        self.num_shared_experts = 0
        self.num_logical_experts = 0
        self.num_physical_experts = 0
        self.num_local_physical_experts = 0
        self.num_routed_experts = 0
        self.num_redundant_experts = 0
        self.is_dense_mtp = True
        self.moe_layers = []

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.get_input_embeddings(input_ids)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids, positions, hidden_states, inputs_embeds, spec_step_idx
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor | None:
        return self.model.compute_logits(hidden_states, spec_step_idx)

    def get_top_tokens(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        return self.model.get_top_tokens(hidden_states, spec_step_idx)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkvg_proj", "q_proj", 0),
            ("qkvg_proj", "k_proj", 1),
            ("qkvg_proj", "v_proj", 2),
            ("qkvg_proj", "g_proj", 3),
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if name.startswith("layers."):
                name = f"model.{name}"
            spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)
            if "embed_tokens" not in name and spec_layer is None:
                continue
            name = self._rewrite_spec_layer_name(spec_layer, name)

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if "experts" in name or "moe" in name:
                    continue
                replaced_name = name.replace(weight_name, param_name)
                if replaced_name not in params_dict:
                    continue
                name = replaced_name
                if name.endswith(".bias") and name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Handle other weights
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if "tok_embeddings" in name:
                    continue

                if spec_layer is not None and ".transformer." in name:
                    name = name.replace(".transformer.", ".")
                if "shared_head" in name:
                    name = name.replace("shared_head.output", "shared_head.head")
                if "embed_tokens" in name:
                    name = "model.embed_tokens.weight"
                if name not in params_dict:
                    logger.warning_once(
                        "Skipping unexpected MTP checkpoint weight: %s", name
                    )
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)

        # Report embed_tokens as loaded
        loaded_params.update(
            name for name in params_dict if name.startswith("model.embed_tokens.")
        )
        return loaded_params

    def _rewrite_spec_layer_name(self, spec_layer: int | None, name: str) -> str:
        """
        Rewrite the weight name to match the format of the original model.
        Add .mtp_block for modules in transformer layer block for spec layer
        """
        if spec_layer is None:
            return name

        spec_layer_weight_names = [
            "embed_tokens",
            "enorm",
            "hnorm",
            "eh_proj",
            "shared_head",
        ]
        spec_layer_weight = False
        for weight_name in spec_layer_weight_names:
            if weight_name in name:
                spec_layer_weight = True
                break
        if not spec_layer_weight:
            # Treat rest weights as weights for transformer layer block
            name = name.replace(
                f"model.layers.{spec_layer}.", f"model.layers.{spec_layer}.mtp_block."
            )
        return name
