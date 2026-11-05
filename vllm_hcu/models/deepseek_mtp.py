# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# Modified by Hygon Information Technology Co., Ltd., 2026.
import typing
from collections.abc import Callable, Iterable
from copy import copy
from types import MethodType

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from vllm._aiter_ops import rocm_aiter_ops
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.model_executor.models.utils import maybe_prefix
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors

import vllm_hcu.platforms.envs as henvs
from vllm_hcu.v1.attention.lightly_cp_utils import lightly_cp_inputs_splitting

from .deepseek_v2 import (
    DeepseekV2DecoderLayer,
    DeepseekV2MixtureOfExperts,
    DeepseekV2MoE,
    _try_load_quantized_indexer_wk,
    get_spec_layer_idx_from_weight_name,
)

logger = init_logger(__name__)


def _mtp_quantization_config(vllm_config: VllmConfig, prefix: str) -> VllmConfig:
    """Match this MTP layer against canonical checkpoint quantization rules."""
    if not henvs.VLLM_HCU_GLM_MTP_QUANT_PREFIX_FIX:
        return vllm_config
    # Eagle replaces model_config with the converted draft config, whose type
    # is deepseek_mtp for GLM DSA as well as DeepSeek. Identify the original
    # target through SpeculativeConfig rather than widening the architecture
    # gate to every model sharing that draft type.
    speculative_config = getattr(vllm_config, "speculative_config", None)
    model_config = getattr(speculative_config, "target_model_config", None)
    if model_config is None:
        model_config = getattr(vllm_config, "model_config", None)
    hf_config = getattr(model_config, "hf_config", None)
    if getattr(hf_config, "model_type", None) != "glm_moe_dsa":
        return vllm_config
    quant_config = vllm_config.quant_config
    if quant_config is None:
        return vllm_config
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
        CompressedTensorsConfig,
    )

    if not isinstance(quant_config, CompressedTensorsConfig):
        return vllm_config

    from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
        find_matched_target,
    )
    from vllm.model_executor.layers.quantization.utils.config_utils import (
        find_matching_patterns,
    )

    original_match = type(quant_config).get_scheme_dict

    def get_scheme_dict(self, layer, layer_name=None):
        if layer_name is None or not layer_name.startswith(prefix + "."):
            return original_match(self, layer, layer_name)
        canonical = "model.decoder" + layer_name[len(prefix):]
        mapping = self.packed_modules_mapping
        # Resolve both spellings before validating fused shards. A complete
        # fused-name ignore takes precedence over redundant partial ignores.
        ignores = [find_matching_patterns(name, self.ignore, mapping)
                   for name in (layer_name, canonical)]
        if any(all(shards) for shards in ignores):
            return None
        ignored_shards = [any(matches) for matches in zip(*ignores)]
        if any(ignored_shards):
            if not all(ignored_shards):
                raise ValueError(
                    "Found different quantization schemes for the shards of "
                    f"{layer_name}. vLLM requires all to use the same scheme."
                )
            return None

        canonical_targets = {}
        actual_targets = {}
        for target, scheme in self.target_scheme_map.items():
            actual_matches = find_matching_patterns(layer_name, [target], mapping)
            canonical_matches = find_matching_patterns(canonical, [target], mapping)
            if any(canonical_matches) and not any(actual_matches):
                canonical_targets[target] = scheme
            if any(actual_matches) and not any(canonical_matches):
                actual_targets[target] = scheme

        def depends_on(name, targets, candidates):
            # Use the full native rule table for fusion completeness and its
            # first matching rule for each shard. A decoder rule can cooperate
            # with a generic rule, or be shadowed by an earlier generic rule.
            shards = find_matching_patterns(name, targets, mapping)
            first_matches = [
                next((target for target in targets if target in matches), None)
                for matches in shards
            ]
            return all(first_matches) and any(
                target in candidates for target in first_matches
            )

        # Configs without decoder-specific targets keep native runtime matching.
        if not canonical_targets:
            return original_match(self, layer, layer_name)

        # Matching only the actual spelling does not make a rule MTP-specific:
        # model.layers.\d+ also targets the backbone. Only rules excluding all
        # target-model backbone layers can override a canonical decoder rule.
        parent, index = prefix.rsplit(".", 1)
        num_layers = getattr(hf_config, "num_hidden_layers", None)
        if num_layers is None:
            num_layers = int(index) if index.isdecimal() else 0
        suffix = layer_name[len(prefix):]
        overrides = {
            target: scheme for target, scheme in actual_targets.items()
            if not any(
                any(find_matching_patterns(f"{parent}.{i}{suffix}", [target], mapping))
                for i in range(num_layers)
            )
        }
        # Fully explicit fused overrides retain priority over generic rules
        # matching the fused name directly, as in the existing MTP contract.
        targets = dict(overrides)
        explicit_match = find_matched_target(layer_name, layer, targets, mapping)
        if explicit_match is None:
            # A partial explicit override can cooperate with generic shard
            # rules; use the complete native table to resolve that combination.
            targets.update(self.target_scheme_map)
        if explicit_match is not None or depends_on(layer_name, targets, overrides):
            explicit = copy(self)
            explicit.target_scheme_map = targets
            return original_match(explicit, layer, layer_name)
        if not depends_on(canonical, self.target_scheme_map, canonical_targets):
            return original_match(self, layer, layer_name)
        return original_match(self, layer, canonical)

    # Keep the target model and other MTP layers' quantization rules unchanged.
    # copy(), unlike dataclasses.replace(), does not rerun engine validation.
    mtp_quant_config = copy(quant_config)
    mtp_quant_config.target_scheme_map = dict(quant_config.target_scheme_map)
    mtp_quant_config.ignore = list(quant_config.ignore)
    mtp_quant_config.get_scheme_dict = MethodType(get_scheme_dict, mtp_quant_config)
    mtp_vllm_config = copy(vllm_config)
    mtp_vllm_config.quant_config = mtp_quant_config
    logger.info_once(
        "Matching compressed-tensors MTP prefix %s as model.decoder", prefix
    )
    return mtp_vllm_config


class SharedHead(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        prefix: str,
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__()
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "head"),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.norm(hidden_states)


class DeepSeekMultiTokenPredictorLayer(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str) -> None:
        super().__init__()

        config = vllm_config.speculative_config.draft_model_config.hf_config
        self.config = config
        vllm_config = _mtp_quantization_config(vllm_config, prefix)
        quant_config = vllm_config.quant_config

        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)

        self.device = current_platform.device_type

        self.is_v32 = hasattr(config, "index_topk") and not henvs.VLLM_HCU_DISABLE_DSA
        if self.is_v32:
            topk_tokens = config.index_topk
            topk_indices_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                topk_tokens,
                dtype=torch.int32,
                device=self.device,
            )
        else:
            topk_indices_buffer = None

        self.shared_head = SharedHead(
            config=config, prefix=prefix, quant_config=quant_config
        )
        self.mtp_block = DeepseekV2DecoderLayer(
            vllm_config,
            prefix,
            config=self.config,
            topk_indices_buffer=topk_indices_buffer,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_index: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert inputs_embeds is not None
        # masking inputs at position 0, as not needed by MTP
        inputs_embeds = torch.where(positions.unsqueeze(-1) == 0, 0, inputs_embeds)
        inputs_embeds = self.enorm(inputs_embeds)
        previous_hidden_states = self.hnorm(previous_hidden_states)

        hidden_states = self.eh_proj(
            torch.cat([inputs_embeds, previous_hidden_states], dim=-1)
        )

        hidden_states, residual = self.mtp_block(
            positions=positions, hidden_states=hidden_states, residual=None
        )
        hidden_states = residual + hidden_states  # pre-final-norm (logits hidden)
        # Target vLLM's DeepSeek MTP proposer recycles the post-final-norm
        # hidden state into the next draft step, while compute_logits applies
        # shared_head to the pre-norm element.
        return hidden_states, self.shared_head(hidden_states)


class DeepSeekMultiTokenPredictor(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = config.num_nextn_predict_layers
        # to map the exact layer index from weights

        self.layers = torch.nn.ModuleDict(
            {
                str(idx): DeepSeekMultiTokenPredictorLayer(
                    vllm_config, f"{prefix}.layers.{idx}"
                )
                for idx in range(
                    self.mtp_start_layer_idx,
                    self.mtp_start_layer_idx + self.num_mtp_layers,
                )
            }
        )
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)

        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()

    def _iter_mla_attn(self) -> Iterable[object]:
        for layer in self.layers.values():
            mtp_block = getattr(layer, "mtp_block", None)
            self_attn = getattr(mtp_block, "self_attn", None)
            mla_attn = getattr(self_attn, "mla_attn", None)
            if mla_attn is not None:
                yield mla_attn

    def set_skip_topk(self, skip: bool) -> None:
        """Toggle skip_topk on all MTP layers with sparse attention."""
        for mla_attn in self._iter_mla_attn():
            if hasattr(mla_attn, "skip_topk"):
                mla_attn.skip_topk = skip

    def compact_topk_indices(self, slot_ids: torch.Tensor) -> None:
        """Compact per-token index rows for subsequent MTP draft steps."""
        num_slots = slot_ids.numel()
        for mla_attn in self._iter_mla_attn():
            if hasattr(mla_attn, "topk_indices_buffer"):
                topk_indices_buffer = mla_attn.topk_indices_buffer
                topk_indices_buffer[:num_slots] = topk_indices_buffer[slot_ids]

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        current_step_idx = spec_step_idx % self.num_mtp_layers

        enable_lightly_cp = get_forward_context().enable_lightly_cp
        if enable_lightly_cp:
            previous_hidden_states, positions, _, inputs_embeds = (
                lightly_cp_inputs_splitting(
                    previous_hidden_states,
                    positions,
                    None,
                    inputs_embeds,
                    self.tp_size,
                    self.tp_rank,
                )
            )

        last_hidden_states, hidden_states = self.layers[
            str(self.mtp_start_layer_idx + current_step_idx)
        ](
            input_ids,
            positions,
            previous_hidden_states,
            inputs_embeds,
            current_step_idx,
        )

        if enable_lightly_cp:
            last_hidden_states = tensor_model_parallel_all_gather(
                last_hidden_states.contiguous(), dim=0
            )
            hidden_states = tensor_model_parallel_all_gather(
                hidden_states.contiguous(), dim=0
            )
            gather_indexes_tensor = get_forward_context().gather_indexes_tensor
            if gather_indexes_tensor is not None:
                hidden_states = torch.index_select(
                    hidden_states, 0, gather_indexes_tensor
                )
                last_hidden_states = torch.index_select(
                    last_hidden_states, 0, gather_indexes_tensor
                )

        return last_hidden_states, hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        current_step_idx = spec_step_idx % self.num_mtp_layers
        mtp_layer = self.layers[str(self.mtp_start_layer_idx + current_step_idx)]
        logits = self.logits_processor(
            mtp_layer.shared_head.head, mtp_layer.shared_head(hidden_states)
        )
        return logits


@support_torch_compile
class DeepSeekMTP(nn.Module, DeepseekV2MixtureOfExperts, SupportsPP):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.quant_config = vllm_config.quant_config
        self.model = DeepSeekMultiTokenPredictor(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        # Set MoE hyperparameters
        self.set_moe_parameters()
        self.use_pp = vllm_config.parallel_config.pipeline_parallel_size > 1

    def set_moe_parameters(self):
        self.expert_weights = []
        self.num_moe_layers = self.config.num_nextn_predict_layers
        self.num_expert_groups = self.config.n_group

        self.moe_layers = []
        self.moe_mlp_layers = []
        example_moe = None
        for layer in self.model.layers.values():
            assert isinstance(layer, DeepSeekMultiTokenPredictorLayer)
            layer = layer.mtp_block
            assert isinstance(layer, DeepseekV2DecoderLayer)
            if isinstance(layer.mlp, DeepseekV2MoE):
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
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        rocm_aiter_moe_shared_expert_enabled = (
            rocm_aiter_ops.is_fusion_moe_shared_experts_enabled()
        )
        stacked_params_mapping = [
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
            ("fused_qkv_a_proj", "q_a_proj", 0),
            ("fused_qkv_a_proj", "kv_a_proj_with_mqa", 1),
        ]

        # Fused indexer wk + weights_proj (shard 0 = wk, shard 1 = weights_proj)
        indexer_fused_mapping = [
            ("wk_weights_proj", "wk", 0),
            ("wk_weights_proj", "weights_proj", 1),
        ]
        stacked_params_mapping.extend(indexer_fused_mapping)

        expert_params_mapping = fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts
            + (
                self.config.n_shared_experts
                if rocm_aiter_moe_shared_expert_enabled
                else 0
            ),
        )

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        # Determine whether to load the indexer weight
        mtp_has_indexer = any("indexer" in param_name for param_name in params_dict.keys())
        _pending_wk_fp8: dict = {}  # FP8 indexer wk dequant buffer
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            #  Skip loading mtp indexer weights
            if "indexer" in name and not mtp_has_indexer:
                logger.info(f"Skipping mtp indexer weight (DSA disabled): {name}")
                continue
            spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)
            if spec_layer is None:
                # load embed_tokens weight from target model in pp mode
                if "embed_tokens" in name and self.use_pp:
                    for local_name in params_dict.keys():
                        if "embed_tokens" in local_name:
                            param = params_dict[local_name]
                            weight_loader = getattr(
                                param, "weight_loader", default_weight_loader
                            )
                            weight_loader(param, loaded_weight)
                            break
                continue
            is_fusion_moe_shared_experts_layer = (
                rocm_aiter_moe_shared_expert_enabled and ("mlp.shared_experts" in name)
            )
            name = self._rewrite_spec_layer_name(spec_layer, name)

            if _try_load_quantized_indexer_wk(
                name, loaded_weight, _pending_wk_fp8, params_dict, loaded_params
            ):
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
                if is_fusion_moe_shared_experts_layer:
                    continue
                name_mapped = name.replace(weight_name, param_name)

                # QKV fusion is optional, fall back to normal
                # weight loading if it's not enabled
                if (
                    param_name == "fused_qkv_a_proj"
                ) and name_mapped not in params_dict:
                    continue
                else:
                    name = name_mapped

                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Special handling: when AITER fusion_shared_experts is enabled,
                # checkpoints may provide a single widened shared_experts tensor
                # without explicit expert indices
                # (e.g. ...mlp.shared_experts.gate_proj.weight).
                # For models with multiple shared experts, split that tensor
                # evenly into per-shared-expert slices and load them into
                # appended expert slots mlp.experts.{n_routed_experts + j}.*
                # accordingly.
                num_chunks = 1
                if is_fusion_moe_shared_experts_layer:
                    num_chunks = getattr(self.config, "n_shared_experts", 1) or 1
                    # Determine split axis based on op type
                    # gate/up: ColumnParallel → split along dim 0
                    # down: RowParallel → split along dim 1
                    split_dim = (
                        1
                        if ("down_proj.weight" in name and loaded_weight.ndim > 1)
                        else 0
                    )
                    total = loaded_weight.shape[split_dim]
                    assert total % num_chunks == 0, (
                        f"Shared expert weight dim {total} "
                        f"not divisible by num_chunks {num_chunks}"
                    )
                    chunk_size = total // num_chunks

                for j in range(num_chunks):
                    chunk_name = name
                    weight_to_load = loaded_weight

                    if is_fusion_moe_shared_experts_layer:
                        chunk_slice = slice(j * chunk_size, (j + 1) * chunk_size)
                        if loaded_weight.ndim == 1:
                            weight_to_load = loaded_weight[chunk_slice]
                        elif split_dim == 0:
                            weight_to_load = loaded_weight[chunk_slice, :]
                        else:
                            weight_to_load = loaded_weight[:, chunk_slice]
                        # Synthesize an expert-style name so expert mapping
                        # can route it
                        chunk_name = name.replace(
                            "mlp.shared_experts",
                            f"mlp.experts.{self.config.n_routed_experts + j}",
                        )

                    # Use expert_params_mapping to locate the destination
                    # param and delegate to its expert-aware weight_loader
                    # with expert_id.
                    is_expert_weight = False
                    for mapping in expert_params_mapping:
                        param_name, weight_name, expert_id, shard_id = mapping
                        if weight_name not in chunk_name:
                            continue

                        # Anyway, this is an expert weight and should not be
                        # attempted to load as other weights later
                        is_expert_weight = True

                        # Do not modify `name` since the loop may continue here
                        # Instead, create a new variable
                        name_mapped = chunk_name.replace(weight_name, param_name)

                        param = params_dict[name_mapped]
                        # We should ask the weight loader to return success or
                        # not here since otherwise we may skip experts with
                        # other available replicas.
                        weight_loader = typing.cast(
                            Callable[..., bool], param.weight_loader
                        )
                        success = weight_loader(
                            param,
                            weight_to_load,
                            name_mapped,
                            shard_id=shard_id,
                            expert_id=expert_id,
                            return_success=True,
                        )
                        if success:
                            if not is_fusion_moe_shared_experts_layer:
                                name = name_mapped
                            else:
                                loaded_params.add(name_mapped)
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

                        name = maybe_remap_kv_scale_name(name, params_dict)
                        if name is None:
                            continue

                        # According to DeepSeek-V3 Technical Report, MTP modules
                        # shares embedding layer. We only load the first weights.
                        if (
                            spec_layer != self.model.mtp_start_layer_idx
                            and ".layers" not in name
                        ):
                            continue

                        param = params_dict[name]
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, loaded_weight)
            if not is_fusion_moe_shared_experts_layer:
                loaded_params.add(name)

        # Validate that weights were loaded for each expected MTP layer.
        loaded_layers: set[int] = set()
        for param_name in loaded_params:
            spec_layer = get_spec_layer_idx_from_weight_name(self.config, param_name)
            if spec_layer is not None:
                loaded_layers.add(spec_layer)
        for layer_idx in range(
            self.model.mtp_start_layer_idx,
            self.model.mtp_start_layer_idx + self.model.num_mtp_layers,
        ):
            if layer_idx not in loaded_layers:
                raise ValueError(
                    f"MTP speculative decoding layer {layer_idx} weights "
                    f"missing from checkpoint. The checkpoint may have "
                    f"been quantized without including the MTP layers. "
                    f"Use a checkpoint that includes MTP layer weights, "
                    f"or disable speculative decoding."
                )

        return loaded_params

    def _rewrite_spec_layer_name(self, spec_layer: int, name: str) -> str:
        """
        Rewrite the weight name to match the format of the original model.
        Add .mtp_block for modules in transformer layer block for spec layer
        and rename shared layer weights to be top level.
        """
        spec_layer_weight_names = [
            "embed_tokens",
            "enorm",
            "hnorm",
            "eh_proj",
            "shared_head",
        ]
        shared_weight_names = ["embed_tokens"]
        spec_layer_weight = False
        shared_weight = False
        for weight_name in spec_layer_weight_names:
            if weight_name in name:
                spec_layer_weight = True
                if weight_name in shared_weight_names:
                    shared_weight = True
                break
        if not spec_layer_weight:
            # treat rest weights as weights for transformer layer block
            name = name.replace(
                f"model.layers.{spec_layer}.", f"model.layers.{spec_layer}.mtp_block."
            )
        elif shared_weight:
            # treat shared weights as top level weights
            name = name.replace(f"model.layers.{spec_layer}.", "model.")
        return name
