# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pinned PR #68296 DFly derivation for the v0.33 HCU plugin."""
from copy import deepcopy
from functools import partial
from transformers import PreTrainedConfig
from .configs import SPARSE_ATTENTION
from vllm.transformers_utils.configs.hy_v4 import HYV4Config
from huggingface_hub.dataclasses import strict


def _copy_derived_dfly_config(derived, _loaded_config):
    """Keep draft HF overrides pickleable across Core/worker spawn boundaries."""
    return deepcopy(derived)


@strict
class HYV4DFlyConfig(HYV4Config):
    """Core HYV4 text draft, without target IndexPool/MoE-only validation."""
    model_type = "hy_v4_dfly"

    def rope_for_layer(self, layer_idx):
        if not 0 <= layer_idx < self.num_hidden_layers:
            raise ValueError("DFly RoPE layer is out of range")
        return deepcopy(self.rope_parameters)


def _derive_dfly_config(
    hf_config: PreTrainedConfig,
    *,
    target_config: PreTrainedConfig,
    target_model_type: str = "hy_v4",
) -> PreTrainedConfig:
    """Derive HYV4's text-only draft without changing the target configuration."""
    config = deepcopy(target_config.get_text_config())
    if config.model_type != target_model_type:
        raise ValueError(f"DFly requires a {target_model_type} target model")
    num_layers = getattr(target_config, "dfly_num_hidden_layers", None)
    if type(num_layers) is not int or num_layers < 1:
        raise ValueError("DFly requires a positive dfly_num_hidden_layers")
    config.target_num_hidden_layers = config.num_hidden_layers
    config.target_hidden_size = config.hidden_size
    config.dfly_intermediate_size = getattr(
        target_config, "dfly_intermediate_size", None
    )
    if config.dfly_intermediate_size is not None and (
        type(config.dfly_intermediate_size) is not int
        or config.dfly_intermediate_size < 1
    ):
        raise ValueError("DFly intermediate size must be positive")
    for name, default in (
        ("enable_ihc", False),
        ("mtp_fusion", "slot0"),
        ("target_index_topk", config.index_topk),
        ("context_local_window", None),
        ("enable_hidden_correction", True),
        ("mlp_type", "dense"),
        ("intermediate_size", config.intermediate_size),
        ("attn_post_proj_norm", None),
        ("mask_token_id", None),
        ("target_index_layer", None),
        ("target_kv_layers", None),
        ("hidden_correction_intermediate_size", config.hidden_size),
    ):
        setattr(config, name, deepcopy(getattr(target_config, f"dfly_{name}", default)))
    if (
        config.attn_post_proj_norm is not None
        and type(config.attn_post_proj_norm) is not bool
    ):
        raise ValueError("DFly attention post-projection norm must be boolean")
    if config.mlp_type != "dense" or config.enable_ihc is not False:
        raise ValueError("DFly requires dense FFNs and dfly_enable_ihc=false")
    if getattr(target_config, "dfly_attn_causal", True) is not True:
        raise ValueError("DFly requires causal MLA attention")
    if config.mtp_fusion != "slot0" or config.enable_hidden_correction is not True:
        raise ValueError("DFly requires slot0 fusion and hidden correction")
    if type(config.mask_token_id) is not int or not (
        0 <= config.mask_token_id < config.vocab_size
    ):
        raise ValueError("DFly dfly_mask_token_id is outside the vocabulary")
    if (
        type(config.target_index_topk) is not int
        or config.target_index_topk <= 0
        or (
            config.context_local_window is not None
            and (
                type(config.context_local_window) is not int
                or config.context_local_window < 0
            )
        )
    ):
        raise ValueError("DFly needs positive target top-k and nonnegative window")
    if (
        type(config.hidden_correction_intermediate_size) is not int
        or config.hidden_correction_intermediate_size <= 0
    ):
        raise ValueError("DFly hidden correction intermediate size must be positive")
    config.model_type = "hy_v4_dfly"
    config.architectures = ["HYV4DFlyModel"]
    config.num_hidden_layers = num_layers
    config.n_routed_experts = 0
    config.n_shared_experts = 0
    config.num_experts = 0
    config.num_nextn_predict_layers = 0
    config.n_predict = 0
    config.mlp_layer_types = ["dense"] * num_layers
    config.layer_types = ["deepseek_sparse_attention"] * num_layers
    config.indexer_types = ["full"] * num_layers
    return config


def derive_flash_dfly_config(target_config):
    draft = _derive_dfly_config(target_config, target_config=target_config,
                                target_model_type="hy_v4_flash")
    target = target_config.get_text_config()
    ropes = []
    for field in ("target_index_layer", "target_kv_layers"):
        indices = getattr(draft, field)
        if field == "target_index_layer" and type(indices) is int:
            indices = [indices] * draft.num_hidden_layers
        if not isinstance(indices, list) or len(indices) != draft.num_hidden_layers:
            raise ValueError(f"DFly {field} needs one target layer per draft layer")
        for index in indices:
            if type(index) is not int or not 1 <= index < target.num_hidden_layers:
                raise ValueError(f"DFly {field} has invalid target layer {index}")
            if target.layer_types[index] != SPARSE_ATTENTION:
                raise ValueError(f"DFly target layer {index} must be sparse, not SWA")
            ropes.append(target.rope_for_layer(index))
        setattr(draft, field, indices)
    if len(set(draft.target_kv_layers)) != draft.num_hidden_layers:
        raise ValueError("DFly requires distinct target KV layers")
    if any(p != ropes[0] for p in ropes[1:]):
        raise ValueError("DFly selected target layers must use identical RoPE")
    draft.rope_parameters = deepcopy(ropes[0])
    draft.engram_layer_ids = None
    draft.routed_expert_hidden_size = None
    draft.sandwich_norm = True
    draft.dsa_skip_attn_post_norm = False
    draft.layer_types = [SPARSE_ATTENTION] * draft.num_hidden_layers
    values = draft.to_dict()
    values["model_type"] = "hy_v4_dfly"
    return HYV4DFlyConfig.from_dict(values)


from typing import Literal

import torch
from vllm.config.model import ModelConfig
from vllm.config.speculative import SpeculativeConfig
from vllm.config.utils import config
from .quantization import validate_flash_quant_config


@config
class HCUDFlySpeculativeConfig(SpeculativeConfig):
    """DFly's behavior on Core v0.33, not an alias for DFlash or MTP.

    Instantiated only by the explicitly installed plugin compatibility hooks.
    Parent field validators remain active; other speculative methods continue
    to construct the untouched upstream class.
    """

    method: Literal["dfly"] = "dfly"

    def __post_init__(self):
        target = self.target_model_config
        parallel = self.target_parallel_config
        if target is None or parallel is None:
            raise ValueError("DFly requires target model and parallel configuration")
        family = target.hf_text_config.model_type
        if family not in ("hy_v4_flash", "hy_v4"):
            raise ValueError("HCU DFly requires a standard HYV4 or HYV4 Flash target")
        if target.dtype != torch.bfloat16:
            raise ValueError("DFly requires bfloat16 compute")
        if family == "hy_v4":
            from .dfly_topology import validate_standard_dfly_parallel
            validate_standard_dfly_parallel(parallel, enforce_eager=target.enforce_eager)
        else:
            for name in ("data_parallel_size", "pipeline_parallel_size",
                         "prefill_context_parallel_size", "decode_context_parallel_size"):
                if getattr(parallel, name) != 1:
                    raise ValueError(f"HCU v0.33 DFly requires {name}=1")
        if family == "hy_v4_flash" and target.supports_multimodal_inputs:
            raise NotImplementedError("First qualify HCU DFly with image/video limits set to zero")
        # None inherits the target policy; an explicitly eager draft is still
        # useful with target graphs. Never force target graphs off for DFly.
        draft_eager = (target.enforce_eager if self.enforce_eager is None
                       else self.enforce_eager)
        if target.enforce_eager and not draft_eager:
            raise ValueError("DFly draft graphs require target graphs to be enabled")
        if (self.parallel_drafting
                or self.enable_adaptive_verification
                or self.rejection_sample_method != "standard"
                or self.use_local_argmax_reduction
                or self.draft_lm_head_quantization is not None
                or self.use_heterogeneous_vocab
                or self.num_speculative_tokens_per_batch_size is not None):
            raise ValueError("Unsupported DFly verification, logits or scheduling option")
        if self.attention_backend is not None:
            raise ValueError("DFly inherits the target attention backend")
        if self.kv_cache_dtype not in (None, "auto", "bfloat16", "fp8_ds_mla"):
            raise ValueError("DFly requires BF16 or fp8_ds_mla KV cache")
        n = self.num_speculative_tokens
        if family == "hy_v4":
            from vllm.models.hy_v4.dfly_config import resolve_dfly_width
            n = resolve_dfly_width(target.hf_config, n)
        if n is None:
            block = getattr(target.hf_config, "dfly_block_size", None)
            if type(block) is not int or not 2 <= block <= 17:
                raise ValueError("DFly needs num_speculative_tokens or dfly_block_size")
            n = block - 1
        if type(n) is not int or not 1 <= n <= 16:
            raise ValueError("DFly requires 1..16 speculative tokens")
        self.num_speculative_tokens = n
        if self.model not in (None, target.model, target.model_weights):
            raise ValueError("DFly draft weights must come from the target checkpoint")
        if self.quantization not in (None, target.quantization):
            raise ValueError("DFly must preserve checkpoint quantization")
        if target.quantization not in (None, "fp8", "compressed-tensors", "hcu_hy4_ct_w4a8"):
            raise ValueError("Unsupported HCU DFly target quantization")
        if family == "hy_v4_flash":
            if target.quantization is not None:
                validate_flash_quant_config(target.model_arch_config.quantization_config)
            derived = derive_flash_dfly_config(target.hf_config)
        else:
            from vllm.models.hy_v4.dfly_config import derive_standard_dfly_config
            if target.quantization is not None:
                from ...quantization.hy4_ct_policy import accepts_checkpoint
                if not accepts_checkpoint(target.model_arch_config.quantization_config,
                                          target.quantization, target.hf_config):
                    raise ValueError("Standard DFly requires the admitted HY4 CT checkpoint recipe")
            derived = derive_standard_dfly_config(target.hf_config)
        for size in (derived.num_attention_heads, derived.dfly_intermediate_size,
                     derived.hidden_correction_intermediate_size):
            if size is not None and size % parallel.tensor_parallel_size:
                raise ValueError("DFly heads and FFN widths must be divisible by TP")
        if self.draft_tensor_parallel_size not in (None, parallel.tensor_parallel_size):
            raise ValueError("DFly draft must use target TP")
        # Draft config and weights must describe the very same target revision.
        # Core v0.33 does not fill these speculative fields from EngineArgs.
        for field in ("revision", "code_revision"):
            target_revision = getattr(target, field, None)
            requested_revision = getattr(self, field)
            if requested_revision not in (None, target_revision):
                raise ValueError(f"DFly {field} must match the target checkpoint")
            setattr(self, field, target_revision)
        self.model = target.model_weights or target.model
        self.quantization = target.quantization
        self.enforce_eager = draft_eager
        self.prompt_lookup_min = None
        self.prompt_lookup_max = None
        # This Core flag selects P-EAGLE input padding; DFly builds its own
        # parallel query block in HYV4DFlySpeculator and must not select it.
        self.parallel_drafting = False
        self.draft_tensor_parallel_size = parallel.tensor_parallel_size
        self.draft_model_config = ModelConfig(
            model=self.model, runner="draft", tokenizer=target.tokenizer,
            tokenizer_mode=target.tokenizer_mode, trust_remote_code=target.trust_remote_code,
            skip_tokenizer_init=target.skip_tokenizer_init,
            allowed_local_media_path=target.allowed_local_media_path,
            allowed_media_domains=target.allowed_media_domains,
            dtype=target.dtype, seed=target.seed, revision=self.revision,
            code_revision=self.code_revision, tokenizer_revision=target.tokenizer_revision,
            max_model_len=self.max_model_len or target.max_model_len,
            spec_target_max_model_len=target.max_model_len, quantization=self.quantization,
            enforce_eager=draft_eager, max_logprobs=target.max_logprobs,
            hf_overrides=partial(_copy_derived_dfly_config, derived), config_format=target.config_format,
        )
        self.draft_parallel_config = self.create_draft_parallel_config(
            parallel, parallel.tensor_parallel_size, self.draft_model_config)
        if family == "hy_v4" and parallel.data_parallel_size > 1:
            from .dfly_topology import make_rank_local_draft_parallel
            self.draft_parallel_config = make_rank_local_draft_parallel(parallel)

    def use_eagle(self) -> bool:
        # In v0.33 this public predicate means "consumes target hidden states".
        return True

    def needs_eagle_block_drop(self) -> bool:
        return False

    def draft_shares_target_positions(self) -> bool:
        return False

    @property
    def max_num_new_slots_for_drafting(self) -> int:
        # DFly uses a separate six-query pass, not extra target-batch queries.
        return 0
