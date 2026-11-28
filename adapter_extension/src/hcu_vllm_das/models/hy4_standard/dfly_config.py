# SPDX-License-Identifier: Apache-2.0
"""Standard HY4 checkpoint-native DFly derivation, PR #68296 / Core033."""

from copy import deepcopy

from huggingface_hub.dataclasses import strict
from transformers import PreTrainedConfig
from vllm.transformers_utils.configs.hy_v4 import HYV4Config


@strict
class HYV4StandardDFlyConfig(HYV4Config):
    model_type = "hy_v4_dfly"


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


def derive_standard_dfly_config(target_config):
    draft = _derive_dfly_config(target_config, target_config=target_config)
    target = target_config.get_text_config()
    for field in ("target_index_layer", "target_kv_layers"):
        selected = getattr(draft, field)
        if field == "target_index_layer" and type(selected) is int:
            selected = [selected] * draft.num_hidden_layers
        if (
            not isinstance(selected, list)
            or len(selected) != draft.num_hidden_layers
            or any(
                type(i) is not int or not 1 <= i < target.num_hidden_layers
                for i in selected
            )
        ):
            raise ValueError(
                f"DFly {field} needs one valid target layer per draft layer"
            )
        if target.layer_types is not None and any(
            target.layer_types[i]
            not in (
                "sparse_attention",
                "sparse",
                "indexed_attention",
                "deepseek_sparse_attention",
            )
            for i in selected
        ):
            raise ValueError("DFly target layers must use sparse MLA")
        setattr(draft, field, deepcopy(selected))
    if len(set(draft.target_kv_layers)) != draft.num_hidden_layers:
        raise ValueError("DFly target KV layers must be distinct")
    indexers = getattr(target, "indexer_types", None)
    if indexers is not None and any(
        indexers[i] != "full" for i in draft.target_index_layer
    ):
        raise ValueError("DFly selected target indexer must produce its own top-k")
    draft.architectures = ["HYV4DFlyModel"]
    draft.dfly_target_model_type = "hy_v4"
    values = draft.to_dict()
    values["model_type"] = "hy_v4_dfly"
    return HYV4StandardDFlyConfig.from_dict(values)


def resolve_dfly_width(target_config, requested=None):
    """N draft tokens; checkpoint block width also contains one anchor."""
    n = requested
    if n is None:
        block = getattr(target_config, "dfly_block_size", None)
        if type(block) is not int or not 2 <= block <= 17:
            raise ValueError("DFly needs num_speculative_tokens or dfly_block_size")
        n = block - 1
    if type(n) is not int or not 1 <= n <= 16:
        raise ValueError("DFly requires 1..16 speculative tokens")
    return n
