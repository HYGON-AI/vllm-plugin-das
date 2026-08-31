# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from huggingface_hub.dataclasses import strict
from transformers.configuration_utils import PreTrainedConfig
from transformers.modeling_rope_utils import RopeParameters


@strict
class HYV4Config(PreTrainedConfig):
    """Configuration for the HY V4 mixture-of-experts language model."""

    model_type = "hy_v4"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {"num_local_experts": "n_routed_experts"}
    base_model_tp_plan = {
        "layers.*.self_attn.q_b_proj": "colwise",
        "layers.*.self_attn.kv_a_proj_with_mqa": "mla_kv_a_proj",
        "layers.*.self_attn.kv_b_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.self_attn.linear_gate": "colwise",
        "layers.*.mlp.experts.gate_up_proj": "packed_colwise",
        "layers.*.mlp.experts.down_proj": "rowwise",
        "layers.*.mlp.experts": "moe_tp_experts",
        "layers.*.mlp.shared_experts.gate_proj": "colwise",
        "layers.*.mlp.shared_experts.up_proj": "colwise",
        "layers.*.mlp.shared_experts.down_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    vocab_size: int = 120832
    hidden_size: int = 2816
    intermediate_size: int = 6912
    moe_intermediate_size: int = 768
    num_hidden_layers: int = 34
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    head_dim: int = 256
    hidden_act: str = "silu"
    max_position_embeddings: int = 262144
    initializer_range: float = 0.006
    rms_norm_eps: float = 1e-5
    use_cache: bool = True
    pad_token_id: int | None = 120002
    bos_token_id: int | None = 120000
    eos_token_id: int | list[int] | None = 120025
    tie_word_embeddings: bool = False
    attention_bias: bool = False
    attention_dropout: float = 0.0
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    num_experts_per_tok: int = 8
    routed_scaling_factor: float = 2.827
    norm_topk_prob: bool = True
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 192
    qk_rope_head_dim: int = 64
    v_head_dim: int = 256
    mlp_layer_types: list[str] | None = None
    layer_types: list[str] | None = None
    index_topk: int = 2048
    index_head_dim: int = 128
    index_n_heads: int = 16
    indexer_types: list[str] | None = None
    enable_lm_head_fp32: bool = True
    enable_ihc: bool = True
    hc_mult: int = 4
    hc_magnitude: float = 2.0
    hc_eps: float = 1e-6
    gated_mla: bool = True
    gating_type: str = "elementwise"
    learnable_sink: bool = True
    learnable_sink_init: float = 0.0
    swiglu_limit: float = 10.0
    rope_parameters: RopeParameters | dict | None = None
    num_nextn_predict_layers: int = 1
    mtp_loss_factor: float = 0.1

    def __post_init__(self, **kwargs) -> None:
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.head_dim = self.qk_rope_head_dim

        if self.mlp_layer_types is None:
            self.mlp_layer_types = ["dense"] * min(1, self.num_hidden_layers) + [
                "sparse"
            ] * max(self.num_hidden_layers - 1, 0)
        if self.indexer_types is None:
            self.indexer_types = [
                "full" if layer_idx == 0 or (layer_idx - 1) % 4 == 0 else "shared"
                for layer_idx in range(self.num_hidden_layers)
            ]
        if self.layer_types is not None:
            self.layer_types = [
                "deepseek_sparse_attention"
                if layer_type in ("sparse_attention", "sparse")
                else layer_type
                for layer_type in self.layer_types
            ]

        if self.enable_lm_head_fp32 and getattr(self, "head_dtype", None) is None:
            self.head_dtype = "float32"

        super().__post_init__(**kwargs)


def register_hy_v4_config() -> None:
    """Register HY V4 with Transformers and vLLM configuration loaders."""

    from transformers import AutoConfig
    from vllm.transformers_utils import config as vllm_config

    vllm_config._CONFIG_REGISTRY[HYV4Config.model_type] = HYV4Config
    AutoConfig.register(HYV4Config.model_type, HYV4Config, exist_ok=True)


__all__ = ["HYV4Config", "register_hy_v4_config"]
