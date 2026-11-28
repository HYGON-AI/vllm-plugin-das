# SPDX-License-Identifier: Apache-2.0
"""Standard HY4 DFly bindings: Core MLA semantics, HCU sparse provider.

The shared Core033 DFly loader/proposer retains checkpoint and KV guards.
This module supplies constructor factories only, without a second forward.
"""

from vllm.models.hy_v4.nvidia.attention import HYV4MLAAttention

from ..quantization.hy4_ct_w4a8 import HcuHY4CTW4A8Config
from .hy_v4_flash.dfly import HYV4DFlyDecoderLayer, HYV4DFlyQuantizationConfig
from .hy_v4_flash.runtime import validate_flash_platform
from .hy_v4_flash.swa import FlashSparseBackend


class HYV4StandardDFlyAttention(HYV4MLAAttention):
    def __init__(self, vllm_config, config, layer_idx, topk_indices_buffer, prefix):
        validate_flash_platform()
        super().__init__(
            vllm_config=vllm_config,
            config=config,
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            q_lora_rank=config.q_lora_rank,
            kv_lora_rank=config.kv_lora_rank,
            max_position_embeddings=vllm_config.model_config.max_model_len,
            cache_config=vllm_config.cache_config,
            quant_config=vllm_config.quant_config,
            prefix=prefix,
            topk_indices_buffer=topk_indices_buffer,
            layer_idx=layer_idx,
        )
        if not self.external_topk or self.indexer is not None:
            raise ValueError(
                "Standard DFly must consume target top-k without draft index weights"
            )

    def _resolve_sink_backend(self, kv_cache_dtype):
        # Same qualified HCU sparse MLA provider as the standard target.
        return FlashSparseBackend


class HYV4StandardDFlyDecoderLayer(HYV4DFlyDecoderLayer):
    attention_cls = HYV4StandardDFlyAttention


class HYV4StandardDFlyQuantizationConfig(
    HYV4DFlyQuantizationConfig, HcuHY4CTW4A8Config
):
    """Canonical draft path matching with the admitted HCU INT8 linear method."""
