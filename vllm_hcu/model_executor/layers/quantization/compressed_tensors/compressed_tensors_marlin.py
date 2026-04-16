# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import torch
from typing import TYPE_CHECKING, Any, Literal, Optional, cast
from compressed_tensors.config import SparsityCompressionConfig
from compressed_tensors.quantization import QuantizationArgs
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.vocab_parallel_embedding import UnquantizedEmbeddingMethod
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (  # noqa: E501
    QuantizationConfig, QuantizeMethodBase)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig, CompressedTensorsLinearMethod, CompressedTensorsKVCacheMethod)
from vllm_hcu.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe_marlin import (
    CompressedTensorsMarlinMoEMethod,
)
from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
    should_ignore_layer)
from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod
from vllm import _custom_ops as ops

if TYPE_CHECKING:
    from vllm.model_executor.models.utils import WeightsMapper

logger = init_logger(__name__)

__all__ = ["CompressedTensorsLinearMethod"]

SPARSITY_CONFIG_NAME: Literal["sparsity_config"] = "sparsity_config"
QUANTIZATION_SCHEME_MAP_TYPE = dict[str, Optional[dict[str, QuantizationArgs]]]


class SlimQuantCompressedTensorsMarlinConfig(CompressedTensorsConfig):
    def __init__(
        self,
        target_scheme_map: dict[str, Any],
        ignore: list[str],
        quant_format: str,
        sparsity_scheme_map: dict[str, SparsityCompressionConfig],
        sparsity_ignore_list: list[str],
        kv_cache_scheme: Optional[dict[str, Any]] = None,
        config: Optional[dict[str, Any]] = None,
        transform_config: Optional[dict[str, Any]] = None,
        total_num_heads: int | None = None,
        total_num_kv_heads: int | None = None,
    ):
        super().__init__(
            target_scheme_map,
            ignore,
            quant_format,
            sparsity_scheme_map,
            sparsity_ignore_list,
            kv_cache_scheme,
            config,
            transform_config
        )
        self.total_num_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads
        
    @classmethod
    def override_quantization_method(
            cls, hf_quant_cfg, user_quant) -> Optional[QuantizationMethods]:
        if hf_quant_cfg.get("quant_method") == "compressed-tensors" \
                and user_quant == "slimquant_marlin":
            return cls.get_name()
        return None
    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "slimquant_compressed_tensors_marlin"

    def get_quant_method(
            self,
            layer: torch.nn.Module,
            prefix: str,
    ) -> Optional["QuantizeMethodBase"]:
        from vllm.model_executor.layers.attention import Attention

        # Check if the layer is skipped for quantization.

        if should_ignore_layer(prefix,
                               ignore=self.ignore,
                               fused_mapping=self.packed_modules_mapping):
            return UnquantizedEmbeddingMethod()#UnquantizedLinearMethod()
        if isinstance(layer, LinearBase):
            scheme = self.get_scheme(layer=layer, layer_name=prefix)
            if scheme is None:
                return UnquantizedEmbeddingMethod()#UnquantizedLinearMethod()
            layer.scheme = scheme
            return CompressedTensorsLinearMethod(self)
        if isinstance(layer, Attention):
            return CompressedTensorsKVCacheMethod(self)
        if isinstance(layer, FusedMoE):
            return CompressedTensorsMarlinMoEMethod.get_moe_method(self, layer)
        return None
