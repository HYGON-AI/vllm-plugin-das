# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# Modified by Hygon Information Technology Co., Ltd., 2026.
import torch
from typing import TYPE_CHECKING, Optional
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import RoutedExperts
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

if TYPE_CHECKING:
    from vllm.model_executor.models.utils import WeightsMapper

logger = init_logger(__name__)

__all__ = ["CompressedTensorsLinearMethod"]

class SlimQuantCompressedTensorsMarlinConfig(CompressedTensorsConfig):
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
        if isinstance(layer, RoutedExperts):
            moe_backend = getattr(layer.moe_config, "moe_backend", "auto")
            from vllm_hcu.model_executor.layers.fused_moe.aiter_runtime import (
                is_aiter_moe_requested,
            )

            if moe_backend != "auto" or is_aiter_moe_requested(
                layer.moe_config
            ):
                from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe import (
                    CompressedTensorsMoEMethod,
                )

                return CompressedTensorsMoEMethod.get_moe_method(
                    self,
                    layer,
                    prefix,
                )
            return CompressedTensorsMarlinMoEMethod.get_moe_method(self, layer)
        return None
