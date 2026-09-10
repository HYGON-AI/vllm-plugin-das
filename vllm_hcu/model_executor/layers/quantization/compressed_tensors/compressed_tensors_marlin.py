# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# Modified by Hygon Information Technology Co., Ltd., 2026.
import torch
from typing import TYPE_CHECKING, Optional
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    RoutedExperts,
    UnquantizedFusedMoEMethod,
)
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
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


def _add_runtime_prefix_ignore_aliases(ignore: list[str]) -> None:
    """Add aliases for wrappers that prepend ``language_model.`` at runtime."""
    existing = set(ignore)
    aliases = []
    for pattern in tuple(ignore):
        if pattern.startswith("re:^"):
            body = pattern[len("re:^") :]
            if body.startswith(r"language_model\."):
                continue
            alias = r"re:^language_model\." + body
        elif pattern.startswith("language_model."):
            continue
        else:
            alias = "language_model." + pattern
        if alias not in existing:
            existing.add(alias)
            aliases.append(alias)
    ignore.extend(aliases)

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
        _add_runtime_prefix_ignore_aliases(self.ignore)

        if should_ignore_layer(prefix,
                               ignore=self.ignore,
                               fused_mapping=self.packed_modules_mapping):
            if isinstance(layer, RoutedExperts):
                return UnquantizedFusedMoEMethod(layer.moe_config)
            if isinstance(layer, LinearBase):
                return UnquantizedLinearMethod()
            return None
        if isinstance(layer, LinearBase):
            scheme = self.get_scheme(layer=layer, layer_name=prefix)
            if scheme is None:
                return UnquantizedLinearMethod()
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
            return CompressedTensorsMarlinMoEMethod.get_moe_method(
                self,
                layer,
                layer_name=prefix,
            )
        return None
