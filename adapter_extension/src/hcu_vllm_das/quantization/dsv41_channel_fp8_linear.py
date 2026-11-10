# SPDX-License-Identifier: Apache-2.0
"""Keep native CT loading/computation; handle empty PCP-local rows only."""
from math import prod
import torch
from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
from vllm.model_executor.layers.linear import register_weight_loader_v2_supported_method
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import CompressedTensorsLinearMethod


@register_weight_loader_v2_supported_method
class HcuDSV41ChannelFP8LinearMethod(CompressedTensorsLinearMethod):
    def apply(self, layer, x, bias=None):
        shape = x.orig_shape if isinstance(x, QuantizedActivation) else x.shape
        if len(shape) < 2 or prod(shape[:-1]) != 0:
            return super().apply(layer, x, bias)
        if (getattr(layer, "processed_weight_dims", None) != (0, 1)
                or layer.weight.ndim != 2 or shape[-1] != layer.weight.shape[0]):
            raise ValueError("Empty PCP input requires finalized CT (K, N) weights")
        data = x.data if isinstance(x, QuantizedActivation) else x
        dtype = x.orig_dtype if isinstance(x, QuantizedActivation) else x.dtype
        dtype = layer.scheme.fp8_linear.config.out_dtype or dtype
        return torch.empty((*shape[:-1], layer.weight.shape[1]), dtype=dtype, device=data.device)
