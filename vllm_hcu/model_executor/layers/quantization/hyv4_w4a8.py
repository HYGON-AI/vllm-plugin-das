# SPDX-License-Identifier: Apache-2.0
"""Explicit HYV4 custom-format quantization, using aiter W4A8 kernels."""

import torch
from torch.nn import Parameter

from vllm.model_executor.layers.linear import (
    LinearBase, LinearMethodBase, UnquantizedLinearMethod,
)
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.utils import set_weight_attrs

from .slimquant_w4a8 import (
    SlimQuantW4A8Int8Config, SlimQuantW4A8Int8AiterMoEMethod,
)
from .hyv4_w4a8_weights import (
    CHECKPOINT_FORMAT, read_quantized_modules, to_aiter_packing,
)


class HYV4W4A8Config(SlimQuantW4A8Int8Config):
    checkpoint_format = CHECKPOINT_FORMAT

    def __init__(self, conversion_manifest: str):
        self.quantized_modules = read_quantized_modules(conversion_manifest)

    def get_quant_method(self, layer, prefix):
        quantized = prefix.removesuffix(".routed_experts") in self.quantized_modules
        if isinstance(layer, LinearBase):
            return HYV4W4A8LinearMethod() if quantized else UnquantizedLinearMethod()
        if isinstance(layer, RoutedExperts) and quantized:
            return HYV4W4A8MoEMethod(self, layer.moe_config)
        return None


class HYV4W4A8LinearMethod(LinearMethodBase):
    def create_weights(self, layer, input_size_per_partition,
                       output_partition_sizes, input_size, output_size,
                       params_dtype, **extra_weight_attrs):
        if input_size_per_partition % 2:
            raise ValueError("HYV4 INT4 requires an even local input dimension")
        n = sum(output_partition_sizes)
        weight = Parameter(torch.empty(n, input_size_per_partition // 2,
                                       dtype=torch.int8), requires_grad=False)
        scale = Parameter(torch.empty(n, 1, dtype=torch.float32), requires_grad=False)
        set_weight_attrs(weight, {**extra_weight_attrs, "input_dim": 1,
                                  "output_dim": 0, "packed_dim": 1,
                                  "packed_factor": 2})
        set_weight_attrs(scale, {**extra_weight_attrs, "output_dim": 0})
        layer.register_parameter("weight", weight)
        layer.register_parameter("weight_scale", scale)

    def process_weights_after_loading(self, layer):
        layer.weight.data.copy_(to_aiter_packing(layer.weight.data))

    def apply(self, layer, x, bias=None):
        from .hyv4_w4a8_kernels import linear

        shape = x.shape[:-1]
        result = linear(x.reshape(-1, x.shape[-1]), layer.weight,
                        layer.weight_scale, bias)
        return result.reshape(*shape, layer.weight.shape[0])


class HYV4W4A8MoEMethod(SlimQuantW4A8Int8AiterMoEMethod):
    def process_weights_after_loading(self, layer):
        super().process_weights_after_loading(layer)
        # Convert only local shards, expert by expert to bound peak memory.
        for parameter in (layer.w13_weight, layer.w2_weight):
            for expert in parameter.data:
                expert.copy_(to_aiter_packing(expert))

    def apply(self, layer, x, topk_weights, topk_ids,
              shared_experts_input=None, **kwargs):
        from .hyv4_w4a8_kernels import moe

        return moe(
            x, layer.w13_weight, layer.w2_weight,
            layer.w13_weight_scale, layer.w2_weight_scale,
            topk_weights, topk_ids,
            gemm1_limit=self.moe.swiglu_limit,
            expert_map=layer.expert_map,
            global_num_experts=layer.global_num_experts,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
        )
