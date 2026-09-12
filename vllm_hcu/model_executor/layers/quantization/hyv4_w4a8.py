# SPDX-License-Identifier: Apache-2.0
"""HYV4 format/name adapters over the current SlimQuant execution owners."""
from __future__ import annotations

import torch
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.fused_moe import RoutedExperts

from .slimquant_w4a8 import (
    SlimQuantW4A8Int8Config,
    SlimQuantW4A8Int8LinearMethod,
    SlimQuantW4A8Int8AiterMoEMethod,
)
from .hyv4_w4a8_kernels import unpack_int4
from .hyv4_w4a8_weights import (
    CHECKPOINT_FORMAT, adapt_weights, read_quantized_modules,
    to_aiter_packing, validate_channel_scale, validate_metadata,
)


class HYV4W4A8Config(SlimQuantW4A8Int8Config):
    checkpoint_format = CHECKPOINT_FORMAT
    manifest_field = "conversion_manifest"
    adapt_weights = staticmethod(adapt_weights)

    def __init__(self, conversion_manifest: str):
        self.quantized_modules = read_quantized_modules(conversion_manifest)

    @classmethod
    def from_config(cls, config):
        if (config.get("quant_method") != "slimquant_w4a8"
                or config.get("checkpoint_format") != cls.checkpoint_format):
            raise ValueError(f"HYV4 requires explicit {cls.checkpoint_format}")
        validate_metadata(config, "quantization_config")
        path = config.get(cls.manifest_field)
        if not isinstance(path, str) or not path:
            raise ValueError(f"HYV4 requires {cls.manifest_field}")
        return cls(path)

    def get_quant_method(self, layer, prefix):
        prefix = prefix.removesuffix(".routed_experts").replace(".mtp_block.", ".")
        quantized = prefix in self.quantized_modules
        if isinstance(layer, LinearBase):
            if not quantized:
                return UnquantizedLinearMethod()
            # Let the current owner install its compressed INT8 scheme.
            super().get_quant_method(layer, prefix)
            return HYV4W4A8LinearMethod(self)
        if isinstance(layer, RoutedExperts) and quantized:
            return HYV4W4A8MoEMethod(self, layer.moe_config)
        return None

    def remap_mtp_modules(self, start: int, count: int) -> frozenset[str]:
        modules = set(self.quantized_modules)
        for offset in range(count):
            source = f"model.mtp_layers.{offset}."
            target = f"model.layers.{start + offset}."
            modules.update(target + name[len(source):]
                           for name in self.quantized_modules if name.startswith(source))
        return frozenset(modules)


class HYV4W4A8LinearMethod(SlimQuantW4A8Int8LinearMethod):
    """Expand serialized INT4 pieces once into the current INT8 allocation."""

    def create_weights(self, layer, input_size_per_partition, output_partition_sizes,
                       input_size, output_size, params_dtype, **extra_weight_attrs):
        if input_size_per_partition % 2:
            raise ValueError("HYV4 INT4 requires an even local input dimension")
        super().create_weights(layer, input_size_per_partition, output_partition_sizes,
                               input_size, output_size, params_dtype, **extra_weight_attrs)
        load_weight = layer.weight.weight_loader

        def load_int4(param, weight, *args, **kwargs):
            if weight.dtype != torch.uint8 or weight.ndim != 2:
                raise ValueError("HYV4 linear requires rank-2 UINT8 checkpoint packing")
            return load_weight(param, unpack_int4(to_aiter_packing(weight)), *args, **kwargs)

        layer.weight.weight_loader = load_int4


class HYV4W4A8MoEMethod(SlimQuantW4A8Int8AiterMoEMethod):
    """Normalize serialized pieces before the current loader/layout lifecycle.

    Current SlimQuant multiplies canonical channel scales by 16 for INT4
    kernels. HYV4 serializes signed-INT4 multipliers, so divide on loading.
    No post-load conversion can accidentally swap an installed AITER layout.
    supports_eplb remains the current owner's False capability.
    """

    def create_weights(self, layer, num_experts, hidden_size,
                       intermediate_size_per_partition, params_dtype, **extra_weight_attrs):
        super().create_weights(layer, num_experts, hidden_size,
                               intermediate_size_per_partition, params_dtype, **extra_weight_attrs)

        def wrap(parameter, scale):
            load = parameter.weight_loader

            def load_piece(param, value, *args, **kwargs):
                if scale:
                    validate_channel_scale(value, "expert")
                    value = value / 16.0
                else:
                    value = to_aiter_packing(value)
                return load(param, value, *args, **kwargs)

            # The target/MTP ledger reads the bound loader owner to determine
            # which logical experts are local. Preserve that same owner.
            load_piece.__self__ = getattr(load, "__self__", None)
            parameter.weight_loader = load_piece

        for name in ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale"):
            wrap(getattr(layer, name), name.endswith("_scale"))
