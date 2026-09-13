# SPDX-License-Identifier: Apache-2.0
"""HYV4 format/name adapters over the current SlimQuant execution owners."""
from __future__ import annotations

import torch
from vllm.model_executor.layers.linear import (
    LinearBase, MergedColumnParallelLinear, QKVParallelLinear,
    UnquantizedLinearMethod,
)
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
        load_scale = layer.weight_scale.weight_loader

        def load_channel_scale(param, value, *args, **kwargs):
            # Validate after name normalization, including retained/canonical
            # aliases, and before the owner can narrow or mutate a fused shard.
            validate_channel_scale(value, "linear")
            shard = kwargs.get("loaded_shard_id", args[0] if args else None)
            if isinstance(layer, QKVParallelLinear):
                layer.validate_shard_id(shard)
                sizes = [layer.total_num_heads * layer.head_size,
                         layer.total_num_kv_heads * layer.head_size,
                         layer.total_num_kv_heads * layer.v_head_size]
                # KV replication expands allocation, not serialized channels.
                channels = sum(sizes) if shard is None else sizes["qkv".index(shard)]
            elif isinstance(layer, MergedColumnParallelLinear):
                layer.validate_shard_id(shard)
                shards = (range(len(layer.output_sizes)) if shard is None
                          else shard if isinstance(shard, tuple) else (shard,))
                channels = sum(layer.output_sizes[index] for index in shards)
            else:
                channels = output_size
            if value.shape != (channels, 1):
                raise ValueError(f"HYV4 linear scale requires shape {(channels, 1)}, "
                                 f"got {tuple(value.shape)}")
            return load_scale(param, value, *args, **kwargs)

        layer.weight_scale.weight_loader = load_channel_scale


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

        def wrap(parameter, name):
            load = parameter.weight_loader

            def load_piece(param, value, *args, **kwargs):
                if name.endswith("_scale"):
                    validate_channel_scale(value, "expert")
                    shard = kwargs.get("shard_id", args[1] if len(args) > 1 else None)
                    if shard is None:
                        expected = tuple(param.shape)
                    else:
                        down = name == "w2_weight_scale"
                        if shard not in (("w2",) if down else ("w1", "w3")):
                            raise ValueError(f"HYV4 expert scale {name} cannot load {shard}")
                        allocated = hidden_size if down else intermediate_size_per_partition
                        field = ("hidden_dim_unpadded" if down else
                                 "intermediate_size_per_partition_unpadded")
                        # Only declared logical padding permits a smaller
                        # checkpoint. Missing metadata uses allocation; invalid
                        # metadata must not silently take that fallback.
                        channels = getattr(layer.moe_config, field, allocated)
                        if type(channels) is not int or not 0 < channels <= allocated:
                            raise ValueError(f"HYV4 expert scale has invalid {field}: {channels}")
                        if not down:
                            channels *= layer.moe_config.moe_parallel_config.tp_size
                        expected = (channels, 1)
                    if value.shape != expected:
                        raise ValueError(f"HYV4 expert scale requires shape {expected}, "
                                         f"got {tuple(value.shape)}")
                    value = value / 16.0
                else:
                    # The target loader splits fused checkpoints into the
                    # same per-expert projection ABI before reaching here.
                    # Prove the complete serialized extent before the generic
                    # owner's TP slicing/padding-tolerant copy can accept it.
                    shard = kwargs.get("shard_id", args[1] if len(args) > 1 else None)
                    down = name == "w2_weight"
                    if shard not in (("w2",) if down else ("w1", "w3")):
                        raise ValueError(f"HYV4 packed expert {name} cannot load {shard}")
                    if value.dtype != torch.uint8 or value.ndim != 2:
                        raise ValueError("HYV4 packed expert requires rank-2 UINT8")
                    dimensions = []
                    for field, allocated in (
                        ("hidden_dim_unpadded", hidden_size),
                        ("intermediate_size_per_partition_unpadded",
                         intermediate_size_per_partition),
                    ):
                        logical = getattr(layer.moe_config, field, allocated)
                        if type(logical) is not int or not 0 < logical <= allocated:
                            raise ValueError(f"HYV4 packed expert has invalid {field}: {logical}")
                        dimensions.append(logical)
                    hidden, intermediate = dimensions
                    tp = layer.moe_config.moe_parallel_config.tp_size
                    if type(tp) is not int or tp <= 0:
                        raise ValueError(f"HYV4 packed expert has invalid tp_size: {tp}")
                    # Row-parallel INT4 must also divide on a byte boundary
                    # locally; an even global extent alone is insufficient.
                    if (intermediate if down else hidden) % 2:
                        raise ValueError("HYV4 packed expert requires an even logical input dimension per TP rank")
                    expected = ((hidden, intermediate * tp // 2) if down
                                else (intermediate * tp, hidden // 2))
                    if value.shape != expected:
                        raise ValueError(f"HYV4 packed expert {shard} requires shape {expected}, "
                                         f"got {tuple(value.shape)}")
                    value = to_aiter_packing(value)
                return load(param, value, *args, **kwargs)

            # The target/MTP ledger reads the bound loader owner to determine
            # which logical experts are local. Preserve that same owner.
            load_piece.__self__ = getattr(load, "__self__", None)
            parameter.weight_loader = load_piece

        for name in ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale"):
            wrap(getattr(layer, name), name)
