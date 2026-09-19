# SPDX-License-Identifier: Apache-2.0
"""HYV4 format/name adapters over the current SlimQuant execution owners."""
from __future__ import annotations

import torch
from vllm.model_executor.layers.linear import (
    LinearBase, MergedColumnParallelLinear, QKVParallelLinear,
    RowParallelLinear, UnquantizedLinearMethod,
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


_INT4_VALUES_PER_BYTE = 2


def _validate_packed_linear_extent(
    layer,
    weight,
    *,
    input_size_per_partition,
    input_size,
    output_size,
    expected_tp_size,
    expected_tp_rank,
    shard_id,
):
    if weight.dtype != torch.uint8 or weight.ndim != 2:
        raise ValueError("HYV4 linear requires rank-2 UINT8 checkpoint packing")

    if type(input_size) is not int or input_size <= 0:
        raise ValueError(f"HYV4 linear weight has invalid input_size: {input_size}")
    if input_size % _INT4_VALUES_PER_BYTE:
        raise ValueError("HYV4 linear weight requires an even checkpoint input size")

    tp_size = getattr(layer, "tp_size", 1)
    tp_rank = getattr(layer, "tp_rank", 0)
    if type(tp_size) is not int or tp_size <= 0:
        raise ValueError(f"HYV4 linear weight has invalid tp_size: {tp_size}")
    if type(tp_rank) is not int or not 0 <= tp_rank < tp_size:
        raise ValueError(f"HYV4 linear weight has invalid tp_rank: {tp_rank}")
    if isinstance(layer, RowParallelLinear) and input_size % tp_size:
        raise ValueError(
            "HYV4 linear weight input_size must be divisible by tp_size"
        )
    if (tp_size, tp_rank) != (expected_tp_size, expected_tp_rank):
        raise ValueError(
            "HYV4 linear weight TP metadata changed after allocation: "
            f"expected size/rank {(expected_tp_size, expected_tp_rank)}, "
            f"got {(tp_size, tp_rank)}"
        )
    expected_partition = (
        input_size // tp_size if isinstance(layer, RowParallelLinear)
        else input_size
    )
    if input_size_per_partition != expected_partition:
        raise ValueError(
            "HYV4 linear weight has inconsistent TP input partition: "
            f"expected {expected_partition}, got {input_size_per_partition}"
        )

    if isinstance(layer, QKVParallelLinear):
        layer.validate_shard_id(shard_id)
        sizes = {
            "q": layer.total_num_heads * layer.head_size,
            "k": layer.total_num_kv_heads * layer.head_size,
            "v": layer.total_num_kv_heads * layer.v_head_size,
        }
        channels = sum(sizes.values()) if shard_id is None else sizes[shard_id]
    elif isinstance(layer, MergedColumnParallelLinear):
        layer.validate_shard_id(shard_id)
        if isinstance(shard_id, tuple) and not shard_id:
            raise ValueError("HYV4 linear weight requires a nonempty shard tuple")
        shards = (
            range(len(layer.output_sizes)) if shard_id is None
            else shard_id if isinstance(shard_id, tuple)
            else (shard_id,)
        )
        channels = sum(layer.output_sizes[index] for index in shards)
    else:
        if shard_id is not None:
            raise ValueError(f"HYV4 linear weight cannot load shard {shard_id!r}")
        channels = output_size

    expected = (channels, input_size // _INT4_VALUES_PER_BYTE)
    if tuple(weight.shape) != expected:
        raise ValueError(
            "HYV4 linear weight requires packed shape "
            f"{expected}, got {tuple(weight.shape)}"
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
        expected_tp_size = getattr(layer, "tp_size", 1)
        expected_tp_rank = getattr(layer, "tp_rank", 0)

        def load_int4(param, weight, *args, **kwargs):
            shard = kwargs.get("loaded_shard_id", args[0] if args else None)
            _validate_packed_linear_extent(
                layer,
                weight,
                input_size_per_partition=input_size_per_partition,
                input_size=input_size,
                output_size=output_size,
                expected_tp_size=expected_tp_size,
                expected_tp_rank=expected_tp_rank,
                shard_id=shard,
            )
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

    Native HYV4 with the explicit AITER backend retains checkpoint scales and
    packed weights for the calibrated INT4 Triton kernel. Other SlimQuant
    routes multiply canonical scales by 16, so normalize those scales while
    loading. supports_eplb remains the current owner's False capability.
    """

    def _uses_native_packed_aiter(self) -> bool:
        return (
            self.quant_config.checkpoint_format == "hy4_w4a8_v1"
            and getattr(self.moe, "moe_backend", "auto") == "aiter"
        )

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
                    if not self._uses_native_packed_aiter():
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

    def process_weights_after_loading(self, layer):
        if not self._uses_native_packed_aiter():
            return super().process_weights_after_loading(layer)
        # Native HYV4 is calibrated for AITER's contiguous packed-INT4
        # Triton kernel. MOE_C shuffling changes that physical contract.
        for name in (
            "w13_weight",
            "w2_weight",
            "w13_weight_scale",
            "w2_weight_scale",
        ):
            parameter = getattr(layer, name, None)
            if not isinstance(parameter, torch.nn.Parameter):
                raise TypeError(f"HYV4 W4A8 requires Parameter {name}")
            parameter.requires_grad_(False)

    def apply(
        self,
        layer,
        x,
        topk_weights,
        topk_ids,
        shared_experts=None,
        shared_experts_input=None,
        **kwargs,
    ):
        if not self._uses_native_packed_aiter():
            return super().apply(
                layer,
                x,
                topk_weights,
                topk_ids,
                shared_experts,
                shared_experts_input,
                **kwargs,
            )
        del shared_experts, shared_experts_input, kwargs
        from aiter.ops.triton.fused_moe import fused_experts_impl
        from vllm_hcu.model_executor.layers.fused_moe.aiter_moe_dispatch import (
            resolve_aiter_expert_maps,
        )

        global_num_experts = getattr(
            layer, "global_num_experts", layer.w13_weight.size(0)
        )
        native_expert_map, _ = resolve_aiter_expert_maps(
            getattr(layer, "expert_map", None), global_num_experts
        )

        return fused_experts_impl(
            x.contiguous(),
            layer.w13_weight,
            layer.w2_weight,
            topk_weights,
            topk_ids,
            output_dtype=x.dtype,
            use_int4_w4a8=True,
            per_channel_quant=True,
            global_num_experts=global_num_experts,
            expert_map=native_expert_map,
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            activation="silu",
            is_gated=True,
            gemm1_limit=getattr(self.moe, "swiglu_limit", 10.0),
            apply_router_weight_on_input=getattr(
                layer, "apply_router_weight_on_input", False
            ),
        )
