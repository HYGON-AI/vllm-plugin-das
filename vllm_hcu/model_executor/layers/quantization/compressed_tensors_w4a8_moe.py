# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""HCU execution owner for compressed-tensors channel-wise W4A8 MoE."""

from __future__ import annotations

import torch
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
)
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig

from .slimquant_w4a8 import SlimQuantW4A8Int8AiterMoEMethod

HCU_COMPRESSED_W4A8_METHOD_MARKER = (
    "_vllm_hcu_compressed_tensors_w4a8_int8_moe"
)


def repack_compressed_tensors_int4(weight: torch.Tensor) -> torch.Tensor:
    """Convert offset-packed INT32 storage to signed high-first nibbles."""
    if weight.dtype != torch.int32 or weight.ndim != 2:
        raise ValueError(
            "compressed-tensors expert weight_packed must be rank-2 INT32"
        )
    # compressed-tensors stores eight offset-binary nibbles per INT32 in
    # little-endian order.  AITER expects two two's-complement nibbles per
    # byte, high nibble first.  Convert four pairs directly and avoid an 8x
    # INT32 intermediate allocation during large PP checkpoint loads.
    pairs = []
    for shift in (0, 8, 16, 24):
        high = (((weight >> shift) & 0xF) ^ 0x8) << 4
        low = ((weight >> (shift + 4)) & 0xF) ^ 0x8
        pairs.append((high | low).to(torch.uint8))
    return torch.stack(pairs, dim=-1).flatten(-2).view(torch.int8)


class HcuCompressedTensorsW4A8Int8MoEMethod(
    SlimQuantW4A8Int8AiterMoEMethod
):
    """Run packed compressed-tensors W4A8 experts with the HCU MoE path."""

    _vllm_hcu_compressed_tensors_w4a8_int8_moe = True

    def __init__(
        self,
        weight_quant: QuantizationArgs,
        input_quant: QuantizationArgs,
        moe: FusedMoEConfig,
        layer_name: str | None = None,
    ) -> None:
        if (
            weight_quant.num_bits != 4
            or weight_quant.type != QuantizationType.INT
            or weight_quant.strategy != QuantizationStrategy.CHANNEL
            or not weight_quant.symmetric
            or weight_quant.dynamic
            or weight_quant.group_size not in (None, -1)
        ):
            raise ValueError(
                "HCU compressed-tensors W4A8 MoE requires static, symmetric "
                "channel-wise INT4 weights"
            )
        if (
            input_quant.num_bits != 8
            or input_quant.type != QuantizationType.INT
            or input_quant.strategy != QuantizationStrategy.TOKEN
            or not input_quant.symmetric
            or not input_quant.dynamic
        ):
            raise ValueError(
                "HCU compressed-tensors W4A8 MoE requires dynamic per-token "
                "INT8 activations"
            )
        if getattr(moe, "has_bias", False):
            raise ValueError("HCU compressed-tensors W4A8 MoE does not support bias")
        self.weight_quant = weight_quant
        self.input_quant = input_quant
        self.layer_name = layer_name
        super().__init__(quant_config=None, moe=moe)

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        super().create_weights(
            layer,
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            params_dtype,
            **extra_weight_attrs,
        )

        def logical_dimensions():
            dimensions = []
            for field, allocated in (
                ("hidden_dim_unpadded", hidden_size),
                (
                    "intermediate_size_per_partition_unpadded",
                    intermediate_size_per_partition,
                ),
            ):
                logical = getattr(layer.moe_config, field, allocated)
                if type(logical) is not int or not 0 < logical <= allocated:
                    raise ValueError(
                        "compressed-tensors expert has invalid "
                        f"{field}: {logical}"
                    )
                dimensions.append(logical)
            return dimensions

        def shard_id(args, kwargs):
            return kwargs.get("shard_id", args[1] if len(args) > 1 else None)

        def wrap_weight(parameter, name):
            load = parameter.weight_loader

            def load_packed(param, loaded_weight, *args, **kwargs):
                shard = shard_id(args, kwargs)
                down = name == "w2_weight"
                if shard not in (("w2",) if down else ("w1", "w3")):
                    raise ValueError(
                        f"compressed-tensors expert {name} cannot load {shard}"
                    )
                hidden, intermediate = logical_dimensions()
                tp = layer.moe_config.moe_parallel_config.tp_size
                if type(tp) is not int or tp <= 0:
                    raise ValueError(
                        "compressed-tensors expert has invalid "
                        f"tp_size: {tp}"
                    )
                input_size = intermediate * tp if down else hidden
                if input_size % 8:
                    raise ValueError(
                        "compressed-tensors expert INT4 input dimension must "
                        "be divisible by 8"
                    )
                expected = (
                    (hidden, input_size // 8)
                    if down
                    else (intermediate * tp, input_size // 8)
                )
                if (
                    loaded_weight.dtype != torch.int32
                    or tuple(loaded_weight.shape) != expected
                ):
                    raise ValueError(
                        f"compressed-tensors expert {shard} requires INT32 "
                        f"shape {expected}, got {loaded_weight.dtype} "
                        f"{tuple(loaded_weight.shape)}"
                    )
                loaded_weight = repack_compressed_tensors_int4(loaded_weight)
                return load(param, loaded_weight, *args, **kwargs)

            load_packed.__self__ = getattr(load, "__self__", None)
            parameter.weight_loader = load_packed

        def wrap_scale(parameter, name):
            load = parameter.weight_loader

            def load_scale(param, loaded_weight, *args, **kwargs):
                shard = shard_id(args, kwargs)
                down = name == "w2_weight_scale"
                if shard not in (("w2",) if down else ("w1", "w3")):
                    raise ValueError(
                        f"compressed-tensors expert scale {name} cannot load "
                        f"{shard}"
                    )
                hidden, intermediate = logical_dimensions()
                tp = layer.moe_config.moe_parallel_config.tp_size
                if type(tp) is not int or tp <= 0:
                    raise ValueError(
                        "compressed-tensors expert scale has invalid "
                        f"tp_size: {tp}"
                    )
                channels = hidden if down else intermediate * tp
                expected = (channels, 1)
                if tuple(loaded_weight.shape) != expected:
                    raise ValueError(
                        "compressed-tensors expert scale requires shape "
                        f"{expected}, got {tuple(loaded_weight.shape)}"
                    )
                if not loaded_weight.numel() or not torch.all(
                    torch.isfinite(loaded_weight) & (loaded_weight > 0)
                ).item():
                    raise ValueError(
                        "compressed-tensors expert scale must be finite and "
                        "positive"
                    )
                return load(param, loaded_weight / 16.0, *args, **kwargs)

            load_scale.__self__ = getattr(load, "__self__", None)
            parameter.weight_loader = load_scale

        wrap_weight(layer.w13_weight, "w13_weight")
        wrap_weight(layer.w2_weight, "w2_weight")
        wrap_scale(layer.w13_weight_scale, "w13_weight_scale")
        wrap_scale(layer.w2_weight_scale, "w2_weight_scale")


__all__ = [
    "HCU_COMPRESSED_W4A8_METHOD_MARKER",
    "HcuCompressedTensorsW4A8Int8MoEMethod",
    "repack_compressed_tensors_int4",
]
