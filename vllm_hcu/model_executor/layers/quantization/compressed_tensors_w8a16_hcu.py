# SPDX-License-Identifier: Apache-2.0
# Ported from vllm-hcu-main 70f5373: W8A16 (int8 weight-only, per-channel,
# symmetric) scheme for compressed-tensors checkpoints on HCU. Routes
# pack-quantized int8 checkpoints to torch.ops.vllm.w8a16_gemm (deepgemm /
# dequant fallback) instead of the marlin/exllama MP-linear kernel selection,
# which is broken on HCU (exllama crashes with negative tensor dims).

from __future__ import annotations

import torch
from torch.nn.parameter import Parameter

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme,
)
from vllm.model_executor.parameter import (
    BasevLLMParameter,
    ChannelQuantScaleParameter,
    PackedvLLMParameter,
)

logger = init_logger(__name__)

__all__ = ["CompressedTensorsW8A16Hcu", "is_hcu_w8a16_int8_channel"]


def is_hcu_w8a16_int8_channel(weight_quant, input_quant, format: str | None) -> bool:
    """True for compressed-tensors config parts matching ModelOpt INT8 W8A16:
    pack-quantized, static, per-channel symmetric int8 weights, no input
    quantization."""
    try:
        from compressed_tensors.config import CompressionFormat
        from compressed_tensors.quantization import QuantizationStrategy
    except ImportError:
        return False
    return (
        weight_quant is not None
        and input_quant is None
        and format == CompressionFormat.pack_quantized.value
        and weight_quant.num_bits == 8
        and not weight_quant.dynamic
        and weight_quant.strategy == QuantizationStrategy.CHANNEL.value
        and bool(weight_quant.symmetric)
    )


class CompressedTensorsW8A16Hcu(CompressedTensorsScheme):
    """int8 weight-only scheme backed by torch.ops.vllm.w8a16_gemm.

    Checkpoint layout: ``weight_packed`` int32 [out, in/4] (4 int8 per word,
    little-endian), ``weight_scale`` [out, 1] per-output-channel,
    ``weight_shape`` int64 [2].
    """

    def __init__(self, layer_name: str | None = None) -> None:
        self.layer_name = layer_name

    @classmethod
    def get_min_capability(cls) -> int:
        return 0  # dequant fallback works everywhere; HCU reports non-CUDA caps

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size: int,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        output_size: int,
        params_dtype: torch.dtype,
        weight_loader=None,
        **kwargs,
    ) -> None:
        del output_size
        output_size_per_partition = sum(output_partition_sizes)
        assert input_size_per_partition % 4 == 0, (
            "W8A16 HCU scheme expects input dim divisible by 4 (int32 packing), "
            f"got {input_size_per_partition}"
        )

        # Keep the exact upstream parameter names/types so checkpoint loading
        # (including TP sharding of packed weights) is unchanged.
        weight_packed = PackedvLLMParameter(
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
            packed_factor=4,
            packed_dim=1,
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // 4,
                dtype=torch.int32,
            ),
        )
        weight_scale = ChannelQuantScaleParameter(
            output_dim=0,
            weight_loader=weight_loader,
            data=torch.empty(output_size_per_partition, 1, dtype=params_dtype),
        )
        weight_shape = BasevLLMParameter(
            data=torch.empty(2, dtype=torch.int64), weight_loader=weight_loader
        )

        layer.register_parameter("weight_packed", weight_packed)
        layer.register_parameter("weight_scale", weight_scale)
        layer.register_parameter("weight_shape", weight_shape)
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Unpack int32 [out, in/4] -> int8 [out, in]. Little-endian byte view
        # matches llmcompressor pack-quantized layout; compressed-tensors
        # symmetric int8 is uint8b128 (real = u8 - 128), so flip the high bit
        # to get the two's-complement int8 that w8a16_gemm expects.
        packed = layer.weight_packed.data
        unpacked = (
            (packed.view(torch.uint8) ^ 0x80)
            .view(packed.shape[0], packed.shape[1] * 4)
            .view(torch.int8)
            .contiguous()
        )
        scale = layer.weight_scale.data.reshape(-1).contiguous()

        from vllm_hcu.model_executor.layers.quantization.w8a16_gemm import (
            W8A16_LAYOUT_RAW,
            preshuffle_linear_weight,
        )

        weight, layout = preshuffle_linear_weight(unpacked,
                                                  "compressed-tensors")
        layer.weight_layout = layout

        layer.weight = Parameter(weight, requires_grad=False)
        layer.weight_scale = Parameter(scale, requires_grad=False)
        # Release the packed buffers; the custom op only needs the unpacked
        # int8 weight (registered params cannot be assigned None directly).
        layer.register_parameter("weight_packed", None)
        layer.register_parameter("weight_shape", None)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        # Import guarantees the custom-op registration ordering.
        from vllm_hcu.model_executor.layers.quantization.w8a16_gemm import (
            W8A16_LAYOUT_RAW,
            w8a16_gemm,  # noqa: F401
        )

        return torch.ops.vllm.w8a16_gemm(
            x,
            layer.weight,
            layer.weight_scale,
            bias,
            getattr(layer, "weight_layout", W8A16_LAYOUT_RAW),
        )
