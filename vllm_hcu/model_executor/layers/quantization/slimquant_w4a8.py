# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from typing import Any

import torch
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from compressed_tensors.quantization import QuantizationStrategy
from torch.nn.parameter import Parameter
from vllm.model_executor.layers.fused_moe import (
    FusedMoE,
    FusedMoEMethodBase,
    FusedMoeWeightScaleSupported,
)
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsW8A8Int8,
)
from vllm.model_executor.utils import set_weight_attrs

try:
    import aiter.ops.triton.fused_moe as aiter_triton_fused_moe
except ImportError as exc:
    aiter_triton_fused_moe = None
    _AITER_TRITON_IMPORT_ERROR: ImportError | None = exc
else:
    _AITER_TRITON_IMPORT_ERROR = None


class SlimQuantW4A8Int8Config(QuantizationConfig):
    """SlimQuant W4A8 configuration.

    Weights are static, symmetric per-channel INT4 values packed two per byte.
    Activations are dynamically quantized per token to INT8.
    """

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 75

    @classmethod
    def get_name(cls) -> str:
        return "slimquant_w4a8"

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "SlimQuantW4A8Int8Config":
        return cls()

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> QuantizeMethodBase | None:
        if isinstance(layer, LinearBase):
            layer.scheme = CompressedTensorsW8A8Int8(
                QuantizationStrategy.CHANNEL, False, True
            )
            return SlimQuantW4A8Int8LinearMethod(self)
        if isinstance(layer, FusedMoE):
            return SlimQuantW4A8Int8AiterMoEMethod(self, layer.moe_config)
        return None

    def get_scaled_act_names(self) -> list[str]:
        return []


class SlimQuantW4A8Int8LinearMethod(LinearMethodBase):
    def __init__(self, quantization_config: SlimQuantW4A8Int8Config):
        self.quantization_config = quantization_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        """Delegate parameter creation to the layer's compressed scheme."""
        weight_loader = extra_weight_attrs.get("weight_loader")
        layer.scheme.create_weights(
            layer=layer,
            input_size=input_size,
            input_size_per_partition=input_size_per_partition,
            output_partition_sizes=output_partition_sizes,
            output_size=output_size,
            params_dtype=params_dtype,
            weight_loader=weight_loader,
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.scheme.process_weights_after_loading(layer)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ):
        """Apply the layer's compressed linear scheme."""
        scheme = layer.scheme
        if scheme is None:
            raise ValueError("A scheme must be defined for each layer")
        return scheme.apply_weights(layer, x, bias=bias)


class SlimQuantW4A8Int8AiterMoEMethod(FusedMoEMethodBase):
    """Channel-wise AITER Triton MoE for raw packed SlimQuant W4A8 weights."""

    def __init__(self, quant_config, moe):
        self.moe = moe
        self.quant_config = quant_config
        self.moe_quant_config: FusedMoEQuantConfig | None = None
        self.moe_kernel: mk.FusedMoEKernel | None = None

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig:
        self.moe_quant_config = FusedMoEQuantConfig.make(
            torch.int8,
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            a1_scale=layer.w13_input_scale,
            a2_scale=layer.w2_input_scale,
            per_act_token_quant=True,
            per_out_ch_quant=False,
            block_shape=None,
            weight_dtype="int4",
        )
        return self.moe_quant_config

    @property
    def is_monolithic(self) -> bool:
        return False

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        intermediate_size = intermediate_size_per_partition
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size,
                hidden_size // 2,
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size // 2,
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        w13_weight_scale = torch.nn.Parameter(
            torch.ones(num_experts, 2 * intermediate_size, 1, dtype=torch.float32),
            requires_grad=False,
        )
        w2_weight_scale = torch.nn.Parameter(
            torch.ones(num_experts, hidden_size, 1, dtype=torch.float32),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        layer.register_parameter("w2_weight_scale", w2_weight_scale)

        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.CHANNEL.value}
        )

        set_weight_attrs(w13_weight_scale, extra_weight_attrs)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        w13_input_scale = None
        layer.register_parameter("w13_input_scale", w13_input_scale)

        w2_input_scale = None
        layer.register_parameter("w2_input_scale", w2_input_scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Keep SlimQuant W4A8 weights in the raw packed checkpoint layout.
        # AITER's channel-wise W4A8 Triton path consumes the raw [E, N, K // 2]
        # / [E, K, N // 2] tensors directly.  Pre-shuffling here routes the
        # weights into the MOE_C/Marlin layout and produces incorrect values
        # for this path.
        for name in (
            "w13_weight",
            "w2_weight",
            "w13_weight_scale",
            "w2_weight_scale",
        ):
            parameter = getattr(layer, name, None)
            if not isinstance(parameter, Parameter):
                raise TypeError(f"SlimQuant W4A8 requires Parameter {name}")
            parameter.requires_grad_(False)

    def apply(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
        **_,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:

        if x.dim() != 2:
            raise ValueError(
                "SlimQuant W4A8 AITER Triton MoE expects rank-2 hidden states, "
                f"got shape {tuple(x.shape)}"
            )
        if shared_experts_input is not None:
            raise ValueError(
                "SlimQuant W4A8 Triton MoE does not support shared_experts_input"
            )
        if topk_weights.shape != topk_ids.shape:
            raise ValueError(
                "SlimQuant W4A8 requires topk_weights and topk_ids with the "
                f"same shape, got {tuple(topk_weights.shape)} and "
                f"{tuple(topk_ids.shape)}"
            )
        if topk_ids.dim() != 2 or topk_ids.shape[0] != x.shape[0]:
            raise ValueError(
                "SlimQuant W4A8 requires rank-2 top-k tensors with the same "
                f"token count as x, got x={tuple(x.shape)}, "
                f"topk_ids={tuple(topk_ids.shape)}"
            )
        if aiter_triton_fused_moe is None:
            raise RuntimeError(
                "SlimQuant W4A8 requires aiter.ops.triton.fused_moe"
            ) from _AITER_TRITON_IMPORT_ERROR

        return aiter_triton_fused_moe.fused_experts_impl(
            x,
            layer.w13_weight,
            layer.w2_weight,
            topk_weights,
            topk_ids,
            output_dtype=x.dtype,
            use_int4_w4a8=True,
            per_channel_quant=True,
            global_num_experts=layer.w13_weight.size(0),
            expert_map=None,
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            activation="silu",
            is_gated=True,
        )
