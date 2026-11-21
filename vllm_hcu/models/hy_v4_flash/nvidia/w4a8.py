# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Opt-in, eager TP-only CT integer reference lane for HYV4 Flash.

Reuse adapter's native-storage token INT8 quantizer and integer INT4 GEMMs.
No global quantization registration, config mutation, W4A16 conversion, or
changes to installed Core/adapter sources. This bounded, allocation-heavy
reference is a correctness baseline, not a serving performance fast path.
"""

from copy import deepcopy

import torch
from torch import nn

from vllm.model_executor.layers.fused_moe import FusedMoEMethodBase, RoutedExperts
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from vllm.model_executor.utils import set_weight_attrs

from vllm_hcu.models.hy_v4_flash.runtime import _validate_channel_w4a8_config


def packed_byte_view(words: torch.Tensor) -> torch.Tensor:
    """CT stores eight offset-binary nibbles per little-endian INT32 word.

    The adapter consumes the identical low-nibble-first bytes. In particular,
    do not apply SlimQuant's high-nibble signed packing or x16 compensation.
    """
    if words.dtype != torch.int32 or words.ndim != 3 or not words.is_contiguous():
        raise ValueError("CT expert weights must be contiguous INT32 [E, N, K/8]")
    return words.view(torch.uint8)


def routed_reference(
    x, weights, ids, w13, w2, s13, s2, *, swiglu_limit=None, expert_impl=None
):
    """Dispatch in original token order and combine in original router-slot order.

    Scratch capacity is bounded to 32 input rows per local call. Inactive
    expert outputs can be uninitialized: mask them before any gather/combine.
    The adapter owns quantization rounding, two integer dots and the single
    BF16 SiLU-times-up product store. Core still owns TP reduction and shared
    experts; this function must not run either of those itself.
    """
    if (
        x.ndim != 2
        or ids.ndim != 2
        or ids.shape != weights.shape
        or ids.shape[0] != x.shape[0]
    ):
        raise ValueError("W4A8 expects x [M, H] and matching route tensors [M, K]")
    if ids.dtype not in (torch.int32, torch.int64) or weights.dtype != torch.float32:
        raise ValueError("W4A8 requires integer expert IDs and FP32 router weights")
    experts = w13.shape[0]
    if experts < 1 or ids.shape[1] < 1:
        raise ValueError("W4A8 needs nonempty experts and router slots")
    torch._assert_async(
        torch.all((ids >= 0) & (ids < experts)), "W4A8 expert ID out of range"
    )
    if expert_impl is None:
        from hcu_adapter.kernels.w4a8_int8_native_triton import (
            w4a8_int8_native_masked_experts,
        )

        expert_impl = w4a8_int8_native_masked_experts
    output = torch.empty_like(x)
    expert_ids = torch.arange(experts, device=x.device)
    for begin in range(0, x.shape[0], 32):
        end = min(begin + 32, x.shape[0])
        local_ids, local_weights = ids[begin:end], weights[begin:end]
        rows = end - begin
        # A stable sort moves selected tokens to the front of each expert.
        # No .item(), CPU transfer, dynamic nonzero, or data-dependent shape.
        selected = (local_ids[:, :, None] == expert_ids).any(1).T
        counts = selected.sum(1, dtype=torch.int32)
        order = torch.argsort((~selected).to(torch.int32), dim=1, stable=True)
        inverse = torch.argsort(order, dim=1)
        values = x[begin:end][order].contiguous()
        result = expert_impl(
            values, w13, w2, s13, s2, counts, swiglu_limit=swiglu_limit
        )
        active = torch.arange(rows, device=x.device)[None, :] < counts[:, None]
        result = torch.where(active[:, :, None], result, 0)
        token = torch.arange(rows, device=x.device)[:, None]
        slot_output = result[local_ids.long(), inverse[local_ids.long(), token]]
        combined = (slot_output.float() * local_weights[:, :, None]).sum(1)
        output[begin:end].copy_(combined.to(x.dtype))
    return output


class HYV4FlashW4A8Config(CompressedTensorsConfig):
    """A model-local config; dense/DFly linears keep the stock CT selector."""

    @classmethod
    def for_moe(cls, source):
        if not isinstance(source, CompressedTensorsConfig) or source.config is None:
            raise ValueError("Flash W4A8 needs serialized compressed-tensors config")
        _validate_channel_w4a8_config(source.config)
        result = cls(
            target_scheme_map=deepcopy(source.target_scheme_map),
            ignore=source.ignore.copy(),
            quant_format=source.quant_format,
            kv_cache_scheme=None,
            config=deepcopy(source.config),
        )
        result.packed_modules_mapping = deepcopy(source.packed_modules_mapping)
        return result

    def get_quant_method(self, layer, prefix):
        if not isinstance(layer, RoutedExperts):
            return super().get_quant_method(layer, prefix)
        schemes = [
            self.get_scheme_dict(layer, f"{prefix}.0.{projection}")
            for projection in (
                layer.ckpt_gate_proj_name,
                layer.ckpt_up_proj_name,
                layer.ckpt_down_proj_name,
            )
        ]
        if schemes[0] is None or any(s != schemes[0] for s in schemes[1:]):
            raise ValueError(
                "Flash W4A8 requires identical schemes for all three expert projections"
            )
        scheme = schemes[0]
        w, a = scheme["weights"], scheme["input_activations"]
        if (
            w.num_bits != 4
            or w.type != "int"
            or w.strategy != "channel"
            or not w.symmetric
            or w.dynamic
            or a is None
            or a.num_bits != 8
            or a.type != "int"
            or not a.dynamic
            or a.strategy != "token"
            or not a.symmetric
            or scheme["format"] != "pack-quantized"
        ):
            raise ValueError(
                "Flash reference requires channel INT4 / token INT8 experts"
            )
        return HYV4FlashChannelW4A8Method(layer.moe_config)


class HYV4FlashChannelW4A8Method(FusedMoEMethodBase):
    @property
    def skip_forward_padding(self):
        return True

    def maybe_roundup_sizes(
        self,
        hidden_size,
        intermediate_size_per_partition,
        act_dtype,
        moe_parallel_config,
    ):
        if hidden_size % 8 or intermediate_size_per_partition % 8:
            raise ValueError("Channel W4A8 TP dimensions must be divisible by eight")
        return hidden_size, intermediate_size_per_partition

    def get_fused_moe_quant_config(self, layer):
        # This legacy apply boundary does its own activation quantization.
        # Do not advertise an unrelated FP8/W4A16 modular-kernel contract.
        return None

    def create_weights(
        self,
        layer,
        num_experts,
        hidden_size,
        intermediate_size_per_partition,
        params_dtype,
        **extra_weight_attrs,
    ):
        if (
            params_dtype != torch.bfloat16
            or self.moe.ep_size != 1
            or self.moe.dp_size != 1
            or self.moe.is_sequence_parallel
            or self.moe.activation.value != "silu"
        ):
            raise ValueError(
                "Channel W4A8 reference supports BF16 SiLU TP-only experts"
            )
        self.hidden = hidden_size
        self.intermediate = intermediate_size_per_partition
        attrs = dict(extra_weight_attrs, is_transposed=False, quant_method="channel")
        shapes = {
            "w13_weight_packed": (
                (num_experts, 2 * self.intermediate, hidden_size // 8),
                torch.int32,
            ),
            "w2_weight_packed": (
                (num_experts, hidden_size, self.intermediate // 8),
                torch.int32,
            ),
            "w13_weight_scale": ((num_experts, 2 * self.intermediate, 1), params_dtype),
            "w2_weight_scale": ((num_experts, hidden_size, 1), params_dtype),
            "w13_weight_shape": ((num_experts, 2), torch.int64),
            "w2_weight_shape": ((num_experts, 2), torch.int64),
        }
        for name, (shape, dtype) in shapes.items():
            parameter = nn.Parameter(
                torch.empty(shape, dtype=dtype), requires_grad=False
            )
            layer.register_parameter(name, parameter)
            set_weight_attrs(parameter, attrs)

    def process_weights_after_loading(self, layer):
        global_i = self.intermediate * self.moe.tp_size
        for name, expected in (
            ("w13_weight_shape", (global_i, self.hidden)),
            ("w2_weight_shape", (self.hidden, global_i)),
        ):
            value = getattr(layer, name)
            target = torch.tensor(expected, device=value.device, dtype=value.dtype)
            torch._assert_async(
                torch.all(value == target), "Invalid CT expert weight_shape metadata"
            )
        # Byte views alias the original Parameters: no extra full-model copy.
        self.w13 = packed_byte_view(layer.w13_weight_packed)
        self.w2 = packed_byte_view(layer.w2_weight_packed)

    def apply(
        self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input
    ):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "Channel W4A8 reference has not qualified model Graph execution"
            )
        return routed_reference(
            x,
            topk_weights,
            topk_ids,
            self.w13,
            self.w2,
            layer.w13_weight_scale,
            layer.w2_weight_scale,
            swiglu_limit=getattr(layer, "swiglu_limit", None),
        )
