# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Eager TP-only CT channel INT4 / token INT8 method for actual Core033.

Core's RoutedExperts and MoERunner retain loader, router, shared-expert and TP
reduction ownership. Only packed expert arithmetic is Adapter-owned. No marker,
legacy runner, global quant registration, W4A16 conversion, or Core patch.
"""

from copy import deepcopy
import sys

import torch
from torch import nn

from vllm.config import get_current_vllm_config
from vllm.model_executor.layers.fused_moe import FusedMoEMethodBase, RoutedExperts
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from vllm.model_executor.utils import set_weight_attrs

from .quantization import (
    validate_channel_w4a8_config,
    validate_flash_quant_parallelism,
)


def packed_byte_view(words: torch.Tensor) -> torch.Tensor:
    """CT stores eight offset-binary nibbles per little-endian INT32 word.

    The adapter consumes the identical low-nibble-first bytes. In particular,
    do not apply SlimQuant's high-nibble signed packing or x16 compensation.
    """
    if words.dtype != torch.int32 or words.ndim != 3 or not words.is_contiguous():
        raise ValueError("CT expert weights must be contiguous INT32 [E, N, K/8]")
    return words.view(torch.uint8)


def routed_reference(x, weights, ids, w13, w2, s13, s2, *, swiglu_limit=None):
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
        torch.all((ids >= -1) & (ids < experts)), "W4A8 expert ID out of range"
    )
    from hcu_adapter.kernels.w4a8_int8_native_triton import (
        w4a8_int8_native_masked_experts,
    )

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
        result = w4a8_int8_native_masked_experts(
            values, w13, w2, s13, s2, counts, swiglu_limit=swiglu_limit
        )
        active = torch.arange(rows, device=x.device)[None, :] < counts[:, None]
        result = torch.where(active[:, :, None], result, 0)
        token = torch.arange(rows, device=x.device)[:, None]
        valid_routes = local_ids >= 0
        safe_ids = local_ids.clamp_min(0).long()
        slot_output = result[safe_ids, inverse[safe_ids, token]]
        # The router uses -1 for skipped padding. Mask both operands before
        # multiplication: zero times a NaN padding value is still NaN.
        slot_output = torch.where(valid_routes[:, :, None], slot_output, 0)
        local_weights = torch.where(valid_routes, local_weights, 0)
        combined = (slot_output.float() * local_weights[:, :, None]).sum(1)
        output[begin:end].copy_(combined.to(x.dtype))
    return output


class HYV4FlashW4A8Config(CompressedTensorsConfig):
    """Per-model expert override; inherited Core CT owns all other layers."""

    @classmethod
    def for_moe(cls, source):
        if not isinstance(source, CompressedTensorsConfig):
            raise TypeError("Flash W4A8 requires actual Core CompressedTensorsConfig")
        validate_channel_w4a8_config(source.config)
        if (
            source.kv_cache_scheme is not None
            or source.transform_config is not None
            or source.online_quantization_config is not None
        ):
            raise ValueError(
                "Flash W4A8 does not admit transforms/KV/online quantization"
            )
        result = cls(
            target_scheme_map=deepcopy(source.target_scheme_map),
            ignore=deepcopy(source.ignore),
            quant_format=source.quant_format,
            config=deepcopy(source.config),
            total_num_heads=source.total_num_heads,
            total_num_kv_heads=source.total_num_kv_heads,
        )
        result.packed_modules_mapping = deepcopy(source.packed_modules_mapping)
        # Match the real Core CT Linear -> RoutedExperts class-target convention,
        # modifying only this model-local copy and preserving an explicit target.
        if (
            "Linear" in result.target_scheme_map
            and "RoutedExperts" not in result.target_scheme_map
        ):
            result.target_scheme_map["RoutedExperts"] = deepcopy(
                result.target_scheme_map["Linear"]
            )
        return result

    def get_quant_method(self, layer, prefix):
        if not isinstance(layer, RoutedExperts):
            return super().get_quant_method(layer, prefix)
        validate_flash_quant_parallelism(get_current_vllm_config())
        schemes = [
            self.get_scheme_dict(layer, f"{prefix}.{expert}.{projection}")
            for expert in range(layer.moe_config.num_logical_experts)
            for projection in (
                layer.ckpt_gate_proj_name,
                layer.ckpt_up_proj_name,
                layer.ckpt_down_proj_name,
            )
        ]
        if schemes[0] is None or any(s != schemes[0] for s in schemes[1:]):
            raise ValueError(
                "Flash W4A8 requires identical non-ignored expert projection schemes"
            )
        scheme = schemes[0]
        w, a = scheme["weights"], scheme["input_activations"]
        if (
            w.num_bits != 4
            or w.type != "int"
            or w.strategy != "channel"
            or not w.symmetric
            or w.dynamic
            or w.group_size not in (None, -1)
            or a is None
            or a.num_bits != 8
            or a.type != "int"
            or not a.dynamic
            or a.strategy != "token"
            or not a.symmetric
            or a.group_size not in (None, -1)
            or scheme["format"] != "pack-quantized"
            or scheme.get("output_activations") is not None
            or any(
                getattr(q, k, None) is not None
                for q in (w, a)
                for k in ("actorder", "block_structure", "scale_dtype")
            )
        ):
            raise ValueError(
                "Flash W4A8 requires channel INT4 / dynamic token INT8 experts"
            )
        return HYV4FlashChannelW4A8Method(layer.moe_config)


class HYV4FlashChannelW4A8Method(FusedMoEMethodBase):
    """Actual Core modular apply contract; no internal MK/monolithic claim."""

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
        if (
            hidden_size <= 0
            or intermediate_size_per_partition <= 0
            or hidden_size % 8
            or intermediate_size_per_partition % 8
        ):
            raise ValueError(
                "Channel W4A8 TP dimensions must be positive multiples of eight"
            )
        return hidden_size, intermediate_size_per_partition

    def get_fused_moe_quant_config(self, layer):
        # apply does dynamic token INT8 itself. Do not advertise an unrelated
        # FP8/W4A16 MK contract: Core forwards to this method directly.
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
        p = self.moe.moe_parallel_config
        if (
            sys.byteorder != "little"
            or params_dtype != torch.bfloat16
            or self.moe.in_dtype != torch.bfloat16
            or self.moe.ep_size != 1
            or self.moe.dp_size != 1
            or self.moe.pcp_size != 1
            or p.use_ep
            or p.enable_eplb
            or self.moe.is_sequence_parallel
            or self.moe.activation.value != "silu"
            or self.moe.has_bias
            or self.moe.is_lora_enabled
            or self.moe.swiglu_alpha is not None
            or self.moe.swiglu_beta is not None
            or num_experts != self.moe.num_logical_experts
            or layer.apply_router_weight_on_input
        ):
            raise ValueError(
                "Channel W4A8 supports little-endian BF16 SiLU ordinary-TP experts only"
            )
        self.hidden, self.intermediate = hidden_size, intermediate_size_per_partition
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
            # Invalid sentinels make absent shape/scale loads fail closed.
            value = (
                torch.full(
                    shape, -1 if dtype == torch.int64 else float("nan"), dtype=dtype
                )
                if dtype != torch.int32
                else torch.empty(shape, dtype=dtype)
            )
            parameter = nn.Parameter(value, requires_grad=False)
            layer.register_parameter(name, parameter)
            set_weight_attrs(parameter, attrs)
        self._weights_processed = False

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
        for name in ("w13_weight_scale", "w2_weight_scale"):
            scale = getattr(layer, name)
            torch._assert_async(
                torch.all(torch.isfinite(scale) & (scale > 0)),
                "Invalid or unloaded CT expert channel scale",
            )
        packed_byte_view(layer.w13_weight_packed)
        packed_byte_view(layer.w2_weight_packed)
        self._weights_processed = True

    def apply(
        self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input
    ):
        if not self._weights_processed:
            raise RuntimeError("W4A8 weights must be processed after real CT loading")
        if x.dtype != torch.bfloat16 or x.device.type != "cuda":
            raise ValueError("Channel W4A8 apply requires device BF16 activations")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Channel W4A8 model Graph execution is not qualified")
        # Core runs shared experts outside apply and owns final TP reduction.
        # Byte views are recreated, not copied/cached across layer device moves.
        return routed_reference(
            x,
            topk_weights,
            topk_ids,
            packed_byte_view(layer.w13_weight_packed),
            packed_byte_view(layer.w2_weight_packed),
            layer.w13_weight_scale,
            layer.w2_weight_scale,
            swiglu_limit=layer.swiglu_limit,
        )
