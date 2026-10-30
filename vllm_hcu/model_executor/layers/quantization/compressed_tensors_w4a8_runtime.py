# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Bridge channel-wise compressed-tensors INT4 checkpoints to HCU W4A8.

The checkpoint loader stays vLLM-owned. Only its post-load representation and
execution are adapted to the existing HCU dynamic per-token INT8 runtime.
"""

from __future__ import annotations

from types import MethodType


def _token(value: object) -> str:
    return str(getattr(value, "value", value)).lower()


def supports_channel_w4a8(weight: object, activation: object, moe: object) -> bool:
    """Accept only the audited symmetric channel/token INT4/INT8 contract."""
    return bool(
        getattr(weight, "num_bits", None) == 4
        and _token(getattr(weight, "type", None)) == "int"
        and getattr(weight, "symmetric", False)
        and not getattr(weight, "dynamic", True)
        and _token(getattr(weight, "strategy", None)) == "channel"
        and getattr(weight, "group_size", None) in (None, -1)
        and getattr(weight, "actorder", None) is None
        and getattr(activation, "num_bits", None) == 8
        and _token(getattr(activation, "type", None)) == "int"
        and getattr(activation, "symmetric", False)
        and getattr(activation, "dynamic", False)
        and _token(getattr(activation, "strategy", None)) == "token"
        and getattr(activation, "group_size", None) in (None, -1)
        and not getattr(moe, "has_bias", False)
        and _token(getattr(moe, "activation", None)) == "silu"
    )


def repack_ct_int4(packed, *, transposed: bool):
    """Convert CT offset-binary INT32 to HCU signed, high-nibble-first INT8."""
    import torch

    if packed.dtype != torch.int32 or packed.ndim != 3:
        raise ValueError("HCU CT W4A8 expects rank-3 packed INT32 weights")
    words = packed.transpose(1, 2).contiguous() if transposed else packed
    signed = torch.stack(
        [(((words >> (4 * i)) & 15) - 8).to(torch.int8) for i in range(8)],
        dim=-1,
    ).flatten(-2)
    high = (signed[..., 0::2].to(torch.int16) & 15) << 4
    low = signed[..., 1::2].to(torch.int16) & 15
    return (high | low).to(torch.uint8).view(torch.int8).contiguous()


def _create_weights(
    self,
    layer,
    num_experts,
    hidden_size,
    intermediate_size_per_partition,
    params_dtype,
    **extra_weight_attrs,
):
    result = self._hcu_ct_w4a8_create_weights(
        layer,
        num_experts,
        hidden_size,
        intermediate_size_per_partition,
        params_dtype,
        **extra_weight_attrs,
    )
    layer.w13_input_scale = None
    layer.w2_input_scale = None
    return result


def _process_weights_after_loading(self, layer) -> None:
    from vllm.model_executor.utils import replace_parameter

    if getattr(self, "_hcu_ct_w4a8_loaded", False):
        raise RuntimeError("HCU CT W4A8 post-load conversion must run only once")
    for prefix in ("w13", "w2"):
        packed = getattr(layer, prefix + "_weight_packed")
        converted = repack_ct_int4(packed, transposed=self.is_transposed)
        replace_parameter(layer, prefix + "_weight_packed", converted)
        setattr(
            layer, prefix + "_weight", getattr(layer, prefix + "_weight_packed")
        )
        scale = getattr(layer, prefix + "_weight_scale")
        if self.is_transposed:
            scale = scale.transpose(1, 2)
        # SlimQuant's canonical scale represents the signed high-nibble q*16
        # domain. AITER handles that domain; the Triton fallback restores x16.
        replace_parameter(
            layer, prefix + "_weight_scale", scale.float().contiguous() / 16.0
        )
        setattr(layer, prefix + "_input_scale", None)

    delegate = self._hcu_ct_w4a8_delegate
    delegate.process_weights_after_loading(layer)
    # AITER can repack and Triton can expand weights during post-load. Keep
    # loader-name aliases on that final storage, without retaining old buffers.
    for prefix in ("w13", "w2"):
        setattr(
            layer, prefix + "_weight_packed", getattr(layer, prefix + "_weight")
        )
    self.moe_quant_config = delegate.moe_quant_config
    self.moe_kernel = delegate.moe_kernel
    self._hcu_ct_w4a8_loaded = True


def _get_fused_moe_quant_config(self, layer):
    return self._hcu_ct_w4a8_delegate.get_fused_moe_quant_config(layer)


def _apply(
    self,
    layer,
    x,
    topk_weights,
    topk_ids,
    shared_experts=None,
    shared_experts_input=None,
    **kwargs,
):
    if not getattr(self, "_hcu_ct_w4a8_loaded", False):
        raise RuntimeError("HCU CT W4A8 execution requires completed weight post-load")
    return self._hcu_ct_w4a8_delegate.apply(
        layer,
        x,
        topk_weights,
        topk_ids,
        shared_experts,
        shared_experts_input,
        **kwargs,
    )



def make_hcu_w4a8_method(weight_quant, input_quant, moe, layer_name=None):
    """Reuse the audited CT loader, but execute dynamic INT8 HCU MoE."""
    if not supports_channel_w4a8(weight_quant, input_quant, moe):
        raise ValueError("Unsupported HCU compressed-tensors W4A8 quantization scheme")
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_wna16 import (
        CompressedTensorsWNA16MoEMethod,
    )
    from vllm_hcu.model_executor.layers.quantization.slimquant_w4a8 import (
        SlimQuantW4A8Int8AiterMoEMethod,
    )

    # Preserve the loader's exact class identity: RoutedExperts uses it to
    # transpose CT checkpoint tensors and provide intermediate_size_full.
    # None is used only to construct storage; it does not disable A8 execution.
    method = CompressedTensorsWNA16MoEMethod(weight_quant, None, moe, layer_name)
    method.input_quant = input_quant
    method._hcu_ct_w4a8_delegate = SlimQuantW4A8Int8AiterMoEMethod(None, moe)
    method._hcu_ct_w4a8_create_weights = method.create_weights
    method.create_weights = MethodType(_create_weights, method)
    method.process_weights_after_loading = MethodType(
        _process_weights_after_loading, method
    )
    method.get_fused_moe_quant_config = MethodType(
        _get_fused_moe_quant_config, method
    )
    method.apply = MethodType(_apply, method)
    return method


__all__ = ["supports_channel_w4a8", "repack_ct_int4", "make_hcu_w4a8_method"]
