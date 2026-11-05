# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Bridge channel-wise compressed-tensors INT4 checkpoints to HCU W4A8.

The checkpoint loader stays vLLM-owned. Only its post-load representation and
execution are adapted to the existing HCU dynamic per-token INT8 runtime.
"""

from __future__ import annotations

from copy import copy
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


_REPACK_MAX_WORDS = 1 << 20


def repack_ct_int4(packed, *, transposed: bool):
    """Convert CT offset-binary INT32 to HCU signed, high-nibble-first INT8."""
    import torch

    if packed.dtype != torch.int32 or packed.ndim != 3:
        raise ValueError("HCU CT W4A8 expects rank-3 packed INT32 weights")
    words = packed.transpose(1, 2) if transposed else packed
    experts, rows, columns = words.shape
    result = torch.empty((experts, rows, columns * 4), dtype=torch.int8, device=packed.device)
    if not packed.numel():
        return result
    column_step = min(columns, _REPACK_MAX_WORDS)
    row_step = max(1, _REPACK_MAX_WORDS // column_step)
    for expert in range(experts):
        for row in range(0, rows, row_step):
            stop_row = min(row + row_step, rows)
            for column in range(0, columns, column_step):
                stop_column = min(column + column_step, columns)
                block = words[expert, row:stop_row, column:stop_column]
                output = result[expert, row:stop_row, column * 4:stop_column * 4].view(
                    stop_row - row, stop_column - column, 4,
                )
                for byte in range(4):
                    # CT stores offset-binary low nibble first. Swap the pair
                    # and flip each sign bit directly, without unpacking INT4.
                    high = ((block >> (byte * 8)) & 15) << 4
                    low = (block >> (byte * 8 + 4)) & 15
                    output[..., byte].copy_(((high | low) ^ 0x88).to(torch.uint8).view(torch.int8))
    return result


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
    import torch
    from vllm.model_executor.utils import replace_parameter

    # Native checkpoint reload restores INT32 checkpoint Parameters without
    # recreating this method. A lifetime flag would reject every later reload.
    # Validate both projections before mutation: runtime INT8 aliases must never
    # be repacked or have their scales divided by 16 a second time.
    self._hcu_ct_w4a8_loaded = False
    self.supports_int8_allgatherv_input = False
    for prefix in ("w13", "w2"):
        packed = getattr(layer, prefix + "_weight_packed", None)
        if (
            not isinstance(packed, torch.Tensor)
            or packed.dtype != torch.int32
            or packed.ndim != 3
        ):
            raise RuntimeError(
                "HCU CT W4A8 post-load requires fresh rank-3 packed INT32 "
                f"checkpoint weights: {prefix}_weight_packed"
            )
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
    source_scales = (layer.w13_weight_scale, layer.w2_weight_scale)
    previous_config = getattr(delegate, "moe_quant_config", None)
    delegate.process_weights_after_loading(layer)
    from vllm.model_executor.model_loader.reload.layerwise import get_layerwise_info
    from vllm_hcu.model_executor.layers.fused_moe.aiter_moe_dispatch import (
        refresh_aiter_moe_scale_cache,
    )

    kernel_tensors = get_layerwise_info(layer).kernel_tensors
    if kernel_tensors is not None:
        # Native reload will restore these same Parameters after post-load.
        # Captured graphs may already point at their cached shuffled copies:
        # update those allocations now, before any Python-free graph replay.
        parameters, _ = kernel_tensors
        for owner in (
            parameters.get("w13_weight_scale"),
            parameters.get("w2_weight_scale"),
            previous_config,
        ):
            refresh_aiter_moe_scale_cache(owner, *source_scales)
    # AITER can repack and Triton can expand weights during post-load. Keep
    # loader-name aliases on that final storage, without retaining old buffers.
    for prefix in ("w13", "w2"):
        setattr(
            layer, prefix + "_weight_packed", getattr(layer, prefix + "_weight")
        )
    self.moe_quant_config = delegate.moe_quant_config
    self.moe_kernel = delegate.moe_kernel
    self.supports_int8_allgatherv_input = bool(
        getattr(delegate, "supports_int8_allgatherv_input", False)
    )
    self._hcu_ct_w4a8_loaded = True


def refresh_hcu_w4a8_scale_caches(model) -> None:
    """Refresh captured CT W4A8 scales after all kernel-format writes finish."""
    if model is None:
        return
    from vllm_hcu.model_executor.layers.fused_moe.aiter_moe_dispatch import (
        refresh_aiter_moe_scale_cache,
    )

    for layer in model.modules():
        method = getattr(layer, "quant_method", None)
        # Modular MoE wrappers retain the loader method under old_quant_method.
        seen = set()
        while method is not None and id(method) not in seen:
            seen.add(id(method))
            if hasattr(method, "_hcu_ct_w4a8_delegate"):
                break
            method = getattr(method, "old_quant_method", None)
        delegate = getattr(method, "_hcu_ct_w4a8_delegate", None)
        if delegate is None or not getattr(method, "_hcu_ct_w4a8_loaded", False):
            continue
        scales = layer.w13_weight_scale, layer.w2_weight_scale
        owners = (*scales, getattr(delegate, "moe_quant_config", None),
                  getattr(method, "moe_quant_config", None))
        seen.clear()
        for owner in owners:
            if owner is not None and id(owner) not in seen:
                seen.add(id(owner))
                refresh_aiter_moe_scale_cache(owner, *scales, update_generation=True)


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
    from vllm_hcu.model_executor.layers.quantization.compressed_tensors_moe_runtime import (
        _replace_tensor_reference,
    )

    # Native checkpoint reload restores persistent kernel Parameters *after*
    # post-load. Bind here so eager execution and subsequent graph capture use
    # those final owners, including later in-place kernel-format updates.
    # Retain the config/descriptors also held by modular kernel consumers.
    delegate = self._hcu_ct_w4a8_delegate
    config = delegate.moe_quant_config
    _replace_tensor_reference(config, "w1_scale", layer.w13_weight_scale)
    _replace_tensor_reference(config, "w2_scale", layer.w2_weight_scale)
    self.moe_quant_config = config
    return delegate.apply(
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
    parallel = getattr(moe, "moe_parallel_config", None)
    if getattr(parallel, "enable_eplb", False):
        raise NotImplementedError("HCU CT W4A8 does not support EPLB")
    if getattr(parallel, "all2all_backend", None) in (
        "deepep_auto", "deepep_high_throughput", "deepep_low_latency",
    ) or getattr(parallel, "use_deepep_auto_kernels", False):
        raise NotImplementedError(
            "HCU CT W4A8 does not support DeepEP; use the AITER expert path "
            "with allgather_reducescatter for DP+EP"
        )
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_wna16 import (
        CompressedTensorsWNA16MoEMethod,
    )
    from vllm_hcu.model_executor.layers.quantization.slimquant_w4a8 import (
        SlimQuantW4A8Int8AiterMoEMethod,
    )

    # Omitted channel group_size and -1 mean the same thing. Canonicalize a
    # private descriptor before WNA16 construction without changing the config
    # shared by other layers (native create_weights also normalizes this).
    if getattr(weight_quant, "group_size", None) is None:
        weight_quant = copy(weight_quant)
        weight_quant.group_size = -1

    # Preserve the loader's exact class identity: RoutedExperts uses it to
    # transpose CT checkpoint tensors and provide intermediate_size_full.
    # None is used only to construct storage; it does not disable A8 execution.
    method = CompressedTensorsWNA16MoEMethod(weight_quant, None, moe, layer_name)
    method.input_quant = input_quant
    method._hcu_ct_w4a8_delegate = SlimQuantW4A8Int8AiterMoEMethod(None, moe)
    method.supports_int8_allgatherv_input = False
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
