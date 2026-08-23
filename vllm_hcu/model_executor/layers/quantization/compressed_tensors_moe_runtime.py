# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""HCU-owned compressed-tensors MoE runtime helpers.

The vLLM-facing adapters only validate and wrap the audited target methods.
All AITER configuration, weight-layout, and zero-point behavior lives here so
that no vLLM source file needs to be rewritten.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch


class HcuCompressedTensorsMoeError(RuntimeError):
    """An explicitly selected HCU compressed-tensors MoE path is invalid."""


def _required_tensor(owner: object, name: str) -> torch.Tensor:
    value = getattr(owner, name, None)
    if not isinstance(value, torch.Tensor):
        raise HcuCompressedTensorsMoeError(
            f"HCU compressed-tensors MoE requires tensor {name!r}"
        )
    return value


def _enum_token(value: object) -> str:
    """Normalize an Enum, string, or enum-like object for cross-patch use."""

    candidates = (
        getattr(value, "name", None),
        getattr(value, "value", None),
        value,
    )
    for candidate in candidates:
        if candidate is None:
            continue
        token = str(candidate).rsplit(".", 1)[-1].upper()
        if token:
            return token
    return ""


def _tensor_generation(tensor: torch.Tensor) -> tuple[object, ...]:
    """Identity used to invalidate cached shuffled weights after reloads."""

    return (
        id(tensor),
        getattr(tensor, "_version", None),
        tuple(tensor.shape),
        tensor.dtype,
        tensor.device,
    )


def _tensor_cache(
    tensor: torch.Tensor,
    name: str,
) -> dict[tuple[object, ...], object]:
    cache = getattr(tensor, name, None)
    if cache is None:
        cache = {}
        setattr(tensor, name, cache)
    if not isinstance(cache, dict):
        raise HcuCompressedTensorsMoeError(
            f"AITER quantized MoE tensor cache {name!r} has an invalid type"
        )
    return cache


def _get_aiter_quantized_runtime_config(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_ids: torch.Tensor,
    quant_type: object,
    activation: str,
) -> object:
    try:
        from aiter.moe import get_aiter_moe_config
    except Exception as exc:
        raise HcuCompressedTensorsMoeError(
            "AITER quantized MoE is selected, but get_aiter_moe_config is "
            "unavailable"
        ) from exc

    cache = _tensor_cache(w1, "_hcu_aiter_quantized_config_cache")
    cache_key = (
        hidden_states.shape[0],
        w1.shape[0],
        w1.shape[1],
        w2.shape[1],
        w1.shape[2],
        topk_ids.shape[1],
        hidden_states.dtype,
        hidden_states.device,
        _enum_token(quant_type),
        activation,
    )
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    status, aiter_config = get_aiter_moe_config(
        M=hidden_states.shape[0],
        E=w1.shape[0],
        N1=w1.shape[1],
        N2=w2.shape[1],
        K=w1.shape[2],
        top_k=topk_ids.shape[1],
        block_size=0,
        dtype=hidden_states.dtype,
        quant_type=quant_type,
        activation=activation,
    )
    if not status or aiter_config is None:
        raise HcuCompressedTensorsMoeError(
            "AITER quantized MoE found no backend config for "
            f"M={hidden_states.shape[0]}, E={w1.shape[0]}, "
            f"top_k={topk_ids.shape[1]}, dtype={hidden_states.dtype}, "
            f"quant_type={quant_type}"
        )
    if len(cache) >= 128:
        cache.clear()
    cache[cache_key] = aiter_config
    return aiter_config


def _get_aiter_quantized_weights(
    w1: torch.Tensor,
    w2: torch.Tensor,
    aiter_config: object,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not bool(getattr(aiter_config, "need_shuffle", False)):
        return w1, w2

    cache = _tensor_cache(w1, "_hcu_aiter_quantized_weight_cache")
    cache_key = (
        _tensor_generation(w1),
        _tensor_generation(w2),
        _enum_token(getattr(aiter_config, "quant_type", None)),
        _enum_token(getattr(aiter_config, "solution_type", None)),
    )
    cached = cache.get(cache_key)
    if (
        isinstance(cached, tuple)
        and len(cached) == 2
        and isinstance(cached[0], torch.Tensor)
        and isinstance(cached[1], torch.Tensor)
    ):
        return cached

    try:
        from aiter.moe import aiter_moe_shfl_weight
    except Exception as exc:
        raise HcuCompressedTensorsMoeError(
            "AITER selected a shuffled quantized MoE solution, but "
            "aiter_moe_shfl_weight is unavailable"
        ) from exc
    with torch.no_grad():
        shuffled_w1, shuffled_w2 = aiter_moe_shfl_weight(w1, w2, aiter_config)
    if not isinstance(shuffled_w1, torch.Tensor) or not isinstance(
        shuffled_w2, torch.Tensor
    ):
        raise HcuCompressedTensorsMoeError(
            "AITER returned missing shuffled quantized MoE weights"
        )
    if shuffled_w1.shape != w1.shape or shuffled_w2.shape != w2.shape:
        raise HcuCompressedTensorsMoeError(
            "AITER returned incompatible shuffled quantized MoE weight shapes"
        )
    if len(cache) >= 8:
        cache.clear()
    cache[cache_key] = (shuffled_w1, shuffled_w2)
    return shuffled_w1, shuffled_w2


def apply_aiter_quantized_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    vllm_moe_config: object,
    activation: object,
    apply_router_weight_on_input: bool,
    expert_map: torch.Tensor | None,
    quant_config: object,
    a1q_scale: torch.Tensor | None = None,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Run v0.25 FP8/INT8 W8A8 experts through the public HCU AITER API."""

    if (
        topk_weights.ndim != 2
        or topk_ids.ndim != 2
        or topk_weights.shape != topk_ids.shape
        or topk_ids.shape[0] != hidden_states.shape[0]
    ):
        raise HcuCompressedTensorsMoeError(
            "AITER quantized MoE requires matching rank-2 top-k tensors"
        )
    if apply_router_weight_on_input:
        raise HcuCompressedTensorsMoeError(
            "AITER quantized MoE does not support apply_router_weight_on_input=True"
        )
    if getattr(quant_config, "block_shape", None) is not None:
        raise HcuCompressedTensorsMoeError(
            "AITER quantized MoE supports only channel/token W8A8 in this path"
        )

    try:
        from aiter.moe import MoeQuantType, aiter_moe
    except Exception as exc:
        raise HcuCompressedTensorsMoeError(
            "AITER quantized MoE is selected, but the public aiter.moe API "
            "is unavailable"
        ) from exc

    use_fp8 = bool(getattr(quant_config, "use_fp8_w8a8", False))
    use_int8 = bool(getattr(quant_config, "use_int8_w8a8", False))
    if use_fp8 == use_int8:
        raise HcuCompressedTensorsMoeError(
            "AITER quantized MoE requires exactly one FP8-W8A8 or INT8-W8A8 config"
        )
    quant_member = "FP8_W8A8" if use_fp8 else "W8A8"
    quant_type = getattr(MoeQuantType, quant_member, None)
    if quant_type is None:
        raise HcuCompressedTensorsMoeError(
            f"AITER does not expose required MoeQuantType.{quant_member}"
        )

    w1_scale = _required_tensor(quant_config, "w1_scale")
    w2_scale = _required_tensor(quant_config, "w2_scale")
    activation_value = getattr(activation, "value", activation)
    if activation_value is None:
        raise HcuCompressedTensorsMoeError(
            "AITER quantized MoE requires an activation"
        )
    activation_name = str(activation_value)
    aiter_config = _get_aiter_quantized_runtime_config(
        hidden_states,
        w1,
        w2,
        topk_ids,
        quant_type,
        activation_name,
    )
    prepared_w1, prepared_w2 = _get_aiter_quantized_weights(
        w1,
        w2,
        aiter_config,
    )

    from vllm_hcu.model_executor.layers.fused_moe.aiter_runtime import (
        aiter_asm_boltops_int8_quant_context,
    )

    align_int8_quant = bool(
        use_int8
        and _enum_token(getattr(aiter_config, "solution_type", None)) == "ASM"
    )
    with aiter_asm_boltops_int8_quant_context(enabled=align_int8_quant):
        return aiter_moe(
            hidden_states=hidden_states,
            w1=prepared_w1,
            w2=prepared_w2,
            topk_weights=topk_weights.to(torch.float32),
            topk_ids=topk_ids.to(torch.int32),
            moe_config=aiter_config,
            inplace=False,
            activation=activation_name,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            w1_zp=getattr(quant_config, "w1_zp", None),
            w2_zp=getattr(quant_config, "w2_zp", None),
            a1_scale=(
                a1q_scale
                if a1q_scale is not None
                else getattr(quant_config, "a1_scale", None)
            ),
            a2_scale=getattr(quant_config, "a2_scale", None),
            block_shape=None,
            global_num_experts=getattr(
                vllm_moe_config, "num_experts", w1.shape[0]
            ),
            expert_map=expert_map,
            routed_scaling_factor=1.0,
            use_weight_shuffle=bool(
                getattr(aiter_config, "need_shuffle", False)
            ),
            output_dtype=output_dtype,
        )


def get_aiter_w8a8_runtime_config(
    method: object,
    layer: object,
    x: torch.Tensor,
    topk_ids: torch.Tensor,
) -> object:
    """Get a layer-aware, shape-aware AITER FP8-W8A8 MoE config."""

    if x.ndim != 2 or topk_ids.ndim != 2 or x.shape[0] != topk_ids.shape[0]:
        raise HcuCompressedTensorsMoeError(
            "AITER FP8-W8A8 MoE expects x and topk_ids to be matching 2D tensors"
        )
    w1 = _required_tensor(layer, "w13_weight")
    w2 = _required_tensor(layer, "w2_weight")
    if w1.ndim != 3 or w2.ndim != 3 or w1.shape[0] != w2.shape[0]:
        raise HcuCompressedTensorsMoeError(
            "AITER FP8-W8A8 MoE expects compatible rank-3 expert weights"
        )

    activation_value = getattr(getattr(layer, "activation", None), "value", None)
    if activation_value is None:
        activation_value = getattr(layer, "activation", None)
    if activation_value is None:
        raise HcuCompressedTensorsMoeError(
            "AITER FP8-W8A8 MoE requires a layer activation"
        )
    activation = str(activation_value)

    cache = getattr(method, "_hcu_aiter_moe_config_cache", None)
    if cache is None:
        cache = {}
        setattr(method, "_hcu_aiter_moe_config_cache", cache)
    if not isinstance(cache, dict):
        raise HcuCompressedTensorsMoeError(
            "AITER FP8-W8A8 MoE config cache has an invalid type"
        )

    cache_key = (
        id(layer),
        x.shape[0],
        w1.shape[0],
        w1.shape[1],
        w2.shape[1],
        w1.shape[2],
        topk_ids.shape[1],
        x.dtype,
        x.device,
        activation,
    )
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    try:
        from aiter.moe import MoeQuantType, get_aiter_moe_config
    except Exception as exc:
        raise HcuCompressedTensorsMoeError(
            "AITER FP8-W8A8 MoE is enabled, but aiter.moe is unavailable"
        ) from exc
    quant_type = getattr(MoeQuantType, "FP8_W8A8", None)
    if quant_type is None:
        raise HcuCompressedTensorsMoeError(
            "AITER does not expose the required FP8_W8A8 MoE quant type"
        )

    status, moe_config = get_aiter_moe_config(
        M=x.shape[0],
        E=w1.shape[0],
        N1=w1.shape[1],
        N2=w2.shape[1],
        K=w1.shape[2],
        top_k=topk_ids.shape[1],
        block_size=0,
        dtype=x.dtype,
        quant_type=quant_type,
        activation=activation,
    )
    if not status or moe_config is None:
        layer_name = getattr(layer, "layer_name", "unknown")
        raise HcuCompressedTensorsMoeError(
            "AITER FP8-W8A8 MoE found no backend config for "
            f"layer {layer_name!r}, M={x.shape[0]}, "
            f"top_k={topk_ids.shape[1]}, dtype={x.dtype}"
        )
    cache[cache_key] = moe_config
    return moe_config


def get_aiter_weights_for_solution(
    layer: object,
    moe_config: object,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return weights prepared by the public HCU AITER API."""

    w1 = _required_tensor(layer, "w13_weight")
    w2 = _required_tensor(layer, "w2_weight")
    if not bool(getattr(moe_config, "need_shuffle", False)):
        return w1, w2

    cache_key = (
        _tensor_generation(w1),
        _tensor_generation(w2),
        _enum_token(getattr(moe_config, "quant_type", None)),
        _enum_token(getattr(moe_config, "solution_type", None)),
    )
    cache = getattr(layer, "_hcu_aiter_shuffled_weights", None)
    if cache is None:
        cache = {}
        setattr(layer, "_hcu_aiter_shuffled_weights", cache)
    if not isinstance(cache, dict):
        raise HcuCompressedTensorsMoeError(
            "AITER shuffled-weight cache has an invalid type"
        )
    cached = cache.get(cache_key)
    if (
        isinstance(cached, tuple)
        and len(cached) == 2
        and isinstance(cached[0], torch.Tensor)
        and isinstance(cached[1], torch.Tensor)
    ):
        return cached

    try:
        from aiter.moe import aiter_moe_shfl_weight
    except Exception as exc:
        raise HcuCompressedTensorsMoeError(
            "HCU AITER selected a shuffled MoE layout, but "
            "aiter.moe.aiter_moe_shfl_weight is unavailable"
        ) from exc
    with torch.no_grad():
        shuffled_w1, shuffled_w2 = aiter_moe_shfl_weight(w1, w2, moe_config)
    if not isinstance(shuffled_w1, torch.Tensor) or not isinstance(
        shuffled_w2, torch.Tensor
    ):
        raise HcuCompressedTensorsMoeError(
            "HCU AITER returned missing shuffled MoE weights"
        )
    if shuffled_w1.shape != w1.shape or shuffled_w2.shape != w2.shape:
        raise HcuCompressedTensorsMoeError(
            "HCU AITER returned incompatible shuffled MoE weight shapes"
        )
    cache[cache_key] = (shuffled_w1, shuffled_w2)
    return shuffled_w1, shuffled_w2


def apply_aiter_w8a8_fp8_moe(
    method: object,
    layer: object,
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    shared_experts: object | None,
    shared_experts_input: torch.Tensor | None,
    i_q: torch.Tensor | None = None,
    i_s: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the explicit HCU AITER FP8-W8A8 MoE fast path."""

    if (i_q is None) != (i_s is None):
        raise HcuCompressedTensorsMoeError(
            "AITER FP8-W8A8 MoE requires i_q and i_s together"
        )
    if getattr(method, "moe", None) is None:
        raise HcuCompressedTensorsMoeError(
            "AITER FP8-W8A8 MoE requires the vLLM v0.25.1 MoE configuration"
        )
    # vLLM v0.25.1 passes both objects through the quant-method contract.  The
    # HCU MoERunner owns shared-expert launch/stream ordering around this
    # routed-expert kernel, so this backend must accept (but not execute) them.
    del shared_experts, shared_experts_input
    if bool(getattr(layer, "apply_router_weight_on_input", False)):
        raise HcuCompressedTensorsMoeError(
            "AITER FP8-W8A8 MoE does not support "
            "apply_router_weight_on_input=True"
        )
    if topk_weights.ndim != 2 or topk_weights.shape != topk_ids.shape:
        raise HcuCompressedTensorsMoeError(
            "AITER FP8-W8A8 MoE requires matching 2D top-k weights and IDs"
        )
    if i_q is not None:
        if not isinstance(i_q, torch.Tensor) or not isinstance(i_s, torch.Tensor):
            raise HcuCompressedTensorsMoeError(
                "AITER FP8-W8A8 prequantized input must contain tensors"
            )
        if i_q.shape != x.shape or i_s.shape != (*x.shape[:-1], 1):
            raise HcuCompressedTensorsMoeError(
                "AITER FP8-W8A8 prequantized input shapes do not match x"
            )

    moe_config = get_aiter_w8a8_runtime_config(method, layer, x, topk_ids)
    w1, w2 = get_aiter_weights_for_solution(layer, moe_config)
    try:
        from aiter.moe import aiter_moe
    except Exception as exc:
        raise HcuCompressedTensorsMoeError(
            "AITER FP8-W8A8 MoE is enabled, but aiter_moe is unavailable"
        ) from exc
    if not callable(aiter_moe):
        raise HcuCompressedTensorsMoeError("aiter.moe.aiter_moe is not callable")

    activation = getattr(getattr(layer, "activation", None), "value", None)
    if activation is None:
        activation = getattr(layer, "activation", None)
    return aiter_moe(
        hidden_states=x if i_q is None else i_q,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights.to(torch.float32),
        topk_ids=topk_ids.to(torch.int32),
        moe_config=moe_config,
        # vLLM v0.25.1 removed FusedMoEConfig.disable_inplace together with
        # the in-place fused-experts mechanism.  Preserve the target contract
        # and keep x available to the runner/shared-expert lifecycle.
        inplace=False,
        activation=activation,
        w1_scale=_required_tensor(layer, "w13_weight_scale"),
        w2_scale=_required_tensor(layer, "w2_weight_scale"),
        w1_zp=None,
        w2_zp=None,
        a1_scale=i_s if i_s is not None else getattr(layer, "w13_input_scale", None),
        a2_scale=getattr(layer, "w2_input_scale", None),
        block_shape=None,
        global_num_experts=getattr(layer, "global_num_experts", -1),
        expert_map=getattr(layer, "expert_map", None),
        routed_scaling_factor=1.0,
        use_weight_shuffle=bool(getattr(moe_config, "need_shuffle", False)),
        output_dtype=None if i_q is None else x.dtype,
    )


def process_dpsk_deepgemm_weights(method: object, layer: object) -> None:
    """Run the HCU-owned expert post-load step for the DPSK backend."""

    if _enum_token(getattr(method, "fp8_backend", None)) != "DPSK_DEEPGEMM":
        return
    moe_kernel = getattr(method, "moe_kernel", None)
    fused_experts = getattr(moe_kernel, "fused_experts", None)
    experts = getattr(fused_experts, "experts", fused_experts)
    process = getattr(experts, "process_weights_after_loading", None)
    if not callable(process):
        raise HcuCompressedTensorsMoeError(
            "DPSK_DEEPGEMM was selected, but the HCU expert post-load hook "
            "is unavailable"
        )
    process(layer)


def create_aiter_w4a16_qzeros(
    method: object,
    layer: object,
    num_experts: int,
    hidden_size: int,
    intermediate_size_per_partition: int,
    extra_weight_attrs: dict[str, object],
    set_weight_attrs: Callable[[torch.Tensor, dict[str, object]], None],
) -> None:
    """Register initialized packed symmetric-zero tensors for AITER W4A16."""

    if getattr(method, "num_bits", None) != 4:
        raise HcuCompressedTensorsMoeError(
            "VLLM_HCU_USE_AITER_W4A16_MOE requires 4-bit weights"
        )
    group_size = getattr(method, "group_size", None)
    if not isinstance(group_size, int) or group_size <= 0:
        raise HcuCompressedTensorsMoeError(
            "AITER W4A16 MoE requires a positive integer group_size"
        )
    if min(num_experts, hidden_size, intermediate_size_per_partition) <= 0:
        raise HcuCompressedTensorsMoeError(
            "AITER W4A16 MoE weight dimensions must be positive"
        )
    if hidden_size % group_size or intermediate_size_per_partition % group_size:
        raise HcuCompressedTensorsMoeError(
            "AITER W4A16 MoE K dimensions must be divisible by group_size"
        )
    hidden_groups = hidden_size // group_size
    intermediate_groups = intermediate_size_per_partition // group_size
    if hasattr(layer, "w13_qzeros") or hasattr(layer, "w2_qzeros"):
        raise HcuCompressedTensorsMoeError(
            "AITER W4A16 MoE zero-point parameters already exist"
        )
    register_parameter = getattr(layer, "register_parameter", None)
    if not callable(register_parameter):
        raise HcuCompressedTensorsMoeError(
            "AITER W4A16 MoE layer cannot register zero-point parameters"
        )

    moe = getattr(method, "moe", None)
    shards = 2 if bool(getattr(moe, "is_act_and_mul", False)) else 1
    w13_output_size = shards * intermediate_size_per_partition
    if w13_output_size % 2 or hidden_size % 2:
        raise HcuCompressedTensorsMoeError(
            "AITER W4A16 MoE output dimensions must be even for packed zero points"
        )
    device = _required_tensor(layer, "w13_weight_packed").device
    _required_tensor(layer, "w2_weight_packed")
    # 0x88 stores two symmetric int4 zero points in each byte.
    w13_qzeros = torch.nn.Parameter(
        torch.full(
            (
                num_experts,
                w13_output_size // 2,
                hidden_groups,
            ),
            0x88,
            dtype=torch.uint8,
            device=device,
        ),
        requires_grad=False,
    )
    w2_qzeros = torch.nn.Parameter(
        torch.full(
            (num_experts, hidden_size // 2, intermediate_groups),
            0x88,
            dtype=torch.uint8,
            device=device,
        ),
        requires_grad=False,
    )
    attrs = dict(extra_weight_attrs)
    attrs.update(
        {
            "is_transposed": True,
            "quant_method": getattr(method, "strategy", "group"),
        }
    )
    register_parameter("w13_qzeros", w13_qzeros)
    set_weight_attrs(w13_qzeros, attrs)
    register_parameter("w2_qzeros", w2_qzeros)
    set_weight_attrs(w2_qzeros, attrs)


def build_aiter_w4a16_quant_config(
    method: object,
    layer: object,
    config_builder: Callable[..., object],
) -> object:
    """Build vLLM's W4A16 config with the HCU zero-point tensors."""

    if getattr(method, "num_bits", None) != 4:
        raise HcuCompressedTensorsMoeError(
            "VLLM_HCU_USE_AITER_W4A16_MOE requires 4-bit weights"
        )
    group_size = getattr(method, "group_size", None)
    if not isinstance(group_size, int) or group_size <= 0:
        raise HcuCompressedTensorsMoeError(
            "AITER W4A16 MoE requires a positive integer group_size"
        )
    return config_builder(
        w1_scale=_required_tensor(layer, "w13_weight_scale"),
        w2_scale=_required_tensor(layer, "w2_weight_scale"),
        w1_zp=_required_tensor(layer, "w13_qzeros"),
        w2_zp=_required_tensor(layer, "w2_qzeros"),
        block_shape=[0, group_size],
    )


__all__ = [
    "HcuCompressedTensorsMoeError",
    "apply_aiter_quantized_moe",
    "apply_aiter_w8a8_fp8_moe",
    "build_aiter_w4a16_quant_config",
    "create_aiter_w4a16_qzeros",
    "get_aiter_w8a8_runtime_config",
    "get_aiter_weights_for_solution",
    "process_dpsk_deepgemm_weights",
]
