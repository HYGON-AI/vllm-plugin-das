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
from vllm.model_executor.utils import replace_parameter

from vllm_hcu.model_executor.layers.fused_moe.aiter_moe_dispatch import (
    AiterMoeProblem,
    aiter_moe_weight_layout_signature,
    aiter_expert_map_for_solution,
    execute_aiter_moe,
    prewarm_aiter_moe_config,
    prepare_aiter_moe_scales,
    prepare_aiter_moe_weights,
    resolve_aiter_expert_maps,
    select_aiter_moe_config,
)
from vllm_hcu.platforms import envs as henvs


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


_WEIGHT_SOLUTION_ATTR = "_hcu_aiter_moe_solution_type"
_WEIGHT_LAYOUT_ATTR = "_hcu_aiter_moe_weight_layout"
_WEIGHT_LOGICAL_SHAPE_ATTR = "_hcu_aiter_moe_logical_shape"
_SCALE_LAYOUT_ATTR = "_hcu_aiter_moe_scale_layout"


def _scale_layout_signature(config: object) -> tuple[object, ...]:
    config_values = getattr(config, "config", None)
    padded_k = (
        config_values.get("PADDED_K") if isinstance(config_values, dict) else None
    )
    original_k = (
        config_values.get("ORIGINAL_K") if isinstance(config_values, dict) else None
    )
    return (
        _enum_token(getattr(config, "quant_type", None)).lower(),
        _enum_token(getattr(config, "solution_type", None)),
        bool(getattr(config, "need_shuffle_scale", False)),
        padded_k,
        original_k,
    )


def _replace_tensor_reference(owner: object, name: str, value: torch.Tensor) -> None:
    descriptor_name = {"w1_scale": "_w1", "w2_scale": "_w2"}.get(name)
    descriptor = getattr(owner, descriptor_name, None) if descriptor_name else None
    if descriptor is not None and hasattr(descriptor, "scale"):
        descriptor.scale = value
    elif isinstance(owner, torch.nn.Module) and name in owner._parameters:
        replace_parameter(owner, name, value)
    else:
        setattr(owner, name, value)


def install_aiter_moe_scale_layout(
    layer: object,
    quant_config: object | None,
    config: object,
    *,
    prefer_quant_config: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Install the selected scale layout and update every owning reference."""

    layer_has_scales = not prefer_quant_config and isinstance(
        getattr(layer, "w13_weight_scale", None), torch.Tensor
    ) and isinstance(getattr(layer, "w2_weight_scale", None), torch.Tensor)
    owner = layer if layer_has_scales else quant_config
    if owner is None:
        raise HcuCompressedTensorsMoeError(
            "AITER selected scale shuffle without a quantization config"
        )
    first_name, second_name = (
        ("w13_weight_scale", "w2_weight_scale")
        if layer_has_scales
        else ("w1_scale", "w2_scale")
    )
    scale1 = _required_tensor(owner, first_name)
    scale2 = _required_tensor(owner, second_name)
    signature = _scale_layout_signature(config)
    first_layout = getattr(scale1, _SCALE_LAYOUT_ATTR, None)
    second_layout = getattr(scale2, _SCALE_LAYOUT_ATTR, None)
    if first_layout is not None or second_layout is not None:
        if first_layout != signature or second_layout != signature:
            raise HcuCompressedTensorsMoeError(
                "AITER installed scales do not match the selected layout"
            )
        return scale1, scale2
    installed1, installed2 = prepare_aiter_moe_scales(
        scale1,
        scale2,
        config,
        cache_owner=object(),
    )
    assert installed1 is not None and installed2 is not None
    _replace_tensor_reference(owner, first_name, installed1)
    _replace_tensor_reference(owner, second_name, installed2)
    installed1 = _required_tensor(owner, first_name)
    installed2 = _required_tensor(owner, second_name)
    if layer_has_scales and quant_config is not None:
        _replace_tensor_reference(quant_config, "w1_scale", installed1)
        _replace_tensor_reference(quant_config, "w2_scale", installed2)
    for scale in (installed1, installed2):
        setattr(scale, _SCALE_LAYOUT_ATTR, signature)
    return installed1, installed2


def _scales_for_selected_config(
    scale1: torch.Tensor,
    scale2: torch.Tensor,
    config: object,
) -> tuple[torch.Tensor, torch.Tensor]:
    first_layout = getattr(scale1, _SCALE_LAYOUT_ATTR, None)
    second_layout = getattr(scale2, _SCALE_LAYOUT_ATTR, None)
    if first_layout != second_layout:
        raise HcuCompressedTensorsMoeError(
            "AITER scales have inconsistent physical layouts"
        )
    if first_layout is not None:
        if _scale_layout_signature(config) != first_layout:
            raise HcuCompressedTensorsMoeError(
                "AITER config scale layout does not match installed scales"
            )
        return scale1, scale2
    prepared1, prepared2 = prepare_aiter_moe_scales(
        scale1,
        scale2,
        config,
        cache_owner=scale1,
    )
    assert prepared1 is not None and prepared2 is not None
    return prepared1, prepared2


def mark_aiter_moe_native_layout(
    layer: object,
    *,
    w1_name: str = "w13_weight",
    w2_name: str = "w2_weight",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Lock canonical weights to the native fallback without a second layout."""

    w1 = _required_tensor(layer, w1_name)
    w2 = _required_tensor(layer, w2_name)
    logical_shape = (
        int(w1.shape[0]),
        int(w1.shape[1]),
        int(w2.shape[1]),
        int(w1.shape[2]),
    )
    for weight in (w1, w2):
        setattr(weight, _WEIGHT_SOLUTION_ATTR, "native")
        setattr(weight, _WEIGHT_LOGICAL_SHAPE_ATTR, logical_shape)
        weight.is_shuffled = False
    return w1, w2


def install_aiter_moe_weight_layout(
    layer: object,
    config: object,
    *,
    w1_name: str = "w13_weight",
    w2_name: str = "w2_weight",
    block_shape: list[int] | None = None,
    logical_shape: tuple[int, int, int, int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Install one selected AITER layout and record its routing contract."""

    w1 = _required_tensor(layer, w1_name)
    w2 = _required_tensor(layer, w2_name)
    solution = _enum_token(getattr(config, "solution_type", None)).lower()
    if not solution:
        raise HcuCompressedTensorsMoeError(
            "AITER selected a quantized MoE config without a solution type"
        )
    layout = aiter_moe_weight_layout_signature(config)
    installed_solution = getattr(w1, _WEIGHT_SOLUTION_ATTR, None)
    installed_layout = getattr(w1, _WEIGHT_LAYOUT_ATTR, None)
    has_installed_state = any(
        value is not None
        for value in (
            installed_solution,
            getattr(w2, _WEIGHT_SOLUTION_ATTR, None),
            installed_layout,
            getattr(w2, _WEIGHT_LAYOUT_ATTR, None),
        )
    )
    if has_installed_state:
        installed_logical_shape = _installed_weight_logical_shape(w1, w2)
        if (
            installed_solution != solution
            or getattr(w2, _WEIGHT_SOLUTION_ATTR, None) != solution
            or installed_layout != layout
            or getattr(w2, _WEIGHT_LAYOUT_ATTR, None) != layout
            or bool(getattr(w1, "is_shuffled", False))
            != bool(getattr(config, "need_shuffle", False))
            or bool(getattr(w2, "is_shuffled", False))
            != bool(getattr(config, "need_shuffle", False))
            or (
                logical_shape is not None
                and installed_logical_shape != logical_shape
            )
        ):
            raise HcuCompressedTensorsMoeError(
                "AITER installed weights do not match the selected layout"
            )
        return w1, w2
    if logical_shape is None:
        logical_shape = (
            int(w1.shape[0]),
            int(w1.shape[1]),
            int(w2.shape[1]),
            int(w1.shape[2]),
        )
    need_shuffle = bool(getattr(config, "need_shuffle", False))
    if need_shuffle:
        w1, w2 = prepare_aiter_moe_weights(
            w1,
            w2,
            config,
            cache_owner=object(),
            block_shape=block_shape,
        )
        replace_parameter(layer, w1_name, w1)
        replace_parameter(layer, w2_name, w2)
        w1 = _required_tensor(layer, w1_name)
        w2 = _required_tensor(layer, w2_name)

    for weight in (w1, w2):
        setattr(weight, _WEIGHT_SOLUTION_ATTR, solution)
        setattr(
            weight,
            _WEIGHT_LAYOUT_ATTR,
            layout,
        )
        setattr(weight, _WEIGHT_LOGICAL_SHAPE_ATTR, logical_shape)
        weight.is_shuffled = need_shuffle
    return w1, w2


def _installed_weight_logical_shape(
    w1: torch.Tensor,
    w2: torch.Tensor,
    *,
    activation: object = "silu",
) -> tuple[int, int, int, int]:
    first = getattr(w1, _WEIGHT_LOGICAL_SHAPE_ATTR, None)
    second = getattr(w2, _WEIGHT_LOGICAL_SHAPE_ATTR, None)
    if first != second:
        raise HcuCompressedTensorsMoeError(
            "AITER weights have inconsistent logical dimensions"
        )
    if first is None:
        return (
            int(w1.shape[0]),
            int(w1.shape[1]),
            int(w2.shape[1]),
            int(w1.shape[2]),
        )
    if (
        not isinstance(first, tuple)
        or len(first) != 4
        or not all(
            isinstance(value, int) and not isinstance(value, bool) and value > 0
            for value in first
        )
    ):
        raise HcuCompressedTensorsMoeError(
            "AITER weights have invalid logical dimensions"
        )
    logical_e, logical_n1, logical_n2, logical_k = first
    gated = _enum_token(activation) in {
        "SILU",
        "SITU",
        "GELU",
        "SWIGLUOAI",
        "SWIGLUSTEP",
        "GELU_TANH",
    }
    if gated and logical_n1 % 2:
        raise HcuCompressedTensorsMoeError(
            "AITER weights have invalid logical dimensions"
        )
    first_layout = getattr(w1, _WEIGHT_LAYOUT_ATTR, None)
    second_layout = getattr(w2, _WEIGHT_LAYOUT_ATTR, None)
    if first_layout != second_layout:
        raise HcuCompressedTensorsMoeError(
            "AITER weights have inconsistent physical layouts"
        )
    if first_layout is not None and (
        not isinstance(first_layout, tuple) or len(first_layout) != 4
    ):
        raise HcuCompressedTensorsMoeError(
            "AITER weights have invalid physical layout signature"
        )
    padded_k = first_layout[3] if first_layout is not None else None
    physical_k = int(padded_k) if padded_k is not None else logical_k
    expected_w1 = (logical_e, logical_n1, physical_k)
    intermediate_size = logical_n1 // 2 if gated else logical_n1
    expected_w2 = (logical_e, physical_k, intermediate_size)
    if tuple(w1.shape) != expected_w1 or tuple(w2.shape) != expected_w2:
        raise HcuCompressedTensorsMoeError(
            "AITER weights do not match their installed physical layout"
        )
    return first


def _quantized_problem_dimensions(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    *,
    activation: object,
) -> tuple[int, int, int, int]:
    logical_shape = _installed_weight_logical_shape(
        w1,
        w2,
        activation=activation,
    )
    if int(hidden_states.shape[1]) != logical_shape[3]:
        raise HcuCompressedTensorsMoeError(
            "AITER quantized MoE hidden states do not match logical K"
        )
    return logical_shape


def _installed_weight_solution(
    w1: torch.Tensor,
    w2: torch.Tensor,
    *,
    label: str,
) -> str | None:
    first = getattr(w1, _WEIGHT_SOLUTION_ATTR, None)
    second = getattr(w2, _WEIGHT_SOLUTION_ATTR, None)
    if first != second:
        raise HcuCompressedTensorsMoeError(
            f"{label} weights have inconsistent installed solutions"
        )
    return first


def _weights_for_selected_config(
    w1: torch.Tensor,
    w2: torch.Tensor,
    config: object,
    *,
    installed_solution: str | None,
    preserve_inputs: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reuse an installed layout or derive a legacy transient layout."""

    if installed_solution is not None:
        selected = _enum_token(getattr(config, "solution_type", None)).lower()
        if selected != installed_solution:
            raise HcuCompressedTensorsMoeError(
                "AITER selected a solution incompatible with installed weights"
            )
        expected_shuffle = bool(getattr(w1, "is_shuffled", False))
        if expected_shuffle != bool(getattr(w2, "is_shuffled", False)):
            raise HcuCompressedTensorsMoeError(
                "AITER weights have inconsistent shuffle state"
            )
        if expected_shuffle != bool(getattr(config, "need_shuffle", False)):
            raise HcuCompressedTensorsMoeError(
                "AITER config layout does not match installed weights"
            )
        first_layout = getattr(w1, _WEIGHT_LAYOUT_ATTR, None)
        second_layout = getattr(w2, _WEIGHT_LAYOUT_ATTR, None)
        if first_layout != second_layout:
            raise HcuCompressedTensorsMoeError(
                "AITER weights have inconsistent physical layouts"
            )
        if (
            first_layout is not None
            and aiter_moe_weight_layout_signature(config) != first_layout
        ):
            raise HcuCompressedTensorsMoeError(
                "AITER config physical layout does not match installed weights"
            )
        return w1, w2
    return prepare_aiter_moe_weights(
        w1,
        w2,
        config,
        cache_owner=w1,
        preserve_inputs=preserve_inputs,
    )


def prewarm_aiter_quantized_moe(
    layer: object,
    vllm_moe_config: object,
    quant_config: object,
) -> object | None:
    """Probe M=1 while canonical quantized MoE weights are being loaded."""

    w1 = _required_tensor(layer, "w13_weight")
    w2 = _required_tensor(layer, "w2_weight")
    if w1.ndim != 3 or w2.ndim != 3 or w1.shape[0] != w2.shape[0]:
        raise HcuCompressedTensorsMoeError(
            "AITER quantized MoE prewarm expects compatible rank-3 weights"
        )
    use_fp8 = bool(getattr(quant_config, "use_fp8_w8a8", False))
    use_int8 = bool(getattr(quant_config, "use_int8_w8a8", False))
    if use_fp8 == use_int8:
        raise HcuCompressedTensorsMoeError(
            "AITER quantized MoE prewarm requires one W8A8 quantization mode"
        )
    from aiter.moe import MoeQuantType

    quant_member = "FP8_W8A8" if use_fp8 else "W8A8"
    quant_type = getattr(MoeQuantType, quant_member, None)
    if quant_type is None:
        raise HcuCompressedTensorsMoeError(
            f"AITER does not expose required MoeQuantType.{quant_member}"
        )
    top_k = getattr(vllm_moe_config, "experts_per_token", None)
    dtype = getattr(vllm_moe_config, "in_dtype", None)
    if not isinstance(top_k, int) or top_k <= 0 or not isinstance(dtype, torch.dtype):
        raise HcuCompressedTensorsMoeError(
            "AITER quantized MoE prewarm requires valid top-k and input dtype"
        )
    activation = getattr(getattr(layer, "activation", None), "value", None)
    if activation is None:
        activation = getattr(getattr(vllm_moe_config, "activation", None), "value", None)
    if activation is None:
        raise HcuCompressedTensorsMoeError(
            "AITER quantized MoE prewarm requires an activation"
        )
    logical_shape = _installed_weight_logical_shape(
        w1,
        w2,
        activation=activation,
    )
    block_shape = getattr(quant_config, "block_shape", None)
    block_size = int(block_shape[1]) if block_shape else 0
    problem = AiterMoeProblem(
        M=1,
        E=logical_shape[0],
        N1=logical_shape[1],
        N2=logical_shape[2],
        K=logical_shape[3],
        top_k=top_k,
        block_size=block_size,
        dtype=dtype,
        device=w1.device,
        quant_type=quant_type,
        activation=str(activation),
        use_shuffle=bool(henvs.VLLM_HCU_USE_AITER_MOE_SHUFFLE),
    )
    config = prewarm_aiter_moe_config(problem, cache_owner=w1)
    if config is not None:
        install_aiter_moe_weight_layout(
            layer,
            config,
            logical_shape=logical_shape,
        )
        install_aiter_moe_scale_layout(layer, quant_config, config)
    else:
        mark_aiter_moe_native_layout(layer)
    return config


def prewarm_aiter_w4a16_moe(
    method: object,
    layer: object,
    quant_config: object,
) -> object | None:
    """Probe the M=1 W4A16 route after packed weights are loaded."""

    w1 = _required_tensor(layer, "w13_weight_packed")
    w2 = _required_tensor(layer, "w2_weight_packed")
    if w1.ndim != 3 or w2.ndim != 3 or w1.shape[0] != w2.shape[0]:
        raise HcuCompressedTensorsMoeError(
            "AITER W4A16 prewarm expects compatible rank-3 packed weights"
        )
    moe = getattr(method, "moe", None)
    top_k = getattr(moe, "experts_per_token", None)
    hidden_dim = getattr(moe, "hidden_dim", None)
    dtype = getattr(moe, "in_dtype", None)
    if (
        not isinstance(top_k, int)
        or top_k <= 0
        or not isinstance(hidden_dim, int)
        or hidden_dim <= 0
        or not isinstance(dtype, torch.dtype)
    ):
        raise HcuCompressedTensorsMoeError(
            "AITER W4A16 prewarm requires valid MoE dimensions and dtype"
        )
    activation = getattr(getattr(moe, "activation", None), "value", None)
    if activation is None:
        raise HcuCompressedTensorsMoeError(
            "AITER W4A16 prewarm requires an activation"
        )
    block_shape = getattr(quant_config, "block_shape", None)
    if not block_shape or len(block_shape) < 2:
        raise HcuCompressedTensorsMoeError(
            "AITER W4A16 prewarm requires a two-dimensional block shape"
        )
    from aiter.moe import MoeQuantType

    quant_type = getattr(MoeQuantType, "W4A16", None)
    if quant_type is None:
        raise HcuCompressedTensorsMoeError(
            "AITER does not expose required MoeQuantType.W4A16"
        )
    problem = AiterMoeProblem(
        M=1,
        E=int(w1.shape[0]),
        N1=int(w1.shape[1]),
        N2=int(w2.shape[1]),
        K=hidden_dim,
        top_k=top_k,
        block_size=int(block_shape[1]),
        dtype=dtype,
        device=w1.device,
        quant_type=quant_type,
        activation=str(activation),
        use_shuffle=bool(henvs.VLLM_HCU_USE_AITER_MOE_SHUFFLE),
    )
    return prewarm_aiter_moe_config(problem, cache_owner=w1)


def _slimquant_w4a8_metadata(
    method: object,
    layer: object,
) -> tuple[torch.Tensor, torch.Tensor, int, str]:
    """Validate packed SlimQuant weights and return logical MoE metadata."""

    w1 = _required_tensor(layer, "w13_weight")
    w2 = _required_tensor(layer, "w2_weight")
    if w1.dtype != torch.int8 or w2.dtype != torch.int8:
        raise HcuCompressedTensorsMoeError(
            "SlimQuant W4A8 requires packed INT8 storage"
        )
    if w1.ndim != 3 or w2.ndim != 3 or w1.shape[0] != w2.shape[0]:
        raise HcuCompressedTensorsMoeError(
            "SlimQuant W4A8 expects compatible rank-3 packed weights"
        )
    logical_k = int(w1.shape[2]) * 2
    if int(w2.shape[1]) != logical_k or int(w2.shape[2]) * 4 != int(w1.shape[1]):
        raise HcuCompressedTensorsMoeError(
            "SlimQuant W4A8 packed weight dimensions are inconsistent"
        )
    moe = getattr(method, "moe", None)
    hidden_dim = getattr(moe, "hidden_dim", logical_k)
    if not isinstance(hidden_dim, int) or hidden_dim != logical_k:
        raise HcuCompressedTensorsMoeError(
            "SlimQuant W4A8 logical hidden size does not match packed weights"
        )
    activation = getattr(getattr(moe, "activation", None), "value", None)
    if activation is None:
        activation = "silu"
    if str(activation) != "silu":
        raise HcuCompressedTensorsMoeError(
            "SlimQuant W4A8 supports only silu activation"
        )
    return w1, w2, logical_k, str(activation)


def prewarm_aiter_w4a8_moe(
    method: object,
    layer: object,
) -> object | None:
    """Probe SlimQuant W4A8 at M=1 while canonical weights are loaded."""

    w1, w2, logical_k, activation = _slimquant_w4a8_metadata(method, layer)
    moe = getattr(method, "moe", None)
    top_k = getattr(moe, "experts_per_token", None)
    dtype = getattr(moe, "in_dtype", None)
    if not isinstance(top_k, int) or top_k <= 0 or not isinstance(dtype, torch.dtype):
        raise HcuCompressedTensorsMoeError(
            "SlimQuant W4A8 prewarm requires valid top-k and input dtype"
        )
    from aiter.moe import MoeQuantType

    quant_type = getattr(MoeQuantType, "W4A8", None)
    if quant_type is None:
        raise HcuCompressedTensorsMoeError(
            "AITER does not expose required MoeQuantType.W4A8"
        )
    problem = AiterMoeProblem(
        M=1,
        E=int(w1.shape[0]),
        N1=int(w1.shape[1]),
        N2=int(w2.shape[1]),
        K=logical_k,
        top_k=top_k,
        block_size=0,
        dtype=dtype,
        device=w1.device,
        quant_type=quant_type,
        activation=activation,
        use_shuffle=bool(henvs.VLLM_HCU_USE_AITER_MOE_SHUFFLE),
    )
    config = select_aiter_moe_config(
        problem,
        cache_owner=w1,
        solution_type="moe_c",
    )
    return config


def _unpack_slimquant_w4a8_tensor(packed: torch.Tensor) -> torch.Tensor:
    """Expand HIPC high-nibble-first signed INT4 storage to canonical INT8."""

    packed_u8 = packed.view(torch.uint8)
    high = ((packed_u8 >> 4) & 0xF).to(torch.int16)
    low = (packed_u8 & 0xF).to(torch.int16)
    high = torch.where(high >= 8, high - 16, high).to(torch.int8)
    low = torch.where(low >= 8, low - 16, low).to(torch.int8)
    return torch.stack((high, low), dim=-1).flatten(-2).contiguous()


def _vllm_w4a8_fallback_weights(
    w1: torch.Tensor,
    w2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create transient W8 storage consumed by vLLM's Triton fallback."""

    with torch.no_grad():
        return (
            _unpack_slimquant_w4a8_tensor(w1),
            _unpack_slimquant_w4a8_tensor(w2),
        )


def prepare_vllm_w4a8_moe(
    method: object,
    layer: object,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replace packed weights with the sole layout used by explicit Triton."""

    w1, w2, _, _ = _slimquant_w4a8_metadata(method, layer)
    fallback_weights = _vllm_w4a8_fallback_weights(w1, w2)
    replace_parameter(layer, "w13_weight", fallback_weights[0])
    replace_parameter(layer, "w2_weight", fallback_weights[1])
    layer.w13_weight._hcu_vllm_w4a8_unpacked = True
    layer.w2_weight._hcu_vllm_w4a8_unpacked = True
    return layer.w13_weight, layer.w2_weight


def apply_aiter_w4a8_moe(
    method: object,
    layer: object,
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    allow_aiter: bool = True,
) -> torch.Tensor:
    """Route SlimQuant W4A8 through AITER, or vLLM Triton if unsupported."""

    if (
        hidden_states.ndim != 2
        or topk_weights.ndim != 2
        or topk_ids.ndim != 2
        or topk_weights.shape != topk_ids.shape
        or topk_ids.shape[0] != hidden_states.shape[0]
    ):
        raise HcuCompressedTensorsMoeError(
            "SlimQuant W4A8 requires matching rank-2 hidden and top-k tensors"
        )
    explicit_vllm_layout = bool(
        getattr(
            getattr(layer, "w13_weight", None),
            "_hcu_vllm_w4a8_unpacked",
            False,
        )
    )
    if explicit_vllm_layout:
        w1 = _required_tensor(layer, "w13_weight")
        w2 = _required_tensor(layer, "w2_weight")
        if not bool(getattr(w2, "_hcu_vllm_w4a8_unpacked", False)):
            raise HcuCompressedTensorsMoeError(
                "SlimQuant W4A8 Triton weights have inconsistent layouts"
            )
        if w1.dtype != torch.int8 or w2.dtype != torch.int8:
            raise HcuCompressedTensorsMoeError(
                "SlimQuant W4A8 Triton weights require unpacked INT8 storage"
            )
        if (
            w1.ndim != 3
            or w2.ndim != 3
            or w1.shape[0] != w2.shape[0]
            or w1.shape[2] != w2.shape[1]
            or w1.shape[1] != 2 * w2.shape[2]
        ):
            raise HcuCompressedTensorsMoeError(
                "SlimQuant W4A8 Triton weight dimensions are inconsistent"
            )
        logical_k = int(w1.shape[2])
        activation = str(
            getattr(
                getattr(getattr(method, "moe", None), "activation", None),
                "value",
                "silu",
            )
        )
        if activation != "silu":
            raise HcuCompressedTensorsMoeError(
                "SlimQuant W4A8 supports only silu activation"
            )
    else:
        w1, w2, logical_k, activation = _slimquant_w4a8_metadata(method, layer)
    if int(hidden_states.shape[1]) != logical_k:
        raise HcuCompressedTensorsMoeError(
            "SlimQuant W4A8 hidden states do not match the weight logical K"
        )
    quant_config = getattr(method, "moe_quant_config", None)
    w1_scale = _required_tensor(quant_config, "w1_scale")
    w2_scale = _required_tensor(quant_config, "w2_scale")
    aiter_config = None
    if allow_aiter:
        from aiter.moe import MoeQuantType

        quant_type = getattr(MoeQuantType, "W4A8", None)
        if quant_type is None:
            raise HcuCompressedTensorsMoeError(
                "AITER does not expose required MoeQuantType.W4A8"
            )
        problem = AiterMoeProblem(
            M=int(hidden_states.shape[0]),
            E=int(w1.shape[0]),
            N1=int(w1.shape[1]),
            N2=int(w2.shape[1]),
            K=logical_k,
            top_k=int(topk_ids.shape[1]),
            block_size=0,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
            quant_type=quant_type,
            activation=activation,
            use_shuffle=bool(henvs.VLLM_HCU_USE_AITER_MOE_SHUFFLE),
        )
        installed_solution = _installed_weight_solution(
            w1, w2, label="SlimQuant W4A8"
        )
        if installed_solution != "native":
            aiter_config = select_aiter_moe_config(
                problem,
                cache_owner=w1,
                solution_type="moe_c",
            )
    else:
        installed_solution = None
    global_num_experts = getattr(
        layer,
        "global_num_experts",
        getattr(getattr(method, "moe", None), "num_experts", int(w1.shape[0])),
    )
    native_expert_map = getattr(
        layer,
        "_expert_map",
        getattr(layer, "expert_map", None),
    )
    expert_mask = getattr(layer, "expert_mask", None)
    if aiter_config is None:
        if installed_solution not in (None, "native"):
            raise HcuCompressedTensorsMoeError(
                "AITER has no MOE_C solution for installed W4A8 weights"
            )
        if explicit_vllm_layout:
            fallback_weights = (w1, w2)
        else:
            fallback_weights = _vllm_w4a8_fallback_weights(w1, w2)
        fallback_w1, fallback_w2 = fallback_weights
        from vllm.model_executor.layers.fused_moe.fused_moe import (
            fused_experts_impl,
        )
        return fused_experts_impl(
            hidden_states,
            fallback_w1,
            fallback_w2,
            topk_weights,
            topk_ids,
            activation=activation,
            apply_router_weight_on_input=False,
            use_fp8_w8a8=False,
            use_int8_w8a8=True,
            use_int8_w8a16=False,
            use_int4_w4a16=False,
            per_channel_quant=True,
            global_num_experts=global_num_experts,
            expert_map=native_expert_map,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            a1_scale=getattr(quant_config, "a1_scale", None),
            a2_scale=getattr(quant_config, "a2_scale", None),
            block_shape=None,
        )

    prepared_w1, prepared_w2 = _weights_for_selected_config(
        w1,
        w2,
        aiter_config,
        installed_solution=installed_solution,
        preserve_inputs=True,
    )
    prepared_w1_scale, prepared_w2_scale = _scales_for_selected_config(
        w1_scale,
        w2_scale,
        aiter_config,
    )
    aiter_expert_map = aiter_expert_map_for_solution(
        native_expert_map,
        aiter_config,
        int(global_num_experts),
        expert_mask=expert_mask,
    )
    return execute_aiter_moe(
        aiter_config,
        hidden_states=hidden_states,
        w1=prepared_w1,
        w2=prepared_w2,
        topk_weights=topk_weights.to(torch.float32),
        topk_ids=topk_ids.to(torch.int32),
        inplace=False,
        activation=activation,
        w1_scale=prepared_w1_scale,
        w2_scale=prepared_w2_scale,
        a1_scale=getattr(quant_config, "a1_scale", None),
        a2_scale=getattr(quant_config, "a2_scale", None),
        block_shape=None,
        global_num_experts=int(global_num_experts),
        expert_map=aiter_expert_map,
        routed_scaling_factor=1.0,
        use_weight_shuffle=bool(getattr(aiter_config, "need_shuffle", False)),
        output_dtype=hidden_states.dtype,
    )


def apply_vllm_w4a8_moe(
    method: object,
    layer: object,
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    """Run SlimQuant W4A8 directly with vLLM Triton, without AITER probes."""

    return apply_aiter_w4a8_moe(
        method,
        layer,
        hidden_states,
        topk_weights,
        topk_ids,
        allow_aiter=False,
    )


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
    num_local_tokens: torch.Tensor | None = None,
    output_dtype: torch.dtype | None = None,
    moe_sorting_dispatch_policy: int = 0,
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
    if num_local_tokens is not None:
        raise HcuCompressedTensorsMoeError(
            "AITER quantized MoE does not support num_local_tokens"
        )
    if moe_sorting_dispatch_policy != 0:
        raise HcuCompressedTensorsMoeError(
            "AITER quantized MoE does not support "
            "moe_sorting_dispatch_policy != 0"
        )
    if getattr(quant_config, "block_shape", None) is not None:
        raise HcuCompressedTensorsMoeError(
            "AITER quantized MoE supports only channel/token W8A8 in this path"
        )

    from aiter.moe import MoeQuantType

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
    logical_e, logical_n1, logical_n2, logical_k = _quantized_problem_dimensions(
        hidden_states,
        w1,
        w2,
        activation=activation_name,
    )
    problem = AiterMoeProblem(
        M=int(hidden_states.shape[0]),
        E=logical_e,
        N1=logical_n1,
        N2=logical_n2,
        K=logical_k,
        top_k=int(topk_ids.shape[1]),
        block_size=0,
        dtype=hidden_states.dtype,
        device=hidden_states.device,
        quant_type=quant_type,
        activation=activation_name,
        use_shuffle=bool(henvs.VLLM_HCU_USE_AITER_MOE_SHUFFLE),
    )
    installed_solution = _installed_weight_solution(
        w1, w2, label="AITER quantized MoE"
    )
    aiter_config = (
        None
        if installed_solution == "native"
        else select_aiter_moe_config(
            problem,
            cache_owner=w1,
            solution_type=installed_solution,
        )
    )
    global_num_experts = getattr(vllm_moe_config, "num_experts", w1.shape[0])
    native_expert_map, expert_mask = resolve_aiter_expert_maps(
        expert_map,
        int(global_num_experts),
    )
    if aiter_config is None:
        if installed_solution not in (None, "native"):
            raise HcuCompressedTensorsMoeError(
                "AITER has no solution compatible with installed quantized weights"
            )
        from vllm.model_executor.layers.fused_moe.fused_moe import (
            fused_experts_impl,
        )

        return fused_experts_impl(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            activation=activation_name,
            apply_router_weight_on_input=False,
            use_fp8_w8a8=use_fp8,
            use_int8_w8a8=use_int8,
            use_int8_w8a16=False,
            use_int4_w4a16=False,
            per_channel_quant=True,
            global_num_experts=global_num_experts,
            expert_map=native_expert_map,
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
        )

    prepared_w1, prepared_w2 = _weights_for_selected_config(
        w1,
        w2,
        aiter_config,
        installed_solution=installed_solution,
    )
    prepared_w1_scale, prepared_w2_scale = _scales_for_selected_config(
        w1_scale,
        w2_scale,
        aiter_config,
    )

    from vllm_hcu.model_executor.layers.fused_moe.aiter_runtime import (
        aiter_asm_boltops_fp8_quant_context,
        aiter_asm_boltops_int8_quant_context,
    )

    is_asm_solution = (
        _enum_token(getattr(aiter_config, "solution_type", None)) == "ASM"
    )
    align_int8_quant = bool(use_int8 and is_asm_solution)
    align_fp8_quant = bool(use_fp8 and is_asm_solution)
    aiter_expert_map = aiter_expert_map_for_solution(
        native_expert_map,
        aiter_config,
        int(global_num_experts),
        expert_mask=expert_mask,
    )
    with (
        aiter_asm_boltops_int8_quant_context(enabled=align_int8_quant),
        aiter_asm_boltops_fp8_quant_context(enabled=align_fp8_quant),
    ):
        return execute_aiter_moe(
            aiter_config,
            hidden_states=hidden_states,
            w1=prepared_w1,
            w2=prepared_w2,
            topk_weights=topk_weights.to(torch.float32),
            topk_ids=topk_ids.to(torch.int32),
            inplace=False,
            activation=activation_name,
            w1_scale=prepared_w1_scale,
            w2_scale=prepared_w2_scale,
            w1_zp=getattr(quant_config, "w1_zp", None),
            w2_zp=getattr(quant_config, "w2_zp", None),
            a1_scale=(
                a1q_scale
                if a1q_scale is not None
                else getattr(quant_config, "a1_scale", None)
            ),
            a2_scale=getattr(quant_config, "a2_scale", None),
            block_shape=None,
            global_num_experts=global_num_experts,
            expert_map=aiter_expert_map,
            routed_scaling_factor=1.0,
            use_weight_shuffle=bool(
                getattr(aiter_config, "need_shuffle", False)
            ),
            output_dtype=output_dtype,
        )


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

    w1 = _required_tensor(layer, "w13_weight")
    w2 = _required_tensor(layer, "w2_weight")
    if w1.ndim != 3 or w2.ndim != 3 or w1.shape[0] != w2.shape[0]:
        raise HcuCompressedTensorsMoeError(
            "AITER FP8-W8A8 MoE expects compatible rank-3 expert weights"
        )
    activation = getattr(getattr(layer, "activation", None), "value", None)
    if activation is None:
        activation = getattr(layer, "activation", None)
    if activation is None:
        raise HcuCompressedTensorsMoeError(
            "AITER FP8-W8A8 MoE requires a layer activation"
        )
    activation = str(activation)
    from aiter.moe import MoeQuantType

    quant_type = getattr(MoeQuantType, "FP8_W8A8", None)
    if quant_type is None:
        raise HcuCompressedTensorsMoeError(
            "AITER does not expose the required FP8_W8A8 MoE quant type"
        )
    logical_e, logical_n1, logical_n2, logical_k = _quantized_problem_dimensions(
        x,
        w1,
        w2,
        activation=activation,
    )
    problem = AiterMoeProblem(
        M=int(x.shape[0]),
        E=logical_e,
        N1=logical_n1,
        N2=logical_n2,
        K=logical_k,
        top_k=int(topk_ids.shape[1]),
        block_size=0,
        dtype=x.dtype,
        device=x.device,
        quant_type=quant_type,
        activation=activation,
        use_shuffle=bool(henvs.VLLM_HCU_USE_AITER_MOE_SHUFFLE),
    )
    installed_solution = _installed_weight_solution(
        w1, w2, label="AITER FP8-W8A8"
    )
    moe_config = (
        None
        if installed_solution == "native"
        else select_aiter_moe_config(
            problem,
            cache_owner=w1,
            solution_type=installed_solution,
        )
    )
    w1_scale = _required_tensor(layer, "w13_weight_scale")
    w2_scale = _required_tensor(layer, "w2_weight_scale")
    global_num_experts = getattr(layer, "global_num_experts", -1)
    expert_map = getattr(layer, "_expert_map", None)
    expert_mask = getattr(layer, "expert_mask", None)
    if moe_config is None:
        if installed_solution not in (None, "native"):
            raise HcuCompressedTensorsMoeError(
                "AITER has no solution compatible with installed FP8 weights"
            )
        from vllm.model_executor.layers.fused_moe.fused_moe import (
            fused_experts_impl,
        )

        return fused_experts_impl(
            x,
            w1,
            w2,
            topk_weights,
            topk_ids,
            activation=activation,
            apply_router_weight_on_input=False,
            use_fp8_w8a8=True,
            use_int8_w8a8=False,
            use_int8_w8a16=False,
            use_int4_w4a16=False,
            per_channel_quant=True,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            a1_scale=getattr(layer, "w13_input_scale", None),
            a2_scale=getattr(layer, "w2_input_scale", None),
            block_shape=None,
        )

    prepared_w1, prepared_w2 = _weights_for_selected_config(
        w1,
        w2,
        moe_config,
        installed_solution=installed_solution,
    )
    prepared_w1_scale, prepared_w2_scale = _scales_for_selected_config(
        w1_scale,
        w2_scale,
        moe_config,
    )
    aiter_expert_map = aiter_expert_map_for_solution(
        expert_map,
        moe_config,
        int(global_num_experts),
        expert_mask=expert_mask,
    )
    from vllm_hcu.model_executor.layers.fused_moe.aiter_runtime import (
        aiter_asm_boltops_fp8_quant_context,
    )

    with aiter_asm_boltops_fp8_quant_context(
        enabled=_enum_token(getattr(moe_config, "solution_type", None)) == "ASM"
    ):
        return execute_aiter_moe(
            moe_config,
            hidden_states=x if i_q is None else i_q,
            w1=prepared_w1,
            w2=prepared_w2,
            topk_weights=topk_weights.to(torch.float32),
            topk_ids=topk_ids.to(torch.int32),
            inplace=False,
            activation=activation,
            w1_scale=prepared_w1_scale,
            w2_scale=prepared_w2_scale,
            w1_zp=None,
            w2_zp=None,
            a1_scale=(
                i_s
                if i_s is not None
                else getattr(layer, "w13_input_scale", None)
            ),
            a2_scale=getattr(layer, "w2_input_scale", None),
            block_shape=None,
            global_num_experts=global_num_experts,
            expert_map=aiter_expert_map,
            routed_scaling_factor=1.0,
            use_weight_shuffle=bool(
                getattr(moe_config, "need_shuffle", False)
            ),
            output_dtype=None if i_q is None else x.dtype,
        )


def process_dpsk_deepgemm_weights(method: object, layer: object) -> None:
    """Run the HCU-owned expert post-load step for ``deep_gemm``."""

    if _enum_token(getattr(method, "fp8_backend", None)) != "HCU_DEEPGEMM":
        return
    moe_kernel = getattr(method, "moe_kernel", None)
    fused_experts = getattr(moe_kernel, "fused_experts", None)
    experts = getattr(fused_experts, "experts", fused_experts)
    process = getattr(experts, "process_weights_after_loading", None)
    if not callable(process):
        raise HcuCompressedTensorsMoeError(
            "HCU_DEEPGEMM was selected, but the HCU expert post-load hook "
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
    config = config_builder(
        w1_scale=_required_tensor(layer, "w13_weight_scale"),
        w2_scale=_required_tensor(layer, "w2_weight_scale"),
        w1_zp=_required_tensor(layer, "w13_qzeros"),
        w2_zp=_required_tensor(layer, "w2_qzeros"),
        block_shape=[0, group_size],
    )
    prewarm_aiter_w4a16_moe(method, layer, config)
    return config


__all__ = [
    "HcuCompressedTensorsMoeError",
    "apply_aiter_quantized_moe",
    "apply_aiter_w4a8_moe",
    "apply_aiter_w8a8_fp8_moe",
    "apply_vllm_w4a8_moe",
    "build_aiter_w4a16_quant_config",
    "create_aiter_w4a16_qzeros",
    "install_aiter_moe_weight_layout",
    "install_aiter_moe_scale_layout",
    "mark_aiter_moe_native_layout",
    "prewarm_aiter_w4a8_moe",
    "prepare_vllm_w4a8_moe",
    "process_dpsk_deepgemm_weights",
]
