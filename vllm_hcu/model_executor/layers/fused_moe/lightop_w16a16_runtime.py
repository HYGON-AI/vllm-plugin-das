# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""LightOp W16A16 Marlin layout and runtime helpers.

The packed layout is backend-specific.  Selection must therefore happen before
weight conversion; callers must never send these tensors to an official vLLM
or AITER/Triton expert implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


_PACKED_MARKER = "_hcu_lightop_w16a16_packed"
_GENERATION_MARKER = "_hcu_lightop_w16a16_generation"
_LAYOUT_MARKER = "_hcu_lightop_w16a16_layout"


@dataclass(frozen=True, slots=True)
class LightopW16A16Layout:
    logical_w13_shape: tuple[int, ...]
    logical_w2_shape: tuple[int, ...]
    packed_w13_shape: tuple[int, ...]
    packed_w2_shape: tuple[int, ...]


def _weight_permutation(device: torch.device) -> torch.Tensor:
    # One 16x32 tile is distributed across a 64-lane wave.  Keep this literal
    # construction in HCU-owned code so packing does not depend on SGLang or
    # an AITER checkout.
    permutation = [
        ((lane // 16) * 4 + row) * 32 + (lane % 16) * 2 + column
        for lane in range(64)
        for column in range(2)
        for row in range(4)
    ]
    return torch.tensor(permutation, dtype=torch.long, device=device)


def _pack_one_weight(weight: torch.Tensor) -> torch.Tensor:
    # Logical expert matrices are stored [N, K].  LightOp consumes tiled
    # [K, N] data with a 16x32 tile and a lane-local permutation.
    transposed = weight.transpose(0, 1).contiguous()
    size_k, size_n = transposed.shape
    permutation = _weight_permutation(transposed.device)
    packed = transposed.reshape(size_k // 16, 16, size_n // 32, 32)
    packed = packed.permute(0, 2, 1, 3).reshape(size_k // 16, size_n * 16)
    return packed.reshape(-1, permutation.numel())[:, permutation].reshape(
        packed.shape
    ).contiguous()


def _layout_from_markers(
    w13: torch.Tensor,
    w2: torch.Tensor,
) -> LightopW16A16Layout | None:
    if not bool(getattr(w13, _PACKED_MARKER, False)):
        return None
    if not bool(getattr(w2, _PACKED_MARKER, False)):
        raise ValueError("W16A16 packed layout marker is missing from w2")
    layout13 = getattr(w13, _LAYOUT_MARKER, None)
    layout2 = getattr(w2, _LAYOUT_MARKER, None)
    generation13 = getattr(w13, _GENERATION_MARKER, None)
    generation2 = getattr(w2, _GENERATION_MARKER, None)
    if (
        not isinstance(layout13, LightopW16A16Layout)
        or layout13 != layout2
        or generation13 != generation2
    ):
        raise ValueError("W16A16 packed weight markers are inconsistent")
    if layout13.packed_w13_shape != tuple(w13.shape):
        raise ValueError("W16A16 packed w13 shape does not match its layout marker")
    if layout13.packed_w2_shape != tuple(w2.shape):
        raise ValueError("W16A16 packed w2 shape does not match its layout marker")
    return layout13


def mark_lightop_w16a16_weights(
    w13: torch.Tensor,
    w2: torch.Tensor,
    layout: LightopW16A16Layout,
    *,
    generation: int = 1,
) -> None:
    for weight in (w13, w2):
        setattr(weight, _PACKED_MARKER, True)
        setattr(weight, _GENERATION_MARKER, generation)
        setattr(weight, _LAYOUT_MARKER, layout)


def pack_lightop_w16a16_weights(
    w13: torch.Tensor,
    w2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, LightopW16A16Layout]:
    """Pack canonical vLLM expert weights without mutating the inputs."""

    installed = _layout_from_markers(w13, w2)
    if installed is not None:
        return w13, w2, installed
    if w13.dim() != 3 or w2.dim() != 3:
        raise ValueError("W16A16 weights must both be three-dimensional")
    if not w13.is_contiguous() or not w2.is_contiguous():
        raise ValueError("W16A16 logical weights must be contiguous")
    experts, two_n, hidden = map(int, w13.shape)
    experts2, hidden2, intermediate = map(int, w2.shape)
    if experts != experts2:
        raise ValueError("W16A16 w13/w2 expert counts do not match")
    if hidden != hidden2 or two_n != 2 * intermediate:
        raise ValueError("W16A16 weights do not use vLLM [E,2N,K]/[E,K,N] layout")
    if hidden % 32 != 0 or intermediate % 16 != 0:
        raise ValueError("W16A16 Marlin packing requires K % 32 == 0 and N % 16 == 0")
    if w13.dtype != w2.dtype or w13.device != w2.device:
        raise ValueError("W16A16 weights must have the same dtype and device")

    packed13 = torch.stack([_pack_one_weight(weight) for weight in w13])
    packed2 = torch.stack([_pack_one_weight(weight) for weight in w2])
    layout = LightopW16A16Layout(
        logical_w13_shape=tuple(w13.shape),
        logical_w2_shape=tuple(w2.shape),
        packed_w13_shape=tuple(packed13.shape),
        packed_w2_shape=tuple(packed2.shape),
    )
    mark_lightop_w16a16_weights(packed13, packed2, layout)
    return packed13, packed2, layout


def lightop_w16a16_layout(
    w13: torch.Tensor,
    w2: torch.Tensor,
) -> LightopW16A16Layout:
    layout = _layout_from_markers(w13, w2)
    if layout is None:
        raise RuntimeError(
            "LightOp W16A16 experts require weights packed by the HCU oracle"
        )
    return layout


def _load_lightop_w16a16() -> tuple[Any, Any] | None:
    try:
        from lightop import activation as lightop_activation
        from lightop import moe as lightop_moe
    except (ImportError, AttributeError):
        return None
    required_moe = (
        "get_moe_cuda_marlin_config_w16a16",
        "moe_gemm_marlin_w16a16",
        "moe_align_block_size_out",
        "moe_sum",
    )
    if not all(callable(getattr(lightop_moe, name, None)) for name in required_moe):
        return None
    if not callable(getattr(lightop_activation, "fuse_silu_and_mul", None)):
        return None
    return lightop_moe, lightop_activation


def is_lightop_w16a16_available() -> bool:
    return _load_lightop_w16a16() is not None


def _logical_problem(moe_config: object) -> tuple[int, int, int, int]:
    try:
        experts = int(getattr(moe_config, "num_experts"))
        hidden = int(getattr(moe_config, "hidden_dim"))
        intermediate = int(
            getattr(moe_config, "intermediate_size_per_partition")
        )
        topk = int(getattr(moe_config, "experts_per_token"))
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid vLLM W16A16 MoE problem description") from exc
    return experts, hidden, intermediate, topk


def select_lightop_w16a16_config(
    moe_config: object,
    expected_m: int,
    device: torch.device | str | int | None,
) -> tuple[dict[str, object], dict[str, object]] | None:
    """Return validated LightOp configs, or ``None`` before layout mutation."""

    exports = _load_lightop_w16a16()
    if exports is None:
        return None
    if expected_m <= 0:
        return None
    experts, hidden, intermediate, topk = _logical_problem(moe_config)
    if hidden % 32 or intermediate % 16 or experts <= 0 or topk <= 0:
        return None
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    else:
        device = torch.device(device)
    if device.type != "cuda":
        return None
    index = torch.cuda.current_device() if device.index is None else device.index
    properties = torch.cuda.get_device_properties(index)
    dtype = getattr(moe_config, "in_dtype", None)
    lightop_moe, _ = exports
    result = lightop_moe.get_moe_cuda_marlin_config_w16a16(
        experts,
        int(expected_m),
        2 * intermediate,
        hidden,
        hidden,
        intermediate,
        topk,
        torch.cuda.get_device_name(index),
        properties.multi_processor_count,
        dtype,
    )
    if not isinstance(result, tuple) or len(result) != 3:
        raise RuntimeError(
            "LightOp get_moe_cuda_marlin_config_w16a16 returned an incompatible ABI"
        )
    config1, config2, status = result
    if not bool(status):
        return None
    if not isinstance(config1, dict) or not isinstance(config2, dict):
        raise RuntimeError("LightOp W16A16 config lookup did not return dictionaries")
    for config in (config1, config2):
        block_m = config.get("BLOCK_SIZE_M")
        if not isinstance(block_m, int) or block_m <= 0:
            raise RuntimeError("LightOp W16A16 config is missing a valid BLOCK_SIZE_M")
    return config1, config2


def run_lightop_w16a16(
    *,
    output: torch.Tensor,
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    workspace13: torch.Tensor,
    workspace2: torch.Tensor,
    global_num_experts: int,
) -> None:
    """Execute the complete LightOp expert path into vLLM's output buffer."""

    layout = lightop_w16a16_layout(w13, w2)
    exports = _load_lightop_w16a16()
    if exports is None:
        raise RuntimeError(
            "VLLM_HCU_USE_LIGHTOP_W16A16_MOE requires the LightOp W16A16 "
            "config, GEMM, alignment, activation and reduction symbols"
        )
    if hidden_states.dim() != 2 or not hidden_states.is_contiguous():
        raise RuntimeError("LightOp W16A16 hidden states must be contiguous 2-D")
    if hidden_states.dtype != torch.bfloat16:
        raise RuntimeError("LightOp W16A16 vLLM integration supports BF16 only")
    if topk_ids.dim() != 2 or topk_weights.shape != topk_ids.shape:
        raise RuntimeError(
            "LightOp W16A16 top-k ids and weights must have equal 2-D shapes"
        )
    if (
        topk_ids.device != hidden_states.device
        or topk_weights.device != hidden_states.device
    ):
        raise RuntimeError("LightOp W16A16 routing tensors must share the input device")
    experts, two_n, hidden = layout.logical_w13_shape
    _, hidden2, intermediate = layout.logical_w2_shape
    tokens = int(hidden_states.shape[0])
    topk = int(topk_ids.shape[1])
    if hidden != hidden2 or hidden_states.shape[1] != hidden:
        raise RuntimeError(
            "LightOp W16A16 hidden dimension does not match packed weights"
        )
    if two_n != 2 * intermediate or experts != w13.shape[0]:
        raise RuntimeError("LightOp W16A16 logical layout markers are inconsistent")
    if global_num_experts not in (-1, experts):
        raise RuntimeError("LightOp W16A16 TP path requires all experts to be local")

    config_pair = select_lightop_w16a16_config(
        SimpleMoeProblem(experts, hidden, intermediate, topk, hidden_states.dtype),
        tokens,
        hidden_states.device,
    )
    if config_pair is None:
        raise RuntimeError(
            "LightOp W16A16 has no runtime config for the packed MoE shape; "
            "disable VLLM_HCU_USE_LIGHTOP_W16A16_MOE before loading the model"
        )
    config1, config2 = config_pair
    block_m = int(config1["BLOCK_SIZE_M"])
    lightop_moe, lightop_activation = exports

    max_padded = topk_ids.numel() + experts * (block_m - 1)
    max_padded = ((max_padded + block_m - 1) // block_m) * block_m
    sorted_ids = torch.full(
        (max_padded,),
        topk_ids.numel(),
        dtype=torch.int32,
        device=topk_ids.device,
    )
    expert_ids = torch.empty(
        (max_padded // block_m,), dtype=torch.int32, device=topk_ids.device
    )
    num_tokens_post_pad = torch.empty(
        (1,), dtype=torch.int32, device=topk_ids.device
    )
    lightop_moe.moe_align_block_size_out(
        topk_ids,
        experts,
        block_m,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
        None,
        None,
        None,
        is_ep=False,
        is_fuse_fill=False,
    )

    cache1_elements = tokens * topk * two_n
    cache3_elements = tokens * topk * hidden
    if workspace13.numel() < max(cache1_elements, cache3_elements):
        raise RuntimeError("vLLM W16A16 workspace13 is smaller than required")
    if workspace2.numel() < tokens * topk * intermediate:
        raise RuntimeError("vLLM W16A16 workspace2 is smaller than required")
    cache1 = workspace13.reshape(-1)[:cache1_elements].view(-1, two_n)
    cache2 = workspace2.reshape(-1)[: tokens * topk * intermediate].view(
        -1, intermediate
    )
    cache3 = workspace13.reshape(-1)[:cache3_elements].view(-1, hidden)
    lightop_moe.moe_gemm_marlin_w16a16(
        hidden_states,
        w13,
        cache1,
        None,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
        topk,
        config1,
    )
    lightop_activation.fuse_silu_and_mul(cache1, cache2)
    lightop_moe.moe_gemm_marlin_w16a16(
        cache2,
        w2,
        cache3,
        topk_weights,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
        1,
        config2,
    )
    lightop_moe.moe_sum(
        input=cache3.view(tokens, topk, hidden),
        output=output,
        bias=None,
        expert_mask=None,
        num_local_tokens=None,
        factor=1.0,
    )


@dataclass(frozen=True, slots=True)
class SimpleMoeProblem:
    num_experts: int
    hidden_dim: int
    intermediate_size_per_partition: int
    experts_per_token: int
    in_dtype: torch.dtype


__all__ = [
    "LightopW16A16Layout",
    "is_lightop_w16a16_available",
    "lightop_w16a16_layout",
    "mark_lightop_w16a16_weights",
    "pack_lightop_w16a16_weights",
    "run_lightop_w16a16",
    "select_lightop_w16a16_config",
]
