# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Step5 router-bias top-k kernel with configurable index dtype."""

import operator

import torch

from vllm_hcu.ops.step5.driver_cache import (
    KernelLaunchSpec,
    get_driver_launcher,
)
from vllm.triton_utils import tl, triton

# Adapted from optimus_jit==0.1.10.post8+gitcfde41ba's
# optimus_triton/router_bias_topk.py. That version cannot materialize routing
# indices in a caller-selected dtype. This local variant writes int32 or int64
# directly while retaining the source implementation's driver-launch machinery.
# CustomRoutingRouter may still normalize the returned dtype downstream.
# Keep a separate launcher cache because the output index pointer type varies.
_router_bias_topk_driver_launcher = get_driver_launcher("step5_router_bias_topk_cache")


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [1, 2, 4, 8]
        for num_stages in [2, 3, 4, 5]
    ],
    key=["E", "TOPK"],
)
@triton.jit
def router_bias_topk_kernel(
    gating_ptr,
    bias_ptr,
    out_w_ptr,
    out_i_ptr,
    stride_gm,
    stride_om,
    E: tl.constexpr,
    TOPK: tl.constexpr,
    RENORM: tl.constexpr,
    CHECK_NAN: tl.constexpr,
    BLOCK_E: tl.constexpr,
    ROUTED_SCALING_FACTOR: tl.constexpr,
    NAN_ROW_I_OUT: tl.constexpr,
):
    pid = tl.program_id(0)

    offs_e = tl.arange(0, BLOCK_E)
    mask_e = offs_e < E

    row_ptr = gating_ptr + pid * stride_gm + offs_e
    tl.multiple_of(row_ptr, 8)
    tl.max_contiguous(offs_e, 128)

    gating = tl.load(row_ptr, mask=mask_e, other=0)
    gate_prob = tl.sigmoid(gating.to(tl.float32))

    bias = tl.load(bias_ptr + offs_e, mask=mask_e, other=0).to(tl.float32)
    bias = tl.where(mask_e, bias, -float("inf"))
    scores = tl.where(mask_e, gate_prob + bias, -float("inf"))

    if CHECK_NAN:
        gating_nan = gating != gating
        has_bad = tl.max(gating_nan.to(tl.int32), axis=0) > 0

    # Triton vector lengths must be powers of two, while the model's top-k
    # need not be. Keep selecting TOPK experts and mask the padded lanes below.
    BLOCK_TOPK: tl.constexpr = triton.next_power_of_2(TOPK)
    weights = tl.zeros((BLOCK_TOPK,), dtype=tl.float32)
    indices = tl.zeros((BLOCK_TOPK,), dtype=tl.int32)
    weight_sum = 0.0
    topk_offsets = tl.arange(0, BLOCK_TOPK)

    for k in tl.static_range(TOPK):
        max_score, max_index = tl.max(scores, axis=0, return_indices=True)
        max_index = max_index.to(tl.int32)

        selected_gating = tl.load(
            gating_ptr + pid * stride_gm + max_index, mask=True, other=0
        ).to(tl.float32)
        selected_prob = tl.sigmoid(selected_gating)

        weights = tl.where(topk_offsets == k, selected_prob, weights)
        indices = tl.where(topk_offsets == k, max_index, indices)

        weight_sum += selected_prob
        scores = tl.where(offs_e == max_index, -float("inf"), scores)

    if RENORM:
        weights = weights / (weight_sum + 1e-20)

    if ROUTED_SCALING_FACTOR != 1.0:
        weights = weights * ROUTED_SCALING_FACTOR

    if CHECK_NAN:
        weights = tl.where(has_bad, 0.0, weights)
        indices = tl.where(has_bad, NAN_ROW_I_OUT, indices)

    offsets = tl.arange(0, BLOCK_TOPK)
    tl.store(out_w_ptr + pid * stride_om + offsets, weights, mask=offsets < TOPK)
    # Triton casts the int32 expert ID to the output pointer's element type,
    # allowing the caller to request either int32 or int64 without a follow-up
    # tensor conversion.
    tl.store(out_i_ptr + pid * stride_om + offsets, indices, mask=offsets < TOPK)


def router_bias_triton_func(
    gating_output: torch.Tensor,
    router_bias: torch.Tensor | None,
    topk: int,
    renormalize: bool,
    check_nan: bool = True,
    routed_scaling_factor: float = 1.0,
    nan_row_i_out: int = 0,
    indices_dtype: torch.dtype = torch.int32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute router-bias top-k with indices materialized in the requested dtype."""
    assert gating_output.is_cuda, "gating_output must be on CUDA"
    assert gating_output.ndim == 2
    assert router_bias is not None, "router_bias must be provided"
    num_tokens, num_experts = gating_output.shape
    assert 0 < topk <= num_experts
    if indices_dtype not in (torch.int32, torch.int64):
        raise ValueError(
            f"indices_dtype must be torch.int32 or torch.int64, got {indices_dtype}"
        )

    try:
        nan_row_i_out = operator.index(nan_row_i_out)
    except TypeError as exc:
        raise TypeError("nan_row_i_out must be an integer") from exc
    if nan_row_i_out < -(1 << 31) or nan_row_i_out >= (1 << 31):
        raise ValueError("nan_row_i_out must fit in int32")

    bias = router_bias.to(gating_output.device)
    assert bias.numel() == num_experts

    topk_weights = torch.empty(
        (num_tokens, topk), device=gating_output.device, dtype=torch.float32
    )
    topk_ids = torch.empty(
        (num_tokens, topk), device=gating_output.device, dtype=indices_dtype
    )

    block_experts = 1 << (num_experts - 1).bit_length()
    if block_experts > 1024:
        raise ValueError(
            f"num_experts={num_experts} is too large for single-block top-k "
            f"(block_experts={block_experts})"
        )

    grid = (num_tokens,)
    kernel_args = (
        gating_output,
        bias,
        topk_weights,
        topk_ids,
        gating_output.stride(0),
        topk_weights.stride(0),
    )
    kernel_spec = KernelLaunchSpec(
        kernel_id="step5_router_bias_topk",
        runtime_args=kernel_args,
        grid_fn=lambda: grid,
        autotuner=router_bias_topk_kernel,
        kernel_fn=lambda grid_meta: router_bias_topk_kernel[grid_meta](
            *kernel_args,
            E=num_experts,
            TOPK=topk,
            RENORM=renormalize,
            CHECK_NAN=check_nan,
            BLOCK_E=block_experts,
            ROUTED_SCALING_FACTOR=routed_scaling_factor,
            NAN_ROW_I_OUT=nan_row_i_out,
        ),
        meta_fn=lambda _grid_meta: (
            num_experts,
            topk,
            renormalize,
            check_nan,
            block_experts,
            routed_scaling_factor,
            nan_row_i_out,
        ),
        autotuned_meta_fn=lambda _grid_meta: (num_experts, topk),
        enforce_driver_launch_on_register=True,
    )
    _router_bias_topk_driver_launcher(kernel_spec)

    return topk_weights, topk_ids
