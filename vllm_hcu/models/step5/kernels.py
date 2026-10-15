# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Step5 biased router dispatch on HCU."""

import importlib
import torch


def router_bias_func(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    router_bias: torch.Tensor | None = None,
    routed_scaling_factor: float = 1.0,
    nan_row_i_out: int = 0,
    indices_dtype: torch.dtype = torch.int32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dispatch Step5 biased top-k to the model-scoped Triton kernel."""
    del hidden_states
    if indices_dtype not in (torch.int32, torch.int64):
        raise ValueError(
            "Step5 router indices_dtype must be torch.int32 or torch.int64, "
            f"got {indices_dtype}."
        )

    # The Triton kernel also runs on ROCm/HCU (is_cuda is True there) and is
    # 7-10x faster than the eager fallback below, which runs inside the opaque
    # moe_forward op and so never gets fused by Inductor.
    if not gating_output.is_cuda:
        if router_bias is None:
            raise ValueError("Step5 router_bias must be provided.")
        scores = gating_output.float().sigmoid()
        _, topk_ids = torch.topk(
            scores + router_bias.to(device=gating_output.device, dtype=torch.float32),
            k=topk,
            dim=-1,
        )
        topk_weights = scores.gather(1, topk_ids)
        if renormalize:
            topk_weights /= topk_weights.sum(dim=-1, keepdim=True) + 1e-20
        topk_weights *= routed_scaling_factor

        invalid_rows = gating_output.isnan().any(dim=-1, keepdim=True)
        topk_weights = torch.where(invalid_rows, 0, topk_weights)
        topk_ids = torch.where(invalid_rows, nan_row_i_out, topk_ids)
        return topk_weights, topk_ids.to(indices_dtype)

    router_bias_triton_func = importlib.import_module(
        "vllm_hcu.ops.step5.router_bias"
    ).router_bias_triton_func
    if gating_output.stride(-1) != 1:
        gating_output = gating_output.contiguous()
    return router_bias_triton_func(
        gating_output,
        router_bias,
        topk,
        renormalize=renormalize,
        routed_scaling_factor=routed_scaling_factor,
        nan_row_i_out=nan_row_i_out,
        indices_dtype=indices_dtype,
    )
