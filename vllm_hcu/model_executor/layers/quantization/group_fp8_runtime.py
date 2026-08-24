# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""HCU Triton fallback for dynamic per-token-group FP8 quantization."""

from __future__ import annotations

import torch
from vllm.model_executor.layers.quantization.utils import fp8_utils
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    get_fp8_min_max,
)
from vllm.platforms import current_platform
from vllm.triton_utils import triton
from vllm.utils.deep_gemm import is_deep_gemm_e8m0_used


def per_token_group_quant_fp8(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
    dtype: torch.dtype | None = None,
    column_major_scales: bool = False,
    tma_aligned_scales: bool = False,
    out_q: torch.Tensor | None = None,
    use_ue8m0: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a contiguous 2-D tensor without dispatching to NVIDIA ``_C``.

    The kernel is vLLM's own Triton reference implementation, invoked
    directly so HCU does not take vLLM's ``is_cuda_alike`` extension branch.
    HY V4's indexer requires row-major UE8M0 scales with groups of 128.
    """

    if x.ndim != 2:
        raise ValueError(f"HCU group FP8 quantization requires rank 2, got {x.ndim}")
    if group_size <= 0 or x.shape[-1] % group_size != 0:
        raise ValueError(
            f"last dimension {x.shape[-1]} must be divisible by {group_size}"
        )
    if x.stride(-1) != 1:
        raise ValueError("HCU group FP8 quantization requires contiguous groups")
    if column_major_scales or tma_aligned_scales:
        raise ValueError("HCU HY V4 indexer requires row-major FP8 scales")

    dtype = current_platform.fp8_dtype() if dtype is None else dtype
    use_ue8m0 = (
        is_deep_gemm_e8m0_used() if use_ue8m0 is None else use_ue8m0
    )
    if out_q is not None and out_q.shape != x.shape:
        raise ValueError(
            f"out_q shape {tuple(out_q.shape)} must match {tuple(x.shape)}"
        )
    quantized = (
        torch.empty(x.shape, device=x.device, dtype=dtype)
        if out_q is None
        else out_q
    )
    scales = torch.empty(
        (x.shape[0], x.shape[1] // group_size),
        device=x.device,
        dtype=torch.float32,
    )

    kernel = getattr(fp8_utils, "_per_token_group_quant_fp8", None)
    if kernel is None:
        raise RuntimeError(
            "vLLM Triton _per_token_group_quant_fp8 kernel is unavailable"
        )
    fp8_min, fp8_max = get_fp8_min_max()
    block = triton.next_power_of_2(group_size)
    groups = x.numel() // group_size
    kernel[(groups,)](
        x,
        quantized,
        scales,
        group_size,
        x.shape[1],
        x.stride(0),
        eps,
        fp8_min=fp8_min,
        fp8_max=fp8_max,
        use_ue8m0=use_ue8m0,
        BLOCK=block,
        num_warps=min(max(block // 256, 1), 8),
        num_stages=1,
    )
    return quantized, scales


__all__ = ["per_token_group_quant_fp8"]
