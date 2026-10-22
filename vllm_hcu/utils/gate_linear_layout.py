# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Post-load storage preparation for HCU BF16-to-FP32 router GEMMs."""

from __future__ import annotations

import torch

# Set on the audited GateLinear class by patch_gate_linear.apply_to_module.
HCU_GATE_LINEAR_MARKER = "_vllm_hcu_nn_layout_applied"


def prepare_gate_linear_nt_weight(
    layer: torch.nn.Module, *, use_nn: bool
) -> bool:
    """Keep the logical [K,N] parameter, but give its storage an NT layout.

    Called after loading and before profiling/graph capture. The original
    Parameter and its loader/sharding attributes survive; only its storage is
    rearranged. No detached cache is retained, so copy_ (including data.copy_)
    reloads update the very tensor used by forward. CPU dispatch is handled by
    the caller and must not be changed by this optimization.
    """
    weight = getattr(layer, "weight", None)
    if (
        not use_nn
        or not getattr(layer, HCU_GATE_LINEAR_MARKER, False)
        or not getattr(layer, "allow_cublas_router_gemm", False)
        or getattr(layer, "out_dtype", None) != torch.float32
        or not isinstance(weight, torch.Tensor)
        or weight.ndim != 2
        or weight.dtype != torch.bfloat16
        or weight.is_meta
        or weight.layout != torch.strided
    ):
        return False
    if weight.t().is_contiguous():
        return False

    # Two transposes preserve shape and values. The intervening materialization
    # changes [K,N] strides from (N,1) to (1,K), selecting the NT BLAS path.
    # Assign data rather than replacing Parameter: weight loaders and state_dict
    # must continue to see the same registered object and logical dimensions.
    with torch.no_grad():
        weight.data = weight.detach().t().contiguous().t()
    return True
