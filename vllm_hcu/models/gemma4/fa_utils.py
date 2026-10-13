# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""HCU FA gather workspace owned by the Gemma4 plugin, not the FA package."""

from __future__ import annotations

import torch


def gemma4_flash_attention(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_q: int,
    seqused_k: torch.Tensor,
    max_seqlen_k: int,
    softmax_scale: float,
    block_table: torch.Tensor,
) -> torch.Tensor:
    """Run the admitted BF16 D512 causal path with NHD paged KV.

    The caller checks the attention semantics and layout. Capture cannot read
    device sequence lengths: reserve the per-request upper bound so replay can
    grow beyond the dummy one-token KV used during capture. This retains the
    existing minimum capacity and does not impose a new memory-budget policy.
    """
    from flash_attn.flash_attn_interface import flash_attn_cuda

    batch_size = cu_seqlens_q.numel() - 1
    if torch.cuda.is_current_stream_capturing():
        used_tokens = batch_size * max_seqlen_k
    else:
        used_tokens = int(seqused_k.sum().item())
    total_k = max(262144, used_tokens)
    contiguous_k = torch.empty(
        (total_k, k.shape[-2], k.shape[-1]), device=k.device, dtype=k.dtype
    )
    contiguous_v = torch.empty_like(contiguous_k)
    cu_seqlens_k = torch.empty_like(cu_seqlens_q)
    flash_attn_cuda.pagedkv_to_contiguouskv(
        contiguous_k,
        contiguous_v,
        k,
        v,
        block_table,
        seqused_k,
        cu_seqlens_k,
        max_seqlen_k,
        2,  # Native layout 2 is NHD (bshd).
    )
    # Use the HCU native entry point to preserve the caller's output buffer.
    # The public non-paged autograd wrapper does not forward its out argument.
    flash_attn_cuda.varlen_fwd(
        q,
        contiguous_k,
        contiguous_v,
        out,
        cu_seqlens_q,
        cu_seqlens_k,
        None,
        None,
        None,
        None,
        max_seqlen_q,
        max_seqlen_k,
        0.0,
        softmax_scale,
        False,
        True,
        -1,
        -1,
        0.0,
        False,
        None,
        None,
        None,
        None,
        None,
    )
    return out
