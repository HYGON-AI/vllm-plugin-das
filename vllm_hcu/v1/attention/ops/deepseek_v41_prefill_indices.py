# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Fused TopK/SWA index construction for the DSV4.1 AMD prefill path.

Adapted from SGLang's ``kernels/ops/attention/dsv4/sparse_prefill_kernels.py``
at ebe659200 (Apache-2.0). Keep request-parallel device-side construction, but
use vLLM's M/N workspace layout and its Torch fallback's bounds/sentinel rules.
Each row, including padding, is fully written: no separate fill or nonzero.
"""

import math

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _combine_topk_swa_indices_kernel(
    indices_ptr,
    starts_ptr,
    seq_lens_ptr,
    gather_lens_ptr,
    out_ptr,
    lengths_ptr,
    M,
    N,
    NUM_TOKENS: tl.constexpr,
    INPUT_WIDTH: tl.constexpr,
    LOGICAL_TOPK: tl.constexpr,
    OUTPUT_WIDTH: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    req = tl.program_id(0)
    worker = tl.program_id(1)
    workers = tl.num_programs(1)
    base = tl.load(starts_ptr).to(tl.int64)
    start = tl.load(starts_ptr + req).to(tl.int64) - base
    end = tl.load(starts_ptr + req + 1).to(tl.int64) - base
    query_len = end - start
    seq_len = tl.load(seq_lens_ptr + req).to(tl.int64)
    gather_start = seq_len - tl.load(gather_lens_ptr + req).to(tl.int64)
    req_base = M.to(tl.int64) * req
    offsets = tl.arange(0, BLOCK)

    for row in range(start + worker, end, workers):
        # Defense in depth for mixed warmup metadata: never address a row
        # outside this chunk. Valid metadata must partition NUM_TOKENS rows.
        if (row >= 0) & (row < NUM_TOKENS):
            pos = seq_len - query_len + row - start
            topk_len = tl.maximum(
                0, tl.minimum((pos + 1) // COMPRESS_RATIO, LOGICAL_TOPK)
            )
            swa_len = tl.maximum(0, tl.minimum(pos + 1, WINDOW_SIZE))
            topk_mask = offsets < topk_len
            values = tl.load(
                indices_ptr + row * INPUT_WIDTH + offsets,
                mask=topk_mask,
                other=-1,
            ).to(tl.int32)
            valid = topk_mask & (values >= 0) & (values < N)
            combined = tl.where(valid, values + req_base, -1)
            swa_offset = offsets - topk_len
            swa_mask = (swa_offset >= 0) & (swa_offset < swa_len)
            swa_values = (
                req_base + N + swa_offset + pos - swa_len + 1 - gather_start
            )
            combined = tl.where(swa_mask, swa_values, combined)
            tl.store(
                out_ptr + row * OUTPUT_WIDTH + offsets,
                combined.to(tl.int32),
                mask=offsets < OUTPUT_WIDTH,
            )
            tl.store(lengths_ptr + row, (topk_len + swa_len).to(tl.int32))


def combine_topk_swa_indices(
    topk_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor,
    window_size: int,
    compress_ratio: int,
    topk: int,
    M: int,
    N: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match the Torch combiner without device-to-host size queries.

    Metadata must partition all input rows, with nondecreasing query starts;
    their first value may be nonzero. As with the original function, these
    device-side content invariants are the caller's responsibility.
    """
    if topk_indices.device.type != "cuda":
        raise ValueError("DSV4.1 Triton prefill indices require a CUDA/HIP tensor")
    if topk_indices.ndim < 2:
        raise ValueError("topk_indices must have a token and an index dimension")
    if topk == 0 and compress_ratio == 0:
        compress_ratio = 1
    if compress_ratio < 1 or topk < 0 or window_size < 0:
        raise ValueError("require compress_ratio >= 1, topk >= 0 and window_size >= 0")
    tensors = (topk_indices, query_start_loc, seq_lens, gather_lens)
    if any(t.device != topk_indices.device for t in tensors):
        raise ValueError("prefill index tensors must be on the same device")
    if any(t.dtype not in (torch.int32, torch.int64) for t in tensors):
        raise ValueError("prefill index tensors must be int32 or int64")
    if (
        seq_lens.ndim != 1
        or gather_lens.shape != seq_lens.shape
        or query_start_loc.ndim != 1
        or query_start_loc.numel() != seq_lens.numel() + 1
    ):
        raise ValueError("inconsistent prefill request metadata shapes")

    num_tokens = topk_indices.shape[0]
    input_width = math.prod(topk_indices.shape[1:])
    indices = topk_indices.reshape(num_tokens, input_width).contiguous()
    output_width = (topk + window_size + 127) // 128 * 128
    combined_indices = torch.empty(
        (num_tokens, output_width), dtype=torch.int32, device=indices.device
    )
    combined_lens = torch.empty(num_tokens, dtype=torch.int32, device=indices.device)
    if num_tokens == 0:
        return combined_indices, combined_lens
    if seq_lens.numel() == 0:
        raise ValueError("nonempty tokens require at least one request")

    # Use the tensor's device/current stream, not the process's default device.
    with torch.cuda.device(indices.device):
        _combine_topk_swa_indices_kernel[(seq_lens.numel(), 128)](
            indices,
            query_start_loc.contiguous(),
            seq_lens.contiguous(),
            gather_lens.contiguous(),
            combined_indices,
            combined_lens,
            M,
            N,
            NUM_TOKENS=num_tokens,
            INPUT_WIDTH=input_width,
            LOGICAL_TOPK=min(topk, input_width),
            OUTPUT_WIDTH=output_width,
            COMPRESS_RATIO=compress_ratio,
            WINDOW_SIZE=window_size,
            BLOCK=triton.next_power_of_2(max(1, output_width)),
        )
    return combined_indices, combined_lens
