# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# Modified by Hygon Information Technology Co., Ltd., 2026.

import torch
from vllm.triton_utils import tl, triton


def compute_global_topk_indices_and_lens(
    topk_indices: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    is_valid_token: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map local topk indices to global KV cache slots and count valid entries.

    Fuses three operations into a single kernel:
    1. Block-table lookup (local index → global slot id)
    2. Valid-entry counting (topk_lens per token)
    3. Masking padding tokens to length 0
    """
    num_tokens = topk_indices.shape[0]
    global_topk_indices = torch.empty_like(topk_indices)
    topk_lens = torch.empty(num_tokens, dtype=torch.int32, device=topk_indices.device)
    _compute_global_topk_indices_and_lens_kernel[(num_tokens,)](
        global_topk_indices,
        global_topk_indices.stride(0),
        topk_lens,
        topk_indices,
        topk_indices.stride(0),
        topk_indices.shape[-1],
        token_to_req_indices,
        block_table,
        block_table.shape[0],
        block_table.shape[1],
        block_table.stride(0),
        block_table.stride(1),
        block_size,
        is_valid_token,
        TRITON_BLOCK_SIZE=1024,
    )
    return global_topk_indices, topk_lens


@triton.jit
def _compute_global_topk_indices_and_lens_kernel(
    global_topk_indices_ptr,
    global_topk_indices_stride,
    topk_lens_ptr,
    topk_indices_ptr,
    topk_indices_stride,
    topk,
    token_to_req_indices_ptr,
    block_table_ptr,
    num_reqs,
    num_blocks_per_req,
    block_table_stride,
    block_column_stride,
    block_size,
    is_valid_token_ptr,
    TRITON_BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    is_valid_token = tl.load(is_valid_token_ptr + token_idx)
    req_idx = tl.load(token_to_req_indices_ptr + token_idx)

    if not is_valid_token or req_idx < 0 or req_idx >= num_reqs:
        for i in range(0, topk, TRITON_BLOCK_SIZE):
            offset = i + tl.arange(0, TRITON_BLOCK_SIZE)
            mask = offset < topk
            tl.store(
                global_topk_indices_ptr
                + token_idx * global_topk_indices_stride
                + offset,
                -1,
                mask=mask,
            )
        tl.store(topk_lens_ptr + token_idx, 0)
        return

    count = tl.zeros((), dtype=tl.int32)
    for i in range(0, topk, TRITON_BLOCK_SIZE):
        offset = i + tl.arange(0, TRITON_BLOCK_SIZE)
        mask = offset < topk

        local_idx = tl.load(
            topk_indices_ptr + token_idx * topk_indices_stride + offset,
            mask=mask,
            other=-1,
        )

        block_indices = local_idx // block_size
        is_valid = (local_idx >= 0) & (block_indices < num_blocks_per_req)
        block_numbers = tl.load(
            block_table_ptr + req_idx * block_table_stride + block_indices * block_column_stride,
            mask=mask & is_valid,
            other=-1,
        )
        block_offsets = local_idx % block_size

        slot_ids = block_numbers * block_size + block_offsets
        is_valid = mask & is_valid & (block_numbers >= 0)
        # FlashMLA consumes the first topk_lens entries. Keep surviving slots
        # in order and place discarded entries at the far end of the row.
        rank = tl.cumsum(is_valid.to(tl.int32), axis=0)
        dest = tl.where(is_valid, count + rank - 1, topk - 1 - offset + count + rank)
        slot_ids = tl.where(is_valid, slot_ids, -1)
        tl.store(
            global_topk_indices_ptr + token_idx * global_topk_indices_stride + dest,
            slot_ids,
            mask=mask,
        )
        count += tl.sum(is_valid.to(tl.int32), axis=0)

    tl.store(topk_lens_ptr + token_idx, count)
