# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused DFly draft input preparation.

Three launches replace the per-step torch ops: one per request for the query
block and sampling state, one per KV group for the query slots, and one for
the sparse indices of every draft layer.
"""

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import PAD_SLOT_ID


@triton.jit(do_not_specialize=["num_reqs", "padded_batch"])
def _prepare_dfly_inputs_kernel(
    # Outputs
    input_ids_ptr,
    positions_ptr,
    is_padding_ptr,
    query_start_loc_ptr,
    seq_lens_ptr,
    active_ptr,
    source_rows_ptr,
    anchors_ptr,
    hidden_out_ptr,
    hidden_out_stride,
    mrope_ptr,
    mrope_stride,
    draft_idx_mapping_ptr,
    out_temperature_ptr,
    out_seeds_ptr,
    # Inputs
    target_query_start_loc_ptr,
    target_positions_ptr,
    idx_mapping_ptr,
    num_sampled_ptr,
    num_rejected_ptr,
    last_sampled_ptr,
    last_sampled_stride,
    hidden_in_ptr,
    hidden_in_stride,
    mrope_delta_ptr,
    temperature_ptr,
    seeds_ptr,
    max_model_len_ptr,
    mask_token_id,
    hidden_size,
    num_reqs,
    padded_batch,
    QUERY_WIDTH: tl.constexpr,
    QUERY_BLOCK: tl.constexpr,
    MROPE_DIMS: tl.constexpr,
    HAS_MROPE_DELTA: tl.constexpr,
    HIDDEN_BLOCK: tl.constexpr,
):
    req_idx = tl.program_id(0)
    is_real = req_idx < num_reqs
    max_model_len = tl.load(max_model_len_ptr)

    # Rows past num_reqs are padding: every load below is masked off for them.
    req_state_idx = tl.load(idx_mapping_ptr + req_idx, mask=is_real, other=-1).to(
        tl.int64
    )
    query_end = tl.load(target_query_start_loc_ptr + req_idx + 1, mask=is_real, other=0)
    num_rejected = tl.load(num_rejected_ptr + req_idx, mask=is_real, other=0)
    source_row = tl.where(is_real, query_end - num_rejected - 1, 0).to(tl.int64)
    anchor = tl.load(target_positions_ptr + source_row, mask=is_real, other=-1) + 1
    num_sampled = tl.load(num_sampled_ptr + req_idx, mask=is_real, other=0)
    active = is_real & (num_sampled > 0) & (anchor + QUERY_WIDTH <= max_model_len)
    bonus = tl.load(
        last_sampled_ptr + req_state_idx * last_sampled_stride, mask=active, other=0
    ).to(tl.int32)
    tl.store(
        out_temperature_ptr + req_state_idx,
        tl.load(temperature_ptr + req_state_idx, mask=is_real),
        mask=is_real,
    )
    tl.store(
        out_seeds_ptr + req_state_idx,
        tl.load(seeds_ptr + req_state_idx, mask=is_real),
        mask=is_real,
    )

    tl.store(active_ptr + req_idx, active)
    tl.store(source_rows_ptr + req_idx, source_row)
    tl.store(anchors_ptr + req_idx, anchor)
    tl.store(draft_idx_mapping_ptr + req_idx, tl.where(active, req_state_idx, -1))

    offsets = tl.arange(0, QUERY_BLOCK)
    in_block = offsets < QUERY_WIDTH
    token_idx = req_idx * QUERY_WIDTH + offsets
    positions = tl.where(active, anchor + offsets, offsets)
    input_ids = tl.where(offsets == 0, bonus, mask_token_id)
    tl.store(input_ids_ptr + token_idx, input_ids, mask=in_block)
    tl.store(positions_ptr + token_idx, positions, mask=in_block)
    tl.store(is_padding_ptr + token_idx, active == 0, mask=in_block)
    if MROPE_DIMS > 0:
        delta = 0
        if HAS_MROPE_DELTA:
            delta = tl.load(mrope_delta_ptr + req_state_idx, mask=is_real, other=0)
        rope = tl.where(active, positions + delta, 0)
        for dim in tl.static_range(MROPE_DIMS):
            tl.store(mrope_ptr + dim * mrope_stride + token_idx, rope, mask=in_block)

    tl.store(seq_lens_ptr + req_idx, tl.where(active, anchor, 0) + QUERY_WIDTH)
    tl.store(query_start_loc_ptr + req_idx, req_idx * QUERY_WIDTH)
    if req_idx == padded_batch - 1:
        tl.store(query_start_loc_ptr + padded_batch, padded_batch * QUERY_WIDTH)

    out_row = hidden_out_ptr + req_idx * hidden_out_stride
    in_row = hidden_in_ptr + source_row * hidden_in_stride
    for i in range(0, hidden_size, HIDDEN_BLOCK):
        block = i + tl.arange(0, HIDDEN_BLOCK)
        mask = block < hidden_size
        hidden = tl.load(in_row + block, mask=mask & is_real, other=0.0)
        tl.store(out_row + block, hidden, mask=mask)


@triton.jit
def _dfly_query_slots_kernel(
    slot_mapping_ptr,
    positions_ptr,
    active_ptr,
    block_table_ptr,
    block_table_stride,
    num_blocks_per_req,
    block_size,
    max_model_len_ptr,
    QUERY_WIDTH: tl.constexpr,
    QUERY_BLOCK: tl.constexpr,
    PAD_SLOT_ID: tl.constexpr,
):
    req_idx = tl.program_id(0)
    offsets = tl.arange(0, QUERY_BLOCK)
    in_block = offsets < QUERY_WIDTH
    token_idx = req_idx * QUERY_WIDTH + offsets
    positions = tl.load(positions_ptr + token_idx, mask=in_block, other=0)
    active = tl.load(active_ptr + req_idx)
    max_model_len = tl.load(max_model_len_ptr)
    block_idx = positions // block_size
    valid = active & (positions < max_model_len) & (block_idx < num_blocks_per_req)
    block_id = tl.load(
        block_table_ptr
        + req_idx * block_table_stride
        + tl.minimum(block_idx, num_blocks_per_req - 1),
        mask=in_block & valid,
        other=0,
    ).to(tl.int64)
    # Block 0 is the null block and is never a writable slot.
    valid &= block_id != 0
    slots = block_id * block_size + positions % block_size
    tl.store(
        slot_mapping_ptr + token_idx,
        tl.where(valid, slots, PAD_SLOT_ID),
        mask=in_block,
    )


@triton.jit
def _build_dfly_reuse_indices_kernel(
    out_ptr,
    out_layer_stride,
    out_row_stride,
    snapshot_ptr,
    snapshot_slot_stride,
    snapshot_row_stride,
    layer_snapshot_slots_ptr,
    active_ptr,
    source_rows_ptr,
    anchors_ptr,
    TARGET_TOPK: tl.constexpr,
    LOCAL_WINDOW: tl.constexpr,
    QUERY_WIDTH: tl.constexpr,
    OUT_WIDTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Per query row: target top-k history, local window, then causal block."""
    req_idx = tl.program_id(0)
    query_offset = tl.program_id(1)
    layer = tl.program_id(2)

    active = tl.load(active_ptr + req_idx)
    anchor = tl.load(anchors_ptr + req_idx)
    source_row = tl.load(source_rows_ptr + req_idx).to(tl.int64)
    slot = tl.load(layer_snapshot_slots_ptr + layer).to(tl.int64)
    history_ptr = (
        snapshot_ptr + slot * snapshot_slot_stride + source_row * snapshot_row_stride
    )
    out_row = (
        out_ptr
        + layer * out_layer_stride
        + (req_idx * QUERY_WIDTH + query_offset).to(tl.int64) * out_row_stride
    )
    window_start = anchor - LOCAL_WINDOW
    for i in range(0, OUT_WIDTH, BLOCK_SIZE):
        cols = i + tl.arange(0, BLOCK_SIZE)
        is_history = cols < TARGET_TOPK
        history = tl.load(history_ptr + cols, mask=is_history & active, other=-1)
        history = tl.where((history >= 0) & (history < window_start), history, -1)
        window_col = cols - TARGET_TOPK
        is_window = (window_col >= 0) & (window_col < LOCAL_WINDOW)
        window = tl.maximum(window_start + window_col, -1)
        block_col = window_col - LOCAL_WINDOW
        is_causal = (block_col >= 0) & (block_col <= query_offset)
        value = tl.where(
            is_history,
            history,
            tl.where(is_window, window, tl.where(is_causal, anchor + block_col, -1)),
        )
        value = tl.where(active, value, -1)
        tl.store(out_row + cols, value.to(tl.int32), mask=cols < OUT_WIDTH)



def _check_query_capacity(query_width, padded_batch, rows):
    if (type(query_width) is not int or not 2 <= query_width <= 17
            or type(padded_batch) is not int or padded_batch < 0
            or rows < padded_batch * query_width):
        raise ValueError("DFly query shape exceeds allocated K+1 capacity")


def _check_tensors(*tensors):
    present = [tensor for tensor in tensors if tensor is not None]
    if not present:
        return
    if any(not t.is_cuda or t.device != present[0].device for t in present):
        raise ValueError("DFly kernels require same-device CUDA/HIP tensors")
    if any(t.ndim and t.stride(-1) != 1 for t in present):
        raise ValueError("DFly kernel innermost dimensions must be contiguous")


def prepare_dfly_inputs(
    *,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    is_padding: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    active: torch.Tensor,
    source_rows: torch.Tensor,
    anchors: torch.Tensor,
    target_hidden_out: torch.Tensor,
    mrope_positions: torch.Tensor | None,
    draft_idx_mapping: torch.Tensor,
    out_temperature: torch.Tensor,
    out_seeds: torch.Tensor,
    target_query_start_loc: torch.Tensor,
    target_positions: torch.Tensor,
    idx_mapping: torch.Tensor,
    num_sampled: torch.Tensor,
    num_rejected: torch.Tensor,
    last_sampled: torch.Tensor,
    target_hidden: torch.Tensor,
    mrope_delta: torch.Tensor | None,
    temperature: torch.Tensor,
    seeds: torch.Tensor,
    max_model_len: torch.Tensor,
    mask_token_id: int,
    query_width: int,
    num_reqs: int,
    padded_batch: int,
) -> None:
    """Fill the query block of `padded_batch` requests. Rows past `num_reqs`
    are inactive padding, and `num_reqs=0` gives an all-padding block for
    dummy runs and capture."""
    _check_query_capacity(query_width, padded_batch, input_ids.numel())
    if not 0 <= num_reqs <= padded_batch:
        raise ValueError("DFly real request count exceeds padded batch")
    if (positions.numel() < padded_batch * query_width
            or is_padding.numel() < padded_batch * query_width
            or query_start_loc.numel() < padded_batch + 1
            or any(t.numel() < padded_batch for t in (
                seq_lens, active, source_rows, anchors, draft_idx_mapping))
            or target_hidden_out.shape[0] < padded_batch):
        raise ValueError("DFly output buffers are undersized")
    if num_reqs > 0 and target_hidden.shape[-1] != target_hidden_out.shape[-1]:
        raise ValueError("DFly fusion requires the target's final hidden states")
    _check_tensors(input_ids, positions, is_padding, query_start_loc, seq_lens,
                   active, source_rows, anchors, target_hidden_out, mrope_positions,
                   draft_idx_mapping, out_temperature, out_seeds,
                   target_query_start_loc, target_positions, idx_mapping,
                   num_sampled, num_rejected, last_sampled, target_hidden,
                   mrope_delta, temperature, seeds, max_model_len)
    if padded_batch == 0:
        return
    _prepare_dfly_inputs_kernel[(padded_batch,)](
        input_ids,
        positions,
        is_padding,
        query_start_loc,
        seq_lens,
        active,
        source_rows,
        anchors,
        target_hidden_out,
        target_hidden_out.stride(0),
        mrope_positions,
        mrope_positions.stride(0) if mrope_positions is not None else 0,
        draft_idx_mapping,
        out_temperature,
        out_seeds,
        target_query_start_loc,
        target_positions,
        idx_mapping,
        num_sampled,
        num_rejected,
        last_sampled,
        last_sampled.stride(0),
        target_hidden,
        target_hidden.stride(0),
        mrope_delta,
        temperature,
        seeds,
        max_model_len,
        mask_token_id,
        target_hidden_out.shape[-1],
        num_reqs,
        padded_batch,
        QUERY_WIDTH=query_width,
        QUERY_BLOCK=triton.next_power_of_2(query_width),
        MROPE_DIMS=mrope_positions.shape[0] if mrope_positions is not None else 0,
        HAS_MROPE_DELTA=mrope_delta is not None,
        HIDDEN_BLOCK=1024,
    )


def write_dfly_query_slots(
    slot_mapping: torch.Tensor,
    positions: torch.Tensor,
    active: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    max_model_len: torch.Tensor,
    query_width: int,
    padded_batch: int,
) -> None:
    _check_query_capacity(query_width, padded_batch, slot_mapping.numel())
    if (block_size < 1 or block_table.ndim != 2
            or block_table.shape[0] < padded_batch or block_table.shape[1] < 1
            or positions.numel() < padded_batch * query_width
            or active.numel() < padded_batch):
        raise ValueError("DFly slot/block table buffers are incompatible")
    _check_tensors(slot_mapping, positions, active, block_table, max_model_len)
    if padded_batch == 0:
        return
    _dfly_query_slots_kernel[(padded_batch,)](
        slot_mapping,
        positions,
        active,
        block_table,
        block_table.stride(0),
        block_table.shape[1],
        block_size,
        max_model_len,
        QUERY_WIDTH=query_width,
        QUERY_BLOCK=triton.next_power_of_2(query_width),
        PAD_SLOT_ID=PAD_SLOT_ID,
    )


def build_dfly_reuse_indices(
    out: torch.Tensor,
    snapshots: torch.Tensor,
    layer_snapshot_slots: torch.Tensor,
    active: torch.Tensor,
    source_rows: torch.Tensor,
    anchors: torch.Tensor,
    target_topk: int,
    local_window: int,
    query_width: int,
    padded_batch: int,
) -> None:
    """Fill `out[layer, :padded_batch * query_width]` for every draft layer."""
    num_layers, _, out_width = out.shape
    _check_query_capacity(query_width, padded_batch, out.shape[1])
    if (target_topk < 1 or local_window < 0
            or out_width < target_topk + local_window + query_width
            or snapshots.ndim != 3 or snapshots.shape[-1] != target_topk
            or layer_snapshot_slots.numel() != num_layers
            or any(t.numel() < padded_batch for t in (active, source_rows, anchors))):
        raise ValueError("DFly reuse buffers/512-to-768 expansion are incompatible")
    if out.dtype != torch.int32 or snapshots.dtype != torch.int32:
        raise ValueError("DFly sparse logical indices require int32 storage")
    _check_tensors(out, snapshots, layer_snapshot_slots, active, source_rows, anchors)
    if padded_batch == 0 or num_layers == 0:
        return
    _build_dfly_reuse_indices_kernel[(padded_batch, query_width, num_layers)](
        out,
        out.stride(0),
        out.stride(1),
        snapshots,
        snapshots.stride(0),
        snapshots.stride(1),
        layer_snapshot_slots,
        active,
        source_rows,
        anchors,
        TARGET_TOPK=target_topk,
        LOCAL_WINDOW=local_window,
        QUERY_WIDTH=query_width,
        OUT_WIDTH=out_width,
        BLOCK_SIZE=1024,
    )
