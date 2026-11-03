"""Build decode SWA metadata with one Triton launch (vLLM #60731 / #63885).

DeepseekV4ROCMAiterSparseSWAMetadataBuilder.build runs once per SWA KV-cache
group (~8 target + ~3 DSpark draft builds per step). The previous HCU path
already fused the general builder's ~8 small ops into two launches
(``_swa_decode_lengths_kernel`` + ``_swa_decode_indices_kernel``), but the
second launch still left host-side bubbles between pairs.

``_decode_swa_meta_kernel`` writes dense indices/lengths, ``is_valid_token``,
``token_to_req``, and ragged indices/indptr in one program, matching the
upstream ``swa_decode_meta`` contract while keeping HCU extras (in-kernel
request mapping, padded slot rows, replay clamp).

Across groups, host-side Python between 9 serial launches still left ~8ms GPU
bubbles on HCU (~90us/launch). ``SwaMetaBatch`` + ``_decode_swa_meta_batch_kernel``
collect every eligible group in a prepare_attn pass and fire them as **one**
grid (program_id = group), which is what removes those inter-group bubbles.
"""

from __future__ import annotations

import functools
from contextlib import contextmanager
from dataclasses import dataclass

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.metadata import _token_request

# One program holds a ROWS x WIDTH int32 tile. 32 rows x 128 cols is the
# upstream gfx942/gfx950 default; 64 covers our common cudagraph capture sizes.
MAX_ROWS = 64
TAIL_BLOCK = 1024
_BATCH_STACK: list[list["SwaMetaJob"]] = []


@triton.jit(do_not_specialize=["num_tokens", "num_reqs", "num_slots", "lens_size"])
def _decode_swa_meta_kernel(
    query_start_loc_ptr,
    seq_lens_ptr,
    slot_mapping_ptr,
    replay_start_ptr,
    block_table_ptr,
    block_table_stride,
    token_to_req_ptr,
    is_valid_ptr,
    indices_ptr,
    indices_stride,
    lens_ptr,
    lens_size,
    ragged_ptr,
    indptr_ptr,
    num_tokens,
    num_reqs,
    num_slots,
    window_size,
    block_size,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
    ROWS: tl.constexpr,
    TAIL: tl.constexpr,
    NONCAUSAL: tl.constexpr,
):
    rows = tl.arange(0, ROWS)
    cols = tl.arange(0, BLOCK)
    in_batch = rows < num_tokens
    in_slots = rows < num_slots

    mapped_tokens = tl.load(query_start_loc_ptr + num_reqs)
    req = _token_request(
        query_start_loc_ptr, tl.minimum(rows, num_tokens - 1), num_reqs
    )
    req = tl.where(rows < mapped_tokens, req, 0)
    slot = tl.load(slot_mapping_ptr + rows, mask=in_slots, other=-1)
    valid = in_slots & (slot >= 0)
    tl.store(token_to_req_ptr + rows, req, mask=in_batch)
    tl.store(is_valid_ptr + rows, valid, mask=in_slots)

    q_start = tl.load(query_start_loc_ptr + req, mask=valid, other=0)
    q_end = tl.load(query_start_loc_ptr + req + 1, mask=valid, other=0)
    seq_len = tl.load(seq_lens_ptr + req, mask=valid, other=0)
    prefix_len = seq_len - (q_end - q_start)
    if NONCAUSAL:
        start = tl.maximum(prefix_len - window_size, 0)
        swa_len = tl.where(valid & in_batch, seq_len - start, 0)
    else:
        pos = prefix_len + rows - q_start
        start = tl.maximum(pos - (window_size - 1), 0)
        start = tl.maximum(
            start, tl.load(replay_start_ptr + req, mask=valid, other=0)
        )
        swa_len = tl.where(valid & in_batch, pos + 1 - start, 0)
    tl.store(lens_ptr + rows, swa_len, mask=in_batch)

    p = start[:, None] + cols[None, :]
    inside = (cols[None, :] < swa_len[:, None]) & (cols[None, :] < WIDTH)
    block = tl.load(
        block_table_ptr + req[:, None] * block_table_stride + p // block_size,
        mask=inside,
        other=0,
    )
    slots = tl.where(inside, block * block_size + p % block_size, -1)
    tl.store(
        indices_ptr + rows[:, None] * indices_stride + cols[None, :],
        slots,
        mask=in_batch[:, None] & (cols[None, :] < WIDTH),
    )

    # Ragged layout matches the previous two-kernel path: keep every in-window
    # slot (including null-block 0), ordered by row, with indptr = cumsum(len).
    ends = tl.cumsum(swa_len, axis=0)
    tl.store(indptr_ptr, 0)
    tl.store(indptr_ptr + 1 + rows, ends, mask=in_batch)
    row_start = ends - swa_len
    tl.store(
        ragged_ptr + row_start[:, None] + cols[None, :],
        slots,
        mask=inside,
    )

    # Clear lengths past the batch so graph reuse cannot read a stale tail.
    for i in range(num_tokens, lens_size, TAIL):
        offs = i + tl.arange(0, TAIL)
        tl.store(lens_ptr + offs, 0, mask=offs < lens_size)


@triton.jit(do_not_specialize=["num_tokens", "num_reqs", "window_size", "block_size"])
def _decode_swa_meta_batch_kernel(
    query_start_loc_ptr,
    seq_lens_ptr,
    replay_start_ptrs,
    slot_mapping_ptrs,
    block_table_ptrs,
    block_table_strides,
    token_to_req_ptrs,
    is_valid_ptrs,
    indices_ptrs,
    indices_strides,
    lens_ptrs,
    lens_sizes,
    ragged_ptrs,
    indptr_ptrs,
    num_slots_arr,
    num_tokens,
    num_reqs,
    window_size,
    block_size,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
    ROWS: tl.constexpr,
    TAIL: tl.constexpr,
    NONCAUSAL: tl.constexpr,
):
    """One program per SWA KV-cache group; pointers address that group's buffers."""
    g = tl.program_id(0)
    replay_start_ptr = tl.load(replay_start_ptrs + g).to(tl.pointer_type(tl.int32))
    slot_mapping_ptr = tl.load(slot_mapping_ptrs + g).to(tl.pointer_type(tl.int64))
    block_table_ptr = tl.load(block_table_ptrs + g).to(tl.pointer_type(tl.int32))
    block_table_stride = tl.load(block_table_strides + g)
    token_to_req_ptr = tl.load(token_to_req_ptrs + g).to(tl.pointer_type(tl.int32))
    is_valid_ptr = tl.load(is_valid_ptrs + g).to(tl.pointer_type(tl.int8))
    indices_ptr = tl.load(indices_ptrs + g).to(tl.pointer_type(tl.int32))
    indices_stride = tl.load(indices_strides + g)
    lens_ptr = tl.load(lens_ptrs + g).to(tl.pointer_type(tl.int32))
    lens_size = tl.load(lens_sizes + g)
    ragged_ptr = tl.load(ragged_ptrs + g).to(tl.pointer_type(tl.int32))
    indptr_ptr = tl.load(indptr_ptrs + g).to(tl.pointer_type(tl.int32))
    num_slots = tl.load(num_slots_arr + g)

    rows = tl.arange(0, ROWS)
    cols = tl.arange(0, BLOCK)
    in_batch = rows < num_tokens
    in_slots = rows < num_slots

    mapped_tokens = tl.load(query_start_loc_ptr + num_reqs)
    req = _token_request(
        query_start_loc_ptr, tl.minimum(rows, num_tokens - 1), num_reqs
    )
    req = tl.where(rows < mapped_tokens, req, 0)
    slot = tl.load(slot_mapping_ptr + rows, mask=in_slots, other=-1)
    valid = in_slots & (slot >= 0)
    tl.store(token_to_req_ptr + rows, req, mask=in_batch)
    tl.store(is_valid_ptr + rows, valid.to(tl.int8), mask=in_slots)

    q_start = tl.load(query_start_loc_ptr + req, mask=valid, other=0)
    q_end = tl.load(query_start_loc_ptr + req + 1, mask=valid, other=0)
    seq_len = tl.load(seq_lens_ptr + req, mask=valid, other=0)
    prefix_len = seq_len - (q_end - q_start)
    if NONCAUSAL:
        start = tl.maximum(prefix_len - window_size, 0)
        swa_len = tl.where(valid & in_batch, seq_len - start, 0)
    else:
        pos = prefix_len + rows - q_start
        start = tl.maximum(pos - (window_size - 1), 0)
        start = tl.maximum(
            start, tl.load(replay_start_ptr + req, mask=valid, other=0)
        )
        swa_len = tl.where(valid & in_batch, pos + 1 - start, 0)
    tl.store(lens_ptr + rows, swa_len, mask=in_batch)

    p = start[:, None] + cols[None, :]
    inside = (cols[None, :] < swa_len[:, None]) & (cols[None, :] < WIDTH)
    block = tl.load(
        block_table_ptr + req[:, None] * block_table_stride + p // block_size,
        mask=inside,
        other=0,
    )
    slots = tl.where(inside, block * block_size + p % block_size, -1)
    tl.store(
        indices_ptr + rows[:, None] * indices_stride + cols[None, :],
        slots,
        mask=in_batch[:, None] & (cols[None, :] < WIDTH),
    )

    ends = tl.cumsum(swa_len, axis=0)
    tl.store(indptr_ptr, 0)
    tl.store(indptr_ptr + 1 + rows, ends, mask=in_batch)
    row_start = ends - swa_len
    tl.store(
        ragged_ptr + row_start[:, None] + cols[None, :],
        slots,
        mask=inside,
    )

    for i in range(num_tokens, lens_size, TAIL):
        offs = i + tl.arange(0, TAIL)
        tl.store(lens_ptr + offs, 0, mask=offs < lens_size)


@triton.jit(do_not_specialize=["num_tokens", "num_reqs", "num_slots"])
def _swa_decode_lengths_kernel(
    query_start_loc,
    seq_lens,
    slot_mapping,
    replay_start,
    token_to_req,
    is_valid_token,
    lengths,
    indptr,
    num_tokens,
    num_reqs,
    num_slots,
    CAPACITY: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    NON_CAUSAL: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    """Fallback for batches that exceed the one-launch register tile."""
    rows = tl.arange(0, BLOCK_ROWS)
    mapped_tokens = tl.load(query_start_loc + num_reqs)
    req = _token_request(query_start_loc, tl.minimum(rows, num_tokens - 1), num_reqs)
    req = tl.where(rows < mapped_tokens, req, 0)
    query_start = tl.load(query_start_loc + req)
    query_end = tl.load(query_start_loc + req + 1)
    seq_len = tl.load(seq_lens + req)
    prefix_len = seq_len - (query_end - query_start)
    valid = tl.load(slot_mapping + rows, rows < num_slots, other=-1) >= 0
    if NON_CAUSAL:
        start = tl.maximum(prefix_len - WINDOW_SIZE, 0)
        end = seq_len
    else:
        end = prefix_len + rows - query_start + 1
        start = tl.maximum(end - WINDOW_SIZE, 0)
        start = tl.maximum(start, tl.load(replay_start + req))
    length = tl.where(valid & (rows < num_tokens), tl.maximum(end - start, 0), 0)
    offsets = tl.cumsum(length)
    tl.store(token_to_req + rows, req, rows < num_tokens)
    tl.store(is_valid_token + rows, valid, rows < num_slots)
    tl.store(lengths + rows, length, rows < CAPACITY)
    tl.store(indptr, 0)
    tl.store(indptr + rows + 1, offsets, rows < num_tokens)


@triton.jit(do_not_specialize=["dense_stride", "table_stride"])
def _swa_decode_indices_kernel(
    query_start_loc,
    seq_lens,
    block_table,
    token_to_req,
    lengths,
    indptr,
    dense_indices,
    ragged_indices,
    dense_stride,
    table_stride,
    BLOCK_SIZE: tl.constexpr,
    INDEX_WIDTH: tl.constexpr,
    NON_CAUSAL: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    """Fallback companion for ``_swa_decode_lengths_kernel``."""
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK_COLS)
    req = tl.load(token_to_req + row)
    length = tl.load(lengths + row)
    seq_len = tl.load(seq_lens + req)
    if NON_CAUSAL:
        end = seq_len
    else:
        query_end = tl.load(query_start_loc + req + 1)
        end = seq_len - query_end + row + 1
    positions = end - length + columns
    valid = (columns < length) & (columns < INDEX_WIDTH)
    blocks = tl.load(
        block_table + req.to(tl.int64) * table_stride + positions // BLOCK_SIZE,
        valid,
        other=0,
    )
    slots = blocks * BLOCK_SIZE + positions % BLOCK_SIZE
    tl.store(
        dense_indices + row.to(tl.int64) * dense_stride + columns,
        tl.where(valid, slots, -1),
        columns < INDEX_WIDTH,
    )
    offset = tl.load(indptr + row)
    tl.store(ragged_indices + offset + columns, slots, valid)


@dataclass
class SwaMetaJob:
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    slot_mapping: torch.Tensor
    replay_start: torch.Tensor
    block_table: torch.Tensor
    token_to_req: torch.Tensor
    is_valid: torch.Tensor
    indices: torch.Tensor
    lens: torch.Tensor
    ragged: torch.Tensor
    indptr: torch.Tensor
    num_tokens: int
    num_reqs: int
    num_slots: int
    window_size: int
    block_size: int
    width: int
    noncausal: bool
    one_launch: bool
    capacity: int


# Packed pointer scratch: one HtoD for all i64 fields, one for i32 scalars.
# Row order must match _fill_ptr_scratch / _launch_batch indexing.
_I64_ROWS = (
    "replay_start_ptrs",
    "slot_mapping_ptrs",
    "block_table_ptrs",
    "block_table_strides",
    "token_to_req_ptrs",
    "is_valid_ptrs",
    "indices_ptrs",
    "indices_strides",
    "lens_ptrs",
    "ragged_ptrs",
    "indptr_ptrs",
)
_I32_ROWS = ("lens_sizes", "num_slots_arr")
_PTR_SCRATCH: dict[tuple, dict[str, object]] = {}


def _ptr_scratch(device: torch.device, n: int) -> dict[str, object]:
    key = (device.type, device.index, n)
    bufs = _PTR_SCRATCH.get(key)
    if bufs is None:
        bufs = {
            "host_i64": torch.empty(
                (len(_I64_ROWS), n), dtype=torch.int64, pin_memory=True
            ),
            "dev_i64": torch.empty(
                (len(_I64_ROWS), n), dtype=torch.int64, device=device
            ),
            "host_i32": torch.empty(
                (len(_I32_ROWS), n), dtype=torch.int32, pin_memory=True
            ),
            "dev_i32": torch.empty(
                (len(_I32_ROWS), n), dtype=torch.int32, device=device
            ),
            # Graph-stable buffers keep the same data_ptr; skip HtoD when unchanged.
            "sig": None,
        }
        _PTR_SCRATCH[key] = bufs
    return bufs


def _bunch_ptr_sig(bunch: list[SwaMetaJob]) -> tuple:
    """Fingerprint of device addresses / strides used by the batch kernel."""
    return tuple(
        (
            j.replay_start.data_ptr(),
            j.slot_mapping.data_ptr(),
            j.block_table.data_ptr(),
            j.block_table.stride(0),
            j.token_to_req.data_ptr(),
            j.is_valid.data_ptr(),
            j.indices.data_ptr(),
            j.indices.stride(0),
            j.lens.data_ptr(),
            j.lens.shape[0],
            j.ragged.data_ptr(),
            j.indptr.data_ptr(),
            j.num_slots,
        )
        for j in bunch
    )


def _fill_ptr_scratch(bunch: list[SwaMetaJob]) -> dict[str, torch.Tensor]:
    device = bunch[0].slot_mapping.device
    g = len(bunch)
    bufs = _ptr_scratch(device, g)
    host64 = bufs["host_i64"]
    host32 = bufs["host_i32"]
    assert isinstance(host64, torch.Tensor) and isinstance(host32, torch.Tensor)
    sig = _bunch_ptr_sig(bunch)
    if bufs["sig"] != sig:
        # One packed host write (no per-row torch.as_tensor) then a single HtoD.
        vals64 = [
            [j.replay_start.data_ptr() for j in bunch],
            [j.slot_mapping.data_ptr() for j in bunch],
            [j.block_table.data_ptr() for j in bunch],
            [j.block_table.stride(0) for j in bunch],
            [j.token_to_req.data_ptr() for j in bunch],
            [j.is_valid.data_ptr() for j in bunch],
            [j.indices.data_ptr() for j in bunch],
            [j.indices.stride(0) for j in bunch],
            [j.lens.data_ptr() for j in bunch],
            [j.ragged.data_ptr() for j in bunch],
            [j.indptr.data_ptr() for j in bunch],
        ]
        vals32 = [
            [j.lens.shape[0] for j in bunch],
            [j.num_slots for j in bunch],
        ]
        host64[:, :g].copy_(torch.tensor(vals64, dtype=torch.int64))
        host32[:, :g].copy_(torch.tensor(vals32, dtype=torch.int32))
        assert isinstance(bufs["dev_i64"], torch.Tensor)
        assert isinstance(bufs["dev_i32"], torch.Tensor)
        bufs["dev_i64"][:, :g].copy_(host64[:, :g], non_blocking=True)
        bufs["dev_i32"][:, :g].copy_(host32[:, :g], non_blocking=True)
        bufs["sig"] = sig
    out = {
        name: bufs["dev_i64"][i, :g]  # type: ignore[index]
        for i, name in enumerate(_I64_ROWS)
    }
    out.update(
        {
            name: bufs["dev_i32"][i, :g]  # type: ignore[index]
            for i, name in enumerate(_I32_ROWS)
        }
    )
    return out


def _launch_one(
    *,
    query_start_loc,
    seq_lens,
    slot_mapping,
    replay_start,
    block_table,
    token_to_req,
    is_valid,
    indices,
    lens,
    ragged,
    indptr,
    num_tokens,
    num_reqs,
    num_slots,
    window_size,
    block_size,
    width,
    noncausal,
):
    rows = max(num_tokens, num_slots)
    _decode_swa_meta_kernel[(1,)](
        query_start_loc,
        seq_lens,
        slot_mapping,
        replay_start,
        block_table,
        block_table.stride(0),
        token_to_req,
        is_valid,
        indices,
        indices.stride(0),
        lens,
        lens.shape[0],
        ragged,
        indptr,
        num_tokens,
        num_reqs,
        num_slots,
        window_size,
        block_size,
        WIDTH=width,
        BLOCK=triton.next_power_of_2(width),
        ROWS=triton.next_power_of_2(rows),
        TAIL=TAIL_BLOCK,
        NONCAUSAL=noncausal,
        num_warps=4,
    )


def _launch_batch(jobs: list[SwaMetaJob]) -> None:
    """One grid per compatible shape group; large-batch jobs use the 2-kernel path."""
    if not jobs:
        return
    groups: dict[tuple, list[SwaMetaJob]] = {}
    fallbacks: list[SwaMetaJob] = []
    for job in jobs:
        if not job.one_launch:
            fallbacks.append(job)
            continue
        # replay_start is per-group (each builder owns no_replay_start); the
        # batch kernel takes replay_start_ptrs so it must not be in the key.
        key = (
            job.noncausal,
            job.width,
            job.num_tokens,
            job.num_reqs,
            job.window_size,
            job.block_size,
            id(job.query_start_loc),
            id(job.seq_lens),
        )
        groups.setdefault(key, []).append(job)

    for bunch in groups.values():
        if len(bunch) == 1:
            job = bunch[0]
            _launch_one(
                query_start_loc=job.query_start_loc,
                seq_lens=job.seq_lens,
                slot_mapping=job.slot_mapping,
                replay_start=job.replay_start,
                block_table=job.block_table,
                token_to_req=job.token_to_req,
                is_valid=job.is_valid,
                indices=job.indices,
                lens=job.lens,
                ragged=job.ragged,
                indptr=job.indptr,
                num_tokens=job.num_tokens,
                num_reqs=job.num_reqs,
                num_slots=job.num_slots,
                window_size=job.window_size,
                block_size=job.block_size,
                width=job.width,
                noncausal=job.noncausal,
            )
            continue
        g = len(bunch)
        rows = max(bunch[0].num_tokens, max(j.num_slots for j in bunch))
        ptrs = _fill_ptr_scratch(bunch)
        job0 = bunch[0]
        _decode_swa_meta_batch_kernel[(g,)](
            job0.query_start_loc,
            job0.seq_lens,
            ptrs["replay_start_ptrs"],
            ptrs["slot_mapping_ptrs"],
            ptrs["block_table_ptrs"],
            ptrs["block_table_strides"],
            ptrs["token_to_req_ptrs"],
            ptrs["is_valid_ptrs"],
            ptrs["indices_ptrs"],
            ptrs["indices_strides"],
            ptrs["lens_ptrs"],
            ptrs["lens_sizes"],
            ptrs["ragged_ptrs"],
            ptrs["indptr_ptrs"],
            ptrs["num_slots_arr"],
            job0.num_tokens,
            job0.num_reqs,
            job0.window_size,
            job0.block_size,
            WIDTH=job0.width,
            BLOCK=triton.next_power_of_2(job0.width),
            ROWS=triton.next_power_of_2(rows),
            TAIL=TAIL_BLOCK,
            NONCAUSAL=job0.noncausal,
            num_warps=4,
        )

    for job in fallbacks:
        _launch_two(
            query_start_loc=job.query_start_loc,
            seq_lens=job.seq_lens,
            slot_mapping=job.slot_mapping,
            replay_start=job.replay_start,
            block_table=job.block_table,
            token_to_req=job.token_to_req,
            is_valid=job.is_valid,
            dense=job.indices,
            lens=job.lens,
            ragged=job.ragged,
            indptr=job.indptr,
            num_tokens=job.num_tokens,
            num_reqs=job.num_reqs,
            num_slots=job.num_slots,
            window_size=job.window_size,
            block_size=job.block_size,
            width=job.width,
            capacity=job.capacity,
            noncausal=job.noncausal,
        )


def flush_swa_meta_batch() -> None:
    """Launch any deferred SWA meta jobs so later builders can overlap on GPU."""
    if not _BATCH_STACK or not _BATCH_STACK[-1]:
        return
    jobs = _BATCH_STACK[-1]
    to_run = list(jobs)
    jobs.clear()
    with torch.cuda.device(to_run[0].slot_mapping.device):
        _launch_batch(to_run)


def _job_batch_key(job: SwaMetaJob) -> tuple:
    return (
        job.noncausal,
        job.width,
        job.num_tokens,
        job.num_reqs,
        job.window_size,
        job.block_size,
        id(job.query_start_loc),
        id(job.seq_lens),
    )


@contextmanager
def swa_meta_launch_batch():
    """Collect fused SWA meta jobs and flush them as batched Triton launches."""
    jobs: list[SwaMetaJob] = []
    _BATCH_STACK.append(jobs)
    try:
        yield
        flush_swa_meta_batch()
    finally:
        _BATCH_STACK.pop()


_FLUSH_WRAP_MARKER = "_vllm_hcu_swa_meta_flush_wrap"
_FLUSH_WRAPPED_CLASSES: set[type] = set()


def ensure_builder_flush_wrap(builder) -> None:
    """Before non-SWA ``build``, flush pending SWA jobs so the GPU stays busy."""
    cls = type(builder)
    if cls in _FLUSH_WRAPPED_CLASSES:
        return
    _FLUSH_WRAPPED_CLASSES.add(cls)
    original = cls.build

    @functools.wraps(original)
    def build(self, *args, **kwargs):
        # ROCm SWA builders enqueue via build_fused; flush only for others so
        # MLA / fallback builds overlap with the already-queued SWA grid.
        if not (hasattr(self, "no_replay_start") and _BATCH_STACK):
            flush_swa_meta_batch()
        return original(self, *args, **kwargs)

    setattr(build, _FLUSH_WRAP_MARKER, True)
    cls.build = build


def _launch_two(
    *,
    query_start_loc,
    seq_lens,
    slot_mapping,
    replay_start,
    block_table,
    token_to_req,
    is_valid,
    dense,
    lens,
    ragged,
    indptr,
    num_tokens,
    num_reqs,
    num_slots,
    window_size,
    block_size,
    width,
    capacity,
    noncausal,
):
    _swa_decode_lengths_kernel[(1,)](
        query_start_loc,
        seq_lens,
        slot_mapping,
        replay_start,
        token_to_req,
        is_valid,
        lens,
        indptr,
        num_tokens,
        num_reqs,
        num_slots,
        CAPACITY=capacity,
        WINDOW_SIZE=window_size,
        NON_CAUSAL=noncausal,
        BLOCK_ROWS=triton.next_power_of_2(capacity),
        num_warps=4,
    )
    _swa_decode_indices_kernel[(num_tokens,)](
        query_start_loc,
        seq_lens,
        block_table,
        token_to_req,
        lens,
        indptr,
        dense,
        ragged,
        dense.stride(0),
        block_table.stride(0),
        BLOCK_SIZE=block_size,
        INDEX_WIDTH=width,
        NON_CAUSAL=noncausal,
        BLOCK_COLS=triton.next_power_of_2(width),
        num_warps=4,
    )


def build_fused_decode_metadata(builder, common, replay_start, metadata_cls):
    """Return None for batches requiring the general SWA builder."""
    if not hasattr(builder, "no_replay_start"):
        flush_swa_meta_batch()
        return None
    parallel = builder.vllm_config.parallel_config
    num_tokens = common.num_actual_tokens
    num_slots = common.slot_mapping.numel()
    if (
        common.slot_mapping.device.type != "cuda"
        or parallel.prefill_context_parallel_size != 1
        or parallel.decode_context_parallel_size != 1
        or not isinstance(common.causal, bool)
        or common.max_query_len > builder.decode_threshold
        or common.num_reqs == 0
        or num_tokens == 0
        or num_tokens > builder._max_tokens
        or num_slots < num_tokens
        or num_slots > builder._max_tokens
        or builder.max_image_tokens > 0
    ):
        flush_swa_meta_batch()
        return None
    non_causal = not common.causal
    if non_causal and not builder.is_dspark:
        flush_swa_meta_batch()
        return None
    width = builder.noncausal_index_width if non_causal else builder.window_size
    if non_causal:
        if builder.decode_swa_indices_noncausal is None:
            builder.decode_swa_indices_noncausal = torch.empty(
                (builder._max_tokens, 1, width),
                dtype=torch.int32,
                device=builder.device,
            )
        dense = builder.decode_swa_indices_noncausal
    else:
        dense = builder.decode_swa_indices
    if replay_start is None:
        replay_start = builder.no_replay_start
    views = getattr(builder, "_vllm_hcu_swa_decode_views", None)
    if views is None:
        views = builder._vllm_hcu_swa_decode_views = {}
    key = (num_tokens, num_slots, non_causal)
    if key not in views:
        views[key] = (
            builder.decode_swa_ragged_indices_buffer[: num_tokens * width],
            builder.decode_swa_ragged_indptr_buffer[: num_tokens + 1],
            builder.token_to_req_indices[:num_tokens],
            builder.is_valid_token[:num_slots],
            dense[:num_tokens],
            builder.decode_swa_lens[:num_tokens],
        )
    ragged, indptr, mapping, valid, dense_out, lengths_out = views[key]
    one_launch = max(num_tokens, num_slots) <= MAX_ROWS
    job = SwaMetaJob(
        query_start_loc=common.query_start_loc,
        seq_lens=common.seq_lens,
        slot_mapping=common.slot_mapping,
        replay_start=replay_start,
        block_table=common.block_table_tensor,
        token_to_req=builder.token_to_req_indices,
        is_valid=builder.is_valid_token,
        indices=dense,
        lens=builder.decode_swa_lens,
        ragged=ragged,
        indptr=indptr,
        num_tokens=num_tokens,
        num_reqs=common.num_reqs,
        num_slots=num_slots,
        window_size=builder.window_size,
        block_size=builder.block_size,
        width=width,
        noncausal=non_causal,
        one_launch=one_launch,
        capacity=builder._max_tokens,
    )
    if _BATCH_STACK:
        pending = _BATCH_STACK[-1]
        # Flush previous wave (e.g. causal target → noncausal draft) so the
        # GPU can run while this builder prepares the next group's buffers.
        if pending and _job_batch_key(pending[0]) != _job_batch_key(job):
            flush_swa_meta_batch()
            pending = _BATCH_STACK[-1]
        pending.append(job)
    else:
        with torch.cuda.device(common.slot_mapping.device):
            _launch_batch([job])
    common._token_to_req_indices_cache = mapping
    tile_sched = builder.build_tile_scheduler(num_tokens)
    return metadata_cls(
        seq_lens=common.seq_lens,
        query_start_loc=common.query_start_loc,
        query_start_loc_cpu=common.query_start_loc_cpu,
        block_table=common.block_table_tensor,
        slot_mapping=common.slot_mapping,
        is_valid_token=valid,
        token_to_req_indices=mapping,
        decode_swa_indices=dense_out,
        decode_swa_lens=lengths_out,
        decode_swa_width=width,
        replay_start=replay_start,
        block_size=builder.block_size,
        num_decodes=common.num_reqs,
        num_decode_tokens=num_tokens,
        max_decode_query_len=common.max_query_len,
        tile_sched_swaonly=tile_sched["swaonly"],
        tile_sched_c4a=tile_sched["c4a"],
        tile_sched_c128a=tile_sched["c128a"],
        tile_sched_c1a=tile_sched["c1a"],
        tile_sched_c2a=tile_sched["c2a"],
        decode_swa_ragged_indices=ragged,
        decode_swa_ragged_indptr=indptr,
    )
