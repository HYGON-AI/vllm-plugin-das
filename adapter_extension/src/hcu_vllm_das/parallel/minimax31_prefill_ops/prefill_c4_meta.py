"""Prepare the common C4 MSA tensor metadata once on device."""
from collections import Counter

import torch
from vllm.triton_utils import tl, triton

from hcu_vllm_das.parallel import minimax31_msa as msa
from .prefill_compact_meta import _compact as stock


_GEOMETRY = object()
_hits = Counter()
_verified = Counter()
_verify = False


@triton.jit(do_not_specialize=["p0", "p1", "p2", "p3", "p4", "p5"])
def _write_bounds(cu, offsets, ends, query_ends, p0, p1, p2, p3, p4, p5,
                  ROWS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    group = row // 1024
    start = tl.where(group == 0, p0, p1)
    if ROWS == 6144:
        start = tl.where(group == 2, p2, start)
        start = tl.where(group == 3, p3, start)
        start = tl.where(group == 4, p4, start)
        start = tl.where(group == 5, p5, start)
    local_row = row % 1024
    tl.store(query_ends + row, start + local_row + 1, row < ROWS)
    first = (row < ROWS) & (local_row == 0)
    tl.store(offsets + group, start, first)
    tl.store(ends + group, start + 1024, first)
    tl.store(cu + group, row, first)
    tl.store(cu + group + 1, row + 1, row == ROWS - 1)


def _geometry(device, lengths, offsets):
    key = _GEOMETRY, device, offsets
    if key in msa._META:
        return msa._META[key]
    count = len(lengths)
    rows = 1024 * count
    cu = torch.empty(count + 1, device=device, dtype=torch.int32)
    starts = torch.empty(count, device=device, dtype=torch.int32)
    ends = torch.empty_like(starts)
    per_query = torch.empty(rows, device=device, dtype=torch.int32)
    _write_bounds[(triton.cdiv(rows, 256),)](
        cu, starts, ends, per_query, *offsets, *((0,) * (6 - count)), rows, 256,
    )
    segments = tuple((i, i * 1024, start, 1024) for i, start in enumerate(offsets))
    q_cpu = torch.tensor(lengths, dtype=torch.int32)
    kv_cpu = torch.tensor(tuple(start + 1024 for start in offsets), dtype=torch.int32)
    result = segments, q_cpu, kv_cpu, cu, starts, ends, per_query
    msa._META[key] = result
    _hits[count] += 1
    return result


def build(sub, req_start, req_count):
    if msa._HOST is None or req_count not in (2, 6):
        return stock(sub, req_start, req_count)
    lengths, starts, _ = msa._HOST
    lengths = tuple(lengths[req_start:req_start + req_count])
    offsets = tuple(int(start) for start in starts[req_start:req_start + req_count])
    if (lengths != (1024,) * req_count or len(offsets) != req_count
            or any(s < 131072 or s > 146432 for s in offsets)):
        return stock(sub, req_start, req_count)

    key = id(sub), req_start, req_count
    if key in msa._META:
        return msa._META[key]
    table = sub.block_table
    if (table.ndim != 2 or table.shape[0] != req_count
            or table.dtype != torch.int32 or not table.is_cuda):
        return stock(sub, req_start, req_count)
    counts = tuple((s + 1024 + 127) // 128 for s in offsets)
    if max(counts) > table.shape[1]:
        return stock(sub, req_start, req_count)
    if _verify:
        reference = stock(sub, req_start, req_count)
    table = table.contiguous()
    if min(counts) == max(counts):
        packed = table[:, :counts[0]].contiguous().view(-1)
    else:
        packed = torch.cat(tuple(table[i, :n] for i, n in enumerate(counts)))
    torch._assert_async((packed >= 0).all(), "active native page table contains invalid page")
    geometry = _geometry(table.device, lengths, offsets)
    result = msa.ForwardMetadata(*geometry, table, packed, max(offsets) + 1024)
    torch._assert_async((result.seq_lens == sub.seq_lens).all(), "native MSA planner seq_lens mismatch")
    if hasattr(sub, "cu_seqlens_q"):
        torch._assert_async((result.cu_q == sub.cu_seqlens_q).all(), "native MSA planner cu_seqlens mismatch")
    if _verify:
        for name in result.__dataclass_fields__:
            lhs, rhs = getattr(result, name), getattr(reference, name)
            equal = torch.equal(lhs, rhs) if isinstance(lhs, torch.Tensor) else lhs == rhs
            if not equal:
                raise RuntimeError(f"C4 MSA metadata changed {name}")
        _verified[req_count] += 1
    msa._META[key] = result
    return result


def configure(mode):
    global _verify
    if mode == "status":
        return {"enabled": msa._forward_meta is build, "hits": dict(_hits),
                "verified": dict(_verified)}
    if mode is not None and type(mode) is not bool:
        raise ValueError("Use null for stock metadata, true for verification, or false for timing")
    if msa._forward_meta not in (stock, build):
        raise RuntimeError("Enable qualified compact metadata first")
    msa._forward_meta = stock if mode is None else build
    _verify = mode is True
    _hits.clear()
    _verified.clear()
    return configure("status")
