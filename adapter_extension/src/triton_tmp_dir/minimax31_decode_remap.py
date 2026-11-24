# SPDX-License-Identifier: Apache-2.0
"""Map logical Top4 IDs to physical experts and optionally count logical routes."""
import triton
import triton.language as tl


@triton.jit
def _map_experts(Ids, Table, Mapped, Enabled, Counts, SIZE: tl.constexpr,
                 BLOCK: tl.constexpr, RECORD: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = offset < SIZE
    logical = tl.load(Ids + offset, mask=active, other=-1)
    valid = active & (logical >= 0) & (logical < 128)
    physical = tl.load(Table + logical, mask=valid, other=0)
    # Preserve padding and malformed IDs for LowLatencyMoE's existing guards.
    tl.store(Mapped + offset, tl.where(valid, physical, logical), mask=active)
    if RECORD:
        if tl.load(Enabled) != 0:
            histogram = tl.histogram(logical.to(tl.int32), 128, mask=valid)
            tl.atomic_add(Counts + tl.arange(0, 128), histogram, histogram != 0, sem="relaxed")


def map_experts(ids, table, *, counters=None):
    if not ids.numel():
        return ids
    mapped = ids.new_empty(ids.shape)
    enabled, counts = (None, None) if counters is None else (counters.enabled, counters.counts)
    _map_experts[(triton.cdiv(ids.numel(), 128),)](
        ids, table, mapped, enabled, counts, ids.numel(), 128, counters is not None)
    return mapped
