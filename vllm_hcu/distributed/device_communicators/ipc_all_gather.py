# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Small-message all-gather over IPC buffers (graph capturable).

NCCL all-gather costs ~55 us per call at decode sizes, which the naive DP/EP
MoE dispatch pays once per MoE layer.  This one is a single Triton kernel:
every program pushes its chunk of the local rows straight into each peer's
IPC buffer, bumps a per-(source, program) flag on the peer with a
system-scope release, then waits for the peers' matching flags and copies
their chunks out of its own buffer.

* Each program keeps its own call counter, so the flags are monotonic and
  the kernel needs no host-side state: CUDA-graph replays just work.
* Two buffer slots alternate by call parity.  A rank can only start call
  k + 1 after receiving every peer's call-k flags, i.e. after every peer has
  finished call k - 1, the last reader of the slot call k + 1 writes.
* A program is a single wavefront, so the release orders all of its stores.

All ranks must take the same path (custom vs. fallback) for every call; the
decision depends only on the global ``sizes`` and the tensor layout.
"""

import math

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from vllm.triton_utils import tl, triton

_MAX_TENSORS = 4
_MAX_WORLD = 8
_NT = tl.constexpr(_MAX_TENSORS)


@triton.jit
def _ipc_all_gather_kernel(
    in0, in1, in2, in3,
    out0, out1, out2, out3,
    bufs_ptr,  # int64[WORLD]: every RANK's IPC data buffer
    flags_ptr,  # int64[WORLD]: every RANK's IPC flag buffer
    cnt_ptr,  # int64[NPROG]: local per-program call counters
    s0, s1, s2, s3, s4, s5, s6, s7,  # rows per RANK
    slot_elems,  # int32 elements per buffer slot
    region_elems,  # int32 elements per source region within a slot
    max_spins,  # int64 bound on the flag poll (effectively infinite)
    W0: tl.constexpr, W1: tl.constexpr, W2: tl.constexpr, W3: tl.constexpr,
    RANK: tl.constexpr,
    WORLD: tl.constexpr,
    NPROG: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    call = tl.atomic_add(cnt_ptr + pid, 1)
    slot_off = (call % 2) * slot_elems
    ar = tl.arange(0, BLOCK)

    # Row start of every RANK in the gathered output and this RANK's rows.
    n_my = 0
    start_my = 0
    acc = 0
    for r in tl.static_range(WORLD):
        n_r = (s0 if r == 0 else s1 if r == 1 else s2 if r == 2 else s3 if r == 3
               else s4 if r == 4 else s5 if r == 5 else s6 if r == 6 else s7)
        if r == RANK:
            n_my = n_r
            start_my = acc
        acc += n_r

    my_buf = tl.load(bufs_ptr + RANK).to(tl.pointer_type(tl.int32))
    my_flags = tl.load(flags_ptr + RANK).to(tl.pointer_type(tl.int64))

    # ---- push local rows to every peer (and to the local output) ----
    t_off = 0
    for t in tl.static_range(_NT):
        W = W0 if t == 0 else W1 if t == 1 else W2 if t == 2 else W3
        if W > 0:
            src = in0 if t == 0 else in1 if t == 1 else in2 if t == 2 else in3
            dst = out0 if t == 0 else out1 if t == 1 else out2 if t == 2 else out3
            E = n_my * W
            chunk = tl.cdiv(tl.cdiv(E, NPROG), BLOCK) * BLOCK
            lo = pid * chunk
            hi = tl.minimum(E, lo + chunk)
            for off in range(lo, hi, BLOCK):
                idx = off + ar
                m = idx < hi
                x = tl.load(src + idx, mask=m)
                tl.store(dst + start_my * W + idx, x, mask=m)
                for r in tl.static_range(WORLD):
                    if r != RANK:
                        peer = tl.load(bufs_ptr + r).to(tl.pointer_type(tl.int32))
                        tl.store(peer + slot_off + RANK * region_elems + t_off + idx,
                                 x, mask=m)
            t_off += E
    for r in tl.static_range(WORLD):
        if r != RANK:
            peer_flags = tl.load(flags_ptr + r).to(tl.pointer_type(tl.int64))
            tl.atomic_add(peer_flags + RANK * NPROG + pid, 1, sem="release",
                          scope="sys")

    # ---- wait for every peer's chunk and copy it out ----
    acc = 0
    for r in tl.static_range(WORLD):
        n_r = (s0 if r == 0 else s1 if r == 1 else s2 if r == 2 else s3 if r == 3
               else s4 if r == 4 else s5 if r == 5 else s6 if r == 6 else s7)
        if r != RANK:
            flag = my_flags + r * NPROG + pid
            # The runtime spin bound keeps the loop finite: LLVM assumes a
            # side-effect-free loop terminates and otherwise hoists the poll.
            seen = tl.atomic_add(flag, 0, sem="acquire", scope="sys")
            spins = 0
            while seen <= call and spins < max_spins:
                seen = tl.atomic_add(flag, 0, sem="acquire", scope="sys")
                spins += 1
            t_off = 0
            for t in tl.static_range(_NT):
                W = W0 if t == 0 else W1 if t == 1 else W2 if t == 2 else W3
                if W > 0:
                    dst = out0 if t == 0 else out1 if t == 1 else out2 if t == 2 else out3
                    E = n_r * W
                    chunk = tl.cdiv(tl.cdiv(E, NPROG), BLOCK) * BLOCK
                    lo = pid * chunk
                    hi = tl.minimum(E, lo + chunk)
                    for off in range(lo, hi, BLOCK):
                        idx = off + ar
                        m = idx < hi
                        x = tl.load(my_buf + slot_off + r * region_elems + t_off + idx,
                                    mask=m)
                        tl.store(dst + acc * W + idx, x, mask=m)
                    t_off += E
        acc += n_r


@triton.jit
def _zero_i64_kernel(ptrs, rank, n, BLOCK: tl.constexpr):
    p = tl.load(ptrs + rank).to(tl.pointer_type(tl.int64))
    for off in range(0, n, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        tl.store(p + idx, tl.zeros([BLOCK], tl.int64), mask=idx < n)


class IpcAllGather:
    """dim-0 all-gather(v) of up to four small tensors over IPC memory."""

    NPROG = 64
    BLOCK = 512

    def __init__(
        self,
        group: ProcessGroup,
        device: torch.device,
        max_bytes_per_rank: int = 4 * 1024 * 1024,
    ) -> None:
        from vllm_hcu.distributed.device_communicators.custom_all_reduce import (
            CustomAllreduce,
        )

        self.group = group
        self.rank = dist.get_rank(group=group)
        self.world_size = dist.get_world_size(group=group)
        assert 2 <= self.world_size <= _MAX_WORLD
        self.device = device
        self.region_elems = max_bytes_per_rank // 4
        self.slot_elems = self.region_elems * self.world_size
        data_bytes = 2 * self.slot_elems * 4
        flag_bytes = self.world_size * self.NPROG * 8
        self._data_ptrs = CustomAllreduce.create_shared_buffer(data_bytes, group=group)
        self._flag_ptrs = CustomAllreduce.create_shared_buffer(flag_bytes, group=group)
        self._create_shared_buffer_free = CustomAllreduce.free_shared_buffer
        self._bufs = torch.tensor(self._data_ptrs, dtype=torch.int64, device=device)
        self._flags = torch.tensor(self._flag_ptrs, dtype=torch.int64, device=device)
        # Flags must start at zero on every rank before any peer bumps them.
        _zero_i64_kernel[(1,)](self._flags, self.rank, flag_bytes // 8, BLOCK=256)
        torch.cuda.synchronize(device)
        dist.barrier(group=group)
        self._cnt = torch.zeros(self.NPROG, dtype=torch.int64, device=device)
        self.disabled = False

    def can_gather(self, tensors: list[torch.Tensor], sizes: list[int]) -> bool:
        if self.disabled or not tensors or len(tensors) > _MAX_TENSORS:
            return False
        row_bytes = 0
        for t in tensors:
            if (
                t.dim() == 0
                or t.shape[0] != sizes[self.rank]
                or not t.is_contiguous()
                or t.device != self.device
                or _row_bytes(t) % 4
            ):
                return False
            row_bytes += _row_bytes(t)
        return max(sizes) * row_bytes <= self.region_elems * 4

    def gather(self, tensors: list[torch.Tensor], sizes: list[int]) -> list[torch.Tensor]:
        total = sum(sizes)
        outs = [
            torch.empty((total,) + tuple(t.shape[1:]), dtype=t.dtype, device=t.device)
            for t in tensors
        ]
        ins = [t.view(-1).view(torch.int32) for t in tensors]
        outs_i32 = [o.view(-1).view(torch.int32) for o in outs]
        widths = [_row_bytes(t) // 4 for t in tensors]
        pad = _MAX_TENSORS - len(tensors)
        ins += [ins[0]] * pad
        outs_i32 += [outs_i32[0]] * pad
        widths += [0] * pad
        s = list(sizes) + [0] * (_MAX_WORLD - len(sizes))
        _ipc_all_gather_kernel[(self.NPROG,)](
            *ins, *outs_i32,
            self._bufs, self._flags, self._cnt,
            *s,
            self.slot_elems, self.region_elems, 1 << 62,
            *widths,
            RANK=self.rank, WORLD=self.world_size, NPROG=self.NPROG, BLOCK=self.BLOCK,
            num_warps=1,
        )
        return outs

    def close(self) -> None:
        if getattr(self, "_data_ptrs", None) is None:
            return
        self.disabled = True
        self._create_shared_buffer_free(self._data_ptrs, rank=self.rank)
        self._create_shared_buffer_free(self._flag_ptrs, rank=self.rank)
        self._data_ptrs = None


def _row_bytes(t: torch.Tensor) -> int:
    return t.element_size() * math.prod(t.shape[1:])
