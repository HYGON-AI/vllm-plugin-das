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
decision depends only on the global ``sizes`` and common logical row layouts.
All ranks must supply the same tensor count, dtypes, trailing dimensions and
sizes. Strided dense inputs are packed locally without changing that decision.
Invalid device/shape contracts fail the call instead of falling back locally.
"""

import math
from contextlib import suppress

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
        self.disabled = True
        self._owned_buffers: list[int] = []
        self._imported_buffers: list[int] = []
        self._needs_coordinated_close = False
        self._collective_failed = False
        self._data_ptrs = self._flag_ptrs = None
        self._bufs = self._flags = self._cnt = None
        self.group = group
        self.rank = dist.get_rank(group=group)
        self.world_size = dist.get_world_size(group=group)
        if not 2 <= self.world_size <= _MAX_WORLD:
            raise ValueError(f"IPC all-gather requires 2..{_MAX_WORLD} ranks")
        if max_bytes_per_rank <= 0 or max_bytes_per_rank % 4:
            raise ValueError("IPC buffer capacity must be a positive multiple of 4")
        self.device = device
        self._free_buffer = torch.ops.hcu_ops.free_shared_buffer
        try:
            self._close_mapping = torch.ops.hcu_ops.close_mem_handle
        except AttributeError as exc:
            raise RuntimeError(
                "Rebuild vllm-plugin-das: IPC all-gather requires close_mem_handle"
            ) from exc
        self.region_elems = max_bytes_per_rank // 4
        self.slot_elems = self.region_elems * self.world_size
        data_bytes = 2 * self.slot_elems * 4
        flag_bytes = self.world_size * self.NPROG * 8
        self._needs_coordinated_close = True
        try:
            self._data_ptrs = self._share_buffer(data_bytes)
            self._flag_ptrs = self._share_buffer(flag_bytes)
            error = None
            try:
                self._bufs = torch.tensor(self._data_ptrs, dtype=torch.int64, device=device)
                self._flags = torch.tensor(self._flag_ptrs, dtype=torch.int64, device=device)
                _zero_i64_kernel[(1,)](self._flags, self.rank, flag_bytes // 8, BLOCK=256)
                self._cnt = torch.zeros(self.NPROG, dtype=torch.int64, device=device)
                torch.cuda.synchronize(device)
            except Exception as exc:
                error = exc
            # Initialization only: peers must all have zeroed flags/counters
            # before any rank can enter the first device-side gather.
            self._check_initialization(error, "initializing buffers")
        except BaseException:
            # A failed rank must not silently choose NCCL while peers use IPC.
            # If the group is broken, coordinated teardown cannot be proved
            # safe. The worker must exit rather than freeing an exporter while
            # peers may still hold its mappings.
            if not self._collective_failed:
                with suppress(Exception):
                    self.close()
            raise
        self.disabled = False

    def _exchange_status(self, payload):
        peers = [None] * self.world_size
        try:
            dist.all_gather_object(peers, payload, group=self.group)
        except BaseException:
            self._collective_failed = True
            raise
        return peers

    def _check_initialization(self, error: Exception | None, stage: str) -> None:
        errors = self._exchange_status(
            None if error is None else f"{type(error).__name__}: {error}"
        )
        failed = [(rank, message) for rank, message in enumerate(errors) if message]
        if failed:
            raise RuntimeError(f"IPC all-gather failed {stage}: {failed}") from error

    def _share_buffer(self, nbytes: int) -> list[int]:
        # Own the raw allocation as soon as the native allocator returns. The
        # generic CustomAllreduce helper cannot expose partial ownership when
        # a later handle exchange/open fails.
        pointer, handle, error = None, None, None
        try:
            pointer, handle = torch.ops.hcu_ops.allocate_shared_buffer_and_handle(nbytes)
            self._owned_buffers.append(pointer)
        except Exception as exc:
            error = exc
        payloads = self._exchange_status(
            (None if error is None else f"{type(error).__name__}: {error}", handle)
        )
        failed = [(rank, payload[0]) for rank, payload in enumerate(payloads) if payload[0]]
        if failed:
            raise RuntimeError(f"IPC all-gather failed allocating buffers: {failed}") from error
        pointers = []
        error = None
        try:
            for rank, (_, peer_handle) in enumerate(payloads):
                if rank == self.rank:
                    pointers.append(pointer)
                else:
                    imported = torch.ops.hcu_ops.open_mem_handle(peer_handle)
                    self._imported_buffers.append(imported)
                    pointers.append(imported)
        except Exception as exc:
            error = exc
        self._check_initialization(error, "opening IPC handles")
        return pointers

    def can_gather(self, tensors: list[torch.Tensor], sizes: list[int]) -> bool:
        if self.disabled:
            raise RuntimeError("IPC all-gather communicator is closed")
        if len(sizes) != self.world_size or any(
            not isinstance(size, int) or size < 0 for size in sizes
        ):
            raise ValueError("all-gather sizes must contain one non-negative integer per rank")
        row_bytes = 0
        for tensor in tensors:
            if (
                tensor.layout != torch.strided
                or tensor.dim() == 0
                or tensor.shape[0] != sizes[self.rank]
                or tensor.device != self.device
            ):
                raise ValueError("all-gather inputs must be dense tensors on this rank's device with the declared rows")
            row_bytes += _row_bytes(tensor)
        # Tensor count, row widths, dtypes and sizes are a collective contract.
        # Contiguity is deliberately absent: e.g. identical column slices can
        # be contiguous for M=1 on one rank but strided for M=2 on another.
        return (
            0 < len(tensors) <= _MAX_TENSORS
            and all(_row_bytes(tensor) % 4 == 0 for tensor in tensors)
            and max(sizes) * row_bytes <= self.region_elems * 4
        )

    def gather(self, tensors: list[torch.Tensor], sizes: list[int]) -> list[torch.Tensor]:
        if not self.can_gather(tensors, sizes):
            raise ValueError("tensor row layout or size exceeds the IPC all-gather capacity")
        tensors = [tensor.contiguous() for tensor in tensors]
        # .contiguous() can preserve a one-row view with a byte offset that
        # cannot be reinterpreted as int32. A clone gives that view aligned storage.
        tensors = [
            tensor.clone() if tensor.storage_offset() * tensor.element_size() % 4 else tensor
            for tensor in tensors
        ]
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
        """Collectively quiesce and close imports before freeing exports.

        All ranks must call explicit close/destroy. A broken group or failed
        import close leaves exports allocated for worker-exit cleanup; freeing
        them before every importer closes is undefined in HIP/CUDA.
        """
        self.disabled = True
        if self._needs_coordinated_close:
            if self._collective_failed:
                raise RuntimeError("Cannot safely close IPC buffers after group failure")
            error = None
            try:
                torch.cuda.synchronize(self.device)
            except Exception as exc:
                error = exc
            self._check_initialization(error, "quiescing IPC users")

            error = None
            remaining = []
            for pointer in self._imported_buffers:
                try:
                    self._close_mapping(pointer)
                except Exception as exc:
                    remaining.append(pointer)
                    error = error or exc
            self._imported_buffers = remaining
            # This exchange is also the barrier that ensures *all* importers
            # have closed before any rank frees its local exported allocations.
            self._check_initialization(error, "closing IPC mappings")
            self._needs_coordinated_close = False

        self._data_ptrs = self._flag_ptrs = None
        self._bufs = self._flags = self._cnt = None
        remaining = []
        error = None
        for pointer in self._owned_buffers:
            try:
                self._free_buffer(pointer)
            except Exception as exc:
                remaining.append(pointer)
                error = error or exc
        self._owned_buffers = remaining
        if error is not None:
            raise error

    def __del__(self):
        # GC cannot coordinate with peers. Explicit close/destroy owns teardown;
        # abandoned allocations are left to worker exit rather than freed early.
        self.disabled = True


def _row_bytes(t: torch.Tensor) -> int:
    return t.element_size() * math.prod(t.shape[1:])
