# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Fused per-token FP8 quantization and DP IPC all-gather (opt-in).

The Step5 decode MoE dispatch performs a per-token FP8 quantization followed by a DP IPC
all-gather.  This module folds the quantization into the all-gather prologue: each all-gather
program owns whole activation rows (row % NPROG == pid), applies the same per-token quantization
math (same BK block, same fp32 ops), writes the fp8 row and fp32 scale to the local output and
pushes the same bytes into every peer's IPC slot.  The scale tensor is pushed row-owned as well;
the topk weights/ids keep the production element-chunk push/copy so their bytes and layout are
unchanged.

Output layout and every payload byte are identical to the production chain: the gathered
activation/scale/weights/ids are bytewise equal to `token_fp8_quant` + `IpcAllGather`.

The flag protocol, slot parity, per-program call counter and all-ranks-same-path rule are copied
verbatim from `ipc_all_gather.py`; nothing here invents a new synchronization scheme.

The path is opt-in (`VLLM_HCU_UP02_FUSED_DISPATCH=1`) and only engages when a world-2 DP IPC
quant all-gather exists; any capacity/shape mismatch falls back to a communicator-local quant
plus the standard gather.
"""

import torch
from torch.distributed import ProcessGroup

from vllm.triton_utils import tl, triton

from vllm_hcu.distributed.device_communicators.ipc_all_gather import IpcAllGather, _MAX_WORLD


@triton.jit
def _ipc_quant_gather_prod_kernel(
    x_ptr,
    out_q_ptr, out_q_i32, out_s_ptr, out_s_i32,
    in_w, in_ids, out_w, out_ids,
    bufs_ptr, flags_ptr, cnt_ptr,
    s0, s1, s2, s3, s4, s5, s6, s7,
    slot_elems, region_elems, max_spins,
    K: tl.constexpr, BK: tl.constexpr, WP: tl.constexpr, RPP: tl.constexpr,
    RANK: tl.constexpr, WORLD: tl.constexpr, NPROG: tl.constexpr, BLOCK: tl.constexpr,
):
    """Production dispatch order [activation, weights, ids, scale]: quant + gather."""
    pid = tl.program_id(0)
    call = tl.atomic_add(cnt_ptr + pid, 1)
    slot_off = (call % 2) * slot_elems
    ar = tl.arange(0, BLOCK)
    KW = K // 4
    SPI = 2 * WP  # words per row of weights + ids

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
    region_base = RANK * region_elems

    for k in tl.static_range(RPP):
        row = pid + k * NPROG
        if row < n_my:
            offs = tl.arange(0, BK)
            m = offs < K
            x = tl.load(x_ptr + row.to(tl.int64) * K + offs, mask=m, other=0.0).to(tl.float32)
            s = tl.maximum(tl.max(tl.abs(x), axis=0) / 448.0, 1.0 / (448.0 * 512.0))
            q = tl.minimum(tl.maximum(x / s, -448.0), 448.0).to(tl.float8e4nv)
            tl.store(out_q_ptr + (start_my + row).to(tl.int64) * K + offs, q, mask=m)
            tl.store(out_s_ptr + start_my + row, s)
            tl.debug_barrier()
            base_i32 = (start_my + row) * KW
            peer_row_off = slot_off + region_base + row * KW
            for off in range(0, KW, BLOCK):
                idx = off + ar
                mm = idx < KW
                v = tl.load(out_q_i32 + base_i32 + idx, mask=mm)
                for r in tl.static_range(WORLD):
                    if r != RANK:
                        peer = tl.load(bufs_ptr + r).to(tl.pointer_type(tl.int32))
                        tl.store(peer + peer_row_off + idx, v, mask=mm)
            s_word = tl.load(out_s_i32 + start_my + row)
            for r in tl.static_range(WORLD):
                if r != RANK:
                    peer = tl.load(bufs_ptr + r).to(tl.pointer_type(tl.int32))
                    tl.store(peer + slot_off + region_base + n_my * KW + n_my * SPI + row, s_word)

    for t in tl.static_range(2):
        src = in_w if t == 0 else in_ids
        dst = out_w if t == 0 else out_ids
        base = n_my * KW + t * n_my * WP
        E = n_my * WP
        chunk = tl.cdiv(tl.cdiv(E, NPROG), BLOCK) * BLOCK
        lo = pid * chunk
        hi = tl.minimum(E, lo + chunk)
        for off in range(lo, hi, BLOCK):
            idx = off + ar
            m = idx < hi
            v = tl.load(src + idx, mask=m)
            tl.store(dst + start_my * WP + idx, v, mask=m)
            for r in tl.static_range(WORLD):
                if r != RANK:
                    peer = tl.load(bufs_ptr + r).to(tl.pointer_type(tl.int32))
                    tl.store(peer + slot_off + region_base + base + idx, v, mask=m)

    for r in tl.static_range(WORLD):
        if r != RANK:
            peer_flags = tl.load(flags_ptr + r).to(tl.pointer_type(tl.int64))
            tl.atomic_add(peer_flags + RANK * NPROG + pid, 1, sem="release", scope="sys")

    acc = 0
    for r in tl.static_range(WORLD):
        n_r = (s0 if r == 0 else s1 if r == 1 else s2 if r == 2 else s3 if r == 3
               else s4 if r == 4 else s5 if r == 5 else s6 if r == 6 else s7)
        if r != RANK:
            flag = my_flags + r * NPROG + pid
            seen = tl.atomic_add(flag, 0, sem="acquire", scope="sys")
            spins = 0
            while seen <= call and spins < max_spins:
                seen = tl.atomic_add(flag, 0, sem="acquire", scope="sys")
                spins += 1
            peer_region = slot_off + r * region_elems
            for k in tl.static_range(RPP):
                row = pid + k * NPROG
                if row < n_r:
                    base_out = (acc + row) * KW
                    peer_row_off = peer_region + row * KW
                    for off in range(0, KW, BLOCK):
                        idx = off + ar
                        mm = idx < KW
                        v = tl.load(my_buf + peer_row_off + idx, mask=mm)
                        tl.store(out_q_i32 + base_out + idx, v, mask=mm)
                    s_word = tl.load(my_buf + peer_region + n_r * KW + n_r * SPI + row)
                    tl.store(out_s_i32 + acc + row, s_word)
            for t in tl.static_range(2):
                dst = out_w if t == 0 else out_ids
                base = n_r * KW + t * n_r * WP
                E = n_r * WP
                chunk = tl.cdiv(tl.cdiv(E, NPROG), BLOCK) * BLOCK
                lo = pid * chunk
                hi = tl.minimum(E, lo + chunk)
                for off in range(lo, hi, BLOCK):
                    idx = off + ar
                    m = idx < hi
                    v = tl.load(my_buf + peer_region + base + idx, mask=m)
                    tl.store(dst + acc * WP + idx, v, mask=m)
        acc += n_r


"""Set by the IPC quant all-gather once a world-2 (DP2 decode) communicator exists.

The deferred-quant branch is only useful where the fused path is valid (world == 2).  Other
layouts (notably the dedicated prefill contract TP1/DP8, world == 8) keep the certified
local-quant path: deferring there regressed prefill TPS by ~3.1 % in validation.
"""
UP02_DEFER_ALLOWED = False


class IpcQuantAllGather(IpcAllGather):
    """IpcAllGather with the per-token FP8 quant folded into the push prologue."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        global UP02_DEFER_ALLOWED
        if self.world_size == 2:
            UP02_DEFER_ALLOWED = True

    def can_quant_gather(self, x: torch.Tensor, w: torch.Tensor, ids: torch.Tensor,
                         sizes: list[int]) -> bool:
        """Capacity/shape gate for the production-order fused gather (all ranks agree)."""
        if self.disabled or not sizes or len(sizes) != self.world_size:
            return False
        # The fusion is validated at the DP2 decode layout (world == 2).  The dedicated
        # prefill contract runs TP1/DP8 (world == 8), where the fused prologue regressed
        # prefill TPS by ~3.1 % in validation; fall back to the certified quant + gather there.
        if self.world_size != 2:
            return False
        if x.dim() != 2 or x.dtype is not torch.bfloat16 or not x.is_contiguous():
            return False
        if w.dtype is not torch.float32 or ids.dtype is not torch.int32:
            return False
        if w.shape != ids.shape or w.shape[0] != x.shape[0] or x.shape[1] % 4:
            return False
        if any(not isinstance(v, int) or v < 0 for v in sizes):
            return False
        row_bytes = x.shape[1] + 8 * w.shape[1] + 4
        return max(sizes) * row_bytes <= self.region_elems * 4

    def gather_quant_prod(self, x: torch.Tensor, w: torch.Tensor, ids: torch.Tensor,
                          scale_ph: torch.Tensor, sizes: list[int]) -> list[torch.Tensor]:
        """Production dispatch order [activation, weights, ids, scale] -> [q, w', ids', s']."""
        if self.disabled:
            raise RuntimeError("IPC fused all-gather communicator is closed")
        M, K = x.shape
        TP = w.shape[1]
        assert x.is_contiguous() and x.dtype == torch.bfloat16
        assert w.shape == ids.shape == (M, TP), (w.shape, ids.shape)
        assert ids.dtype == torch.int32 and w.dtype == torch.float32
        assert scale_ph.shape == (M, 1) and scale_ph.dtype == torch.float32
        assert sizes[self.rank] == M
        row_bytes = K + 8 * TP + 4
        if max(sizes) * row_bytes > self.region_elems * 4 or K % 4:
            raise ValueError("fused all-gather capacity/shape contract violated")
        total = sum(sizes)
        q = torch.empty((total, K), dtype=torch.float8_e4m3fn, device=x.device)
        s = torch.empty((total, 1), dtype=torch.float32, device=x.device)
        ow = torch.empty((total, TP), dtype=torch.float32, device=x.device)
        oids = torch.empty((total, TP), dtype=torch.int32, device=x.device)
        s_list = list(sizes) + [0] * (_MAX_WORLD - len(sizes))
        _ipc_quant_gather_prod_kernel[(self.NPROG,)](
            x, q, q.view(-1).view(torch.int32), s, s.view(torch.int32),
            w.view(torch.int32), ids, ow.view(torch.int32), oids,
            self._bufs, self._flags, self._cnt,
            *s_list,
            self.slot_elems, self.region_elems, 1 << 62,
            K=K, BK=triton.next_power_of_2(K), WP=TP,
            RPP=-(-max(sizes) // self.NPROG),
            RANK=self.rank, WORLD=self.world_size, NPROG=self.NPROG, BLOCK=self.BLOCK,
            num_warps=8,
        )
        return [q, ow, oids, s]
