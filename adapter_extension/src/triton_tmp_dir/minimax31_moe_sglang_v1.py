# SPDX-License-Identifier: Apache-2.0
"""Standalone MiniMax31 W4A8 EP module, following SGLang's INT8 normal path.

No model runner, service, IPC owner, or global monkeypatch is installed here.
Weights must be packed once by this process's own DeepGEMM build. Checkpoint
scales remain scale/16. Route weights are applied *after* GEMM2's BF16 store,
matching the SGLang contiguous HIPC/gather contract, rather than the older
plugin's weight-before-store contract.

Each instance owns non-reentrant scratch. The received-result view is valid
until the next invocation; the EP wrapper consumes it immediately in combine.
Only verified DeepEP local-minus-one IDs enter the unchecked native scatter.
The external physical Top4 entry is guarded on GPU before dispatch.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass

import torch
import triton
import triton.language as tl


@triton.jit
def _route_guard(Ids, Weights, Status, N: tl.constexpr, E: tl.constexpr,
                 B: tl.constexpr):
    ix = tl.program_id(0) * B + tl.arange(0, B)
    ids = tl.load(Ids + ix, ix < N, other=-1)
    w = tl.load(Weights + ix, ix < N, other=0)
    valid = (ids >= -1) & (ids < E) & (w == w) & (tl.abs(w) < float('inf'))
    valid = valid & ((ids >= 0) | (w == 0))
    tl.store(Status + tl.program_id(0), tl.min(valid.to(tl.int32), 0) != 0)


@dataclass(frozen=True)
class PackedExperts:
    w13: torch.Tensor
    w2: torch.Tensor
    s13: torch.Tensor
    s2: torch.Tensor

    @classmethod
    def from_checkpoint_bytes(cls, w13, w2, s13, s2):
        from deepgemm import pack_w4a8_moe_hipc_weight
        # Never mutate the caller's checkpoint/IPC weights.
        return cls(pack_w4a8_moe_hipc_weight(w13.clone()),
                   pack_w4a8_moe_hipc_weight(w2.clone()), s13, s2)

    def validate(self):
        e, two_i, half_h = self.w13.shape
        h, i = half_h * 2, two_i // 2
        if (two_i % 2 or h % 128 or i % 128 or
                self.w2.shape != (e, h, i // 2) or
                self.s13.shape != (e, two_i, 1) or self.s2.shape != (e, h, 1)):
            raise ValueError('invalid split HIPC expert weight/scale dimensions')
        for t, dtype in ((self.w13, torch.int8), (self.w2, torch.int8),
                         (self.s13, torch.float32), (self.s2, torch.float32)):
            if t.dtype != dtype or not t.is_cuda or not t.is_contiguous() or t.device != self.w13.device:
                raise ValueError('HIPC weights/scales must be contiguous on the same GPU')
        return e, h, i


class ReceivedExperts:
    """INT8 recv -> native scatter -> two HIPC GEMMs -> weighted gather.

    counts are the *256-aligned* list returned by DeepEP expert_alignment=256,
    not the unpadded valid counts consumed by the legacy argsort implementation.
    The instance and weights are bound to one device and one CUDA stream.
    """
    alignment = 256

    def __init__(self, weights: PackedExperts, *, mode=1006):
        from lightop import op
        from deepgemm import m_grouped_w4a8_gemm_nt_contiguous_hipc
        from lightop.activation.swiglu_oai_int8_quant import swiglu_oai_int8_quant_contiguous
        self.e, self.h, self.i = weights.validate()
        if not hasattr(op, 'ep_scatter') or not hasattr(op, 'ep_gather'):
            raise RuntimeError('this LightOP build lacks native EP scatter/gather')
        self.weights, self.op, self.gemm = weights, op, m_grouped_w4a8_gemm_nt_contiguous_hipc
        self.activation = swiglu_oai_int8_quant_contiguous
        self.mode = mode
        self.device = weights.w13.device
        self.stream = torch.cuda.current_stream(self.device).cuda_stream
        self._lock = threading.Lock()
        self._capacity = (0, 0)
        self._host_counts = torch.empty(self.e, dtype=torch.int32, pin_memory=True)
        self._host_view = self._host_counts.numpy()
        self._counts = torch.empty(self.e, dtype=torch.int32, device=self.device)
        self._copy_done = torch.cuda.Event()
        self._has_copy = False

    def bind_weights(self, weights):
        """Reuse one stream-ordered workspace across serial model layers."""
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('cannot rebind inflight expert workspace')
        try:
            if weights.validate() != (self.e, self.h, self.i) or weights.w13.device != self.device:
                raise ValueError('shared expert workspace geometry/device changed')
            if torch.cuda.current_stream(self.device).cuda_stream != self.stream:
                raise RuntimeError('cannot rebind expert workspace on another stream')
            self.weights = weights
        finally:
            self._lock.release()

    def _reserve(self, rows, padded):
        old_rows, old_padded = self._capacity
        if rows <= old_rows and padded <= old_padded:
            return
        rows, padded = max(rows, old_rows), max(padded, old_padded)
        self._q = torch.empty((padded, self.h), dtype=torch.int8, device=self.device)
        self._scale = torch.empty((padded, 1), dtype=torch.float32, device=self.device)
        self._mid = torch.empty((padded,), dtype=torch.int32, device=self.device)
        self._index = torch.empty((rows, 4), dtype=torch.int32, device=self.device)
        self._gateup = torch.empty((padded, self.i*2), dtype=torch.bfloat16, device=self.device)
        self._down = torch.empty((padded, self.h), dtype=torch.bfloat16, device=self.device)
        self._result = torch.empty((rows, self.h), dtype=torch.bfloat16, device=self.device)
        self._capacity = rows, padded

    @torch.no_grad()
    def __call__(self, x, scale, ids, route_weights, counts):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('ReceivedExperts scratch is non-reentrant')
        try:
            return self._run(x, scale, ids, route_weights, counts)
        finally:
            self._lock.release()

    def _run(self, x, scale, ids, route_weights, counts):
        rows = x.shape[0]
        if (x.ndim != 2 or x.shape[1] != self.h or x.dtype != torch.int8 or
                scale.shape not in ((rows,), (rows, 1)) or scale.dtype != torch.float32 or
                ids.shape != (rows, 4) or ids.dtype != torch.int64 or
                route_weights.shape != ids.shape or route_weights.dtype != torch.float32):
            raise ValueError('expected INT8 recv + per-token FP32 scale + local INT64 Top4')
        if any(t.device != self.device or not t.is_contiguous() for t in (x,scale,ids,route_weights)):
            raise ValueError('received tensors must be contiguous on the weight GPU')
        if torch.cuda.current_stream(self.device).cuda_stream != self.stream:
            raise RuntimeError('workspace cannot be shared across CUDA streams')
        if (len(counts) != self.e or any(type(n) is not int or n < 0 or n % 256 for n in counts)):
            raise ValueError('DeepEP counts must be nonnegative integers aligned to 256')
        padded = sum(counts)
        if padded > rows*4 + self.e*255:
            raise ValueError('aligned counts exceed received route capacity')
        if rows == 0:
            return torch.empty((0, self.h), dtype=torch.bfloat16, device=self.device)
        self._reserve(rows, padded)
        result = self._result[:rows]
        if padded == 0:
            return result.zero_()
        # Pinned host storage must not be overwritten while an older H2D reads it.
        if self._has_copy and not self._copy_done.query():
            self._copy_done.synchronize()
        self._host_view[:] = counts
        self._counts.copy_(self._host_counts, non_blocking=True)
        self._copy_done.record()
        self._has_copy = True
        q, qs = self._q[:padded], self._scale[:padded]
        mid, index = self._mid[:padded], self._index[:rows]
        mid.fill_(-1)
        index.fill_(-1)
        self.op.ep_scatter(x, scale, ids, None, self._counts,
                           q, qs, mid, index, self.e, 256)
        w = self.weights
        gateup, down = self._gateup[:padded], self._down[:padded]
        kwargs = {} if self.mode is None else {'config': {'MODE': self.mode}}
        self.gemm((q, qs), (w.w13, w.s13), gateup, mid, **kwargs)
        aq, asc = self.activation(gateup, alpha=1.702, beta=1.0, limit=7.0)
        self.gemm((aq, asc), (w.w2, w.s2), down, mid, **kwargs)
        self.op.ep_gather(down, ids, route_weights, index, None, result)
        return result


class AlignedMoE:
    """Eight-rank normal EP, accepting already selected physical Top4 routes.

    Gate/TopK and static source-rank replica selection are supplied by the
    caller, so comparisons can use exactly the same route tensor. Optional
    shared_fn runs on an owned auxiliary stream, with a conservative join before
    return. This module is eager-only; no service or distributed group is created.
    """
    def __init__(self, group, weights: PackedExperts, *, num_sms=64,
                 mode=1006, routed_scale=1.0, shared_fn=None, expert_factory=None):
        import deep_ep
        from lightop.quant import per_token_quant_int8
        from lightop.quant.int8 import _wide_int8
        if group.size() != 8 or num_sms not in (24, 64):
            raise ValueError('validated topology is intranode EP8 with 24 or 64 blocks')
        self.experts = (ReceivedExperts(weights, mode=mode) if expert_factory is None
                        else expert_factory(weights))
        self.physical_experts = self.experts.e * 8
        self.device = self.experts.device
        self.stream = torch.cuda.current_stream(self.device).cuda_stream
        self._fallback_quant = per_token_quant_int8
        self._wide_quant = _wide_int8
        if self.experts.h == 6144 and self._wide_quant is None:
            raise RuntimeError('matching SGLang native width-6144 INT8 quantizer is unavailable')
        self.quant = self._quantize
        self.dispatch_config = deep_ep.Config(num_sms, 6, 256, 6, 128)
        self.combine_config = deep_ep.Config(num_sms, 4, 256, 6, 128)
        hbytes = self.experts.h*2  # combine is BF16; reserve for the larger payload
        nvl = max(c.get_nvl_buffer_size_hint(hbytes,8) for c in (self.dispatch_config,self.combine_config))
        rdma = max(c.get_rdma_buffer_size_hint(hbytes,8) for c in (self.dispatch_config,self.combine_config))
        if rdma:
            raise RuntimeError('this module supports intranode EP only')
        self.buffer = deep_ep.Buffer(group, num_nvl_bytes=nvl, num_rdma_bytes=0,
                                     explicitly_destroy=True)
        self.routed_scale = float(routed_scale)
        self.shared_fn = shared_fn
        self.shared_stream = torch.cuda.Stream(device=self.device) if shared_fn else None
        self._guard = None
        self._lock = threading.Lock()
        self.closed = False

    def _quantize(self, hidden):
        if hidden.shape[1] == 6144 and hidden.shape[0] >= 1024:
            q = torch.empty_like(hidden, dtype=torch.int8)
            scale = torch.empty((hidden.shape[0],1), device=self.device, dtype=torch.float32)
            self._wide_quant(q, hidden, scale, 64)
            return q, scale
        return self._fallback_quant(hidden)

    @torch.no_grad()
    def __call__(self, hidden, ids, route_weights, *, expert_weights=None, before_combine=None):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('AlignedMoE owns one non-reentrant DeepEP buffer')
        try:
            if self.closed:
                raise RuntimeError('MoE buffer has been closed')
            if torch.cuda.current_stream(self.device).cuda_stream != self.stream:
                raise RuntimeError('MoE buffer cannot be shared across CUDA streams')
            if expert_weights is not None:
                self.experts.bind_weights(expert_weights)
            if before_combine is None:
                return self._run(hidden, ids, route_weights)
            return self._run(hidden, ids, route_weights, before_combine)
        finally:
            self._lock.release()

    def _run(self, hidden, ids, route_weights, before_combine=None):
        rows = hidden.shape[0]
        if (hidden.ndim != 2 or hidden.shape[1] != self.experts.h or hidden.dtype != torch.bfloat16 or
                ids.shape != (rows,4) or ids.dtype != torch.int64 or
                route_weights.shape != ids.shape or route_weights.dtype != torch.float32):
            raise ValueError('expected BF16 hidden, physical INT64 Top4 and FP32 route weights')
        if any(t.device != self.device or not t.is_contiguous() for t in (hidden,ids,route_weights)):
            raise ValueError('MoE inputs must be contiguous on the weight GPU')
        blocks = triton.cdiv(ids.numel(), 1024)
        if blocks:
            if self._guard is None or self._guard.numel() < blocks:
                self._guard = torch.empty(blocks, dtype=torch.bool, device=self.device)
            guard = self._guard[:blocks]
            _route_guard[(blocks,)](ids, route_weights, guard, ids.numel(), self.physical_experts, 1024)
            torch._assert_async(guard.all(), 'invalid physical Top4 or nonzero padding weight')
        shared_pending = False
        try:
            shared = None
            if self.shared_fn:
                self.shared_stream.wait_stream(torch.cuda.current_stream(self.device))
                hidden.record_stream(self.shared_stream)
                shared_pending = True
                with torch.cuda.stream(self.shared_stream):
                    shared = self.shared_fn(hidden)
            quantized = self.quant(hidden)
            nr, nrdma, ne, mask, event = self.buffer.get_dispatch_layout(ids, self.physical_experts,
                                                                       async_finish=False)
            recv, rid, rw, counts, handle, event = self.buffer.dispatch(
                quantized, topk_idx=ids, topk_weights=route_weights,
                num_tokens_per_rank=nr, num_tokens_per_rdma_rank=nrdma,
                is_token_in_rank=mask, num_tokens_per_expert=ne,
                expert_alignment=256, config=self.dispatch_config, async_finish=False)
            self.last_dispatch = {'aligned_counts': tuple(counts), 'recv_rows': rid.shape[0]}
            if not isinstance(recv, tuple) or len(recv) != 2:
                raise RuntimeError('unverified DeepEP INT8 tuple receive ABI')
            local = self.experts(recv[0], recv[1], rid, rw, counts)
            if before_combine is not None:
                before_combine()
            result, _, event = self.buffer.combine(local, handle, config=self.combine_config,
                                                    async_finish=False)
            if self.routed_scale != 1.0:
                result.mul_(self.routed_scale)
            if shared is not None:
                torch.cuda.current_stream(self.device).wait_stream(self.shared_stream)
                shared_pending = False
                shared.record_stream(torch.cuda.current_stream(self.device))
                result.add_(shared)
            return result
        finally:
            # Join outstanding shared work on both callback and routed failures.
            if shared_pending:
                torch.cuda.current_stream(self.device).wait_stream(self.shared_stream)

    def close(self):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('cannot close inflight MoE buffer')
        try:
            if not self.closed:
                torch.cuda.synchronize(self.device)
                self.buffer.destroy()
                self.closed = True
        finally:
            self._lock.release()
