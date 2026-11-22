# SPDX-License-Identifier: Apache-2.0
"""Packed SlimQuant expert TP checkpoint preparation (load-only candidate).

These functions operate on one canonical unsharded expert, before a vLLM
Parameter loader and are not registered or enabled in serving/preload.
"""
from __future__ import annotations
import torch


def shard_expert(w1, w3, w2, scale1, scale3, scale2, *, rank, tp_size, padded_intermediate):
    """Return gate/up and down slices in high-nibble-first packed INT4 layout.

    w1,w3 [I,H/2], w2 [H,I/2]; per-output scales [I,1], [I,1], [H,1].
    I must divide tp_size, and local I and padded_intermediate must be even.
    Only padding is zeroed; weight and scale bytes for the local logical shard
    are copied exactly. Padded gate/up scales are one and padded weights zero.
    These are reference tensors, not vLLM loader or IPC guarantees.
    """
    weights = (w1, w3, w2); scales = (scale1, scale3, scale2)
    if any(w.dtype != torch.int8 or w.ndim != 2 or not w.is_contiguous() for w in weights):
        raise ValueError("weights must be contiguous rank-2 int8")
    if tp_size < 1 or not 0 <= rank < tp_size:
        raise ValueError("invalid tensor-parallel rank or size")
    inter, packed_h = w1.shape
    hidden = packed_h * 2
    if w3.shape != w1.shape or inter % 2 or w2.shape != (hidden, inter // 2):
        raise ValueError("incompatible expert weight shapes")
    if inter % tp_size or (inter // tp_size) % 2:
        raise ValueError("intermediate must yield even unpadded shards")
    local = inter // tp_size
    if padded_intermediate < local or padded_intermediate % 2:
        raise ValueError("padded intermediate must be even and at least local shard")
    if any(s.dtype != torch.float32 or tuple(s.shape) != (n, 1) for s,n in zip(scales, (inter, inter, hidden))):
        raise ValueError("expected fp32 per-channel scales")
    if any(t.device != w1.device for t in weights + scales):
        raise ValueError("all tensors must share device")
    a = torch.zeros((2*padded_intermediate, packed_h), dtype=torch.int8, device=w1.device)
    b = torch.zeros((hidden, padded_intermediate//2), dtype=torch.int8, device=w1.device)
    sa = torch.ones((2*padded_intermediate, 1), dtype=torch.float32, device=w1.device)
    start = rank * local
    a[:local].copy_(w1[start:start+local])
    a[padded_intermediate:padded_intermediate+local].copy_(w3[start:start+local])
    b[:,:local//2].copy_(w2[:,start//2:(start+local)//2])
    sa[:local].copy_(scale1[start:start+local])
    sa[padded_intermediate:padded_intermediate+local].copy_(scale3[start:start+local])
    return a, b, sa, scale2.clone()
