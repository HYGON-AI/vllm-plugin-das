# SPDX-License-Identifier: Apache-2.0
"""Fused interleaved-pair M-RoPE with native PyTorch rounding semantics."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _mrope_kernel(
    Q,
    K,
    Q_OUT,
    K_OUT,
    POS,
    CACHE,
    Q_TOKEN: tl.constexpr,
    Q_HEAD: tl.constexpr,
    K_TOKEN: tl.constexpr,
    K_HEAD: tl.constexpr,
    P_AXIS: tl.constexpr,
    P_TOKEN: tl.constexpr,
    Q_HEADS: tl.constexpr,
    K_HEADS: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    ROTARY_DIM: tl.constexpr,
    SECTION_T: tl.constexpr,
    SECTION_H: tl.constexpr,
    SECTION_W: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    MULTIMODAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0)
    index = tl.arange(0, BLOCK)
    head = index // HEAD_SIZE
    dim = index % HEAD_SIZE
    frequency = dim // 2
    if INTERLEAVED:
        axis = tl.where((frequency % 3 == 1) & (frequency < 3 * SECTION_H), 1, 0)
        axis = tl.where((frequency % 3 == 2) & (frequency < 3 * SECTION_W), 2, axis)
    else:
        axis = tl.where(
            frequency < SECTION_T, 0, tl.where(frequency < SECTION_T + SECTION_H, 1, 2)
        )
    if not MULTIMODAL:
        axis = tl.full((BLOCK,), 0, tl.int32)
    position = tl.load(POS + axis * P_AXIS + token * P_TOKEN)
    rotary = dim < ROTARY_DIM
    cos = tl.load(CACHE + position * ROTARY_DIM + frequency, mask=rotary, other=1).to(
        tl.float32
    )
    sin = tl.load(
        CACHE + position * ROTARY_DIM + ROTARY_DIM // 2 + frequency,
        mask=rotary,
        other=0,
    ).to(tl.float32)
    partner_dim = dim ^ 1
    q = tl.load(Q + token * Q_TOKEN + head * Q_HEAD + dim, head < Q_HEADS, other=0).to(
        tl.float32
    )
    qp = tl.load(
        Q + token * Q_TOKEN + head * Q_HEAD + partner_dim, head < Q_HEADS, other=0
    ).to(tl.float32)
    k = tl.load(K + token * K_TOKEN + head * K_HEAD + dim, head < K_HEADS, other=0).to(
        tl.float32
    )
    kp = tl.load(
        K + token * K_TOKEN + head * K_HEAD + partner_dim, head < K_HEADS, other=0
    ).to(tl.float32)
    # Eager PyTorch rounds each product before the add/subtract in BF16/FP16.
    # Keep those casts and disable FMA to preserve the validated native path.
    qc = (q * cos).to(Q.dtype.element_ty).to(tl.float32)
    qs = (qp * sin).to(Q.dtype.element_ty).to(tl.float32)
    kc = (k * cos).to(K.dtype.element_ty).to(tl.float32)
    ks = (kp * sin).to(K.dtype.element_ty).to(tl.float32)
    qo = tl.where(dim % 2 == 0, qc - qs, qc + qs)
    ko = tl.where(dim % 2 == 0, kc - ks, kc + ks)
    tl.store(
        Q_OUT + token * Q_HEADS * HEAD_SIZE + index,
        tl.where(rotary, qo, q),
        head < Q_HEADS,
    )
    tl.store(
        K_OUT + token * K_HEADS * HEAD_SIZE + index,
        tl.where(rotary, ko, k),
        head < K_HEADS,
    )


def forward_hcu_mrope(self, positions, query, key=None, offsets=None):
    """Use one kernel for GPT-J pairs; retain native fallback for other layouts."""
    if (
        not query.is_cuda
        or key is None
        or query.dtype not in (torch.bfloat16, torch.float16, torch.float32)
        or key.dtype != query.dtype
        or self.is_neox_style
        or self.apply_rotary_emb.enable_fp32_compute
        or offsets is not None
    ):
        return self.forward_native(positions, query, key, offsets)
    count = positions.shape[-1]
    q = query.view(count, -1, self.head_size)
    k = key.view(count, -1, self.head_size)
    if q.stride(-1) != 1 or k.stride(-1) != 1:
        return self.forward_native(positions, query, key, offsets)
    cache = self._match_cos_sin_cache_dtype(query)
    out_q = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    out_k = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    sections = self.mrope_section or [self.rotary_dim // 2, 0, 0]
    _mrope_kernel[(count,)](
        q,
        k,
        out_q,
        out_k,
        positions,
        cache,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        positions.stride(0) if positions.ndim == 2 else 0,
        positions.stride(-1),
        q.shape[1],
        k.shape[1],
        self.head_size,
        self.rotary_dim,
        *sections,
        self.mrope_interleaved,
        positions.ndim == 2,
        triton.next_power_of_2(max(q.shape[1], k.shape[1]) * self.head_size),
        enable_fp_fusion=False,
    )
    return out_q.reshape(query.shape), out_k.reshape(key.shape)
