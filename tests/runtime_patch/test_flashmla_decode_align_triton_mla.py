# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Numerical alignment of the HCU FlashMLA bf16 decode vs the TRITON_MLA decode.

kimi3 precision trace plan, stage 6 acceptance:
  "同一固定 bf16 decode 输入下, FlashMLA decode 与 TRITON_MLA decode 输出数值对齐
   (沿用 max_abs <= 1e-2 / 同 call 激活口径)."

Both ``forward_mqa`` decode paths are reproduced on identical synthetic bf16
inputs with the exact kernel calls each backend makes for a bf16 KV cache:

* FlashMLA bf16 dense decode (gfx93 split-KV/combine), i.e. the plugin
  ``FlashMLAImpl.forward_mqa`` non-fp8 branch:
    - ``q`` (num_decodes, num_q_heads, 576) = concat(q_nope, q_pe) via the same
      lightop ``concat_helper_decode`` the plugin selects under
      ``VLLM_USE_OPT_CAT``;
    - ``reshape_query_for_spec_decode`` -> (B, 1, num_q_heads, 576);
    - ``flash_mla_with_kvcache(..., is_fp8_kvcache=False)`` with a fresh
      ``FlashMLASchedMeta`` (the builder's ``get_mla_metadata`` is a lazy no-op
      that returns an empty scheduler).
* TRITON_MLA decode, i.e. ``TritonMLAImpl.forward_mqa``:
    - ``decode_attention_fwd(..., is_mla=True)`` with ``q`` (B, H_Q, 576),
      k cache (num_blocks, block, 1, 576) and v = first ``kv_lora_rank`` cols.

An fp32 manual MLA reference (over the paged cache) validates that the two
kernels are computing the intended attention rather than merely agreeing on a
shared bug.

Run from the repository root with vLLM and the HCU plugin installed, using
one free HCU:

    HIP_VISIBLE_DEVICES=0 \\
      python -m pytest \\
        tests/runtime_patch/test_flashmla_decode_align_triton_mla.py -q
"""

from __future__ import annotations

import pytest
import torch

import vllm  # noqa: F401  (import vllm first so the HCU plugin activates)

from vllm.v1.attention.ops.triton_decode_attention import decode_attention_fwd
from vllm_hcu.v1.attention.ops.flashmla import (
    FlashMLASchedMeta,
    flash_mla_with_kvcache,
    is_flashmla_dense_supported,
)

# gfx938 / gfx93 MLA decode geometry used by the precision runs (TP2 rank):
# num_q_heads = 48, qk_head_dim = 576 (latent 512 + rope 64), kv_lora_rank 512,
# MQA, block size 64.
NUM_Q_HEADS = 48
QK_HEAD_DIM = 576
KV_LORA_RANK = 512
ROPE_DIM = QK_HEAD_DIM - KV_LORA_RANK
BLOCK_SIZE = 64

_HAS_FLASHMLA = is_flashmla_dense_supported()[0]
_FLASHMLA_REASON = (
    is_flashmla_dense_supported()[1]
    if not _HAS_FLASHMLA
    else "FlashMLA is supported"
)

pytestmark = [
    pytest.mark.hcu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU available"),
    pytest.mark.skipif(not _HAS_FLASHMLA, reason=_FLASHMLA_REASON),
]


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def _decode_case(B: int, seq_lens: list[int], seed: int):
    """Build a fixed random bf16 paged MLA decode case.

    Returns (kv, block_table, seq_lens_t, q_nope, q_pe, pages_per_seq) on
    ``cuda:0``.  kv layout mirrors the vLLM unified MLA cache:
    ``(num_blocks, block_size, qk_head_dim)``, latent first then rope.
    """
    torch.manual_seed(seed)
    dev = torch.device("cuda:0")
    dtype = torch.bfloat16
    pages_per_seq = [_cdiv(sl, BLOCK_SIZE) for sl in seq_lens]
    total_blocks = sum(pages_per_seq)
    kv = torch.randn(total_blocks, BLOCK_SIZE, QK_HEAD_DIM, dtype=dtype, device=dev)
    max_pages = max(pages_per_seq)
    block_table = torch.zeros(B, max_pages, dtype=torch.int32, device=dev)
    base = 0
    for i, np in enumerate(pages_per_seq):
        block_table[i, :np] = torch.arange(
            base, base + np, dtype=torch.int32, device=dev
        )
        base += np
    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32, device=dev)
    q_nope = torch.randn(B, NUM_Q_HEADS, KV_LORA_RANK, dtype=dtype, device=dev)
    q_pe = torch.randn(B, NUM_Q_HEADS, ROPE_DIM, dtype=dtype, device=dev)
    return kv, block_table, seq_lens_t, q_nope, q_pe, pages_per_seq


def _flash_decode(kv, block_table, seq_lens_t, q_nope, q_pe, B, scale):
    """Reproduce the plugin FlashMLAImpl bf16 decode branch (no fp8, no CAT)."""
    from vllm_hcu.ops.test_concat import concat_helper_decode

    # q_nope/q_pe arrive as (B, num_heads, 512/64); forward_mqa concats dim=-1.
    q = concat_helper_decode(q_nope, q_pe, dim=2)  # (B, heads, 576)

    # reshape_query_for_spec_decode(q, num_decodes): decode = one token per seq.
    assert q.dim() == 3 and q.shape[0] % B == 0
    seq_len = q.shape[0] // B
    q = q.view(B, seq_len, q.shape[1], q.shape[2])
    sched = FlashMLASchedMeta()
    out, lse = flash_mla_with_kvcache(
        q=q,
        k_cache=kv.unsqueeze(-2),  # (num_blocks, block, 1, 576)
        block_table=block_table,
        cache_seqlens=seq_lens_t,
        head_dim_v=KV_LORA_RANK,
        tile_scheduler_metadata=sched,
        softmax_scale=scale,
        causal=True,
        is_fp8_kvcache=False,
    )
    # (B, seq_len, heads, d_v) -> (B, heads, d_v) for seq_len == 1
    total = out.shape[0] * out.shape[1]
    out = out.reshape(total, out.shape[2], out.shape[3])
    lse = lse.squeeze(-1)
    return out, lse


def _triton_decode(kv, block_table, seq_lens_t, q_nope, q_pe, B, scale,
                   num_kv_splits):
    """Reproduce the TritonMLAImpl decode path (is_mla=True)."""
    q = torch.cat((q_nope, q_pe), dim=-1)  # (B, heads, 576)
    dev = q.device
    kv_h = kv.unsqueeze(2)  # (num_blocks, block, 1, 576)
    kv_c = kv_h[..., :KV_LORA_RANK]  # (num_blocks, block, 1, 512)
    out = torch.zeros(B, NUM_Q_HEADS, KV_LORA_RANK, dtype=q.dtype, device=dev)
    lse = torch.zeros(B, NUM_Q_HEADS, dtype=q.dtype, device=dev)
    attn_logits = torch.empty(
        (B, NUM_Q_HEADS, num_kv_splits, KV_LORA_RANK + 1),
        dtype=torch.float32,
        device=dev,
    )
    decode_attention_fwd(
        q,
        kv_h,
        kv_c,
        out,
        lse,
        block_table,
        seq_lens_t,
        attn_logits,
        num_kv_splits,
        scale,
        BLOCK_SIZE,
        is_mla=True,
    )
    return out, lse


def _ref_decode(q_nope, q_pe, kv, block_table, seq_lens_t, pages_per_seq, scale):
    """fp32 manual MQA decode over the paged cache. Returns fp64 out/lse."""
    dev = kv.device
    B = q_nope.shape[0]
    q = torch.cat((q_nope, q_pe), dim=-1).float()  # (B, heads, 576)
    kv32 = kv.float()
    outs = torch.zeros(B, NUM_Q_HEADS, KV_LORA_RANK, dtype=torch.float64, device=dev)
    lses = torch.zeros(B, NUM_Q_HEADS, dtype=torch.float64, device=dev)
    for i in range(B):
        sl = int(seq_lens_t[i])
        blocks = block_table[i, : pages_per_seq[i]].tolist()
        kk = torch.cat([kv32[b] for b in blocks])[:sl]  # (sl, 576)
        vv = kk[:, :KV_LORA_RANK]  # value = latent part
        scores = q[i] @ kk.t() * scale  # (heads, sl)
        m = scores.max(dim=-1, keepdim=True).values
        p = torch.exp(scores - m)
        ssum = p.sum(dim=-1, keepdim=True)
        w = p / ssum  # softmax weights
        outs[i] = (w @ vv).double()
        lses[i] = (m + torch.log(ssum)).squeeze(-1).double()
    return outs, lses


def _seqs_for_b(B: int) -> list[int]:
    return {
        1: [22],
        4: [22, 64, 128, 200],
        8: [22, 63, 64, 127, 200, 257, 320, 512],
    }[B]


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.double() - b.double()).abs().max().item()


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(
        a.double().flatten(), b.double().flatten(), dim=0
    ).item()


def _check(name: str, got: torch.Tensor, ref: torch.Tensor) -> tuple[float, float]:
    max_abs = _max_abs(got, ref)
    cos = _cos(got, ref)
    print(f"{name}: max_abs={max_abs:.6e} cosine={cos:.6f}")
    return max_abs, cos


@pytest.mark.parametrize("num_kv_splits", [1, 4])
@pytest.mark.parametrize("B", [1, 4, 8])
def test_flashmla_bf16_decode_matches_triton_mla(B, num_kv_splits):
    """Same fixed bf16 decode input: FlashMLA decode ~ TRITON_MLA decode."""
    seq_lens = _seqs_for_b(B)
    scale = QK_HEAD_DIM ** (-0.5)
    kv, block_table, seq_lens_t, q_nope, q_pe, pages_per_seq = _decode_case(
        B, seq_lens, seed=0
    )
    ref_out, ref_lse = _ref_decode(
        q_nope, q_pe, kv, block_table, seq_lens_t, pages_per_seq, scale
    )

    fl_out, fl_lse = _flash_decode(kv, block_table, seq_lens_t, q_nope, q_pe, B, scale)
    tr_out, tr_lse = _triton_decode(
        kv, block_table, seq_lens_t, q_nope, q_pe, B, scale, num_kv_splits
    )

    # FlashMLA vs TRITON_MLA (the stage-6 alignment gate).
    ft_max, ft_cos = _check("flash-vs-triton out", fl_out, tr_out)
    assert ft_cos >= 0.9999, f"flash vs triton cosine too low: {ft_cos}"
    assert ft_max <= 1e-2, f"flash vs triton max_abs too large: {ft_max}"

    # Both must also match the fp32 manual attention (guards against a shared bug).
    f_max, f_cos = _check("flash-vs-fp32ref out", fl_out, ref_out)
    t_max, t_cos = _check("triton-vs-fp32ref out", tr_out, ref_out)
    assert f_cos >= 0.999 and t_cos >= 0.999
    assert f_max <= 2e-2 and t_max <= 2e-2

    # LSE should be consistent between the two decode paths.
    lse_max = _max_abs(fl_lse, tr_lse)
    print(f"lse flash-vs-triton max_abs={lse_max:.6e}")
    assert lse_max <= 2e-2, f"flash vs triton lse too large: {lse_max}"


@pytest.mark.parametrize("B", [1, 8])
def test_plugin_decode_concat_matches_torch_cat(B):
    """The lightop concat path used by FlashMLAImpl.forward_mqa is a plain cat."""
    from vllm_hcu.ops.test_concat import concat_helper_decode

    dev = torch.device("cuda:0")
    torch.manual_seed(0)
    a = torch.randn(B, NUM_Q_HEADS, KV_LORA_RANK, dtype=torch.bfloat16, device=dev)
    b = torch.randn(B, NUM_Q_HEADS, ROPE_DIM, dtype=torch.bfloat16, device=dev)
    got = concat_helper_decode(a, b, dim=2)
    exp = torch.cat((a, b), dim=-1)
    assert got.shape == exp.shape
    assert torch.equal(got, exp), "concat_helper_decode diverges from torch.cat"
