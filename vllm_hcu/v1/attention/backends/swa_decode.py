"""Sliding-window decode attention on the paged FP8 (e4m3) head_dim-192 cache.

Serves the Step5 sliding layers for decode / spec-decode batches (q_len <= 4).
The cutlass varlen_fwd HG prefix kernel spends ~120us per layer on this shape
(it is latency bound with one program per token and casts q to FP8, ~5%
output error); this split-KV kernel takes ~27us at bs16 x 4 tokens with ~0.4%
error against an exact reference on the same FP8 bytes.
"""

import torch

from vllm.triton_utils import tl, triton

_BLOCK_N = 32
_CHUNK = 64
MAX_QUERY_LEN = 4


@triton.jit
def _fp8e4m3_to_bf16(x):
    # Bit-level e4m3fn -> fp32 (exponent rebias via a 2^120 multiply, which also
    # covers subnormals); the generic conversion is far slower on this target.
    b = x.to(tl.uint8, bitcast=True).to(tl.uint32)
    f = (((b & 0x7F) << 20) | ((b & 0x80) << 24)).to(tl.float32, bitcast=True)
    y = (f * 1.329227995784916e36).to(tl.bfloat16)
    # Opaque identity: keeps the compiler from sinking the conversion past the
    # shared-memory staging of the dot operand (byte-wise LDS reads are slow).
    return tl.inline_asm_elementwise("", "=v,0", [y], dtype=tl.bfloat16, is_pure=False, pack=1)


@triton.jit
def _swa_split_kernel(
    q_ptr, k_cache, v_cache,
    m_ws, l_ws, o_ws,
    cu_seqlens_q, seqused_k, block_table,
    softmax_scale,
    stride_qt, stride_qh,
    stride_kb, stride_ks, stride_kh,
    stride_vb, stride_vs, stride_vh,
    stride_bt,
    WINDOW_LEFT: tl.constexpr,
    GROUP: tl.constexpr,
    QL_MAX: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CHUNK: tl.constexpr,
    PAGE: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
):
    # One program per (request, kv head, key chunk).  The rows of the tile are
    # all (query token, q head) pairs of the request that share the kv head, so
    # each K/V tile is loaded once for every query token of the request.
    BLOCK_M: tl.constexpr = QL_MAX * GROUP
    r = tl.program_id(0)
    kvh = tl.program_id(1)
    sp = tl.program_id(2)
    num_kv_heads = tl.num_programs(1)

    q_start = tl.load(cu_seqlens_q + r)
    q_len = tl.load(cu_seqlens_q + r + 1) - q_start
    seq_len = tl.load(seqused_k + r)

    offs_m = tl.arange(0, BLOCK_M)
    tok = offs_m // GROUP
    head = kvh * GROUP + offs_m % GROUP
    row_ok = tok < q_len
    pos = seq_len - q_len + tok
    lo_row = tl.maximum(pos - WINDOW_LEFT, 0)

    offs_d1 = tl.arange(0, 128)
    offs_d2 = 128 + tl.arange(0, 64)
    q_base = q_ptr + (q_start + tok)[:, None] * stride_qt + head[:, None] * stride_qh
    qa = tl.load(q_base + offs_d1[None, :], mask=row_ok[:, None], other=0.0)
    qb = tl.load(q_base + offs_d2[None, :], mask=row_ok[:, None], other=0.0)
    # Fold the softmax scale and log2(e) into q so the loop uses exp2.  An fp8
    # query (quantized by the attention layer) is taken as unit-scaled.
    qk_scale = softmax_scale * 1.4426950408889634
    qa = (qa.to(tl.float32) * qk_scale).to(tl.bfloat16)
    qb = (qb.to(tl.float32) * qk_scale).to(tl.bfloat16)

    req_lo = tl.maximum(seq_len - q_len - WINDOW_LEFT, 0)
    start = (req_lo // BLOCK_N) * BLOCK_N + sp * CHUNK
    end = tl.minimum(start + CHUNK, seq_len)

    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc_a = tl.zeros([BLOCK_M, 128], tl.float32)
    acc_b = tl.zeros([BLOCK_M, 64], tl.float32)
    offs_n = tl.arange(0, BLOCK_N)
    bt_row = block_table + r * stride_bt
    # Manually double-buffered: the raw fp8 tile for the next iteration is in
    # flight while the current one is being processed.
    k_off = offs_n[None, :] * stride_ks + kvh * stride_kh
    v_off = offs_n[:, None] * stride_vs + kvh * stride_vh
    # Clamp so empty splits (and padded requests with seq_len 0) still read a
    # valid block-table entry; their tile is never used.
    n1 = (tl.maximum(tl.minimum(start, seq_len - 1), 0) // BLOCK_N) * BLOCK_N
    page = tl.load(bt_row + n1 // PAGE).to(tl.int64)
    kp = k_cache + page * stride_kb + (n1 % PAGE) * stride_ks + k_off
    vp = v_cache + page * stride_vb + (n1 % PAGE) * stride_vs + v_off
    ka_r = tl.load(kp + offs_d1[:, None])
    kb_r = tl.load(kp + offs_d2[:, None])
    va_r = tl.load(vp + offs_d1[None, :])
    vb_r = tl.load(vp + offs_d2[None, :])
    for n0 in range(start, end, BLOCK_N):
        ka = _fp8e4m3_to_bf16(ka_r)
        kb = _fp8e4m3_to_bf16(kb_r)
        va = _fp8e4m3_to_bf16(va_r)
        vb = _fp8e4m3_to_bf16(vb_r)
        n1 = tl.minimum(n0 + BLOCK_N, end - 1)
        page = tl.load(bt_row + n1 // PAGE).to(tl.int64)
        n1 = (n1 // BLOCK_N) * BLOCK_N
        kp = k_cache + page * stride_kb + (n1 % PAGE) * stride_ks + k_off
        vp = v_cache + page * stride_vb + (n1 % PAGE) * stride_vs + v_off
        ka_r = tl.load(kp + offs_d1[:, None])
        kb_r = tl.load(kp + offs_d2[:, None])
        va_r = tl.load(vp + offs_d1[None, :])
        vb_r = tl.load(vp + offs_d2[None, :])
        s = tl.dot(qa, ka) + tl.dot(qb, kb)
        kpos = n0 + offs_n
        valid = (kpos[None, :] >= lo_row[:, None]) & (kpos[None, :] <= pos[:, None])
        s = tl.where(valid, s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp2(m_i - m_safe)
        p = tl.exp2(s - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        pb = p.to(tl.bfloat16)
        acc_a = acc_a * alpha[:, None] + tl.dot(pb, va)
        acc_b = acc_b * alpha[:, None] + tl.dot(pb, vb)
        m_i = m_new

    inv = tl.where(l_i > 0, 1.0 / l_i, 0.0)
    ws = (r * num_kv_heads + kvh) * NUM_SPLITS + sp
    tl.store(m_ws + ws * BLOCK_M + offs_m, m_i)
    tl.store(l_ws + ws * BLOCK_M + offs_m, l_i)
    o_base = o_ws + (ws * BLOCK_M + offs_m)[:, None] * 192
    tl.store(o_base + offs_d1[None, :], (acc_a * inv[:, None]).to(tl.bfloat16))
    tl.store(o_base + offs_d2[None, :], (acc_b * inv[:, None]).to(tl.bfloat16))


@triton.jit
def _swa_combine_kernel(
    m_ws, l_ws, o_ws, out_ptr,
    cu_seqlens_q,
    stride_ot, stride_oh,
    GROUP: tl.constexpr,
    QL_MAX: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    SPLITS_P2: tl.constexpr,
):
    # One program per (request, query token, q head): merge the per-split
    # partial softmax results of that row.
    BLOCK_M: tl.constexpr = QL_MAX * GROUP
    r = tl.program_id(0)
    tok = tl.program_id(1)
    h = tl.program_id(2)
    num_heads = tl.num_programs(2)
    kvh = h // GROUP
    row = tok * GROUP + h % GROUP
    q_start = tl.load(cu_seqlens_q + r)
    q_len = tl.load(cu_seqlens_q + r + 1) - q_start
    if tok >= q_len:
        return
    offs_s = tl.arange(0, SPLITS_P2)
    s_ok = offs_s < NUM_SPLITS
    offs_d = tl.arange(0, 256)
    d_ok = offs_d < 192
    num_kv_heads = num_heads // GROUP
    idx = ((r * num_kv_heads + kvh) * NUM_SPLITS + offs_s) * BLOCK_M + row
    m = tl.load(m_ws + idx, mask=s_ok, other=float("-inf"))
    l = tl.load(l_ws + idx, mask=s_ok, other=0.0)
    o = tl.load(o_ws + idx[:, None] * 192 + offs_d[None, :], mask=s_ok[:, None] & d_ok[None, :], other=0.0)
    m_max = tl.max(m, axis=0)
    m_max = tl.where(m_max == float("-inf"), 0.0, m_max)
    w = l * tl.exp2(m - m_max)
    l_sum = tl.sum(w, axis=0)
    acc = tl.sum(w[:, None] * o.to(tl.float32), axis=0)
    acc = acc / tl.where(l_sum > 0, l_sum, 1.0)
    tl.store(out_ptr + (q_start + tok) * stride_ot + h * stride_oh + offs_d, acc.to(tl.bfloat16), mask=d_ok)


def can_use_swa_decode_fp8_d192(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    max_seqlen_q: int,
    causal: bool,
    window_size: list[int] | None,
    s_aux: torch.Tensor | None,
) -> bool:
    if window_size is None or window_size[0] < 0 or window_size[1] != 0:
        return False
    num_kv_heads = key_cache.shape[2]
    group = query.shape[1] // num_kv_heads
    return (
        causal is True
        and s_aux is None
        and max_seqlen_q <= MAX_QUERY_LEN
        and query.dtype in (torch.bfloat16, torch.float8_e4m3fn)
        and query.shape[-1] == 192
        and query.shape[1] % num_kv_heads == 0
        and group == triton.next_power_of_2(group)
        and key_cache.shape[1] % _BLOCK_N == 0
    )


def _swa_split(q, k, v, cu_seqlens_q, seqused_k, block_table, softmax_scale, window_left):
    num_heads = q.shape[1]
    num_kv_heads = k.shape[2]
    group = num_heads // num_kv_heads
    num_seqs = cu_seqlens_q.shape[0] - 1
    # Always size for MAX_QUERY_LEN so one compiled variant serves every
    # decode batch (and cudagraph replays with shorter requests).
    ql_max = MAX_QUERY_LEN
    num_splits = triton.cdiv(window_left + ql_max + _BLOCK_N - 1, _CHUNK)
    n = num_seqs * num_kv_heads * num_splits * ql_max * group
    m_ws = torch.empty(n, dtype=torch.float32, device=q.device)
    l_ws = torch.empty(n, dtype=torch.float32, device=q.device)
    o_ws = torch.empty(n, 192, dtype=torch.bfloat16, device=q.device)
    _swa_split_kernel[(num_seqs, num_kv_heads, num_splits)](
        q, k, v, m_ws, l_ws, o_ws, cu_seqlens_q, seqused_k, block_table,
        softmax_scale,
        q.stride(0), q.stride(1),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        block_table.stride(0),
        WINDOW_LEFT=window_left, GROUP=group, QL_MAX=ql_max,
        BLOCK_N=_BLOCK_N, CHUNK=_CHUNK, PAGE=k.shape[1], NUM_SPLITS=num_splits,
        num_warps=4, num_stages=1,
    )
    return m_ws, l_ws, o_ws, group, ql_max, num_splits


def swa_decode_fp8_d192(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seqused_k: torch.Tensor,
    block_table: torch.Tensor,
    softmax_scale: float,
    window_left: int,
) -> torch.Tensor:
    num_heads = q.shape[1]
    num_seqs = cu_seqlens_q.shape[0] - 1
    m_ws, l_ws, o_ws, group, ql_max, num_splits = _swa_split(
        q, k, v, cu_seqlens_q, seqused_k, block_table, softmax_scale, window_left
    )
    _swa_combine_kernel[(num_seqs, ql_max, num_heads)](
        m_ws, l_ws, o_ws, out, cu_seqlens_q,
        out.stride(0), out.stride(1),
        GROUP=group, QL_MAX=ql_max, NUM_SPLITS=num_splits,
        SPLITS_P2=triton.next_power_of_2(num_splits),
        num_warps=1,
    )
    return out


# ---------------------------------------------------------------------------
# Step5 attention epilogue: out * sigmoid(gate[head]) -> per-token fp8 for the
# o_proj W8A8 GEMM. Same arithmetic as the Inductor kernels it replaces
# (fp32, no rounding between gate and quant; scale = max(amax / 448, 1e-12)).
# ---------------------------------------------------------------------------

@triton.jit
def _gate_quant_store(x, t, gate_ptr, stride_gt, xq_ptr, xs_ptr,
                      NH: tl.constexpr, D: tl.constexpr, BD: tl.constexpr):
    # x: [NH, BD] fp32 (bf16-representable), columns >= D are zero.
    offs_h = tl.arange(0, NH)
    offs_d = tl.arange(0, BD)
    d_ok = offs_d < D
    g = tl.load(gate_ptr + t.to(tl.int64) * stride_gt + offs_h).to(tl.float32)
    x = x * tl.sigmoid(g)[:, None]
    amax = tl.max(tl.max(tl.abs(x), axis=1), axis=0)
    # 1/448 rounded to fp32, as Inductor folds the division.
    scale = tl.maximum(amax * 0.002232142857142857, 1e-12)
    xq = tl.minimum(tl.maximum(x / scale, -448.0), 448.0).to(tl.float8e4nv)
    dst = xq_ptr + t.to(tl.int64) * (NH * D) + offs_h[:, None] * D + offs_d[None, :]
    tl.store(dst, xq, mask=d_ok[None, :])
    tl.store(xs_ptr + t, scale)


@triton.jit
def _zero_row(t, xq_ptr, xs_ptr, NH: tl.constexpr, D: tl.constexpr, BD: tl.constexpr):
    offs_h = tl.arange(0, NH)
    offs_d = tl.arange(0, BD)
    dst = xq_ptr + t.to(tl.int64) * (NH * D) + offs_h[:, None] * D + offs_d[None, :]
    tl.store(dst, tl.zeros([NH, BD], tl.float32).to(tl.float8e4nv),
             mask=(offs_d < D)[None, :])
    tl.store(xs_ptr + t, 1.0)


@triton.jit
def _swa_combine_gate_quant_kernel(
    m_ws, l_ws, o_ws,
    cu_seqlens_q, num_seqs,
    gate_ptr, stride_gt,
    xq_ptr, xs_ptr,
    NH: tl.constexpr,
    GROUP: tl.constexpr,
    QL_MAX: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    SEQS_P2: tl.constexpr,
):
    # One program per token row (all heads): merge the split partials (single
    # pass, online rescaling, so all split loads are issued up front), round to
    # bf16 as the bf16 attention output would be, then gate + quantize the row.
    # Rows past the last request (cudagraph padding) get zeros.
    BLOCK_M: tl.constexpr = QL_MAX * GROUP
    t = tl.program_id(0)
    offs_r = tl.arange(0, SEQS_P2)
    ends = tl.load(cu_seqlens_q + 1 + offs_r, mask=offs_r < num_seqs, other=2147483647)
    done = ends <= t
    r = tl.sum(done.to(tl.int32), axis=0)
    if r >= num_seqs:
        _zero_row(t, xq_ptr, xs_ptr, NH, 192, 256)
        return
    # Start of request r = end of the last finished one (no dependent load).
    tok = t - tl.max(tl.where(done, ends, 0), axis=0)
    offs_h = tl.arange(0, NH)
    offs_d = tl.arange(0, 256)
    d_ok = offs_d < 192
    num_kv_heads: tl.constexpr = NH // GROUP
    base = ((r * num_kv_heads + offs_h // GROUP) * NUM_SPLITS) * BLOCK_M + tok * GROUP + offs_h % GROUP
    m_run = tl.full([NH], float("-inf"), tl.float32)
    l_sum = tl.zeros([NH], tl.float32)
    acc = tl.zeros([NH, 256], tl.float32)
    for s in tl.static_range(NUM_SPLITS):
        idx = base + s * BLOCK_M
        m = tl.load(m_ws + idx)
        l = tl.load(l_ws + idx)
        o = tl.load(o_ws + idx[:, None] * 192 + offs_d[None, :], mask=d_ok[None, :], other=0.0)
        m_new = tl.maximum(m_run, m)
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        a = tl.exp2(m_run - m_safe)
        w = l * tl.exp2(m - m_safe)
        acc = acc * a[:, None] + w[:, None] * o.to(tl.float32)
        l_sum = l_sum * a + w
        m_run = m_new
    acc = acc / tl.where(l_sum > 0, l_sum, 1.0)[:, None]
    x = acc.to(tl.bfloat16).to(tl.float32)
    _gate_quant_store(x, t, gate_ptr, stride_gt, xq_ptr, xs_ptr, NH, 192, 256)


@triton.jit
def _gate_quant_kernel(
    x_ptr, stride_xt, num_valid,
    gate_ptr, stride_gt,
    xq_ptr, xs_ptr,
    NH: tl.constexpr,
    D: tl.constexpr,
    BD: tl.constexpr,
):
    t = tl.program_id(0)
    if t >= num_valid:
        _zero_row(t, xq_ptr, xs_ptr, NH, D, BD)
        return
    offs_h = tl.arange(0, NH)
    offs_d = tl.arange(0, BD)
    x = tl.load(x_ptr + t.to(tl.int64) * stride_xt + offs_h[:, None] * D + offs_d[None, :],
                mask=(offs_d < D)[None, :], other=0.0).to(tl.float32)
    _gate_quant_store(x, t, gate_ptr, stride_gt, xq_ptr, xs_ptr, NH, D, BD)


def gate_quant_fp8(
    x: torch.Tensor,
    gate: torch.Tensor,
    xq: torch.Tensor,
    xs: torch.Tensor,
    num_valid: int | None = None,
    num_warps: int = 4,
) -> None:
    """x bf16 [T, NH * D], gate [T, NH] (row-strided) -> xq fp8 [T, NH * D],
    xs fp32 [T, 1]. Rows >= num_valid are written as zeros (scale 1)."""
    num_tokens = xq.shape[0]
    if num_tokens == 0:
        return
    nh = gate.shape[-1]
    d = x.shape[-1] // nh
    _gate_quant_kernel[(num_tokens,)](
        x, x.stride(0), num_tokens if num_valid is None else num_valid,
        gate, gate.stride(0), xq, xs,
        NH=nh, D=d, BD=triton.next_power_of_2(d),
        num_warps=num_warps,
    )


def swa_decode_fp8_d192_gate_quant(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seqused_k: torch.Tensor,
    block_table: torch.Tensor,
    softmax_scale: float,
    window_left: int,
    gate: torch.Tensor,
    xq: torch.Tensor,
    xs: torch.Tensor,
    num_warps: int = 8,
) -> None:
    """swa_decode_fp8_d192 whose combine also applies the per-head sigmoid
    gate and quantizes each token row to fp8 (xq [T, NH * 192], xs [T, 1])."""
    num_heads = q.shape[1]
    num_seqs = cu_seqlens_q.shape[0] - 1
    m_ws, l_ws, o_ws, group, ql_max, num_splits = _swa_split(
        q, k, v, cu_seqlens_q, seqused_k, block_table, softmax_scale, window_left
    )
    _swa_combine_gate_quant_kernel[(xq.shape[0],)](
        m_ws, l_ws, o_ws, cu_seqlens_q, num_seqs,
        gate, gate.stride(0), xq, xs,
        NH=num_heads, GROUP=group, QL_MAX=ql_max, NUM_SPLITS=num_splits,
        SEQS_P2=max(triton.next_power_of_2(num_seqs), 2),
        num_warps=num_warps,
    )
