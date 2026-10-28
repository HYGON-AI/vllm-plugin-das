"""
Fused RMSNorm + MRoPE Triton kernel for Qwen3.5-35B-A3B on Hygon DCU.

Model params (from config.json text_config):
  head_dim         = 256
  rotary_dim       = 64   (partial_rotary_factor = 0.25)
  half_rd          = 32   = rotary_dim // 2
  mrope_section    = [11, 11, 10]  (T, H, W)
  mrope_interleaved = True

Interleaved frequency assignment for i in [0..31]:
  T: i % 3 == 0  (11 total: 0,3,6,...,30)
  H: i % 3 == 1  (11 total: 1,4,7,...,31)
  W: i % 3 == 2  (10 total: 2,5,8,...,29)

cos_sin_cache layout from vllm: [max_pos, head_dim]
  The vllm cache stores cos and sin interleaved per the base class.
  Specifically, for neox-style: cache[pos] = [cos(0..half_hd), sin(0..half_hd)]
  where half_hd = head_dim // 2 = 128.
  We only need the first half_rd=32 entries of cos and sin.

positions: [3, num_tokens]  (row 0=T, row 1=H, row 2=W), any strides.
  With VLLM_HCU_1D_MROPE the runner passes a token-major view
  (buf.view(N, 3).T, stride (1, 3)), so the kernel reads it through strides.
"""

import triton
import triton.language as tl
import torch


@triton.jit
def _fused_rmsnorm_mrope_fwd(
    # Separate q and k tensors, modified in-place
    q_ptr,             # [num_tokens, num_qh * head_dim]
    k_ptr,             # [num_tokens, num_kvh * head_dim]
    q_weight_ptr,      # [head_dim]  RMSNorm scale for Q
    k_weight_ptr,      # [head_dim]  RMSNorm scale for K
    cos_sin_cache_ptr, # [max_pos, rotary_dim]  col[0..half_rd-1]=cos, col[half_rd..rotary_dim-1]=sin
    pos_ptr,           # [3, num_tokens]  T/H/W positions, arbitrary strides
    # Scalar strides
    pos_stride_dim,    # positions.stride(0)  (T -> H -> W)
    pos_stride_tok,    # positions.stride(1)
    q_stride_tok,      # = num_qh  * head_dim
    k_stride_tok,      # = num_kvh * head_dim
    cache_stride,      # = rotary_dim (stride along position axis in cos_sin_cache)
    eps,               # RMSNorm epsilon
    # Compile-time constants
    num_qh:   tl.constexpr,
    num_kvh:  tl.constexpr,
    head_dim: tl.constexpr,   # 256
    rotary_dim: tl.constexpr, # 64
    half_rd:  tl.constexpr,   # 32  (= rotary_dim//2; also sin column offset in cache)
    BLOCK: tl.constexpr,      # >= head_dim, power-of-2  (256)
):
    """
    Grid: (num_tokens, num_qh + num_kvh)
    Each CTA handles one (token, head) pair.
    """
    pid_tok  = tl.program_id(0)
    pid_head = tl.program_id(1)

    is_k = pid_head >= num_qh
    local_head = pid_head - num_qh if is_k else pid_head

    # Base pointer into this head's buffer
    if is_k:
        base = k_ptr + pid_tok * k_stride_tok + local_head * head_dim
        weight_ptr = k_weight_ptr
    else:
        base = q_ptr + pid_tok * q_stride_tok + local_head * head_dim
        weight_ptr = q_weight_ptr

    # ── 1. Load full head ─────────────────────────────────────────────────
    offs = tl.arange(0, BLOCK)
    mask = offs < head_dim
    x = tl.load(base + offs, mask=mask, other=0.0).to(tl.float32)

    # ── 2. RMSNorm (GemmaRMSNorm style: x * (1 + w)) ─────────────────────
    var  = tl.sum(x * x, axis=0) * (1.0 / head_dim)
    rstd = tl.rsqrt(var + eps)
    w    = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    xn   = x * rstd * (1.0 + w)   # GemmaRMSNorm: weight is initialized to 0, scale = 1+w

    # ── 3. Build interleaved T/H/W cos/sin ────────────────────────────────
    pos_tok = pos_ptr + pid_tok * pos_stride_tok
    pos_t = tl.load(pos_tok).to(tl.int64)
    pos_h = tl.load(pos_tok + pos_stride_dim).to(tl.int64)
    pos_w = tl.load(pos_tok + 2 * pos_stride_dim).to(tl.int64)

    rd_offs = tl.arange(0, half_rd)  # [0..31]

    # Load cos/sin for T/H/W
    # cos_sin_cache[pos, 0..half_rd-1] = cos, [pos, half_rd..rotary_dim-1] = sin
    cos_t = tl.load(cos_sin_cache_ptr + pos_t * cache_stride + rd_offs).to(tl.float32)
    sin_t = tl.load(cos_sin_cache_ptr + pos_t * cache_stride + half_rd + rd_offs).to(tl.float32)
    cos_h = tl.load(cos_sin_cache_ptr + pos_h * cache_stride + rd_offs).to(tl.float32)
    sin_h = tl.load(cos_sin_cache_ptr + pos_h * cache_stride + half_rd + rd_offs).to(tl.float32)
    cos_w = tl.load(cos_sin_cache_ptr + pos_w * cache_stride + rd_offs).to(tl.float32)
    sin_w = tl.load(cos_sin_cache_ptr + pos_w * cache_stride + half_rd + rd_offs).to(tl.float32)

    # Interleaved selection: i%3==0→T, i%3==1→H, i%3==2→W
    # Compute mod-3 inline
    mod3 = rd_offs % 3  # [0..31], values in {0,1,2}
    t_sel = (mod3 == 0).to(tl.float32)
    h_sel = (mod3 == 1).to(tl.float32)
    w_sel = (mod3 == 2).to(tl.float32)

    cos_row = t_sel * cos_t + h_sel * cos_h + w_sel * cos_w  # [half_rd]
    sin_row = t_sel * sin_t + h_sel * sin_h + w_sel * sin_w  # [half_rd]

    # ── 4. NeoX-style RoPE on first rotary_dim=64 elements ───────────────
    # x1 = xn[0..31],  x2 = xn[32..63]
    # new_x1[i] = x1[i]*cos[i] - x2[i]*sin[i]
    # new_x2[i] = x2[i]*cos[i] + x1[i]*sin[i]
    #
    # Re-load x1, x2 as half_rd-wide vectors to avoid vector slicing.

    x_raw1 = tl.load(base + rd_offs).to(tl.float32)
    w1     = tl.load(weight_ptr + rd_offs).to(tl.float32)
    x1     = x_raw1 * rstd * (1.0 + w1)   # GemmaRMSNorm: 1+w

    x_raw2 = tl.load(base + half_rd + rd_offs).to(tl.float32)
    w2     = tl.load(weight_ptr + half_rd + rd_offs).to(tl.float32)
    x2     = x_raw2 * rstd * (1.0 + w2)   # GemmaRMSNorm: 1+w

    new_x1 = x1 * cos_row - x2 * sin_row
    new_x2 = x2 * cos_row + x1 * sin_row

    # ── 5. Store results ──────────────────────────────────────────────────
    out_dtype = q_ptr.dtype.element_ty

    # Passthrough region [rotary_dim..head_dim)
    tl.store(base + offs, xn.to(out_dtype),
             mask=mask & (offs >= rotary_dim))

    # Rotary region
    tl.store(base + rd_offs,           new_x1.to(out_dtype))
    tl.store(base + half_rd + rd_offs, new_x2.to(out_dtype))


def fused_rmsnorm_mrope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    head_dim: int,
    rotary_dim: int,
    eps: float,
    num_warps: int = 4,
) -> tuple:
    """
    In-place fused RMSNorm + interleaved MRoPE.

    Accepts q/k in any shape whose last dimension is num_heads*head_dim;
    reshapes and ensures contiguity internally.

    Args:
        q:             [..., num_qh  * head_dim]
        k:             [..., num_kvh * head_dim]
        q_weight:      [head_dim]
        k_weight:      [head_dim]
        cos_sin_cache: [max_pos, rotary_dim], cos||sin packed
        positions:     [3, num_tokens], any strides (no copy is made)
        head_dim:      256
        rotary_dim:    64
        eps:           RMSNorm epsilon

    Returns:
        (q, k) modified in-place, flattened to [num_tokens, num_*h * head_dim]
    """
    assert positions.ndim == 2 and positions.shape[0] == 3, \
        "positions must be [3, num_tokens] for mrope"

    q = q.contiguous().view(-1, q.shape[-1])
    k = k.contiguous().view(-1, k.shape[-1])

    num_tokens = q.shape[0]
    num_qh  = q.shape[1] // head_dim
    num_kvh = k.shape[1] // head_dim
    half_rd = rotary_dim // 2

    grid = (num_tokens, num_qh + num_kvh)

    _fused_rmsnorm_mrope_fwd[grid](
        q, k,
        q_weight, k_weight,
        cos_sin_cache,
        positions,
        pos_stride_dim = positions.stride(0),
        pos_stride_tok = positions.stride(1),
        q_stride_tok  = q.shape[1],
        k_stride_tok  = k.shape[1],
        cache_stride  = cos_sin_cache.shape[1],
        eps           = eps,
        num_qh        = num_qh,
        num_kvh       = num_kvh,
        head_dim      = head_dim,
        rotary_dim    = rotary_dim,
        half_rd       = half_rd,
        BLOCK         = triton.next_power_of_2(head_dim),
        num_warps     = num_warps,
    )
    return q, k


@triton.jit
def _fused_qkv_split_rmsnorm_mrope_fwd(
    qkv_ptr,           # [num_tokens, ld]: per q head [q | gate] (gated) or [q],
                       # then k heads, then v heads
    q_out_ptr,         # [num_tokens, num_qh * head_dim]
    k_out_ptr,         # [num_tokens, num_kvh * head_dim]
    v_out_ptr,         # [num_tokens, num_kvh * head_dim]
    q_weight_ptr,
    k_weight_ptr,
    cos_sin_cache_ptr,
    pos_ptr,
    pos_stride_dim,
    pos_stride_tok,
    qkv_stride_tok,
    cache_stride,
    eps,
    num_qh: tl.constexpr,
    num_kvh: tl.constexpr,
    q_head_stride: tl.constexpr,   # 2 * head_dim when gated, else head_dim
    head_dim: tl.constexpr,
    rotary_dim: tl.constexpr,
    half_rd: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Grid: (num_tokens, num_qh + 2 * num_kvh).

    Reads q/k/v straight out of the packed projection (no contiguous copies)
    and writes contiguous q/k (RMSNorm + interleaved MRoPE) and v.
    """
    pid_tok = tl.program_id(0).to(tl.int64)
    pid_head = tl.program_id(1)
    offs = tl.arange(0, BLOCK)
    mask = offs < head_dim
    row = qkv_ptr + pid_tok * qkv_stride_tok
    k_base_col = num_qh * q_head_stride
    v_base_col = k_base_col + num_kvh * head_dim

    if pid_head >= num_qh + num_kvh:
        vh = pid_head - num_qh - num_kvh
        xv = tl.load(row + v_base_col + vh * head_dim + offs, mask=mask)
        tl.store(v_out_ptr + pid_tok * (num_kvh * head_dim) + vh * head_dim
                 + offs, xv, mask=mask)
    else:
        if pid_head >= num_qh:
            local_head = pid_head - num_qh
            src = row + k_base_col + local_head * head_dim
            dst = (k_out_ptr + pid_tok * (num_kvh * head_dim)
                   + local_head * head_dim)
            weight_ptr = k_weight_ptr
        else:
            src = row + pid_head * q_head_stride
            dst = q_out_ptr + pid_tok * (num_qh * head_dim) + pid_head * head_dim
            weight_ptr = q_weight_ptr

        x = tl.load(src + offs, mask=mask, other=0.0).to(tl.float32)
        var = tl.sum(x * x, axis=0) * (1.0 / head_dim)
        rstd = tl.rsqrt(var + eps)
        w = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        xn = x * rstd * (1.0 + w)

        pos_tok = pos_ptr + pid_tok * pos_stride_tok
        pos_t = tl.load(pos_tok).to(tl.int64)
        pos_h = tl.load(pos_tok + pos_stride_dim).to(tl.int64)
        pos_w = tl.load(pos_tok + 2 * pos_stride_dim).to(tl.int64)
        rd_offs = tl.arange(0, half_rd)
        cos_t = tl.load(cos_sin_cache_ptr + pos_t * cache_stride + rd_offs).to(tl.float32)
        sin_t = tl.load(cos_sin_cache_ptr + pos_t * cache_stride + half_rd + rd_offs).to(tl.float32)
        cos_h = tl.load(cos_sin_cache_ptr + pos_h * cache_stride + rd_offs).to(tl.float32)
        sin_h = tl.load(cos_sin_cache_ptr + pos_h * cache_stride + half_rd + rd_offs).to(tl.float32)
        cos_w = tl.load(cos_sin_cache_ptr + pos_w * cache_stride + rd_offs).to(tl.float32)
        sin_w = tl.load(cos_sin_cache_ptr + pos_w * cache_stride + half_rd + rd_offs).to(tl.float32)
        mod3 = rd_offs % 3
        t_sel = (mod3 == 0).to(tl.float32)
        h_sel = (mod3 == 1).to(tl.float32)
        w_sel = (mod3 == 2).to(tl.float32)
        cos_row = t_sel * cos_t + h_sel * cos_h + w_sel * cos_w
        sin_row = t_sel * sin_t + h_sel * sin_h + w_sel * sin_w

        x1 = tl.load(src + rd_offs).to(tl.float32) * rstd * (
            1.0 + tl.load(weight_ptr + rd_offs).to(tl.float32))
        x2 = tl.load(src + half_rd + rd_offs).to(tl.float32) * rstd * (
            1.0 + tl.load(weight_ptr + half_rd + rd_offs).to(tl.float32))
        out_dtype = q_out_ptr.dtype.element_ty
        tl.store(dst + offs, xn.to(out_dtype), mask=mask & (offs >= rotary_dim))
        tl.store(dst + rd_offs, (x1 * cos_row - x2 * sin_row).to(out_dtype))
        tl.store(dst + half_rd + rd_offs,
                 (x2 * cos_row + x1 * sin_row).to(out_dtype))


def fused_qkv_split_rmsnorm_mrope(
    qkv: torch.Tensor,
    num_qh: int,
    num_kvh: int,
    gated: bool,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    head_dim: int,
    rotary_dim: int,
    eps: float,
    num_warps: int = 4,
) -> tuple:
    """Out-of-place split + RMSNorm + interleaved MRoPE from packed qkv.

    Returns contiguous (q, k, v) of shapes [T, num_qh*hd], [T, num_kvh*hd]
    and [T, num_kvh*hd]; the attention gate (if any) stays inside ``qkv``.
    """
    assert positions.ndim == 2 and positions.shape[0] == 3
    assert qkv.stride(-1) == 1
    num_tokens = qkv.shape[0]
    q = qkv.new_empty(num_tokens, num_qh * head_dim)
    k = qkv.new_empty(num_tokens, num_kvh * head_dim)
    v = qkv.new_empty(num_tokens, num_kvh * head_dim)
    if num_tokens == 0:
        return q, k, v
    _fused_qkv_split_rmsnorm_mrope_fwd[(num_tokens, num_qh + 2 * num_kvh)](
        qkv, q, k, v,
        q_weight, k_weight,
        cos_sin_cache,
        positions,
        pos_stride_dim=positions.stride(0),
        pos_stride_tok=positions.stride(1),
        qkv_stride_tok=qkv.stride(0),
        cache_stride=cos_sin_cache.shape[1],
        eps=eps,
        num_qh=num_qh,
        num_kvh=num_kvh,
        q_head_stride=(2 if gated else 1) * head_dim,
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        half_rd=rotary_dim // 2,
        BLOCK=triton.next_power_of_2(head_dim),
        num_warps=num_warps,
    )
    return q, k, v


@triton.jit
def _strided_sigmoid_gate_mul_kernel(x_ptr, qkv_ptr, qkv_stride_tok, out_ptr,
                                     num_qh: tl.constexpr,
                                     head_dim: tl.constexpr):
    """out[t, h*hd + i] = x[t, h*hd + i] * sigmoid(qkv[t, h*2*hd + hd + i])."""
    t = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1)
    offs = tl.arange(0, head_dim)
    x = tl.load(x_ptr + t * (num_qh * head_dim) + h * head_dim + offs).to(tl.float32)
    g = tl.load(qkv_ptr + t * qkv_stride_tok + h * 2 * head_dim + head_dim
                + offs).to(tl.float32)
    tl.store(out_ptr + t * (num_qh * head_dim) + h * head_dim + offs,
             (x / (1.0 + tl.exp(-g))).to(out_ptr.dtype.element_ty))


def strided_sigmoid_gate_mul(attn_out: torch.Tensor, qkv: torch.Tensor,
                             num_qh: int, head_dim: int) -> torch.Tensor:
    """attn_out * sigmoid(gate) with the gate read in place from packed qkv."""
    num_tokens = attn_out.shape[0]
    x = attn_out.reshape(num_tokens, num_qh * head_dim)
    assert x.is_contiguous() and qkv.stride(-1) == 1
    out = torch.empty_like(x)
    if num_tokens:
        _strided_sigmoid_gate_mul_kernel[(num_tokens, num_qh)](
            x, qkv, qkv.stride(0), out, num_qh=num_qh, head_dim=head_dim,
            num_warps=4)
    return out.view(attn_out.shape)
