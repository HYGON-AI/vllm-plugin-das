"""DeepGEMM HIPC MiniMax local experts, layout version hipc-split-v1.

Raw checkpoint scales remain in the signed high-nibble INT8 domain; HIPC
internally supplies the factor 16. Repack weights only before IPC publication.
This module owns neither shared experts nor the model routed scaling factor.
"""
import math
import torch
import triton
import triton.language as tl

LAYOUT = "signed-high-nibble-first-hipc-split-v1"


def prepare_hipc_weights(w13, w2):
    from deepgemm import pack_w4a8_moe_hipc_weight
    if w13.dtype != torch.int8 or w2.dtype != torch.int8:
        raise ValueError("HIPC requires raw split INT8-packed checkpoint weights")
    # Packer may mutate its argument. Never change shared IPC/loader storage.
    return (pack_w4a8_moe_hipc_weight(w13.clone()),
            pack_w4a8_moe_hipc_weight(w2.clone()))


def padded_layout(counts, rows, topk):
    if not isinstance(counts, (list, tuple)) or len(counts) not in (16,17):
        raise ValueError("expected16 or17 exact host DeepEP counts")
    if any(type(n) is not int or n < 0 for n in counts):
        raise ValueError("invalid host expert counts")
    total = sum(counts)
    if total > rows * topk:
        raise ValueError("routes exceed receive capacity")
    # MODE1006 needs expert-contiguous blocks; 128 rows matches its M tile.
    padded = [(n + 127) // 128 * 128 for n in counts]
    actual_start, padded_start = [0], [0]
    for n, p in zip(counts, padded):
        actual_start.append(actual_start[-1] + n)
        padded_start.append(padded_start[-1] + p)
    return actual_start, padded_start


@triton.jit(do_not_specialize=["ROWS", "INVALID"])
def _pad_sorted_routes(X, W, Order, Actual, Bounds, PX, PW, MI, Inv,
                       ROWS, INVALID, H: tl.constexpr, TOPK: tl.constexpr,
                       C: tl.constexpr, NE: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1).to(tl.int64) * C + tl.arange(0, C)
    expert = tl.full((), 0, tl.int32)
    for e in range(NE):
        lo = tl.load(Bounds + e).to(tl.int64)
        hi = tl.load(Bounds + e + 1).to(tl.int64)
        expert = tl.where((row >= lo) & (row < hi), e, expert)
    pstart = tl.load(Bounds + expert).to(tl.int64)
    begin = tl.load(Actual + expert).to(tl.int64)
    end = tl.load(Actual + expert + 1).to(tl.int64)
    offset = row - pstart
    valid = offset < end - begin
    slot = tl.load(Order + INVALID + begin + offset, valid, other=0).to(tl.int64)
    x = tl.load(X + slot // TOPK * H + col, valid & (col < H), other=0)
    tl.store(PX + row * H + col, x, col < H)
    if tl.program_id(1) == 0:
        weight = tl.load(W + slot, valid, other=0)
        tl.store(PW + row, weight)
        tl.store(MI + row, tl.where(valid, expert, -1))
        tl.store(Inv + slot, row.to(tl.int32), valid)



@triton.jit
def _quant_reciprocal(X, Q, Scale, H: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    col = tl.arange(0,B)
    x = tl.load(X + row*H + col,col<H,other=0).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(x),0),1.0e-10)
    scale = tl.div_rn(amax,127.0)
    reciprocal = tl.div_rn(1.0,scale)
    # Preserve original FP32 x*reciprocal(amax/127), not x*(127/amax).
    value = tl.extra.libdevice.nearbyint(x*reciprocal)
    value = tl.minimum(tl.maximum(value,-128.),127.)
    tl.store(Q+row*H+col,value.to(tl.int8),col<H)
    tl.store(Scale+row,scale)


def quant_reciprocal(x):
    q=torch.empty_like(x,dtype=torch.int8)
    scale=torch.empty((x.shape[0],1),dtype=torch.float32,device=x.device)
    _quant_reciprocal[(x.shape[0],)](x,q,scale,x.shape[1],
                                   triton.next_power_of_2(x.shape[1]),
                                   enable_fp_fusion=False)
    return q,scale

def apply_hipc_packed(px, route_weights, m_indices, w13, w2, s13, s2,
                      *, alpha=1.702, beta=1.0, limit=7.0, chunk=32768):
    """Owned sorted padded BF16 rows; output aliases px, no caller input alias."""
    from deepgemm import m_grouped_w4a8_gemm_nt_contiguous_hipc as gemm
    from lightop.activation.swiglu_oai_int8_quant import swiglu_oai_int8_quant_contiguous
    if beta != 1.0 or not math.isfinite(alpha) or alpha <= 0 or not math.isfinite(limit) or limit <= 0:
        raise ValueError("invalid OAI parameters")
    if chunk <= 0 or chunk % 128:
        raise ValueError("HIPC chunk must preserve 128-row expert boundaries")
    if (px.ndim != 2 or px.dtype != torch.bfloat16 or not px.is_cuda
            or not px.is_contiguous() or px.shape[0] % 128
            or m_indices.shape != (px.shape[0],) or m_indices.dtype != torch.int32
            or route_weights.shape != (px.shape[0],1) or route_weights.dtype != torch.float32):
        raise ValueError("invalid sorted HIPC activation inputs")
    e, two_i, half_h = w13.shape
    h, i = px.shape[1], two_i // 2
    if (two_i % 2 or half_h*2 != h or w2.shape != (e,h,i//2)
            or w13.dtype != torch.int8 or w2.dtype != torch.int8
            or s13.shape != (e,two_i,1) or s2.shape != (e,h,1)
            or s13.dtype != torch.float32 or s2.dtype != torch.float32):
        raise ValueError("invalid HIPC packed weights/scales")
    if any(t.device != px.device or not t.is_contiguous()
           for t in (route_weights,m_indices,w13,w2,s13,s2)):
        raise ValueError("HIPC tensors must be contiguous on the same device")
    for start in range(0, px.shape[0], chunk):
        end = min(start + chunk, px.shape[0])
        x, mid = px[start:end], m_indices[start:end]
        q, scale = quant_reciprocal(x)
        # Padding IDs are -1. Zero initialization prevents an uncomputed
        # padded GEMM1 row from feeding arbitrary NaNs into OAI quantization.
        z = torch.zeros((end-start,two_i),dtype=torch.bfloat16,device=px.device)
        gemm((q,scale),(w13,s13),z,mid,config={"MODE":1006})
        aq, asc = swiglu_oai_int8_quant_contiguous(z,alpha=alpha,beta=beta,limit=limit)
        # Preserve Bolt's weighting before final BF16 store. FP32 scale
        # reassociation is checked independently, rather than assumed exact.
        asc.mul_(route_weights[start:end])
        gemm((aq,asc),(w2,s2),x,mid,config={"MODE":1006})
    return px


def run_received_hipc_routes(hidden, ids, weights, counts, w13, w2, s13, s2,
                            *, alpha=1.702, beta=1.0, limit=7.0):
    """Sort/pad local routes, compute two actual HIPC GEMMs and restore slots."""
    from triton_tmp_dir.minimax31_deepep_pack import _restore_routes
    if (hidden.ndim != 2 or hidden.dtype != torch.bfloat16 or not hidden.is_cuda
            or ids.shape != weights.shape or ids.shape != (hidden.shape[0],4)
            or ids.dtype not in (torch.int32,torch.int64) or weights.dtype != torch.float32
            or any(t.device != hidden.device or not t.is_contiguous() for t in (hidden,ids,weights))):
        raise ValueError("invalid received local route tensors")
    rows, h = hidden.shape
    if len(counts)!=w13.shape[0]:raise ValueError("HIPC expert count/weight capacity mismatch")
    actual, bounds = padded_layout(counts,rows,4)
    cap = actual[-1]
    torch._assert_async(((ids >= -1) & (ids < w13.shape[0])).all(),"invalid local HIPC ID")
    torch._assert_async((ids >= 0).sum() == cap,"HIPC host count mismatch")
    if cap == 0:
        return torch.zeros_like(hidden)
    # Invalid -1 slots sort first; only exact host-count valid routes are read.
    # Sorting is part of measured full path, never an offline timing exclusion.
    order = torch.argsort(ids.reshape(-1),stable=False)
    ad = torch.tensor(actual,dtype=torch.int64,device=hidden.device)
    bd = torch.tensor(bounds,dtype=torch.int64,device=hidden.device)
    padded = bounds[-1]
    px = torch.empty((padded,h),dtype=hidden.dtype,device=hidden.device)
    pw = torch.empty((padded,1),dtype=torch.float32,device=hidden.device)
    mid = torch.empty((padded,),dtype=torch.int32,device=hidden.device)
    inv = torch.full_like(ids,-1,dtype=torch.int32)
    _pad_sorted_routes[(padded,triton.cdiv(h,256))](hidden,weights,order,ad,bd,px,pw,mid,inv,
                                                 rows,rows*4-cap,h,4,256,len(counts))
    py = apply_hipc_packed(px,pw,mid,w13,w2,s13,s2,alpha=alpha,beta=beta,limit=limit)
    result = torch.empty_like(hidden)
    _restore_routes[(rows,triton.cdiv(h,256))](py,inv,result,h,4,padded,256)
    return result
