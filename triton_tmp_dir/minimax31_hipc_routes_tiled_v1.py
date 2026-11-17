"""Runtime-only tiled pack. Permanent HIPC weight packer stays unchanged."""
import torch
import triton
import triton.language as tl
from triton_tmp_dir.minimax31_hipc_moe import padded_layout,apply_hipc_packed

@triton.jit(do_not_specialize=["P","INVALID"])
def _pad_sorted_routes_tiled(X,W,Order,Actual,Bounds,PX,PW,MI,Inv,
                            P,INVALID,H:tl.constexpr,TOPK:tl.constexpr,
                            B:tl.constexpr,C:tl.constexpr,NE:tl.constexpr,LOGE:tl.constexpr):
    row=tl.program_id(0).to(tl.int64)*B+tl.arange(0,B)
    col=tl.program_id(1).to(tl.int64)*C+tl.arange(0,C)
    expert=tl.full((B,),0,tl.int32)
    # Upper-bound search with explicit physical capacity; mask out-of-range
    # trials for17 slots and preserve duplicate zero-count expert starts.
    for shift in tl.static_range(LOGE-1,-1,-1):
        trial=expert+(1<<shift)
        bound=tl.load(Bounds+trial,trial<NE,other=0).to(tl.int64)
        expert=tl.where((trial<NE)&(row>=bound),trial,expert)
    pstart=tl.load(Bounds+expert).to(tl.int64)
    begin=tl.load(Actual+expert).to(tl.int64)
    end=tl.load(Actual+expert+1).to(tl.int64)
    offset=row-pstart
    live=(row<P)&(offset>=0)&(offset<end-begin)
    slot=tl.load(Order+INVALID+begin+offset,live,other=0).to(tl.int64)
    x=tl.load(X+(slot//TOPK)[:,None]*H+col[None,:],
              live[:,None]&(col[None,:]<H),other=0)
    tl.store(PX+row[:,None]*H+col[None,:],x,
             (row[:,None]<P)&(col[None,:]<H))
    if tl.program_id(1)==0:
        weight=tl.load(W+slot,live,other=0)
        tl.store(PW+row,weight,row<P)
        tl.store(MI+row,tl.where(live,expert,-1),row<P)
        tl.store(Inv+slot,row.to(tl.int32),live)

@triton.jit(do_not_specialize=["ROWS","CAP"])
def _restore_routes_tiled(PackedY,Inverse,Y,ROWS,H:tl.constexpr,
                          TOPK:tl.constexpr,CAP,B:tl.constexpr,C:tl.constexpr):
    row=tl.program_id(0).to(tl.int64)*B+tl.arange(0,B)
    col=tl.program_id(1).to(tl.int64)*C+tl.arange(0,C)
    acc=tl.zeros((B,C),tl.float32)
    for k in range(TOPK):
        route=tl.load(Inverse+row*TOPK+k,row<ROWS,other=-1).to(tl.int64)
        value=tl.load(PackedY+route[:,None]*H+col[None,:],
                      (row[:,None]<ROWS)&(route[:,None]>=0)&(route[:,None]<CAP)&(col[None,:]<H),other=0)
        acc+=value.to(tl.float32)
    tl.store(Y+row[:,None]*H+col[None,:],
             acc.to(Y.dtype.element_ty,fp_downcast_rounding="rtne"),
             (row[:,None]<ROWS)&(col[None,:]<H))

def run_received_hipc_routes(hidden, ids, weights, counts, w13, w2, s13, s2,
                            *, alpha=1.702, beta=1.0, limit=7.0):
    """Sort/pad local routes, compute two actual HIPC GEMMs and restore slots."""
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
    _pad_sorted_routes_tiled[(triton.cdiv(padded,32),triton.cdiv(h,256))](hidden,weights,order,ad,bd,px,pw,mid,inv,
                                                 padded,rows*4-cap,h,4,32,256,len(counts),(len(counts)-1).bit_length(),num_warps=8)
    py = apply_hipc_packed(px,pw,mid,w13,w2,s13,s2,alpha=alpha,beta=beta,limit=limit)
    result = torch.empty_like(hidden)
    _restore_routes_tiled[(triton.cdiv(rows,32),triton.cdiv(h,256))](py,inv,result,rows,h,4,padded,32,256,num_warps=8)
    return result
