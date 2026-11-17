# SPDX-License-Identifier: Apache-2.0
"""Frozen uint64 selector; exactly one physical replica per logical route."""
import torch
import triton
import triton.language as tl
from .minimax31_static_runtime import layout

@triton.jit
def _mix(x):
    x=x+0x9e3779b97f4a7c15
    x=(x^(x>>30))*0xbf58476d1ce4e5b9
    x=(x^(x>>27))*0x94d049bb133111eb
    return x^(x>>31)

@triton.jit
def _select(Ids,Pos,Base,Counts,Slots,Out,N:tl.constexpr,REAL:tl.constexpr,LAYER:tl.constexpr,B:tl.constexpr):
    ix=tl.program_id(0)*B+tl.arange(0,B);row=ix//4
    logical=tl.load(Ids+ix,ix<N*4,other=-1).to(tl.int32)
    live=(ix<N*4)&(row<REAL)&(logical>=0)&(logical<128)
    h=tl.load(Base+row,row<N,other=0).to(tl.uint64)
    pos=tl.load(Pos+row,row<N,other=0).to(tl.uint64)
    h=_mix(h^pos);h=_mix(h^tl.full((),LAYER,tl.uint64));h=_mix(h^logical.to(tl.uint64))
    count=tl.load(Counts+logical,live,other=1).to(tl.uint64)
    replica=(h%count).to(tl.int32)
    physical=tl.load(Slots+logical*9+replica,live,other=-1)
    tl.store(Out+ix,tl.where(live,physical,-1),ix<N*4)

def make_tables(layer):
    if layer.w13_weight.is_meta or not layer.w13_weight.is_cuda:raise RuntimeError("static selector requires mapped device weights")
    row=layer._static_placement.row(layer._static_layer_id);counts=[];slots=[]
    for logical in range(128):
        copies=[p for p,e in enumerate(row) if e==logical]
        if not 1<=len(copies)<=9:raise RuntimeError('invalid static replica multiplicity')
        counts.append(len(copies));slots.append(copies+[-1]*(9-len(copies)))
    with torch.inference_mode(False):
        a=torch.tensor(counts,dtype=torch.int32,device=layer.w13_weight.device);b=torch.tensor(slots,dtype=torch.int32,device=layer.w13_weight.device)
    layer._static_selector_tables=(a,b,a._version,b._version,a.data_ptr(),b.data_ptr())

def select_physical(layer,logical_ids):
    if not hasattr(layer,'_static_selector_tables'):make_tables(layer)
    counts,slots,cv,sv,cp,sp=layer._static_selector_tables
    if (counts._version,slots._version,counts.data_ptr(),slots.data_ptr())!=(cv,sv,cp,sp):raise RuntimeError('immutable static selector table was modified')
    if logical_ids.ndim!=2 or logical_ids.shape[1]!=4 or logical_ids.dtype not in (torch.int32,torch.int64) or not logical_ids.is_contiguous() or logical_ids.device!=counts.device:raise ValueError('static selector requires contiguous Top4')
    torch._assert_async(((logical_ids>=-1)&(logical_ids<128)).all(),'static logical Top4 out of range')
    batch=layout(logical_ids.shape[0])
    if (batch['positions'].device!=logical_ids.device or batch['base'].device!=logical_ids.device or batch['positions'].dtype!=torch.int64 or not batch['positions'].is_contiguous() or batch['positions'].numel()<logical_ids.shape[0]):raise ValueError('static selector position/device binding mismatch')
    out=torch.empty_like(logical_ids,dtype=torch.int64)
    if out.numel():_select[(triton.cdiv(out.numel(),256),)](logical_ids,batch['positions'],batch['base'],counts,slots,out,logical_ids.shape[0],batch['rows'],layer._static_layer_id,256)
    return out
