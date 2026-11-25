"""Fuse route predicates while retaining both native asynchronous assertions."""
import triton
import triton.language as tl


@triton.jit
def _route_checks(Ids,Weights,Checks,N:tl.constexpr,EXPERTS:tl.constexpr,BLOCK:tl.constexpr):
    offset=tl.arange(0,BLOCK);mask=offset<N
    ids=tl.load(Ids+offset,mask=mask,other=0)
    weights=tl.load(Weights+offset,mask=mask,other=0)
    bits=weights.to(tl.uint32,bitcast=True)&0x7fffffff
    finite=bits<0x7f800000
    zero=bits==0
    bad_id=mask&((ids < -1)|(ids>=EXPERTS))
    bad_weight=mask&(~finite|((ids<0)&~zero))
    tl.store(Checks,tl.sum(bad_id.to(tl.int32),0)==0)
    tl.store(Checks+1,tl.sum(bad_weight.to(tl.int32),0)==0)


def route_checks(ids,weights,checks,experts):
    n=ids.numel()
    _route_checks[(1,)](ids,weights,checks,n,experts,max(1,triton.next_power_of_2(n)),num_warps=4)
