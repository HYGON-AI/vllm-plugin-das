"""Exact integer route preparation for the qualified GPU INT8 EP boundary."""
import torch
import triton
import triton.language as tl


@triton.jit
def _route_kernel(ids, mapping, output, n, GLOBAL: tl.constexpr,
                  HAS_MAP: tl.constexpr, BLOCK: tl.constexpr):
    offsets=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    active=offsets<n
    gid=tl.load(ids+offsets,mask=active,other=-1)
    valid=(gid>=0)&(gid<GLOBAL)
    if HAS_MAP:
        safe=tl.minimum(tl.maximum(gid,0),GLOBAL-1)
        mapped=tl.load(mapping+safe,mask=active&valid,other=-1)
    else:
        mapped=gid
    local=tl.where(valid,mapped,-1).to(tl.int32)
    tl.store(output+offsets,local,mask=active)


def prepare(topk_ids,expert_map,global_num_experts,local_num_experts):
    assert topk_ids.is_cuda and topk_ids.is_contiguous()
    assert topk_ids.dtype in(torch.int32,torch.int64)and global_num_experts>0
    if expert_map is None and topk_ids.dtype==torch.int32:
        return topk_ids,global_num_experts
    if expert_map is not None:
        assert expert_map.is_cuda and expert_map.device==topk_ids.device
        assert expert_map.is_contiguous()and expert_map.dtype==torch.int32
        assert expert_map.numel()>=global_num_experts
    output=torch.empty_like(topk_ids,dtype=torch.int32)
    if topk_ids.numel():
        _route_kernel[(triton.cdiv(topk_ids.numel(),256),)](
            topk_ids,expert_map,output,topk_ids.numel(),global_num_experts,
            expert_map is not None,256,num_warps=4)
    return output,global_num_experts if expert_map is None else local_num_experts
