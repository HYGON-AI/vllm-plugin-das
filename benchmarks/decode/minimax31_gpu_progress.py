"""Optional eager GPU breadcrumbs readable without waiting on the GPU queue."""
import ctypes
import json
import os
from pathlib import Path
import struct
import torch
import triton
import triton.language as tl
from hcu_vllm_das.parallel.minimax31_hisparse import MappedHostStorage,loaded_hip_runtime


@triton.jit(do_not_specialize=['address','value'])
def _mark(address,value):
    pointer=address.to(tl.pointer_type(tl.int32))
    tl.atomic_xchg(pointer,value,sem='release',scope='sys')


@triton.jit(do_not_specialize=['address','value'])
def _routes(address,value,ids,weights,COUNT:tl.constexpr,EXPERTS:tl.constexpr,BLOCK:tl.constexpr):
    pointer=address.to(tl.pointer_type(tl.int32))
    offset=tl.arange(0,BLOCK);live=offset<COUNT
    expert=tl.load(ids+offset,live,other=-1);weight=tl.load(weights+offset,live,other=0).to(tl.float32)
    bad=live&((expert < -1)|(expert>=EXPERTS)|(weight!=weight)|(tl.abs(weight)==float('inf'))|((expert<0)&(weight!=0)))
    tl.store(pointer+1,tl.sum(bad.to(tl.int32),0))
    tl.store(pointer+2,tl.sum((live&(expert<0)).to(tl.int32),0))
    tl.store(pointer+3,tl.sum((live&(weight==0)).to(tl.int32),0))
    tl.atomic_xchg(pointer,value,sem='release',scope='sys')


class GPUProgress:
    def __init__(self,root,rank):
        self.backing=MappedHostStorage((4096,),layer_bytes=4096,chunk_bytes=4096,zero=True)
        self.host_pointer=self.backing.storage.data_ptr();library=loaded_hip_runtime()
        library.hipHostGetDevicePointer.argtypes=[ctypes.POINTER(ctypes.c_void_p),ctypes.c_void_p,ctypes.c_uint]
        library.hipHostGetDevicePointer.restype=ctypes.c_int
        alias=ctypes.c_void_p()
        status=library.hipHostGetDevicePointer(ctypes.byref(alias),ctypes.c_void_p(self.host_pointer),0)
        if status or not alias.value:raise RuntimeError('cannot map owned GPU progress mailbox')
        self.alias=alias.value;self.forward_id=0;self.moe_index=0
        self.mark(0)
        # Compile the full VERIFY route-check signature before peers can enter
        # the first collective. Its tiny probe uses a different COUNT value.
        ids=torch.zeros(512,device='cuda',dtype=torch.int64)
        weights=torch.zeros(512,device='cuda',dtype=torch.float32)
        _routes[(1,)](self.alias,0,ids,weights,512,128,512,num_warps=4)
        torch.cuda.current_stream().synchronize()
        if self.read()!=(0,0,0,512):raise RuntimeError('GPU mailbox initialization mismatch')
        self.backing.storage.view(torch.int32)[1:4].zero_()
        if torch.distributed.is_initialized():
            from vllm.distributed.parallel_state import get_world_group
            get_world_group().barrier()
        record={'rank':rank,'pid':os.getpid(),'birth':Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],
                'host_pointer':self.host_pointer,'bytes':4096,
                'decode':'forward*1000000 + 10000 + layer*32 + attention_phase; or +20000 + moe_index*16 + phase; 999999=end',
                'attention_phases':['begin','qkv','qk','index1','topk','plan','swap','attention','output'],
                'moe_phases':['routes','dispatch_send','dispatch_receive','gemm1','oai','gemm2','combine_send','combine_receive']}
        root=Path(root);root.mkdir(exist_ok=True,parents=True)
        (root/('rank%d.gpu_progress.json'%rank)).write_text(json.dumps(record,indent=2)+'\n')

    def read(self):return struct.unpack('<4i',ctypes.string_at(self.host_pointer,16))
    def mark(self,value):_mark[(1,)](self.alias,int(value),num_warps=1)
    def start_forward(self):
        self.forward_id+=1;self.moe_index=0
        if self.forward_id>2000:raise RuntimeError('GPU breadcrumb diagnostic limit reached')
        self.mark(self.forward_id*1000000+1)
    def end_forward(self):self.mark(self.forward_id*1000000+999999)
    def attention(self,layer,phase):self.mark(self.forward_id*1000000+10000+layer*32+phase)
    def begin_moe(self,ids,weights,experts):
        index=self.moe_index;self.moe_index+=1
        value=self.forward_id*1000000+20000+index*16
        _routes[(1,)](self.alias,value,ids,weights,ids.numel(),experts,triton.next_power_of_2(max(1,ids.numel())),num_warps=4)
        return index
    def moe(self,index,phase):self.mark(self.forward_id*1000000+20000+index*16+phase)
