"""Regression: serial layers share scratch but must consume their own weights."""
import json,torch
from pathlib import Path
from triton_tmp_dir.minimax31_moe_sglang_v1 import PackedExperts,ReceivedExperts
from lightop.quant import per_token_quant_int8

torch.set_num_threads(4);torch.manual_seed(20261115);torch.cuda.set_device(0)
root=Path(__file__).resolve().parent
def weights(seed):
 g=torch.Generator().manual_seed(seed)
 ts=[torch.randint(-128,128,(17,512,256),generator=g,dtype=torch.int8),
     torch.randint(-128,128,(17,512,128),generator=g,dtype=torch.int8),
     torch.rand((17,512,1),generator=g)*.001+.0003,
     torch.rand((17,512,1),generator=g)*.0003+.0001]
 return PackedExperts.from_checkpoint_bytes(*(t.cuda() for t in ts))
w1,w2=weights(101),weights(102)
x=torch.randn(513,512,device='cuda',dtype=torch.bfloat16);q,s=per_token_quant_int8(x)
ids=(torch.arange(513*4,device='cuda').reshape(513,4)%17).long()
rw=torch.full((513,4),.25,device='cuda')
counts=[((int((ids==e).sum())+255)//256)*256 for e in range(17)]
shared=ReceivedExperts(w1);a=shared(q,s,ids,rw,counts).clone();ptrs=[shared._q.data_ptr(),shared._gateup.data_ptr(),shared._down.data_ptr()]
shared.bind_weights(w2);b=shared(q,s,ids,rw,counts).clone()
fresh=ReceivedExperts(w2);expected=fresh(q,s,ids,rw,counts).clone()
assert torch.equal(b,expected) and not torch.equal(a,b)
shared.bind_weights(w1);again=shared(q,s,ids,rw,counts).clone();assert torch.equal(a,again)
assert ptrs==[shared._q.data_ptr(),shared._gateup.data_ptr(),shared._down.data_ptr()]
bad=PackedExperts(w1.w13[:16].contiguous(),w1.w2[:16].contiguous(),w1.s13[:16].contiguous(),w1.s2[:16].contiguous())
try:shared.bind_weights(bad)
except ValueError:pass
else:raise AssertionError('changed geometry accepted')
torch.cuda.synchronize()
r={'status':'pass','different_layers_use_different_weights':True,'layer2_matches_fresh_workspace_bit_exact':True,'rebind_back_bit_exact':True,'same_scratch_pointers':True,'geometry_change_rejected':True}
(root/'rebind_gpu_result.json').write_text(json.dumps(r,indent=2)+'\n');print('REBIND_GPU_PASS',flush=True)
