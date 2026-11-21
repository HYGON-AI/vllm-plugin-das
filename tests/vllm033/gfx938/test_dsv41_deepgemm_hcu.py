"""Single-device gfx938 numerical regression for Channel-FP8 DeepGEMM experts."""
import pytest


@pytest.mark.hcu
def test_signed_experts_match_reference_with_clamp_and_remote_routes():
    from types import SimpleNamespace as NS
    import torch
    from deepgemm import marlin_fp8_contiguous_weight
    from hcu_vllm_das.moe.dsv41_deepgemm import ChannelDeepGemmExperts
    from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation,ApplyMoEActivationConfig
    import vllm._custom_ops

    torch.cuda.set_device(0)
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    assert arch == "gfx938", f"DeepGEMM numerical contract requires gfx938, got {arch}"
    torch.manual_seed(42)
    e,k,i,topk=2,5120,2304,2
    w1=torch.randint(-2,3,(e,2*i,k),device='cuda').to(torch.float8_e4m3fn)
    w2=torch.randint(-2,3,(e,k,i),device='cuda').to(torch.float8_e4m3fn)
    s1=torch.rand(e,2*i,device='cuda')*.02+.01;s2=torch.rand(e,k,device='cuda')*.02+.01
    expert=object.__new__(ChannelDeepGemmExperts)
    expert.moe_config=NS(hidden_dim=k,intermediate_size_per_partition=i,num_local_experts=e)
    expert.quant_config=FusedMoEQuantConfig.make(torch.float8_e4m3fn,w1_scale=s1,w2_scale=s2,per_act_token_quant=True,per_out_ch_quant=True,gemm1_clamp_limit=.5)
    expert.activation_config=ApplyMoEActivationConfig(clamp_limit=.5)
    expert.packed_w1=marlin_fp8_contiguous_weight(w1);expert.packed_w2=marlin_fp8_contiguous_weight(w2)
    results=[]
    for rows in [1,7,257]:
     x=torch.randn(rows,k,device='cuda',dtype=torch.bfloat16)*.2
     ids=torch.tensor([0,2],device='cuda').expand(rows,-1).contiguous();ids[0,0]=1
     weights=torch.tensor([.3,.7],device='cuda').expand(rows,-1).contiguous()
     mapping=torch.tensor([0,-1,1,-1],device='cuda',dtype=torch.int32)
     shapes=expert.workspace_shapes(rows,2*i,k,topk,4,e,None,MoEActivation.SILU)
     ws1=torch.empty(shapes[0],device='cuda',dtype=torch.bfloat16);ws2=torch.empty(shapes[1],device='cuda',dtype=torch.bfloat16)
     out=torch.full_like(x,float('nan'))
     expert.apply(out,x,w1,w2,weights,ids,MoEActivation.SILU,4,mapping,None,None,ws1,ws2,None,False)
     torch.cuda.synchronize()
     scale=x.float().abs().amax(-1,keepdim=True).clamp_min(1e-10)/448
     dq=(x.float()/scale).to(torch.float8_e4m3fn).float()*scale
     expected=torch.zeros_like(x,dtype=torch.float32)
     for j in range(e):
      first=(dq@(w1[j].float()*s1[j,:,None]).T).to(torch.bfloat16).float()
      gate=first[:,:i].clamp(max=.5);up=first[:,i:].clamp(-.5,.5)
      activated=(torch.nn.functional.silu(gate)*up).to(torch.bfloat16).float()
      scale2=activated.abs().amax(-1,keepdim=True).clamp_min(1e-10)/448
      dq2=(activated/scale2).to(torch.float8_e4m3fn).float()*scale2
      second=(dq2@(w2[j].float()*s2[j,:,None]).T).to(torch.bfloat16).float()
      weight=((ids==2*j)*weights).sum(-1)
      expected+=second*weight[:,None]
     rel=float(torch.linalg.vector_norm(out.float()-expected)/torch.linalg.vector_norm(expected))
     finite=bool(torch.isfinite(out).all());print(dict(rows=rows,rel_l2=rel,finite=finite),flush=True)
     assert finite and rel<.04,(rows,rel)
