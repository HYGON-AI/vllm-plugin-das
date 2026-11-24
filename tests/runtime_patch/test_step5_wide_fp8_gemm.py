"""Launch policy tests without allocating an EP64 activation arena."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def provider():
    path=Path(__file__).resolve().parents[2]/'vllm_hcu/model_executor/layers/fused_moe/step5_wide_fp8_gemm.py'
    spec=importlib.util.spec_from_file_location('step5_wide_tested',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('elements,size,expected', [
    ((1<<31)-1,1,False), (1<<31,1,True),
    ((1<<31)+4096,1,True), ((1<<30),4,True), (16,2,False),
])
def test_signed_element_and_byte_thresholds(provider,elements,size,expected):
    tensor=SimpleNamespace(numel=lambda:elements,element_size=lambda:size)
    assert provider.needs_wide_a(tensor) is expected


def test_large_fp8_input_is_read_above_the_signed_32bit_boundary(provider):
    if not torch.cuda.is_available():pytest.skip('GPU required (about 2.1 GiB)')
    import triton.language as tl
    rows=(1<<31)//4096+1
    a=torch.empty((rows,4096),dtype=torch.float8_e4m3fn,device='cuda')
    a[-1].copy_(torch.ones(4096,dtype=torch.bfloat16,device='cuda').to(a.dtype))
    b=torch.ones((1,64,4096),dtype=torch.bfloat16,device='cuda').to(a.dtype)
    c=torch.empty((rows,64),dtype=torch.bfloat16,device='cuda')
    a_scale=torch.full((rows,1),0.5,dtype=torch.float32,device='cuda')
    b_scale=torch.full((1,64,1),0.25,dtype=torch.float32,device='cuda')
    ids=torch.zeros((rows,1),dtype=torch.int32,device='cuda')
    sorted_ids=torch.full((16,),rows,dtype=torch.int32,device='cuda')
    sorted_ids[0]=rows-1
    expert_ids=torch.zeros((1,),dtype=torch.int32,device='cuda')
    nvalid=torch.tensor([16],dtype=torch.int32,device='cuda')
    weights=torch.ones((rows,1),dtype=torch.float32,device='cuda')
    cfg=dict(BLOCK_SIZE_M=16,BLOCK_SIZE_N=64,BLOCK_SIZE_K=256,
             GROUP_SIZE_M=1,num_warps=4,num_stages=2,USE_MLS_LOAD=False)
    provider.invoke_wide_fp8(a,b,c,a_scale,b_scale,weights,ids,sorted_ids,None,
                             expert_ids,nvalid,1,tl.bfloat16,cfg)
    torch.cuda.synchronize()
    torch.testing.assert_close(c[-1],torch.full_like(c[-1],512),rtol=0,atol=0)
