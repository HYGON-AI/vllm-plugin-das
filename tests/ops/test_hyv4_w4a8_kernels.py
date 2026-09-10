"""Numerical coverage: nibble order, routing, gate/up order and clamp."""
import importlib.util
from pathlib import Path
import pytest
import torch

MODULE = Path(__file__).resolve().parents[2] / 'vllm_hcu/model_executor/layers/quantization/hyv4_w4a8_kernels.py'


def runtime():
    assert MODULE.exists(), 'HYV4 aiter W4A8 runtime is missing'
    spec = importlib.util.spec_from_file_location('hyv4_w4a8_kernels', MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pack(values):
    return (((values[..., ::2].to(torch.int16) & 15) << 4)
            | (values[..., 1::2].to(torch.int16) & 15)).to(torch.int8)


def test_signed_nibble_reference():
    values = runtime().unpack_int4(torch.tensor([[-113, 7, -91]], dtype=torch.int8))
    torch.testing.assert_close(values, torch.tensor([[-8, -1, 0, 7, -6, 5]], dtype=torch.int8))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires Hygon GPU')
@pytest.mark.parametrize('m,k,n', [(1,128,64), (35,256,96), (2,6144,512), (1,256,6144)])
def test_linear_matches_quantized_reference(m,k,n):
    torch.manual_seed(12)
    x = torch.randn(m,k,device='cuda',dtype=torch.bfloat16)
    w = torch.randint(-8,8,(n,k),device='cuda',dtype=torch.int8)
    scale = torch.rand(n,1,device='cuda') * .03
    from aiter import per_token_quant_triton
    q, xs = per_token_quant_triton(x,quant_dtype=torch.int8)
    expected = ((q.float() @ w.float().T) * xs * scale.T).to(x.dtype)
    actual = runtime().linear(x,pack(w),scale)
    torch.testing.assert_close(actual,expected,rtol=.008,atol=.002)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires Hygon GPU')
def test_moe_preserves_gate_up_clamp_and_routing():
    torch.manual_seed(22)
    m,k,n,e = 5,128,128,3
    x = torch.randn(m,k,device='cuda',dtype=torch.bfloat16) * 3
    w13 = torch.randint(-8,8,(e,2*n,k),device='cuda',dtype=torch.int8)
    w2 = torch.randint(-8,8,(e,k,n),device='cuda',dtype=torch.int8)
    s13 = torch.full((e,2*n,1),.15,device='cuda')
    s2 = torch.full((e,k,1),.015,device='cuda')
    ids = torch.tensor([[0,1],[1,2],[2,0],[0,2],[1,0]],device='cuda',dtype=torch.int32)
    weights = torch.tensor([[.2,.8]]*m,device='cuda')
    from aiter import per_token_quant_triton
    q,xs = per_token_quant_triton(x,quant_dtype=torch.int8)
    expected = torch.zeros_like(x)
    routes = []
    for j in range(2):
        route = []
        for i in range(m):
            expert = ids[i,j].item()
            pre = ((q[i].float() @ w13[expert].float().T)*xs[i]*s13[expert,:,0]).to(x.dtype).float()
            gate,up = pre.chunk(2)
            bridge = (torch.nn.functional.silu(gate.clamp(max=10))*up.clamp(-10,10)).to(x.dtype)[None,:]
            bq,bs = per_token_quant_triton(bridge,quant_dtype=torch.int8)
            out = ((bq.float() @ w2[expert].float().T)*bs*s2[expert,:,0]*weights[i,j]).to(x.dtype)
            route.append(out)
        routes.append(torch.cat(route))
    expected = (routes[0].float()+routes[1].float()).to(x.dtype)
    actual = runtime().moe(x,pack(w13),pack(w2),s13,s2,weights,ids,gemm1_limit=10.)
    torch.testing.assert_close(actual,expected,rtol=.025,atol=.125)
