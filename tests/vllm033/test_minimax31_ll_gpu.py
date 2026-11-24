"""Independent constant-weight oracle and changed-count HIP graph replay."""
import torch
import pytest
from hcu_vllm_das.moe.minimax31_ll import MaskedExperts


@pytest.mark.skipif(not torch.cuda.is_available(), reason="HCU required")
def test_masked_hipc_live_counts_and_changed_input_graph():
    from deepgemm import pack_w4a8_moe_hipc_weight
    torch.cuda.set_device(0)
    torch.manual_seed(413)
    e, cap, h, i = 16, 64, 6144, 3072
    # Each packed checkpoint nibble is +1, represented in the signed high
    # nibble domain (+16). The channel scales make the logical matrices
    # constant 1/H and 1/I, so an independent sum oracle needs no native GEMM.
    raw13 = torch.full((e,2*i,h//2), 0x11, device="cuda", dtype=torch.int8)
    raw2 = torch.full((e,h,i//2), 0x11, device="cuda", dtype=torch.int8)
    w13, w2 = pack_w4a8_moe_hipc_weight(raw13), pack_w4a8_moe_hipc_weight(raw2)
    s13 = torch.full((e,2*i,1), 1/(16*h), device="cuda", dtype=torch.float32)
    s2 = torch.full((e,h,1), 1/(16*i), device="cuda", dtype=torch.float32)
    q = torch.randint(-7,8,(e,cap,h), device="cuda", dtype=torch.int8)
    scales = torch.full((e,cap,1), 0.1, device="cuda", dtype=torch.float32)
    counts = torch.tensor([0,1,2,3,7,16,31,32,33,63,64,4,8,12,24,48], device="cuda", dtype=torch.int32)
    executor = MaskedExperts(device="cuda", experts=e, hidden=h, intermediate=i, capacity=cap)

    def check(actual):
        first = (q.float().sum(-1,keepdim=True)*scales/h).to(torch.bfloat16).float()
        gate = first.clamp(max=7)
        up = first.clamp(-7,7)
        activated = gate*torch.sigmoid(1.702*gate)*(up+1)
        # Constant features are exactly represented by their token INT8
        # quantization up to the vendor FP32 scale and final BF16 store.
        reference = activated.to(torch.bfloat16).expand(-1,-1,h)
        live = torch.arange(cap,device="cuda")[None,:] < counts[:,None]
        torch.testing.assert_close(actual[live].float(),reference[live].float(),rtol=0.02,atol=2e-4)
        assert torch.isfinite(actual[live]).all()

    weights = w13,w2,s13,s2
    for _ in range(2):
        check(executor((q,scales),counts,weights,32))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = executor((q,scales),counts,weights,32)
    graph.replay(); check(output)
    q.copy_(torch.randint(-11,12,q.shape,device="cuda",dtype=torch.int8))
    counts.copy_(torch.tensor([64,0,1,60,5,2,44,0,16,32,7,63,1,20,0,9],device="cuda",dtype=torch.int32))
    graph.replay(); check(output)
