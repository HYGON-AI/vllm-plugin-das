"""One idle GPU: TP4 router/shared dimensions and changed-input graph replay."""
import json
import pathlib
from types import SimpleNamespace
import torch
import vllm
from vllm.platforms import current_platform
from vllm.plugins import load_general_plugins
from vllm_hcu.patch.worker import apply_worker_patches
load_general_plugins()
apply_worker_patches()
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
from vllm_hcu.models.step5_text import BF16FP32RouterLinear, Step5MLP, SwigluStepAndMul
from vllm_hcu.ops.step5.router_gemm import RSG_MAX_M, init_router_gemm_workspace, router_gemm_bf16_fp32
from contextlib import nullcontext
torch.set_num_threads(1)
torch.cuda.set_device(0)
torch.manual_seed(6479)
(K, NR, I) = (4096, 352, 384)

class Projection(torch.nn.Module):

    def __init__(self, weight=None):
        super().__init__()
        self.weight = weight
        self.calls = 0
        if weight is not None:
            self.output_size_per_partition = weight.shape[1]

    def forward(self, x):
        self.calls += 1
        return (x if self.weight is None else x @ self.weight, None)
shared = Step5MLP.__new__(Step5MLP)
torch.nn.Module.__init__(shared)
shared.fused_act = None
shared.limit = 7.0
shared._fused_router_shared_max_m = None
shared.gate_up_proj = Projection(torch.nn.Parameter(torch.randn(K, 2 * I, device='cuda', dtype=torch.bfloat16) * 0.02, requires_grad=False))
shared.down_proj = Projection()
# Use the production GPU provider; forward_native rounds BF16 intermediates.
shared.act_fn = lambda x: SwigluStepAndMul.forward_cuda(SimpleNamespace(limit=7.0), x)
gate = BF16FP32RouterLinear.__new__(BF16FP32RouterLinear)
torch.nn.Module.__init__(gate)
gate.weight = torch.nn.Parameter(torch.randn(K, NR, device='cuda', dtype=torch.bfloat16) * 0.02, requires_grad=False)
gate._gemm = router_gemm_bf16_fp32
init_router_gemm_workspace(torch.device('cuda', 0), NR)
gate.fuse_shared_expert(shared)
runner = MoERunner.__new__(MoERunner)
torch.nn.Module.__init__(runner)
runner.gate = gate
runner._shared_experts = SimpleNamespace(_layer=shared)
runner._fse_fuse_gate = False
runner.moe_config = SimpleNamespace(pcp_size=1)
runner.routed_experts = SimpleNamespace(_ensure_moe_quant_config_init=lambda : None)
runner._maybe_sync_shared_experts_stream = lambda *args, **kwargs: None
runner._sequence_parallel_context = lambda : nullcontext()
runner._maybe_dispatch = lambda x, logits: (x, logits)
runner._pcp_routing_inputs = lambda *args: nullcontext(None)
runner._apply_quant_method = lambda **kwargs: (shared(kwargs['shared_experts_input']), None)
runner._maybe_combine = lambda shared_output, routed_output: shared_output

def reference(x):
    return shared.act_fn(x @ shared.gate_up_proj.weight)

def error(actual, expected):
    delta = actual.float() - expected.float()
    l2 = (delta.norm() / expected.float().norm().clamp_min(1e-09)).item()
    assert l2 < 0.002, (l2, delta.abs().max().item())
    return dict(relative_l2=l2, max_abs=delta.abs().max().item())
results = []
for m in [16, 32, 64, 65]:
    x = torch.randn(m, K, device='cuda', dtype=torch.bfloat16)
    before = shared.gate_up_proj.calls
    actual = runner._forward_impl(x, None, x)
    row = dict(rows=m, eager=error(actual, reference(x)))
    assert shared.gate_up_proj.calls - before == (0 if m <= RSG_MAX_M else 1)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = runner._forward_impl(x, None, x)
    row['replays'] = []
    for value in [0, 1]:
        x.copy_(torch.randn_like(x) * (0.5 + value))
        out.zero_()
        graph.replay()
        torch.cuda.synchronize()
        row['replays'].append(error(out, reference(x)))
    assert shared.fused_act is None
    results.append(row)
    del graph
print(json.dumps(dict(gpu_numerics_passed=True, changed_input_zeroed_output_graph_replay=True, results=results)))
