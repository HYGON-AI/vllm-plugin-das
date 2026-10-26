"""Idle GPU check of the real internal runner and shared-activation handoff.

The routed-expert test double overwrites the input on the main stream. The
shared down projection must read its independent router activation instead.
This validates ordering and changed-input graph replay, not model quality.
"""
import json
import os
from contextlib import nullcontext
from types import SimpleNamespace

import torch
import vllm
from vllm.platforms import current_platform
from vllm.plugins import load_general_plugins
from vllm_hcu.patch.worker import apply_worker_patches

load_general_plugins()
apply_worker_patches()
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
from vllm.model_executor.layers.fused_moe.runner.shared_experts import SharedExperts
from vllm_hcu.models.step5_text import BF16FP32RouterLinear, Step5MLP
from vllm.model_executor.layers.activation import SwigluStepAndMul
from vllm_hcu.ops.step5 import router_gemm as rg
from vllm_hcu.model_executor.layers.fused_moe import shared_experts as se

torch.set_num_threads(1)
torch.cuda.set_device(0)
torch.manual_seed(647920)
rg.RSG_MAX_M = 64
se._LATE_STREAM_MIN_TOKENS = 4096
K, NR, I = 4096, 352, 384


class Projection(torch.nn.Module):
    def __init__(self, k, n):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.randn(k, n, device='cuda', dtype=torch.bfloat16) * .02,
            requires_grad=False,
        )

    def forward(self, x):
        return x @ self.weight, None


shared = Step5MLP.__new__(Step5MLP)
torch.nn.Module.__init__(shared)
shared.gate_up_proj = Projection(K, I * 2)
shared.down_proj = Projection(I, K)
shared.limit = 7.
shared.act_fn = lambda x: SwigluStepAndMul.forward_cuda(SimpleNamespace(limit=7.), x)
shared.fused_act = None
shared.enable_fused_router_shared(64)

gate = BF16FP32RouterLinear.__new__(BF16FP32RouterLinear)
torch.nn.Module.__init__(gate)
gate.weight = torch.nn.Parameter(
    torch.randn(K, NR, device='cuda', dtype=torch.bfloat16) * .02,
    requires_grad=False,
)
gate._fused_shared = [shared]
gate._rsg = rg.router_shared_swiglustep
gate._rsg_workspace = rg.make_router_shared_workspace('cuda', NR, I)
gate._gemm = rg.router_gemm_bf16_fp32
rg.init_router_gemm_workspace('cuda', NR)

wrapper = SharedExperts.__new__(SharedExperts)
torch.nn.Module.__init__(wrapper)
wrapper.enable_dbo = False
wrapper._layer = shared
wrapper._stream = torch.cuda.Stream()
wrapper._output = [None, None]
wrapper._output_pending_on_stream = [False, False]
wrapper._output_pending_late = [False, False]
wrapper._output_pending_fused_router = [False, False]
wrapper._moe_config = SimpleNamespace(moe_parallel_config=SimpleNamespace(
    enable_eplb=False, use_fi_nvl_two_sided_kernels=False,
))
wrapper._mk_can_overlap_shared_experts = lambda: False
wrapper._is_multistream_safe = lambda: False
projection_streams = []
project = shared.forward_fused_router_activation


def observe_projection(activation):
    projection_streams.append(torch.cuda.current_stream().cuda_stream)
    return project(activation)


shared.forward_fused_router_activation = observe_projection

runner = MoERunner.__new__(MoERunner)
torch.nn.Module.__init__(runner)
runner.gate = gate
runner._shared_experts = wrapper
runner._fse_fuse_gate = False
runner._ep_fused_combine = False
runner.moe_config = SimpleNamespace(pcp_size=1)


def routed(**kwargs):
    # Mutating hidden_states while the aux stream runs catches an accidental
    # re-read of the input instead of the independent gate activation.
    x = kwargs['x']
    x.fill_(-11)
    return torch.zeros_like(x)


runner.routed_experts = SimpleNamespace(
    _ensure_moe_quant_config_init=lambda: None,
    quant_method=SimpleNamespace(is_monolithic=False, topk_indices_dtype=torch.int32),
    forward_modular=routed,
)
runner.router = SimpleNamespace(select_experts=lambda **kw: (
    torch.ones((kw['hidden_states'].shape[0], 8), device='cuda'),
    torch.zeros((kw['hidden_states'].shape[0], 8), device='cuda', dtype=torch.int32),
))
runner._sequence_parallel_context = lambda: nullcontext()
runner._maybe_dispatch = lambda x, logits: (x, logits)
runner._pcp_routing_inputs = lambda *args: nullcontext(None)
runner._maybe_combine = lambda s, r: s + r


def call(x):
    projection_streams.clear()
    producer = torch.cuda.current_stream().cuda_stream
    output = runner._forward_impl(x, None, x)
    wanted = wrapper._stream.cuda_stream if os.environ['HCU_STEP5_FUSED_ROUTER_STREAM']=='1' and x.shape[0] <= 64 else producer
    assert projection_streams == [wanted], (projection_streams, wanted)
    assert shared.fused_act is None
    assert wrapper._output == [None, None]
    assert not any(wrapper._output_pending_fused_router)
    assert not any(wrapper._output_pending_on_stream)
    return output


results = []
with torch.inference_mode():
    for m in [16, 32, 64, 65]:
        source = torch.randn(m, K, device='cuda', dtype=torch.bfloat16) * .2
        x = source.clone()
        for _ in range(3):
            os.environ['HCU_STEP5_FUSED_ROUTER_STREAM'] = '0'
            x.copy_(source)
            expected = call(x).clone()
            os.environ['HCU_STEP5_FUSED_ROUTER_STREAM'] = '1'
            x.copy_(source)
            actual = call(x)
            torch.cuda.synchronize()
            assert torch.equal(actual, expected), ('eager', m)
        graph = torch.cuda.CUDAGraph()
        x.copy_(source)
        with torch.cuda.graph(graph):
            out = call(x)
        replays = []
        for amplitude in [0., .2, 1.]:
            source.copy_(torch.randn_like(source) * amplitude)
            os.environ['HCU_STEP5_FUSED_ROUTER_STREAM'] = '0'
            x.copy_(source)
            expected = call(x).clone()
            os.environ['HCU_STEP5_FUSED_ROUTER_STREAM'] = '1'
            for _ in range(3):
                out.zero_()
                x.copy_(source)
                graph.replay()
                torch.cuda.synchronize()
                assert torch.equal(out, expected), ('graph', m, amplitude)
            replays.append(dict(amplitude=amplitude, exact=True))
        results.append(dict(rows=m, eager_exact=True, changed_input_replays=replays))

assert se._fused_router_stream_announced
print(json.dumps(dict(passed=True, source=__file__, results=results,
    native_activation_announced=se._fused_router_stream_announced,
    logger_name=se.logger.name, logger_effective_level=se.logger.getEffectiveLevel())))
