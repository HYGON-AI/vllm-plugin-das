"""Real public CT loader, postprocessing and complete forward/Graph contract.

A fresh process activates the real Adapter owner and registered INT8 expert.
These are single-GPU TP1 component tests; model/router/padded-serving/reload
acceptance and performance require their own gates.
"""
import json
import os
import signal
import subprocess
import sys

import pytest

pytestmark = pytest.mark.hcu
TAG = 'HCU_CT_INT8_FORWARD='


@pytest.mark.parametrize('dtype', ['bfloat16', 'float16'])
@pytest.mark.parametrize('tokens', [1, 16, 128])
def test_public_ct_raw_loading_full_forward_and_changed_graph(dtype, tokens):
    _run_component(dtype, tokens, False)


def test_public_ct_small_config_miss_full_forward_and_changed_graph():
    _run_component('bfloat16', 3, True)


def _run_component(dtype, tokens, small):
    if os.environ.get('HCU_VLLM033_AITER_INT8_LIVE') != '1':
        pytest.skip('set HCU_VLLM033_AITER_INT8_LIVE=1 for public CT forward')
    env = dict(os.environ, HCU_ADAPTER_VLLM_REQUIRED_CAPABILITIES='hcu-aiter-int8-moe',
        HCU_ADAPTER_VLLM_REQUIRED_EXTENSIONS='das_hcu_aiter_int8')
    command = [sys.executable, '-P', '-B', __file__, '--component',
        '--dtype', dtype, '--tokens', str(tokens)] + (['--small'] if small else [])
    process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, start_new_session=True)
    try:
        stdout, _ = process.communicate(timeout=600)
    finally:
        if process.returncode is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)
    assert process.returncode == 0, stdout
    rows = [json.loads(line[len(TAG):]) for line in stdout.splitlines() if line.startswith(TAG)]
    assert len(rows) == 1 and rows[0]['graph_replays'] == 6, stdout
    assert rows[0]['raw_generations'] == 2
    assert rows[0]['route'] == ('official-triton' if small else 'asm')


def _metrics(actual, reference):
    import torch
    assert actual.shape == reference.shape and actual.dtype == reference.dtype
    assert torch.isfinite(actual).all() and torch.isfinite(reference).all()
    a, b = actual.float(), reference.float()
    assert torch.count_nonzero(b), 'nondegenerate reference must not be all zero'
    delta = a - b
    rms = b.square().mean().sqrt().item()
    nmae = (delta.abs().mean() / b.abs().mean()).item()
    nrmse = (delta.square().mean().sqrt() / rms).item()
    error = delta.abs().max().item()
    assert rms >= 1e-3, rms
    assert nmae <= .03 and nrmse <= .03 and error <= .001, (rms, nmae, nrmse, error)
    return dict(reference_rms=rms, nmae=nmae, nrmse=nrmse, max_abs_error=error)


def _layer(config, dtype, backend, experts, hidden, width, topk):
    import copy
    import torch
    from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig, FusedMoEParallelConfig, RoutingMethodType, MoEActivation
    from vllm.model_executor.layers.fused_moe.expert_map_manager import ExpertMapManager
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import CompressedTensorsConfig
    parallel = FusedMoEParallelConfig.make_no_parallel()
    moe = FusedMoEConfig(num_experts=experts, num_local_experts=experts,
        num_logical_experts=experts, experts_per_token=topk, hidden_dim=hidden,
        intermediate_size=width, activation=MoEActivation.SILU, device='cuda',
        routing_method=RoutingMethodType.Default, moe_parallel_config=parallel,
        in_dtype=dtype, moe_backend=backend)
    maps = ExpertMapManager(hidden, topk, experts, 0, None, parallel, 'linear', False)
    quant = CompressedTensorsConfig.from_config(copy.deepcopy(config))
    with torch.device('cuda'):
        layer = RoutedExperts('model.layers.0.mlp.experts', dtype, moe, quant, maps)
    assert type(layer.quant_method).__name__ == 'CompressedTensorsW8A8Int8MoEMethod'
    return layer


def _load(layer, raw):
    w13, w2, s1, s2 = raw
    width = w13.shape[1] // 2
    def weights():
        for expert in range(w13.shape[0]):
            yield f'{expert}.gate_proj.weight', w13[expert, :width]
            yield f'{expert}.up_proj.weight', w13[expert, width:]
            yield f'{expert}.down_proj.weight', w2[expert]
            yield f'{expert}.gate_proj.weight_scale', s1[expert, :width]
            yield f'{expert}.up_proj.weight_scale', s1[expert, width:]
            yield f'{expert}.down_proj.weight_scale', s2[expert]
    assert len(list(layer.load_weights(weights()))) == 6 * w13.shape[0]
    layer.quant_method.process_weights_after_loading(layer)
    import torch
    for name, source in zip(['w13_weight', 'w2_weight', 'w13_weight_scale', 'w2_weight_scale'], raw):
        assert torch.equal(getattr(layer, name).cpu(), source)


def _component(dtype_name, tokens, small):
    import ctypes
    import hashlib
    import importlib.metadata
    from pathlib import Path
    import torch
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    props = torch.cuda.get_device_properties(0)
    architecture = props.gcnArchName.split(':')[0]
    assert architecture in {'gfx936', 'gfx938'}
    budget = int(os.environ.get('HCU_INT8_TEST_ALLOCATOR_MIB', '6144'))
    torch.cuda.set_per_process_memory_fraction(budget * 1024**2 / props.total_memory)
    hip = ctypes.CDLL('/opt/dtk/hip/lib/libamdhip64.so')
    pci = ctypes.create_string_buffer(64)
    assert hip.hipDeviceGetPCIBusId(pci, 64, 0) == 0
    if os.environ.get('HCU_AITER_EXPECTED_PCI'):
        assert pci.value.decode() == os.environ['HCU_AITER_EXPECTED_PCI']
    from hcu_adapter.bridges.vllm import hcu_runtime_plugin, extension_reports
    hcu_runtime_plugin()
    from hcu_adapter.extensions import ExtensionStatus
    reports = extension_reports()
    assert any('hcu-aiter-int8-moe' in report.capabilities and report.status is ExtensionStatus.ACTIVE
        for report in reports)
    from hcu_vllm_das import int8_activation
    from hcu_vllm_das.moe import int8_asm
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.config.kernel import KernelConfig
    from vllm.v1.worker.workspace import init_workspace_manager
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts_impl
    from vllm.platforms import current_platform
    import vllm
    import aiter.moe as vendor
    from aiter.jit.utils.chip_info import get_gfx
    init_workspace_manager(torch.device('cuda:0'))
    config = VllmConfig(kernel_config=KernelConfig(moe_backend='aiter'))
    dtype = getattr(torch, dtype_name)
    experts, hidden, width, topk = (8, 256, 128, 2) if small else (256, 2048, 512, 8)
    scheme = dict(quant_method='compressed-tensors', format='int-quantized', ignore=[],
        config_groups={'group_0':dict(targets=['Linear'],
            weights=dict(num_bits=8, type='int', strategy='channel', symmetric=True),
            input_activations=dict(num_bits=8, type='int', strategy='token', symmetric=True, dynamic=True))})
    states, plans = [], []
    counts = dict(asm=0, official_triton=0)
    def observe(frame, event, result):
        if event == 'call' and frame.f_code is int8_asm.execute.__code__: counts['asm'] += 1
        if event == 'call' and frame.f_code is fused_experts_impl.__code__: counts['official_triton'] += 1
        if (event == 'return' and frame.f_code is vendor.get_aiter_moe_config.__code__
                and isinstance(result, tuple) and len(result) == 2):
            found, selected = result
            plans.append(dict(found=found, solution=str(getattr(selected,'solution_type',None)), config=getattr(selected,'config',None)))
    with set_current_vllm_config(config), torch.inference_mode():
        layer = _layer(scheme, dtype, 'aiter', experts, hidden, width, topk)
        twin = _layer(scheme, dtype, 'triton', experts, hidden, width, topk)
        assert layer.quant_method.int8_backend.value == 'aiter'
        x = torch.empty((tokens, hidden), device='cuda', dtype=dtype)
        weights = torch.empty((tokens, topk), device='cuda', dtype=torch.float32)
        ids = torch.empty((tokens, topk), device='cuda', dtype=torch.int32)
        def change(generation, state):
            rng = torch.Generator().manual_seed(8303 + generation * 10 + state)
            x.copy_(torch.randn((tokens, hidden), generator=rng, dtype=dtype) * (.5 if small else .1))
            weights.copy_(torch.softmax(torch.randn((tokens, topk), generator=rng), -1))
            ids.copy_((torch.arange(tokens)[:, None] + torch.arange(topk)[None, :] + state) % experts)
            if state == 2: weights[0].zero_()
        def forward(): return layer.forward_modular(x, weights, ids)
        graph = None
        pointers = None
        for generation in (0, 1):
            rng = torch.Generator().manual_seed(20261029 + generation)
            raw = (torch.randint(-16, 17, (experts, 2 * width, hidden), dtype=torch.int8, generator=rng),
                torch.randint(-16, 17, (experts, hidden, width), dtype=torch.int8, generator=rng),
                torch.rand((experts, 2 * width, 1), generator=rng) * .006 + .0015,
                torch.rand((experts, hidden, 1), generator=rng) * .006 + .0015)
            for target in (layer, twin): _load(target, raw)
            del target, raw
            current = {name:p.data_ptr() for name,p in layer.named_parameters()}
            if pointers is None: pointers = current
            else: assert current == pointers
            assert type(layer.quant_method.moe_kernel.fused_experts).__name__ == 'HcuAiterInt8Experts'
            change(generation, 0)
            if graph is None:
                sys.setprofile(observe)
                try: initial = forward()
                finally: sys.setprofile(None)
                _metrics(initial, twin.forward_modular(x, weights, ids))
                del initial
                for _ in range(2): forward()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph): output = forward()
            for state in (0, 1, 2):
                change(generation, state)
                eager = forward().clone()
                reference = twin.forward_modular(x, weights, ids).clone()
                graph.replay(); torch.cuda.synchronize()
                saved = output.clone()
                if state == 2 and tokens == 1:
                    assert not torch.count_nonzero(reference)
                    assert not torch.count_nonzero(eager)
                    assert not torch.count_nonzero(saved)
                    states.append(dict(generation=generation,state=state,
                        deliberate_zero_route=True))
                else:
                    states.append(dict(generation=generation,state=state,
                        official_triton=_metrics(saved,reference),eager=_metrics(saved,eager)))
                if state == 2: assert not torch.count_nonzero(saved[0])
                del eager, reference, saved
    route = 'official-triton' if small else 'asm'
    assert counts['asm'] == (0 if small else 1) and counts['official_triton'] == (1 if small else 0), counts
    report = dict(architecture=architecture,provider_architecture=get_gfx(),pci=pci.value.decode(),
        core=importlib.metadata.version('vllm'),core_origin=vllm.__file__,
        platform=type(current_platform).__module__+'.'+type(current_platform).__name__,
        actual_extension_reports=[dict(status=report.status.value,capabilities=sorted(report.capabilities)) for report in reports],
        plugin_origin=int8_activation.__file__,plugin_sha256=hashlib.sha256(Path(int8_activation.__file__).read_bytes()).hexdigest(),
        dtype=dtype_name,shape=[tokens,experts,hidden,width,topk],raw_generations=2,
        graph_replays=6,route=route,observed_initial_eager_calls=counts,plans=plans,states=states,
        factor=int8_activation.compile_factor(),padding_policy=os.environ.get('VLLM_MOE_SKIP_PADDING'),
        scope='Actual public host activation, CT raw loader/postprocess, complete official kernel forward and6changed Graph replays including2raw loader generations. No physicalTP/router/padded-serving/public engine reload/model/performance/clean claim.')
    print(TAG + json.dumps(report,default=str),flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--component', action='store_true', required=True)
    parser.add_argument('--dtype', choices=['bfloat16','float16'], required=True)
    parser.add_argument('--tokens', type=int, required=True)
    parser.add_argument('--small', action='store_true')
    args = parser.parse_args()
    _component(args.dtype,args.tokens,args.small)
