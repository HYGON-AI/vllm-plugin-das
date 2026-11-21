"""Installed public activation/CT/expert regression, isolated from warm caches.

The existing operator tests exercise ASM arithmetic. This test exercises the
production owner and expert-entry fallback. Complete CT loading/forward/Graph
is covered separately by test_aiter_int8_ct_graph_hcu.py.
"""

import json
import os
import signal
import subprocess
import sys

import pytest

pytestmark = pytest.mark.hcu
TAG = 'HCU_PUBLIC_INT8_RESULT='


def test_installed_public_int8_owner_and_changed_input_graph():
    if os.environ.get('HCU_VLLM033_AITER_INT8_LIVE') != '1':
        pytest.skip('set HCU_VLLM033_AITER_INT8_LIVE=1 for the installed HCU contract')
    env = dict(os.environ, HCU_ADAPTER_VLLM_REQUIRED_CAPABILITIES='hcu-aiter-int8-moe',
               HCU_ADAPTER_VLLM_REQUIRED_EXTENSIONS='das_hcu_aiter_int8',
               PYTHONDONTWRITEBYTECODE='1')
    # A fresh exec is part of the supported startup contract. Other tests may
    # already have populated vendor caches in this pytest interpreter.
    process = subprocess.Popen([sys.executable, '-P', '-B', __file__, '--component'],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        start_new_session=True)
    try:
        stdout, _ = process.communicate(timeout=600)
    finally:
        if process.returncode is None:
            # communicate() leaves a timed-out child unreaped. Its PID/PGID
            # still belong to this Popen; kill the complete owned test session.
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)
    assert process.returncode == 0, stdout
    records = [json.loads(line[len(TAG):]) for line in stdout.splitlines() if line.startswith(TAG)]
    assert len(records) == 1 and records[0]['changed_replays'] == 3, stdout
    assert records[0]['backend'] == 'aiter' and records[0]['expert'] == 'HcuAiterInt8Experts'
    assert records[0]['observed_initial_fallback_calls'] == 1


def _component():
    import torch
    from compressed_tensors.quantization import QuantizationArgs
    from hcu_adapter.bridges.vllm import hcu_runtime_plugin
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEConfig, FusedMoEParallelConfig, RoutingMethodType,
    )
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts_impl
    from vllm.model_executor.layers.fused_moe.oracle.int8 import (
        convert_to_int8_moe_kernel_format, make_int8_moe_kernel,
        make_int8_moe_quant_config,
    )
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w8a8_int8 import (
        CompressedTensorsW8A8Int8MoEMethod,
    )
    from vllm.v1.worker.workspace import init_workspace_manager

    torch.cuda.set_device(0)
    hcu_runtime_plugin()
    config = FusedMoEConfig(num_experts=8, experts_per_token=2, hidden_dim=256,
        intermediate_size=128, num_local_experts=8, num_logical_experts=8,
        activation=MoEActivation.SILU, device='cuda', in_dtype=torch.bfloat16,
        routing_method=RoutingMethodType.Default, moe_backend='aiter',
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel())
    method = CompressedTensorsW8A8Int8MoEMethod(
        QuantizationArgs(num_bits=8, type='int', strategy='channel', symmetric=True),
        QuantizationArgs(num_bits=8, type='int', strategy='token', symmetric=True, dynamic=True), config)
    assert method.int8_backend.value == 'aiter' and not method.supports_pre_processed_weights
    with set_current_vllm_config(VllmConfig()):
        init_workspace_manager(0)
        torch.manual_seed(261110)
        w1 = torch.randint(-16, 17, (8, 256, 256), dtype=torch.int8, device='cuda')
        w2 = torch.randint(-16, 17, (8, 256, 128), dtype=torch.int8, device='cuda')
        a, b = convert_to_int8_moe_kernel_format(method.int8_backend, w1, w2)
        assert a is w1 and b is w2
        s1 = torch.full((8, 256, 1), .003, device='cuda')
        s2 = torch.full((8, 256, 1), .004, device='cuda')
        quant = make_int8_moe_quant_config(method.int8_backend, s1, s2, per_act_token_quant=True)
        kernel = make_int8_moe_kernel(method.int8_backend, quant, config, method.experts_cls)
        expert = kernel.fused_experts
        assert expert.expects_unquantized_inputs
        x = torch.randn((3, 256), dtype=torch.bfloat16, device='cuda') * .1
        weights = torch.full((3, 2), .5, device='cuda')
        ids = torch.tensor([[0, 1], [2, 3], [4, 5]], dtype=torch.int32, device='cuda')
        output = torch.empty_like(x)
        empty = torch.empty(0, device='cuda')

        def execute():
            expert.apply(output, x, w1, w2, weights, ids, MoEActivation.SILU,
                         8, None, None, None, empty, empty, None, False)

        def reference():
            return fused_experts_impl(x, w1, w2, weights, ids, activation='silu',
                use_int8_w8a8=True, per_channel_quant=True, w1_scale=s1, w2_scale=s2,
                global_num_experts=8)

        fallback_calls = 0
        def observe(frame, event, argument):
            nonlocal fallback_calls
            if event == 'call' and frame.f_code is fused_experts_impl.__code__:
                fallback_calls += 1
        with torch.inference_mode():
            sys.setprofile(observe)
            try:
                execute()
            finally:
                sys.setprofile(None)
            assert fallback_calls == 1
            assert torch.equal(output, reference())
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                execute()
            addresses = (x.data_ptr(), output.data_ptr(), weights.data_ptr(), ids.data_ptr())
            for _ in range(3):
                x.copy_(torch.randn_like(x) * .1)
                weights.copy_(torch.rand_like(weights))
                ids.copy_(torch.randint(0, 8, ids.shape, dtype=ids.dtype, device='cuda'))
                graph.replay()
                torch.cuda.synchronize()
                saved = output.clone()
                assert torch.equal(saved, reference())
                assert addresses == (x.data_ptr(), output.data_ptr(), weights.data_ptr(), ids.data_ptr())
        print(TAG + json.dumps(dict(backend=method.int8_backend.value,
              expert=type(expert).__name__, changed_replays=3,
              fallback='observed official whole-call Triton', observed_initial_fallback_calls=fallback_calls)))


if __name__ == '__main__':
    assert sys.argv[1:] == ['--component']
    _component()
