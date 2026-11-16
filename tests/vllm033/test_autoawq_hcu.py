# SPDX-License-Identifier: Apache-2.0
"""Opt-in real CUDA dispatcher/compile/replay contracts, without a model."""

import os

import pytest
import torch

pytestmark = pytest.mark.hcu


@pytest.fixture(scope="module", autouse=True)
def require_explicit_device_run():
    if os.environ.get("HCU_VLLM033_AUTOAWQ_LIVE") != "1":
        pytest.skip("set HCU_VLLM033_AUTOAWQ_LIVE=1 for the actual HCU component")
    assert torch.cuda.is_available(), "Requested AutoAWQ device run has no device"


@pytest.mark.parametrize("k,n", [(1024, 8192), (4096, 8192)])
@pytest.mark.parametrize("m", [1, 17, 65, 128])
def test_official_loader_custom_op_compile_and_changed_input_graph(
    k, n, m, monkeypatch
):
    import vllm.model_executor.parameter as parameter
    from hcu_vllm_das import autoawq_ops  # noqa: F401
    from hcu_vllm_das.autoawq_linear import LightOpAutoAWQLinearMethod
    from vllm.model_executor.layers.quantization.auto_awq import (
        AutoAWQConfig,
        AutoAWQLinearMethod,
    )
    from vllm.model_executor.layers.quantization.awq_triton import (
        awq_dequantize_triton,
        awq_gemm_triton,
    )

    monkeypatch.setattr(parameter, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_world_size", lambda: 1)
    torch.manual_seed(7)
    config = AutoAWQConfig(4, 128, True, False)
    method = LightOpAutoAWQLinearMethod(AutoAWQLinearMethod(config), config)
    with torch.device("cuda"):
        layer = torch.nn.Module()
        method.create_weights(
            layer,
            k,
            [n],
            k,
            n,
            torch.float16,
            weight_loader=lambda p, w: p.data.copy_(w),
        )
    # Original packed checkpoint values are used by both reference paths.
    qw = torch.randint(
        -(2**31), 2**31 - 1, layer.qweight.shape, device="cuda", dtype=torch.int32
    )
    qz = torch.randint(
        -(2**31), 2**31 - 1, layer.qzeros.shape, device="cuda", dtype=torch.int32
    )
    scales = torch.empty_like(layer.scales).uniform_(0.0005, 0.002)
    layer.qweight.data.copy_(qw)
    layer.qzeros.data.copy_(qz)
    layer.scales.data.copy_(scales)
    method.process_weights_after_loading(layer)
    assert method._lightop_active, "Fallback cannot qualify the LightOp path"
    dequant = awq_dequantize_triton(qw, scales, qz)
    bias = torch.randn(n, device="cuda", dtype=torch.float16) * 0.001

    def call(x):
        return method.apply(layer, x, bias)

    with torch.inference_mode():
        x = torch.randn((m, k), device="cuda", dtype=torch.float16) * 0.1
        compiled = torch.compile(call, fullgraph=True, dynamic=True)
        expected = call(x)
        actual = compiled(x)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                compiled(x)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = compiled(x)
        address = x.data_ptr()
        previous = None
        for scale in (0.05, 0.1, 0.15):
            x.copy_(torch.randn_like(x) * scale)
            expected = call(x)
            official = awq_gemm_triton(x, qw, scales, qz, 8) + bias
            reference = x @ dequant + bias
            graph.replay()
            torch.cuda.synchronize()
            assert x.data_ptr() == address
            torch.testing.assert_close(output, expected, atol=0, rtol=0)
            assert float((output - official).abs().max()) <= 0.015625
            assert float((output - reference).abs().max()) <= 0.015625
            assert torch.isfinite(output).all()
            if previous is not None:
                assert not torch.equal(output, previous)
            previous = output.clone()
        del graph, compiled, output
    torch._dynamo.reset()
