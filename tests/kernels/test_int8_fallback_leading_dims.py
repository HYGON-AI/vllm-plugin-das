# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Native INT8 fallback parity for the multidimensional MTP projection contract."""

from types import ModuleType

import pytest
import torch

pytestmark = pytest.mark.hcu


@pytest.fixture
def fallback(monkeypatch, request):
    if not torch.cuda.is_available():
        pytest.skip("HCU is required")
    from compressed_tensors.quantization import QuantizationStrategy
    from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
        compressed_tensors_w8a8_int8 as target,
    )

    from vllm_hcu.patch.worker.op_opt import (
        patch_compressed_tensors_w8a8_int8 as adapter,
    )

    static, symmetric = request.param
    core = target.CompressedTensorsW8A8Int8
    original_apply = getattr(
        core, "_vllm_hcu_original_apply_weights", core.apply_weights
    )
    original_process = getattr(
        core, "_vllm_hcu_original_process_weights", core.process_weights_after_loading
    )
    # Patch an isolated class, not the process-global Core class. Keep the real
    # scheme, quantizer, post-loader and Triton GEMM; no native boundaries mocked.
    local_class = type(
        "CompressedTensorsW8A8Int8",
        (),
        {
            "__init__": core.__init__,
            "process_weights_after_loading": original_process,
            "apply_weights": original_apply,
        },
    )
    module = ModuleType(adapter.TARGET_MODULE)
    module.CompressedTensorsW8A8Int8 = local_class
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "0")
    adapter.apply_to_module(module)
    scheme = local_class(QuantizationStrategy.CHANNEL, static, symmetric)
    torch.manual_seed(19)
    layer = torch.nn.Module()

    def parameter(value):
        return torch.nn.Parameter(value, requires_grad=False)

    layer.weight = parameter(
        torch.randint(-16, 17, (64, 128), device="cuda", dtype=torch.int8)
    )
    layer.weight_scale = parameter(torch.full((64, 1), 0.01, device="cuda"))
    layer.input_scale = (
        parameter(torch.tensor([0.02], device="cuda")) if static else None
    )
    layer.input_zero_point = (
        parameter(torch.tensor([7], device="cuda", dtype=torch.int8))
        if static and not symmetric
        else None
    )
    layer.azp_adj = None
    layer.params_dtype = torch.bfloat16
    layer.logical_widths = [64]
    scheme.process_weights_after_loading(layer)
    assert type(scheme.kernel).__name__ == "TritonInt8ScaledMMLinearKernel"
    return scheme, layer, original_apply


_MODES = [(False, True), (False, False), (True, True), (True, False)]


@pytest.mark.parametrize("fallback", _MODES, indirect=True)
@pytest.mark.parametrize("with_bias", [False, True])
def test_int8_fallback_native_leading_dims(fallback, with_bias):
    scheme, layer, core_apply = fallback
    bias = torch.randn(64, device="cuda", dtype=torch.bfloat16) if with_bias else None
    for shape in ((128,), (6, 128), (2, 3, 128), (2, 1, 3, 128)):
        for strided in (False, True):
            x = torch.randn(
                *shape[:-1],
                shape[-1] * (2 if strided else 1),
                device="cuda",
                dtype=torch.bfloat16,
            )
            if strided:
                x = x[..., ::2]
                assert not x.is_contiguous()
            expected = core_apply(scheme, layer, x.reshape(-1, 128), bias)
            result = scheme.apply_weights(layer, x, bias)
            torch.cuda.synchronize()
            assert result.shape == shape[:-1] + (64,)
            torch.testing.assert_close(result.reshape(-1, 64), expected, rtol=0, atol=0)


@pytest.mark.parametrize("fallback", _MODES, indirect=True)
def test_int8_fallback_native_graph_replay(fallback):
    scheme, layer, core_apply = fallback
    x = torch.randn(2, 3, 128, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(64, device="cuda", dtype=torch.bfloat16)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            scheme.apply_weights(layer, x, bias)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = scheme.apply_weights(layer, x, bias)
    for _ in range(3):
        x.copy_(torch.randn_like(x))
        graph.replay()
        expected = core_apply(scheme, layer, x.reshape(-1, 128), bias)
        torch.cuda.synchronize()
        assert result.shape == (2, 3, 64)
        torch.testing.assert_close(result.reshape(-1, 64), expected, rtol=0, atol=0)
