"""Exercise official clamp SwiGLU ownership and changing-input HCU Graphs.

Run the installed device tier; no legacy constructor patch or Plugin OOT is
needed. Upstream tests/kernels/core/test_activation.py owns the eager matrix.
"""

from __future__ import annotations

import os

import pytest


pytestmark = pytest.mark.hcu


@pytest.fixture(scope="module")
def official_runtime():
    import torch

    from vllm.config import CompilationConfig, VllmConfig
    from vllm.config.compilation import CompilationMode
    from vllm.model_executor.custom_op import op_registry_oot
    from vllm.model_executor.layers.activation import SiluAndMulWithClamp
    from vllm.platforms import current_platform
    from vllm.plugins import load_general_plugins

    assert os.environ.get("VLLM033_HCU_TEST") == "1"
    assert current_platform.__class__.__module__ == "hcu_adapter.bridges.vllm"
    load_general_plugins()
    assert torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] in {
        "gfx936",
        "gfx938",
    }
    assert SiluAndMulWithClamp.name not in op_registry_oot
    assert SiluAndMulWithClamp.__init__.__module__ == (
        "vllm.model_executor.layers.activation"
    )
    assert not hasattr(SiluAndMulWithClamp, "_vllm_hcu_clamp_swiglu_applied")
    config = VllmConfig(
        compilation_config=CompilationConfig(
            mode=CompilationMode.NONE,
            custom_ops=["none", "+silu_and_mul_with_clamp"],
        )
    )
    return torch, SiluAndMulWithClamp, config


@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
@pytest.mark.parametrize("alpha,beta", [(1.0, 0.0), (1.702, 0.0), (1.0, 1.0)])
@pytest.mark.parametrize("shape", [(0, 256), (7, 1024), (2, 3, 512)])
def test_official_clamp_swiglu_changed_input_graph(
    official_runtime,
    dtype_name,
    alpha,
    beta,
    shape,
):
    torch, operator, config = official_runtime
    from torch.utils._python_dispatch import TorchDispatchMode
    from vllm.config import set_current_vllm_config

    class Observe(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.native_clamp_calls = 0

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            result = func(*args, **(kwargs or {}))
            if func._schema.name == "_C::silu_and_mul_with_clamp":
                self.native_clamp_calls += 1
            return result

    dtype = getattr(torch, dtype_name)
    limit = 3.5
    with set_current_vllm_config(config), torch.inference_mode():
        layer = operator(limit, alpha=alpha, beta=beta, compile_native=False)
        assert type(layer) is operator
        assert layer._forward_method == layer.forward_hip
        generator = torch.Generator(device="cpu").manual_seed(161)
        inputs = [
            (torch.randn(shape, generator=generator) * 8).to(dtype) for _ in range(3)
        ]
        x = inputs[0].cuda()
        pointer = x.data_ptr()
        observer = Observe()
        with observer:
            eager = layer(x)
        assert observer.native_clamp_calls == int(alpha == 1.0 and beta == 0.0)
        assert eager.shape == (*shape[:-1], shape[-1] // 2)
        if not x.numel():
            assert eager.numel() == 0
            return

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                layer(x)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = layer(x)
        previous = None
        for cpu_input in inputs:
            x.copy_(cpu_input)
            before = x.clone()
            graph.replay()
            torch.cuda.synchronize()
            expected_eager = layer(x)
            torch.testing.assert_close(captured, expected_eager, rtol=0, atol=0)
            torch.testing.assert_close(x, before, rtol=0, atol=0)
            assert x.data_ptr() == pointer
            # Independent FP64 math from the actual low-precision inputs;
            # default native dispatch and nondefault Torch rounding differ.
            gate, up = cpu_input.double().chunk(2, dim=-1)
            gate = gate.clamp(max=limit)
            up = up.clamp(min=-limit, max=limit)
            expected = gate / (1 + torch.exp(-alpha * gate)) * (up + beta)
            tolerance = 2e-3 if dtype_name == "float16" else 2e-2
            torch.testing.assert_close(
                captured.cpu().double(),
                expected,
                rtol=tolerance,
                atol=tolerance,
            )
            if previous is not None:
                assert not torch.equal(captured, previous)
            previous = captured.clone()
