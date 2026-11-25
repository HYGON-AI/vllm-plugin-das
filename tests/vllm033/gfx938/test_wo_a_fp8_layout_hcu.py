"""Native WO-A output bytes and the actual Channel-FP8 WO-B consumer."""
from types import MethodType, SimpleNamespace as NS

import pytest
import torch

from hcu_vllm_das import wo_a_layout as layout
from hcu_vllm_das.adapter_api import adapter_module


pytestmark = pytest.mark.hcu


@pytest.fixture
def native_config():
    if not torch.cuda.is_available():
        pytest.skip('HCU required')
    from vllm.config import CompilationConfig, VllmConfig, set_current_vllm_config
    from vllm.plugins import load_general_plugins

    load_general_plugins()
    with set_current_vllm_config(VllmConfig(
            compilation_config=CompilationConfig(mode=0, custom_ops=['all']))):
        yield


@pytest.mark.parametrize('rows,heads', [(1, 64), (7, 64), (17, 64), (42, 64),
                                      (48, 64), (64, 64), (129, 64), (163, 64),
                                      (256, 64), (42, 128), (163, 128)])
def test_native_projection_bytes_and_graph(rows, heads, native_config):
    torch.manual_seed(rows + heads)
    assert layout.qualify_wo_a_layout(torch.device('cuda', 0))
    weight = torch.randn(8192, 4096, device='cuda', dtype=torch.bfloat16) / 64
    owner = NS(n_local_heads=64, n_local_groups=8, o_lora_rank=1024,
               rotary_emb=None, rope_head_dim=64, _wo_b_scale=None,
               wo_a=NS(weight=weight), wo_b=lambda x: x)
    owner._wo_b_after_wo_a = MethodType(layout._CONSUMER, owner)
    x = torch.randn(rows, heads, 512, device='cuda', dtype=torch.bfloat16)
    if heads == 128:
        x[:, 64:] = torch.nan  # Padded heads must never enter WO-A.
    with torch.inference_mode():
        assert layout.supported(owner, x)
        actual = layout.project(owner, x)
        expected = layout._O_PROJ(owner, x, None)
        assert actual.is_contiguous()
        assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = layout.project(owner, x)
        for replay in range(3):
            x[:, :64].normal_(std=.25 + replay)
            weight.normal_(std=.02 + replay * .01)
            graph.replay()
            expected = layout._O_PROJ(owner, x, None)
            assert torch.equal(captured.view(torch.uint8), expected.view(torch.uint8))
        # Preserve the existing larger-row fallback rather than changing it.
        assert not layout.supported(owner, torch.empty(257, 64, 512, device='cuda', dtype=torch.bfloat16))


@pytest.mark.parametrize('rows', [7, 48])
def test_actual_fp8_wo_b_consumes_identical_values(rows, native_config):
    from vllm.model_executor.kernels.linear.scaled_mm.ScaledMMLinearKernel import FP8ScaledMMLinearLayerConfig
    from vllm.model_executor.kernels.linear.scaled_mm.triton import TritonFP8ScaledMMLinearKernel
    from vllm.model_executor.layers.quantization.utils.quant_utils import kFp8DynamicTokenSym, kFp8StaticChannelSym

    torch.manual_seed(231023 + rows)
    assert layout.qualify_wo_a_layout(torch.device('cuda', 0))
    config = FP8ScaledMMLinearLayerConfig(
        weight_quant_key=kFp8StaticChannelSym, activation_quant_key=kFp8DynamicTokenSym,
        weight_shape=(5120, 8192), input_dtype=torch.bfloat16, out_dtype=torch.bfloat16)
    kernel = TritonFP8ScaledMMLinearKernel(config, ('weight', 'weight_scale', 'input_scale', 'input_scale_ub'))
    down = torch.nn.Module()
    down.weight = torch.nn.Parameter(torch.randn(5120, 8192, device='cuda').to(torch.float8_e4m3fn).T,
                                     requires_grad=False)
    down.weight_scale = torch.full((5120, 1), .02, device='cuda')
    owner = NS(n_local_heads=64, n_local_groups=8, o_lora_rank=1024,
               rotary_emb=None, rope_head_dim=64, _wo_b_scale=None,
               wo_a=NS(weight=torch.randn(8192, 4096, device='cuda', dtype=torch.bfloat16) / 64),
               wo_b=lambda x: kernel.apply_weights(down, x))
    owner._wo_b_after_wo_a = MethodType(layout._CONSUMER, owner)
    x = torch.randn(rows, 64, 512, device='cuda', dtype=torch.bfloat16)
    with torch.inference_mode():
        def candidate():
            return owner._wo_b_after_wo_a(layout.project(owner, x))
        expected, actual = layout._O_PROJ(owner, x, None), candidate()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = candidate()
        for magnitude in (0., .1, 8.):
            x.normal_().mul_(magnitude)
            graph.replay()
            expected = layout._O_PROJ(owner, x, None)
            torch.testing.assert_close(captured, expected, rtol=0, atol=0)
