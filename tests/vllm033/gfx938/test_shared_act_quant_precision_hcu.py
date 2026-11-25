# SPDX-License-Identifier: Apache-2.0
"""Real serving-operator checks, including BF16 rounding and FP8 code/scales."""
import pytest
import torch

from hcu_vllm_das.moe.shared_act_quant import silu_clamp_bf16_quant

pytestmark = [pytest.mark.hcu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="HCU required")]


@pytest.mark.parametrize("m", [0, 1, 8, 48, 64, 256])
@pytest.mark.parametrize("magnitude", [0.0, 0.001, 0.25, 1.0, 8.0])
def test_matches_stock_activation_and_quant(m, magnitude):
    if not torch.cuda.is_available():
        pytest.skip("HCU required")
    from vllm import _custom_ops as ops

    torch.cuda.set_device(0)
    torch.manual_seed(42)
    x = (torch.randn(m, 4608, device="cuda") * magnitude).bfloat16()
    if m and magnitude:
        x[:, :6] = torch.tensor([-20, -10, -0.01, 0, 10, 20], device="cuda")
        x[:, 2304:2310] = x[:, :6]
    actual, scale = silu_clamp_bf16_quant(x, 10.0)
    if not m:
        assert actual.shape == (0, 2304) and scale.shape == (0, 1)
        return
    y = torch.empty((m, 2304), dtype=torch.bfloat16, device="cuda")
    torch.ops._C.silu_and_mul_with_clamp(y, x, 10.0, 1.0, 0.0)
    reference, ref_scale = ops.scaled_fp8_quant(y, use_per_token_if_dynamic=True)
    torch.testing.assert_close(actual.view(torch.uint8), reference.view(torch.uint8),
                               rtol=0, atol=0)
    torch.testing.assert_close(scale, ref_scale, rtol=0, atol=0)


def test_replay_reads_new_inputs():
    if not torch.cuda.is_available():
        pytest.skip("HCU required")
    torch.cuda.set_device(0)
    x = torch.randn(48, 4608, device="cuda", dtype=torch.bfloat16)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        silu_clamp_bf16_quant(x, 10.0)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        q, s = silu_clamp_bf16_quant(x, 10.0)
    for replay in range(3):
        x.normal_(std=0.25 + replay)
        graph.replay()
        expected_q, expected_s = stock_activation_quant(x, lightop=False)
        torch.cuda.synchronize()
        torch.testing.assert_close(q.view(torch.uint8), expected_q.view(torch.uint8),
                                   rtol=0, atol=0)
        torch.testing.assert_close(s.view(torch.uint8), expected_s.view(torch.uint8),
                                   rtol=0, atol=0)


@pytest.mark.parametrize('lightop', [False, True])
@pytest.mark.parametrize('n', [120, 2304])
def test_all_finite_bf16_gate_values_preserve_rounding(lightop, n):
    """Every BF16 gate value, crossed with signed zeros and clamp multipliers.

    A constant gate per row makes its rounding visible in the row scale, even
    when a quantized code would otherwise hide an activation discrepancy.
    """
    if not torch.cuda.is_available():
        pytest.skip("HCU required")
    from vllm import _custom_ops as ops

    torch.cuda.set_device(0)
    bits = torch.arange(65536, device="cuda", dtype=torch.int32).to(torch.int16)
    values = bits.view(torch.bfloat16)
    values = values[torch.isfinite(values)]
    x = torch.empty(values.numel(), 2 * n, device="cuda", dtype=torch.bfloat16)
    x[:, :n] = values[:, None]
    multipliers = torch.tensor([-20, -10, -1, -0.0, 0.0, 1 / 512, 1, 10, 20],
                               device="cuda", dtype=torch.bfloat16)
    x[:, n:] = multipliers.repeat((n + 8) // 9)[:n]
    if lightop and n % 16:
        with pytest.raises(ValueError, match='vector-aligned'):
            silu_clamp_bf16_quant(x, 10.0, lightop=True)
        return
    actual, scale = silu_clamp_bf16_quant(x, 10.0, lightop=lightop)
    y = torch.empty((x.shape[0], n), device="cuda", dtype=torch.bfloat16)
    torch.ops._C.silu_and_mul_with_clamp(y, x, 10.0, 1.0, 0.0)
    if lightop:
        from lightop.quant import per_token_quant_fp8

        expected, expected_scale = per_token_quant_fp8(y)
    else:
        expected, expected_scale = ops.scaled_fp8_quant(y, use_per_token_if_dynamic=True)
    torch.testing.assert_close(actual.view(torch.uint8), expected.view(torch.uint8),
                               rtol=0, atol=0)
    torch.testing.assert_close(scale, expected_scale, rtol=0, atol=0)


@pytest.mark.parametrize("m", [1, 7, 35, 42, 48, 257])
def test_fused_activation_is_consumed_by_real_linear(m, native_linear_config):
    if not torch.cuda.is_available():
        pytest.skip("HCU required")
    from types import SimpleNamespace
    from vllm.model_executor.kernels.linear.scaled_mm.ScaledMMLinearKernel import FP8ScaledMMLinearLayerConfig
    from vllm.model_executor.kernels.linear.scaled_mm.triton import TritonFP8ScaledMMLinearKernel
    from vllm.model_executor.layers.activation import SiluAndMulWithClamp
    from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation, get_input_quant_key
    from vllm.model_executor.layers.quantization.utils.quant_utils import kFp8DynamicTokenSym, kFp8StaticChannelSym
    from hcu_vllm_das.moe.shared_act_quant import QuantizedSharedActivation, enable_shared_prequant

    torch.manual_seed(23)
    k, n = 2304, 5120
    config = FP8ScaledMMLinearLayerConfig(
        weight_quant_key=kFp8StaticChannelSym, activation_quant_key=kFp8DynamicTokenSym,
        weight_shape=(n, k), input_dtype=torch.bfloat16, out_dtype=torch.bfloat16)
    kernel = TritonFP8ScaledMMLinearKernel(
        config, ("weight", "weight_scale", "input_scale", "input_scale_ub"))
    down = torch.nn.Module()
    down.weight = torch.nn.Parameter(torch.randn(n, k, device="cuda").to(torch.float8_e4m3fn).T,
                                     requires_grad=False)
    down.weight_scale = torch.full((n, 1), 0.02, device="cuda")
    down.scheme = SimpleNamespace(fp8_linear=kernel)
    x = torch.randn(m, 2 * k, device="cuda", dtype=torch.bfloat16)
    act = SiluAndMulWithClamp(10.0)
    expected = kernel.apply_weights(down, act(x))
    enable_shared_prequant(down)
    assert get_input_quant_key(down) == kFp8DynamicTokenSym
    produced = QuantizedSharedActivation(act, down)(x)
    assert isinstance(produced, QuantizedActivation)
    stock_q, stock_scale = kernel.quant_fp8(act(x))
    torch.testing.assert_close(produced.scale, stock_scale, rtol=0, atol=0)
    torch.testing.assert_close(produced.data.view(torch.uint8), stock_q.view(torch.uint8),
                               rtol=0, atol=0,
                               msg=lambda text: (f"quant={kernel.quant_fp8._forward_method.__name__}, "
                                                 f"aiter={kernel.quant_fp8.use_aiter}: " + text))
    actual = down.scheme.fp8_linear.apply_weights(down, produced)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    wrapper = QuantizedSharedActivation(act, down)
    torch.cuda.synchronize()
    baseline_graph, candidate_graph = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
    with torch.cuda.graph(baseline_graph):
        captured_expected = kernel.apply_weights(down, act(x))
    with torch.cuda.graph(candidate_graph):
        captured_actual = down.scheme.fp8_linear.apply_weights(down, wrapper(x))
    for magnitude in (0., 0.00001, 8.):
        x.normal_().mul_(magnitude)
        baseline_graph.replay()
        candidate_graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(captured_actual, captured_expected, rtol=0, atol=0)


@pytest.fixture
def native_linear_config():
    from vllm.config import CompilationConfig, VllmConfig, set_current_vllm_config
    from vllm.plugins import load_general_plugins

    load_general_plugins()
    with set_current_vllm_config(VllmConfig(
            compilation_config=CompilationConfig(mode=0, custom_ops=["all"]))):
        yield


def test_native_shared_mlp_advertises_quantized_consumer(monkeypatch,
                                                        native_linear_config):
    """Exercise the actual model constructor and quantization scheme together."""
    if not torch.cuda.is_available():
        pytest.skip("HCU required")
    from types import SimpleNamespace
    from vllm.config import get_current_vllm_config
    from vllm.distributed import parallel_state
    from vllm.models.deepseek_v4.amd.model import DeepseekV4MLP
    from vllm.model_executor.layers.fusion.quant_activation import get_input_quant_key
    from vllm.model_executor.layers.quantization.utils.quant_utils import kFp8DynamicTokenSym
    from hcu_vllm_das.adapter_api import adapter_module
    HcuDSV41ChannelFP8Config = adapter_module("dsv41_channel_fp8").HcuDSV41ChannelFP8Config
    from hcu_vllm_das.moe.shared_act_quant import (
        QuantizedSharedActivation, TokenFP8TritonLinearKernel,
        install_shared_activation,
    )

    # Only constructor metadata is needed; the GPU consumer test above checks
    # real weights and outputs. No process group or model-size allocation here.
    monkeypatch.setattr(parallel_state, "_TP",
                        SimpleNamespace(rank_in_group=0, world_size=1))
    monkeypatch.setattr(get_current_vllm_config(), "model_config",
                        SimpleNamespace(dtype=torch.bfloat16))
    config = HcuDSV41ChannelFP8Config.from_config(dict(
        quant_method="fp8", is_per_channel=True, activation_scheme="dynamic"))
    config.packed_modules_mapping = {"gate_up_proj": ["gate_proj", "up_proj"]}
    with torch.device("meta"):
        shared = DeepseekV4MLP(
            5120, 2304, "silu", 10.0, config, reduce_results=False,
            is_sequence_parallel=True, prefix="model.layers.0.ffn.shared_experts")
    install_shared_activation(shared)
    assert isinstance(shared.act_fn, QuantizedSharedActivation)
    assert type(shared.down_proj.scheme.fp8_linear) is TokenFP8TritonLinearKernel
    assert get_input_quant_key(shared.down_proj) == kFp8DynamicTokenSym
    installed = shared.act_fn
    install_shared_activation(shared)
    assert shared.act_fn is installed


@pytest.mark.parametrize("layers", [1, 3, 5])
@pytest.mark.parametrize("dp", [1, 8])
def test_dspark_constructor_binds_native_shared_consumers(
        monkeypatch, native_linear_config, layers, dp):
    """The owned draft constructor must reach every real native MLP consumer.

    Replace the large model allocation, not the MLP, quantization scheme or
    binding. The separate consumer test checks loaded weights and graph output.
    """
    import copy
    from types import SimpleNamespace
    from vllm.config import get_current_vllm_config
    from vllm.distributed import parallel_state
    from vllm.models.deepseek_v4.amd.model import DeepseekV4MLP
    from hcu_vllm_das import dspark_agrs, dspark_backend, dspark_post
    from hcu_vllm_das.adapter_api import adapter_module
    from hcu_vllm_das.models.dsv41_dspark import (
        HcuDSV41DraftWeightAliasModel, HcuDSparkDeepseekV41ForCausalLM,
    )
    from hcu_vllm_das.moe.shared_act_quant import (
        QuantizedSharedActivation, bind_dspark_shared_experts_fp8,
    )

    monkeypatch.setattr(parallel_state, "_TP",
                        SimpleNamespace(rank_in_group=0, world_size=1))
    cfg = get_current_vllm_config()
    monkeypatch.setattr(cfg, "model_config", SimpleNamespace(dtype=torch.bfloat16))
    quant = adapter_module("dsv41_channel_fp8").HcuDSV41ChannelFP8Config.from_config(
        dict(quant_method="fp8", is_per_channel=True, activation_scheme="dynamic"))
    quant.packed_modules_mapping = {"gate_up_proj": ["gate_proj", "up_proj"]}
    monkeypatch.setattr(cfg, "quant_config", quant)
    monkeypatch.setattr(cfg, "speculative_config",
                        SimpleNamespace(target_parallel_config=None))
    monkeypatch.setattr(cfg.parallel_config, "data_parallel_size", dp)
    local = copy.copy(cfg)
    local.parallel_config = copy.copy(cfg.parallel_config)
    local.parallel_config.all2all_backend = "allgather_reducescatter"
    monkeypatch.setattr(dspark_agrs, "draft_config", lambda _: local)
    monkeypatch.setattr(dspark_backend, "native_draft_ll_requested", lambda _: False)
    monkeypatch.setattr(dspark_post, "requested", lambda _: False)
    observed = []

    def allocate(self, *, vllm_config, prefix=""):
        torch.nn.Module.__init__(self)
        observed.append(vllm_config)
        assert get_current_vllm_config() is vllm_config
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList()
        for index in range(layers):
            decoder = torch.nn.Module()
            decoder.ffn = torch.nn.Module()
            with torch.device("meta"):
                decoder.ffn.shared_experts = DeepseekV4MLP(
                    5120, 2304, "silu", 10., quant, reduce_results=False,
                    is_sequence_parallel=True,
                    prefix=f"model.layers.{40 + index}.ffn.shared_experts")
            decoder.ffn.use_mega_moe = False
            decoder.ffn.fuse_heterogeneous_shared_expert = False
            self.model.layers.append(decoder)

    monkeypatch.setattr(HcuDSV41DraftWeightAliasModel, "__init__", allocate)
    draft = HcuDSparkDeepseekV41ForCausalLM(vllm_config=cfg)
    assert observed == [local if dp > 1 else cfg]
    assert get_current_vllm_config() is cfg
    originals = [layer.ffn.shared_experts.act_fn for layer in draft.model.layers]
    assert all(isinstance(act, QuantizedSharedActivation) for act in originals)
    assert bind_dspark_shared_experts_fp8(observed[0], draft.model) == layers
    assert [layer.ffn.shared_experts.act_fn for layer in draft.model.layers] == originals

    # A new unsupported draft remains native rather than silently changing its
    # shared/mega-expert or parallel semantics.
    unsupported = object.__new__(HcuDSparkDeepseekV41ForCausalLM)
    allocate(unsupported, vllm_config=cfg)
    for layer in unsupported.model.layers:
        layer.ffn.use_mega_moe = True
    assert bind_dspark_shared_experts_fp8(cfg, unsupported.model) == 0
    assert all(not isinstance(layer.ffn.shared_experts.act_fn,
                              QuantizedSharedActivation)
               for layer in unsupported.model.layers)


def stock_activation_quant(x, lightop):
    from vllm import _custom_ops as ops

    y = torch.empty((x.shape[0], x.shape[1] // 2), device=x.device, dtype=x.dtype)
    torch.ops._C.silu_and_mul_with_clamp(y, x, 10.0, 1.0, 0.0)
    if lightop:
        from lightop.quant import per_token_quant_fp8

        return per_token_quant_fp8(y)
    return ops.scaled_fp8_quant(y, use_per_token_if_dynamic=True)


@pytest.mark.parametrize('n', [15, 120, 123, 254])
def test_lightop_unaligned_width_keeps_original_activation(n, monkeypatch,
                                                         native_linear_config):
    from types import SimpleNamespace
    from vllm.model_executor.layers.activation import SiluAndMulWithClamp
    from vllm.model_executor.layers.quantization.utils.quant_utils import kFp8DynamicTokenSym
    from hcu_vllm_das.moe import shared_act_quant as fusion

    monkeypatch.setattr(fusion, '_quant_contract', lambda _: ('lightop', True))
    monkeypatch.setattr(fusion, 'get_input_quant_key', lambda _: kFp8DynamicTokenSym)
    down = torch.nn.Module()
    down.scheme = SimpleNamespace(fp8_linear=object())
    original = SiluAndMulWithClamp(10.)
    candidate = fusion.QuantizedSharedActivation(original, down)
    candidate._quantize = lambda *a, **kw: pytest.fail('unaligned LightOp route fused')
    x = torch.randn(7, 2 * n, device='cuda', dtype=torch.bfloat16)
    actual = candidate(x)
    assert isinstance(actual, torch.Tensor)
    torch.testing.assert_close(actual, original(x), rtol=0, atol=0)


@pytest.mark.parametrize('lightop', [False, True])
@pytest.mark.parametrize('m', [0, 1, 7, 42, 48, 64, 257])
@pytest.mark.parametrize('n', [128, 768, 2304, 3072, 8192])
@pytest.mark.parametrize('magnitude', [0.0, 0.00001, 1.0, 8.0])
def test_shapes_and_actual_quantizer_contract(lightop, m, n, magnitude):
    torch.manual_seed(23043 + m + n)
    x = (torch.randn(m, 2 * n, device='cuda') * magnitude).bfloat16()
    before = x.clone()
    q, scale = silu_clamp_bf16_quant(x, 10., lightop=lightop)
    assert q.shape == (m, n) and scale.shape == (m, 1)
    if m:
        expected, expected_scale = stock_activation_quant(x, lightop)
        torch.testing.assert_close(q.view(torch.uint8), expected.view(torch.uint8), rtol=0, atol=0)
        torch.testing.assert_close(scale.view(torch.uint8), expected_scale.view(torch.uint8), rtol=0, atol=0)
    assert torch.equal(x.view(torch.uint8), before.view(torch.uint8))


@pytest.mark.parametrize('lightop', [False, True])
@pytest.mark.parametrize('n', [128, 2304, 3072])
def test_three_graph_replays_match_each_stock_route(lightop, n):
    x = torch.randn(42, 2 * n, device='cuda', dtype=torch.bfloat16)
    silu_clamp_bf16_quant(x, 10., lightop=lightop)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        q, scale = silu_clamp_bf16_quant(x, 10., lightop=lightop)
    for magnitude in (0., 0.00001, 8.):
        x.normal_().mul_(magnitude)
        graph.replay()
        expected, expected_scale = stock_activation_quant(x, lightop)
        torch.cuda.synchronize()
        torch.testing.assert_close(q.view(torch.uint8), expected.view(torch.uint8), rtol=0, atol=0)
        torch.testing.assert_close(scale.view(torch.uint8), expected_scale.view(torch.uint8), rtol=0, atol=0)


@pytest.mark.parametrize('lightop', [False, True])
def test_independent_cpu_activation_and_quantization_oracle(lightop):
    # Mathematical CPU reference independent of either device implementation.
    gate = torch.tensor([-20., -10., -3., -1., -.125, -0., 0., .125, 1., 3., 10., 20.], dtype=torch.bfloat16)
    up = torch.tensor([-20., -10., -1., -.00001, -0., 0., 1 / 512, 1., 10., 20.], dtype=torch.bfloat16)
    gate = gate.repeat_interleave(up.numel())
    up = up.repeat(12)
    # Include zero padding so both native quantizers use their aligned path.
    gate = torch.cat((gate, torch.zeros(8, dtype=gate.dtype)))
    up = torch.cat((up, torch.zeros(8, dtype=up.dtype)))
    x = torch.cat([gate, up])[None].cuda()
    clamped_gate = gate.double().clamp(max=10.)
    clamped_up = up.double().clamp(-10., 10.) + 0.
    activation = (clamped_gate * torch.sigmoid(clamped_gate)).bfloat16()
    oracle = (activation.double() * clamped_up).bfloat16()
    native = torch.empty(1, gate.numel(), device='cuda', dtype=torch.bfloat16)
    torch.ops._C.silu_and_mul_with_clamp(native, x, 10., 1., 0.)
    # One BF16 ULP, measured from adjacent representable numbers.
    spacing = (torch.nextafter(oracle.abs(), torch.full_like(oracle, float('inf'))) - oracle.abs()).float()
    assert torch.all((native.cpu()[0].float() - oracle.float()).abs() <= spacing)
    values = native.cpu().float()
    amax = values.abs().amax(dim=-1, keepdim=True)
    scale = amax / 448.
    if lightop:
        quantized = values * (448. / amax)
    else:
        scale = scale.clamp_min(1. / (448. * 512.))
        quantized = values / scale
    quantized = quantized.clamp(-448., 448.).to(torch.float8_e4m3fn)
    actual, actual_scale = silu_clamp_bf16_quant(x, 10., lightop=lightop)
    torch.testing.assert_close(actual.cpu().view(torch.uint8), quantized.view(torch.uint8), rtol=0, atol=0)
    torch.testing.assert_close(actual_scale.cpu().view(torch.uint8), scale.view(torch.uint8), rtol=0, atol=0)
