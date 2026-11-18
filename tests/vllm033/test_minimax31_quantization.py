"""CPU contracts; run in a fresh HOSTnone, hidden-device interpreter."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from hcu_vllm_das.quantization.minimax31 import MiniMax31SlimQuantConfig, _IGNORES


def config():
    return dict(quant_method="slimquant_w4a8", activation_scheme="dynamic",
                ignore=sorted(_IGNORES), mixed_precision={
                    "experts": dict(weight_bits=4, format="pack-int4-nibble", scope="per_channel",
                                    modules=["language_model.model.layers.3.block_sparse_moe.experts.0.w1"]),
                    "w8a8_include": dict(weight_bits=8, activation_bits=8, format="int8-per-channel",
                                         scope="per_channel", activation_scheme="dynamic_per_token",
                                         modules=[f"language_model.model.layers.0.self_attn.{name}"
                                                  for name in ("q_proj", "k_proj", "v_proj", "index_q_proj", "index_k_proj")])})


def test_precision_contract_and_identity():
    raw = config()
    raw["hcu_weight_cache_identity"] = {"topology": "tp8", "layers": 60}
    actual = MiniMax31SlimQuantConfig.from_config(raw)
    assert actual.get_name() == "slimquant_w4a8"
    assert actual.w8a8_include == raw["mixed_precision"]["w8a8_include"]["modules"]
    assert actual.hcu_weight_cache_identity == raw["hcu_weight_cache_identity"]
    raw["hcu_weight_cache_identity"]["layers"] = 1
    assert actual.hcu_weight_cache_identity["layers"] == 60


@pytest.mark.parametrize("mutation", [
    lambda c: c["ignore"].pop(),
    lambda c: c["mixed_precision"]["experts"].update(format="awq"),
    lambda c: c["mixed_precision"]["w8a8_include"].update(activation_scheme="static"),
    lambda c: c["mixed_precision"]["w8a8_include"]["modules"].append("unsupported.mlp"),
    lambda c: c["mixed_precision"]["experts"]["modules"].append(c["mixed_precision"]["experts"]["modules"][0]),
])
def test_reject_precision_drift(mutation):
    raw = config()
    mutation(raw)
    with pytest.raises(ValueError):
        MiniMax31SlimQuantConfig.from_config(raw)


def test_official_qkv_and_index_weight_loader(monkeypatch):
    """All five projections preserve INT8 bytes and FP32 channel scales."""
    from vllm.model_executor.layers.linear import MinimaxM3QKVParallelLinearWithIndexer
    import vllm.model_executor.parameter as parameter
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_rank", lambda: 5)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_world_size", lambda: 8)
    from hcu_vllm_das.quantization.slimquant_load_only import SlimQuantW8A8LoadOnlyLinear
    # Real vLLM loader on a TP8 rank, without initializing a process group.
    layer = MinimaxM3QKVParallelLinearWithIndexer.__new__(MinimaxM3QKVParallelLinearWithIndexer)
    torch.nn.Module.__init__(layer)
    layer.head_size = layer.index_head_size = 2
    layer.num_heads = 8
    layer.num_kv_heads = layer.num_index_heads = 1
    layer.num_kv_head_replicas = 2
    layer.tp_rank = 5
    layer.tp_size = 8
    layer.replicate_index_q = False
    SlimQuantW8A8LoadOnlyLinear().create_weights(layer, 4, [16, 2, 2, 2, 2], 4, 24, torch.bfloat16,
                                                weight_loader=layer.weight_loader_v2)
    offset = 0
    for name, rows, rank in (("q", 128, 5), ("k", 8, 2), ("v", 8, 2), ("index_q", 8, 2), ("index_k", 2, 0)):
        weight = (torch.arange(rows * 4).reshape(rows, 4) % 127).to(torch.int8)
        scale = torch.arange(rows, dtype=torch.float32).reshape(rows, 1) + 0.25
        n = 16 if name == "q" else 2
        layer.weight_loader_v2(layer.weight, weight, name)
        layer.weight_loader_v2(layer.weight_scale, scale, name)
        assert torch.equal(layer.weight[offset:offset+n], weight[rank*n:(rank+1)*n])
        assert torch.equal(layer.weight_scale[offset:offset+n], scale[rank*n:(rank+1)*n])
        offset += n


def test_registration_is_owned_and_drift_is_fatal(monkeypatch):
    from vllm.model_executor.layers import quantization as registry
    from hcu_adapter.extensions import ExtensionCommitError
    from hcu_vllm_das.quantization import minimax31_activation as activation
    monkeypatch.setattr(registry, "QUANTIZATION_METHODS", list(registry.QUANTIZATION_METHODS))
    monkeypatch.setattr(registry, "_CUSTOMIZED_METHOD_TO_QUANT_CONFIG", dict(registry._CUSTOMIZED_METHOD_TO_QUANT_CONFIG))
    monkeypatch.setattr(activation, "_publication", None)
    monkeypatch.setattr(activation, "_failure", None)
    context = SimpleNamespace(vllm_version="0.33.0", requested_extensions={activation.ENTRY}, requested_capabilities=set())
    activation.activate(context)
    activation.check_health()
    activation.activate(context)
    assert registry._CUSTOMIZED_METHOD_TO_QUANT_CONFIG["slimquant_w4a8"] is MiniMax31SlimQuantConfig
    registry._CUSTOMIZED_METHOD_TO_QUANT_CONFIG["slimquant_w4a8"] = object()
    with pytest.raises(ExtensionCommitError):
        activation.check_health()
    registry._CUSTOMIZED_METHOD_TO_QUANT_CONFIG["slimquant_w4a8"] = MiniMax31SlimQuantConfig
    with pytest.raises(ExtensionCommitError):
        activation.check_health()


def test_w8_runtime_cache_restore_skips_second_transpose(monkeypatch):
    from compressed_tensors.quantization import QuantizationStrategy
    from vllm.model_executor.layers.quantization.compressed_tensors.schemes import CompressedTensorsW8A8Int8
    from vllm.model_executor.kernels.linear.scaled_mm.triton import TritonInt8ScaledMMLinearKernel
    from vllm.model_executor.utils import weights_already_processed
    from hcu_vllm_das.quantization.minimax31 import MiniMax31W8A8LinearMethod
    from hcu_vllm_das.quantization.minimax31_w8a8 import MiniMax31HipBlasLtInt8Kernel
    # Only layout/reload is under test. Admit these inert constructor objects
    # explicitly; this does not exercise or qualify native device support.
    for cls in (TritonInt8ScaledMMLinearKernel, MiniMax31HipBlasLtInt8Kernel):
        monkeypatch.setattr(cls, 'is_supported', classmethod(
            lambda cls, compute_capability=None: (True, None)))
    layer = torch.nn.Module()
    layer.input_size_per_partition = 4
    layer.logical_widths = [3, 3]
    layer.register_parameter("weight", torch.nn.Parameter(torch.arange(24, dtype=torch.int8).reshape(6,4), requires_grad=False))
    layer.register_parameter("weight_scale", torch.nn.Parameter(torch.ones(6,1), requires_grad=False))
    for name in ("input_scale", "input_zero_point", "azp_adj"):
        layer.register_parameter(name, None)
    scheme = CompressedTensorsW8A8Int8(QuantizationStrategy.CHANNEL, False, True)
    kernel = TritonInt8ScaledMMLinearKernel.__new__(TritonInt8ScaledMMLinearKernel)
    kernel.config = SimpleNamespace(is_channelwise=True, is_static_input_scheme=False, input_symmetric=True)
    kernel.layer_param_names = ("weight", "weight_scale", "input_scale", "input_zero_point", "azp_adj")
    scheme.kernel = kernel
    layer.scheme = scheme
    method = MiniMax31W8A8LinearMethod(MiniMax31SlimQuantConfig.from_config(config()))
    method.process_weights_after_loading(layer)
    expected = torch.arange(24,dtype=torch.int8).reshape(6,4).t()
    assert torch.equal(layer.weight, expected)
    pointer = layer.weight.data_ptr()
    with weights_already_processed():
        method.process_weights_after_loading(layer)
    assert layer.weight.data_ptr() == pointer
    assert torch.equal(layer.weight, expected)
    assert layer.weight.stride() == (1,4)


def test_w4_runtime_cache_restore_rebuilds_metadata_without_mutation():
    from vllm.model_executor.utils import weights_already_processed
    from hcu_vllm_das.quantization.minimax31 import MiniMax31W4A8MoEMethod
    moe = SimpleNamespace(hidden_dim=8, intermediate_size_per_partition=4,
                          activation=SimpleNamespace(value="swigluoai_uninterleave"))
    # This tiny CPU fixture specifically checks retained Bolt storage; optimized
    # HIPC defaults are checked separately with real supported dimensions.
    cfg = config()
    cfg["hcu_minimax_routed_backend"] = "bolt"
    method = MiniMax31W4A8MoEMethod(MiniMax31SlimQuantConfig.from_config(cfg), moe)
    layer = torch.nn.Module()
    layer.swiglu_limit = 7.0
    method.create_weights(layer, 2, 8, 4, torch.bfloat16, weight_loader=lambda *a, **k: None)
    for parameter in layer.parameters():
        parameter.data.fill_(1)
    method.process_weights_after_loading(layer)
    snapshot = {name:(value.data_ptr(),value.clone()) for name,value in layer.named_parameters()}
    del layer._minimax31_expert_layout
    with weights_already_processed():
        method.process_weights_after_loading(layer)
    assert layer._minimax31_expert_layout == "signed-high-nibble-first-interleaved-v1"
    for name,value in layer.named_parameters():
        pointer, expected = snapshot[name]
        assert value.data_ptr() == pointer
        assert torch.equal(value, expected)


def test_restore_plain_router_bias_alias():
    from types import SimpleNamespace
    from hcu_vllm_das.quantization.minimax31 import restore_minimax31_router_aliases
    MoE = type("MiniMaxM3MoE", (torch.nn.Module,), {})
    module = MoE()
    module.e_score_correction_bias = torch.nn.Parameter(torch.arange(128, dtype=torch.float32), requires_grad=False)
    stale = torch.empty(128, device="meta")
    module.gate = SimpleNamespace(weight=torch.empty(128, 4))
    module.experts = SimpleNamespace(router=SimpleNamespace(e_score_correction_bias=stale))
    assert restore_minimax31_router_aliases(module) == 1
    assert module.experts.router.e_score_correction_bias is module.e_score_correction_bias
    assert module.e_score_correction_bias.shape == (128,)
    assert restore_minimax31_router_aliases(module) == 0
