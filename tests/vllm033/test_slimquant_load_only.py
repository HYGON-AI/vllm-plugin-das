# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
import pytest
import torch
from torch import nn
from hcu_vllm_das.quantization.slimquant_load_only import SlimQuantW8A8LoadOnlyLinear, SlimQuantW4A8LoadOnlyExperts


def loader(*args, **kwargs):
    pass


def test_linear_parameter_metadata_and_guard(monkeypatch):
    import vllm.model_executor.parameter as parameter
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_world_size", lambda: 1)
    layer=nn.Module(); method=SlimQuantW8A8LoadOnlyLinear()
    method.create_weights(layer, 8, [3,5], 8, 8, torch.bfloat16, weight_loader=loader)
    assert layer.weight.shape == (8,8) and layer.weight.dtype == torch.int8
    assert layer.weight_scale.shape == (8,1) and layer.weight_scale.dtype == torch.float32
    assert layer.weight.weight_loader is loader and layer.weight_scale.weight_loader is loader
    with pytest.raises(RuntimeError, match="does not support forward"):
        method.apply(layer, torch.empty(1,8))


@pytest.mark.parametrize("width", [192, 256])
def test_expert_parameter_width_contract(width):
    layer=nn.Module(); method=SlimQuantW4A8LoadOnlyExperts(SimpleNamespace())
    method.create_weights(layer, 2, 16, width, torch.bfloat16, weight_loader=loader)
    assert layer.w13_weight.shape == (2,2*width,8) and layer.w2_weight.shape == (2,16,width//2)
    assert layer.w13_weight_scale.shape == (2,2*width,1) and layer.w2_weight_scale.shape == (2,16,1)
    assert torch.count_nonzero(layer.w13_weight) == 0 and torch.all(layer.w13_weight_scale == 1)
    assert layer.w13_weight.weight_loader is loader and layer.w2_weight_scale.weight_loader is loader
    with pytest.raises(RuntimeError, match="does not support forward"):
        method.apply(layer,None,None,None,None,None)


def test_candidate_precision_selection_and_mixed_rejection():
    from hcu_vllm_das.quantization.slimquant_load_only import Kimi31SlimQuantLoadOnlyConfig
    quant = {"quant_method":"slimquant_w4a8", "activation_scheme":"dynamic",
             "modules_to_not_convert":list(Kimi31SlimQuantLoadOnlyConfig._EXPECTED_IGNORES)}
    config=Kimi31SlimQuantLoadOnlyConfig(quant)
    assert config.linear_precision("language_model.model.layers.0.self_attn.q_proj") == "w8a8"
    assert config.linear_precision("language_model.model.layers.3.self_attn.fused_qkv_a_proj") == "w8a8"
    assert config.linear_precision("language_model.model.layers.3.self_attn.dsa_indexer.wq_index") == "unquantized"
    assert config.linear_precision("language_model.model.layers.1.block_sparse_moe.shared_experts.gate_up_proj") == "w8a8"
    assert config.linear_precision("language_model.model.layers.0.mlp.gate_up_proj") == "unquantized"
    with pytest.raises(ValueError, match="mixed KDA"):
        config.linear_precision("language_model.model.layers.0.self_attn.in_proj_qkvgfab")
    with pytest.raises(ValueError, match="unsupported checkpoint"):
        config.linear_precision("language_model.model.layers.0.self_attn.unknown")
    with pytest.raises(ValueError, match="unsupported SlimQuant"):
        Kimi31SlimQuantLoadOnlyConfig({**quant, "modules_to_not_convert": []})


def test_candidate_public_registry_only_when_explicitly_invoked():
    from vllm.model_executor.layers.quantization import register_quantization_config, get_quantization_config, _CUSTOMIZED_METHOD_TO_QUANT_CONFIG, QUANTIZATION_METHODS
    from hcu_vllm_das.quantization.slimquant_load_only import Kimi31SlimQuantLoadOnlyConfig
    # The test owns its unique name; importing the candidate publishes nothing.
    name="kimi31_slimquant_test_only"
    assert name not in QUANTIZATION_METHODS
    try:
        register_quantization_config(name)(Kimi31SlimQuantLoadOnlyConfig)
        assert get_quantization_config(name) is Kimi31SlimQuantLoadOnlyConfig
    finally:
        QUANTIZATION_METHODS.remove(name)
        _CUSTOMIZED_METHOD_TO_QUANT_CONFIG.pop(name,None)


def test_official_ipc_checks_and_postprocess_preserve_storage(monkeypatch):
    from vllm.model_executor.model_loader.weight_cache.protocol import check_ipc_quant_support
    from vllm.model_executor.model_loader import utils
    from vllm.model_executor.utils import weights_already_processed
    import vllm.model_executor.parameter as parameter
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(utils, "maybe_retie_word_embeddings", lambda *args: None)
    monkeypatch.setattr(utils, "release_device_memory_under_pressure", lambda *args: None)
    model=nn.Module(); model.linear=nn.Module(); model.experts=nn.Module()
    model.linear.quant_method=SlimQuantW8A8LoadOnlyLinear()
    model.linear.quant_method.create_weights(model.linear,8,[8],8,8,torch.bfloat16,weight_loader=loader)
    model.experts.quant_method=SlimQuantW4A8LoadOnlyExperts(SimpleNamespace())
    model.experts.quant_method.create_weights(model.experts,2,8,4,torch.bfloat16,weight_loader=loader)
    for p in model.parameters(): p.data.fill_(1)
    old={n:(p, p.data_ptr(), p.clone()) for n,p in model.named_parameters()}
    check_ipc_quant_support(model)
    cfg=SimpleNamespace(quantization="slimquant_w4a8",dtype=torch.bfloat16)
    utils.process_weights_after_loading(model,cfg,torch.device("cpu"))
    with weights_already_processed():
        utils.process_weights_after_loading(model,cfg,torch.device("cpu"))
    for name,p in model.named_parameters():
        before,ptr,value=old[name]
        assert p is before and p.data_ptr()==ptr
        torch.testing.assert_close(p,value,rtol=0,atol=0)
