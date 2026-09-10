import importlib
import json

import pytest
import torch

MODULE = 'vllm_hcu.model_executor.layers.quantization.hyv4_w4a8_weights'


def adapter():
    import importlib.util
    assert importlib.util.find_spec(MODULE) is not None, 'custom W4A8 adapter is missing'
    return importlib.import_module(MODULE)


def test_signed_nibble_conversion_preserves_all_values():
    mod = adapter()
    values = torch.arange(-8, 8, dtype=torch.int8)
    packed = ((values[::2].to(torch.uint8) & 15) | ((values[1::2].to(torch.uint8) & 15) << 4)).view(torch.int8)
    actual = mod.to_aiter_packing(packed)
    u = actual.to(torch.uint8)
    unpacked = torch.stack([u >> 4, u & 15], -1).flatten().int()
    unpacked = torch.where(unpacked >= 8, unpacked - 16, unpacked)
    torch.testing.assert_close(unpacked, values.int())


def test_stream_maps_fused_experts_and_shared_linear():
    mod = adapter()
    p = 'layers.1.mlp.'
    packed = torch.tensor([[[0x87, 0x10]]], dtype=torch.uint8)
    scales = torch.tensor([[0.5]])
    source = [(p+'experts.gate_up_proj.int4_packed', packed),
              (p+'experts.gate_up_proj.scale', scales),
              (p+'experts.gate_up_proj.input_scale', torch.ones(1, 4)),
              (p+'shared_experts.gate_proj.weight.int4_packed', packed[0]),
              (p+'shared_experts.gate_proj.weight.scale', scales[0]),
              ('layers.0.self_attn.q_a_proj.weight', torch.ones(2, 2))]
    result = dict(mod.adapt_weights(source))
    assert set(result) == {p+'experts.gate_up_proj', p+'experts.gate_up_proj_scale', p+'shared_experts.gate_proj.weight', p+'shared_experts.gate_proj.weight_scale', 'layers.0.self_attn.q_a_proj.weight'}
    assert result[p+'experts.gate_up_proj'].data_ptr() == packed.data_ptr()
    assert result[p+'experts.gate_up_proj_scale'].shape == (1, 1, 1)
    assert result[p+'shared_experts.gate_proj.weight_scale'].shape == (1, 1)


def test_nonidentity_smoothing_scale_is_rejected():
    with pytest.raises(ValueError, match='identity input_scale'):
        list(adapter().adapt_weights([('layers.1.mlp.experts.down_proj.input_scale', torch.tensor([2.]))]))


def test_manifest_selects_quantized_modules(tmp_path):
    mod = adapter()
    path = tmp_path/'conversion.json'
    path.write_text(json.dumps({'provenance': {'format':'hy4-w4a8-custom-v1'}, 'quantized_tensors':{
        'model.layers.1.mlp.experts.gate_up_proj': {'packing':'signed nibble, even input in low bits'},
        'model.layers.1.mlp.shared_experts.gate_proj.weight': {'packing':'signed nibble, even input in low bits'},
        'model.layers.1.mlp.shared_experts.up_proj.weight': {'packing':'signed nibble, even input in low bits'}}}))
    modules = mod.read_quantized_modules(str(path))
    assert 'model.layers.1.mlp.experts' in modules
    assert 'model.layers.1.mlp.shared_experts.gate_up_proj' in modules
    assert 'model.layers.0.self_attn.q_a_proj' not in modules
    assert 'model.layers.78.mlp.experts' not in modules


def test_custom_quant_config_dispatch_and_linear_storage(tmp_path):
    from vllm_hcu.model_executor.layers.quantization.slimquant_w4a8 import SlimQuantW4A8Int8Config
    path = tmp_path/'conversion.json'
    path.write_text(json.dumps({'provenance': {'format':'hy4-w4a8-custom-v1'}, 'quantized_tensors':{
        'model.layers.1.mlp.shared_experts.down_proj.weight': {'packing':'signed nibble, even input in low bits'}}}))
    config = SlimQuantW4A8Int8Config.from_config({'checkpoint_format':'hy4-w4a8-custom-v1', 'conversion_manifest':str(path)})
    assert getattr(config, 'checkpoint_format', None) == 'hy4-w4a8-custom-v1'
    from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
    layer = LinearBase.__new__(LinearBase)
    torch.nn.Module.__init__(layer)
    method = config.get_quant_method(layer, 'model.layers.1.mlp.shared_experts.down_proj')
    method.create_weights(layer, 128, [256], 128, 256, torch.bfloat16, weight_loader=lambda *a:None)
    assert layer.weight.shape == (256, 64)
    assert layer.weight.dtype == torch.int8
    assert layer.weight_scale.shape == (256, 1)
    assert isinstance(config.get_quant_method(layer, 'model.layers.0.self_attn.q_a_proj'), UnquantizedLinearMethod)
    layer.weight.data.fill_(0x12)
    layer.weight_scale.data.fill_(1.)
    method.process_weights_after_loading(layer)
    assert torch.all(layer.weight == 0x21)


def test_custom_config_selects_routed_experts_and_reorders_local_shards(tmp_path):
    from types import SimpleNamespace
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
    from vllm_hcu.model_executor.layers.quantization.hyv4_w4a8 import HYV4W4A8Config, HYV4W4A8MoEMethod
    path = tmp_path/'conversion.json'
    path.write_text(json.dumps({'provenance': {'format':'hy4-w4a8-custom-v1'}, 'quantized_tensors':{
        'model.layers.1.mlp.experts.down_proj': {'packing':'signed nibble, even input in low bits'}}}))
    config = HYV4W4A8Config(str(path))
    layer = RoutedExperts.__new__(RoutedExperts)
    torch.nn.Module.__init__(layer)
    layer.moe_config = SimpleNamespace(swiglu_limit=10.)
    method = config.get_quant_method(layer, 'model.layers.1.mlp.experts')
    assert isinstance(method, HYV4W4A8MoEMethod)
    method.create_weights(layer, 4, 128, 64, torch.bfloat16)
    assert layer.w13_weight.shape == (4, 128, 64)
    assert layer.w2_weight.shape == (4, 128, 32)
    layer.w13_weight.data.fill_(0x12)
    layer.w2_weight.data.fill_(0x34)
    method.process_weights_after_loading(layer)
    assert torch.all(layer.w13_weight == 0x21)
    assert torch.all(layer.w2_weight == 0x43)
    assert config.get_quant_method(layer, 'model.layers.78.mlp.experts') is None
