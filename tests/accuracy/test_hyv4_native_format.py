import json
from types import SimpleNamespace

import pytest
import torch


def test_native_manifest_selects_main_and_mtp_experts(tmp_path):
    from vllm_hcu.model_executor.layers.quantization.hyv4_native import HYV4NativeW4A8Config
    p = tmp_path / 'hy4-checkpoint.index.json'
    p.write_text(json.dumps({'format':'hy4_w4a8_v1','complete':True,'parameters':{
        'model.layers.1.mlp.experts.0.gate_proj.weight':{'kind':'quantized'},
        'model.mtp_layers.0.mlp.experts.0.up_proj.weight':{'kind':'quantized'},
        'model.mtp_layers.0.mlp.shared_experts.up_proj.weight':{'kind':'quantized'},
        'model.layers.0.self_attn.q_a_proj.weight':{'kind':'retained'},
    }}))
    (tmp_path/'config.json').write_text(json.dumps({'num_hidden_layers':78}))
    c = HYV4NativeW4A8Config(str(p))
    assert c.quantized_modules == frozenset({'model.layers.1.mlp.experts',
        'model.layers.78.mlp.experts','model.layers.78.mlp.shared_experts.gate_up_proj'})


def test_native_weight_mapping_preserves_existing_scale_rank():
    from vllm_hcu.model_executor.layers.quantization.hyv4_native import adapt_native_weights
    q = torch.tensor([[0x21, 0xF0]], dtype=torch.uint8)
    s = torch.tensor([[.5]])
    w = dict(adapt_native_weights([('model.layers.1.mlp.experts.0.gate_proj.weight.packed',q),
        ('model.layers.1.mlp.experts.0.gate_proj.weight.scale',s)]))
    assert w['model.layers.1.mlp.experts.0.gate_proj.weight'].data_ptr() == q.data_ptr()
    assert w['model.layers.1.mlp.experts.0.gate_proj.weight_scale'].shape == (1,1)


def test_native_rejects_invalid_scale():
    from vllm_hcu.model_executor.layers.quantization.hyv4_native import adapt_native_weights
    with pytest.raises(ValueError, match='scale'):
        list(adapt_native_weights([('test.weight.scale',torch.tensor([[float('nan')]]))]))


def test_native_retained_parameter_names_strip_storage_suffix():
    from vllm_hcu.model_executor.layers.quantization.hyv4_native import adapt_native_weights
    w = torch.ones(2)
    values = dict(adapt_native_weights([
        ('lm_head.weight.weight', w),
        ('model.embed_tokens.weight.weight', w),
        ('model.hc_head.hc_head_base.weight', w),
    ]))
    assert set(values) == {'lm_head.weight','model.embed_tokens.weight',
                           'model.hc_head.hc_head_base'}


@pytest.mark.parametrize('weight_block_size', [None, [128, 128]])
def test_fp8_outer_loader_keeps_checkpoint_names(monkeypatch, weight_block_size):
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config
    from vllm_hcu.models.hy_v4 import model as hy_model

    weights = [('lm_head.weight', torch.ones(1)),
               ('model.layers.0.self_attn.q_a_proj.weight_scale', torch.ones(1))]
    received = []

    class Loader:
        def __init__(self, *args, **kwargs):
            pass

        def load_weights(self, values):
            received.extend(values)
            return {name for name, _ in received}

    monkeypatch.setattr(hy_model, 'AutoWeightsLoader', Loader)
    model = SimpleNamespace(
        quant_config=Fp8Config(is_checkpoint_fp8_serialized=True,
                               weight_block_size=weight_block_size),
        config=SimpleNamespace(tie_word_embeddings=False, num_hidden_layers=78,
                               num_nextn_predict_layers=1),
        named_parameters=lambda: iter(weights),
    )
    hy_model.HYV4ForCausalLM.load_weights(model, iter(weights))
    assert [name for name, _ in received] == [name for name, _ in weights]
    assert all(actual is expected for (_, actual), (_, expected) in zip(received, weights))
