# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from torch import nn
from hcu_vllm_das.quantization.mixed_kda import MixedKDAProjection


def test_mixed_projection_routes_native_loaders():
    calls=[]
    def fake_loader(param, value, *args):
        calls.append((param,value.dtype,args))
    def column(n,m,*,quant_config,prefix,**kwargs):
        assert "padding_size" not in kwargs
        x=nn.Module(); x.weight=nn.Parameter(torch.zeros(m,n,dtype=torch.int8),requires_grad=False)
        x.weight_scale=nn.Parameter(torch.ones(m,1),requires_grad=False)
        x.weight.weight_loader=fake_loader; x.weight_scale.weight_loader=fake_loader
        return x
    def merged(n,sizes,*,replicated_shard_id,tp_size,quant_config,prefix,**kwargs):
        assert replicated_shard_id == 1 and quant_config is None and kwargs["padding_size"] == 2
        x=nn.Module(); x.weight=nn.Parameter(torch.zeros(sum(sizes),n,dtype=torch.bfloat16),requires_grad=False)
        x.weight.weight_loader=fake_loader
        return x
    m=MixedKDAProjection(4,[8,8,8,8,4,2,2],4,2,column_linear=column,merged_linear=merged,quant_config=object(),prefix="a.self_attn.in_proj_qkvgfab",padding_size=2)
    for branch in ("q_proj","k_proj","v_proj"):
        assert m.load_branch(branch,"weight",torch.zeros(1,dtype=torch.int8)) == branch+".weight"
        m.load_branch(branch,"weight_scale",torch.ones(1,dtype=torch.float32))
    for i,branch in enumerate(("g_proj","f_a_proj","b_proj")):
        assert m.load_branch(branch,"weight",torch.ones(1,dtype=torch.bfloat16)) == "fp_gate.weight"
        assert calls[-1][2] == (i,)
    with pytest.raises(TypeError): m.load_branch("q_proj","weight",torch.ones(1,dtype=torch.bfloat16))
    with pytest.raises(ValueError): m.load_branch("g_proj","weight",torch.ones(1,dtype=torch.float32))
    with pytest.raises(RuntimeError,match="does not support forward"): m(torch.ones(1,4))


def test_weight_stream_preserves_other_entries_and_rejects_duplicate(monkeypatch):
    from hcu_vllm_das.quantization.mixed_kda import load_mixed_kda_weights
    model = nn.Module(); model.attn = nn.Module()
    projection = MixedKDAProjection.__new__(MixedKDAProjection); nn.Module.__init__(projection)
    model.attn.in_proj_qkvgfab = projection
    seen = []
    def load_branch(self, branch, suffix, tensor):
        seen.append((branch, suffix, tensor))
        return f"{branch}.{suffix}"
    monkeypatch.setattr(MixedKDAProjection, "load_branch", load_branch)
    source = [("attn.q_proj.weight", torch.zeros(1, dtype=torch.int8)),
              ("other.weight", torch.ones(1)),
              ("attn.g_proj.weight", torch.ones(1, dtype=torch.bfloat16))]
    rest = []
    def native(entries):
        rest.extend(entries)
        return {"other.weight"}
    result = load_mixed_kda_weights(model, iter(source), native)
    assert [x[0] for x in rest] == ["other.weight"]
    assert result == {"other.weight", "attn.in_proj_qkvgfab.q_proj.weight", "attn.in_proj_qkvgfab.g_proj.weight"}
    assert [x[0] for x in seen] == ["q_proj", "g_proj"]
    with pytest.raises(ValueError, match="duplicate"):
        load_mixed_kda_weights(model, iter([source[0], source[0]]), lambda entries: set(name for name, _ in entries))
    with pytest.raises(ValueError, match="metadata"):
        load_mixed_kda_weights(model, iter([(source[0][0], source[0][1], {"unexpected": True})]), lambda entries: set(name for name, *_ in entries))
