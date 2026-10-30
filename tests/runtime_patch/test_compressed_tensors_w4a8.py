# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from types import ModuleType, SimpleNamespace

import pytest
import torch

from vllm_hcu.model_executor.layers.quantization import compressed_tensors_w4a8_runtime as runtime
from vllm_hcu.patch.worker.op_opt import patch_compressed_tensors_moe_w4a8_int8 as patch
from vllm_hcu.patch.worker.op_opt._common import PatchCompatibilityError


def _scheme():
    weight = SimpleNamespace(num_bits=4, type="int", symmetric=True,
        dynamic=False, strategy="channel", group_size=-1, actorder=None)
    activation = SimpleNamespace(num_bits=8, type="int", symmetric=True,
        dynamic=True, strategy="token", group_size=None)
    moe = SimpleNamespace(has_bias=False, activation="silu")
    return weight, activation, moe


def _module(function):
    module = ModuleType(patch.TARGET_MODULE)
    module.CompressedTensorsMoEMethod = type(
        "CompressedTensorsMoEMethod", (), {"get_moe_method": staticmethod(function)}
    )
    return module


def _cpu_only(quant_config, layer, layer_name):
    raise NotImplementedError(patch._CPU_ONLY_ERROR)


def test_patch_is_registered_in_worker_inventory():
    from vllm_hcu.patch import worker
    assert any(s.adapter.endswith(".patch_compressed_tensors_moe_w4a8_int8")
               for s in worker._OP_CALLBACKS)


def test_patch_is_idempotent_and_preserves_supported_upstream_path():
    result = object()
    def official(quant_config, layer, layer_name):
        return result
    module = _module(official)
    assert patch.apply_to_module(module)
    assert not patch.apply_to_module(module)
    assert module.CompressedTensorsMoEMethod.get_moe_method(None, None, "experts") is result


def test_hcu_channel_scheme_dispatches_to_w4a8_runtime(monkeypatch):
    module = _module(_cpu_only)
    weight, activation, moe = _scheme()
    layer = SimpleNamespace(moe_config=moe, apply_router_weight_on_input=False)
    qconfig = SimpleNamespace(get_scheme_dict=lambda *_: {
        "format":"pack-quantized", "weights":weight, "input_activations":activation})
    monkeypatch.setattr(patch, "_hcu_enabled", lambda _: True)
    result = object(); calls=[]
    monkeypatch.setattr(runtime, "make_hcu_w4a8_method", lambda *args: calls.append(args) or result)
    patch.apply_to_module(module)
    assert module.CompressedTensorsMoEMethod.get_moe_method(qconfig, layer, "experts") is result
    assert calls == [(weight, activation, moe, "experts")]


def test_non_hcu_preserves_original_cpu_only_error(monkeypatch):
    module = _module(_cpu_only)
    monkeypatch.setattr(patch, "_hcu_enabled", lambda _: False)
    patch.apply_to_module(module)
    with pytest.raises(NotImplementedError, match="only supported on CPU"):
        module.CompressedTensorsMoEMethod.get_moe_method(None, SimpleNamespace(moe_config=None), "experts")


def test_unrelated_error_is_not_swallowed(monkeypatch):
    def official(quant_config, layer, layer_name):
        raise NotImplementedError("different backend failure")
    module = _module(official)
    monkeypatch.setattr(patch, "_hcu_enabled", lambda _: True)
    patch.apply_to_module(module)
    with pytest.raises(NotImplementedError, match="different backend failure"):
        module.CompressedTensorsMoEMethod.get_moe_method(None, None, "experts")


@pytest.mark.parametrize("owner,field,value", [
    ("weight","num_bits",8), ("weight","strategy","group"),
    ("weight","symmetric",False), ("weight","actorder","group"),
    ("activation","dynamic",False), ("activation","type","float"),
    ("activation","strategy","tensor"), ("moe","has_bias",True),
    ("moe","activation","situ"),
])
def test_unsupported_contracts_are_rejected(owner, field, value):
    weight, activation, moe = _scheme()
    assert runtime.supports_channel_w4a8(weight, activation, moe)
    setattr({"weight":weight,"activation":activation,"moe":moe}[owner],field,value)
    assert not runtime.supports_channel_w4a8(weight, activation, moe)


@pytest.mark.parametrize("transposed", [False, True])
def test_packed_weights_preserve_all_signed_int4_values(transposed):
    from compressed_tensors.compressors.pack_quantized.helpers import pack_to_int32
    q = torch.arange(-8,8,dtype=torch.int8).repeat(2,3,4)
    words=pack_to_int32(q,4)
    if transposed: words=words.transpose(1,2).contiguous()
    result=runtime.repack_ct_int4(words,transposed=transposed).view(torch.uint8)
    high=(result>>4).to(torch.int16);low=(result&15).to(torch.int16)
    high=torch.where(high>=8,high-16,high);low=torch.where(low>=8,low-16,low)
    actual=torch.stack([high,low],dim=-1).flatten(-2).to(torch.int8)
    assert torch.equal(actual,q)
    assert result.shape == (2,3,32)


def test_bad_packed_storage_is_rejected():
    with pytest.raises(ValueError,match="rank-3 packed INT32"):
        runtime.repack_ct_int4(torch.zeros(2,4,dtype=torch.int32),transposed=False)


def test_patch_rejects_incompatible_upstream_signature():
    module=_module(lambda x: x)
    with pytest.raises(PatchCompatibilityError,match="signature"):
        patch.apply_to_module(module)


def test_postload_preserves_scale_domain_and_releases_original_buffers(monkeypatch):
    from compressed_tensors.compressors.pack_quantized.helpers import pack_to_int32
    layer=torch.nn.Module()
    originals={}
    for prefix,shape in [("w13",(2,16,32)),("w2",(2,32,8))]:
        q=torch.randint(-8,8,shape,dtype=torch.int8)
        scale=torch.rand(shape[0],shape[1],1,dtype=torch.bfloat16)
        originals[prefix]=(q,scale)
        layer.register_parameter(prefix+"_weight_packed",torch.nn.Parameter(pack_to_int32(q,4).transpose(1,2).contiguous(),requires_grad=False))
        layer.register_parameter(prefix+"_weight_scale",torch.nn.Parameter(scale.transpose(1,2).contiguous(),requires_grad=False))
    class Delegate:
        moe_quant_config=object();moe_kernel=None
        def process_weights_after_loading(self, target):
            for prefix,(q,scale) in originals.items():
                assert torch.equal(getattr(target,prefix+"_weight_scale")*16,scale.float())
                # Simulate the existing Triton fallback expanding its final storage.
                setattr(target,prefix+"_weight",torch.nn.Parameter(q.clone(),requires_grad=False))
    method=SimpleNamespace(is_transposed=True,_hcu_ct_w4a8_delegate=Delegate())
    runtime._process_weights_after_loading(method,layer)
    for prefix in originals:
        assert getattr(layer,prefix+"_weight_packed") is getattr(layer,prefix+"_weight")
    assert method._hcu_ct_w4a8_loaded
    assert method.moe_quant_config is method._hcu_ct_w4a8_delegate.moe_quant_config
    with pytest.raises(RuntimeError,match="only once"):
        runtime._process_weights_after_loading(method,layer)
