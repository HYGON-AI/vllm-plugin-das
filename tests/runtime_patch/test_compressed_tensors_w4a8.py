# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

import sys
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
    with pytest.raises(RuntimeError,match="fresh rank-3 packed INT32"):
        runtime._process_weights_after_loading(method,layer)


@pytest.fixture
def native_wna16(monkeypatch):
    """Use the real CT loader; replace only accelerator backend selection."""
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import compressed_tensors_moe_wna16 as native
    from vllm_hcu.patch.worker.op_opt import patch_compressed_tensors_moe_wna16 as wna16_patch

    cls = native.CompressedTensorsWNA16MoEMethod
    original = dict(vars(cls))
    monkeypatch.setattr(
        native, "select_wna16_moe_backend",
        lambda **_: (native.WNA16MoEBackend.TRITON, None),
    )
    # Exercise the HCU channel-storage wrapper, including its real qzeros
    # creation, without depending on process-global platform/environment flags.
    monkeypatch.setattr(wna16_patch, "_aiter_requested", lambda _: True)
    wna16_patch.apply_to_module(native)
    try:
        yield native
    finally:
        for name in set(vars(cls)) - set(original):
            delattr(cls, name)
        for name, value in original.items():
            if name not in ("__dict__", "__weakref__"):
                setattr(cls, name, value)


def _native_layer(group_size, backend):
    from compressed_tensors.quantization import QuantizationArgs

    weight = QuantizationArgs(num_bits=4, type="int", symmetric=True,
                              dynamic=False, strategy="channel", group_size=group_size)
    activation = QuantizationArgs(num_bits=8, type="int", symmetric=True,
                                  dynamic=True, strategy="token")
    moe = SimpleNamespace(has_bias=False, activation="silu",
                          is_act_and_mul=True, moe_backend=backend)
    method = runtime.make_hcu_w4a8_method(weight, activation, moe, "experts")
    layer = torch.nn.Module()
    layer.moe_config = moe
    layer.quant_method = method
    method.create_weights(layer, 2, 32, 16, torch.bfloat16,
                          intermediate_size_full=16)
    return weight, method, layer


@pytest.mark.parametrize("group_size", [None, -1])
def test_channel_group_size_uses_real_wna16_storage(native_wna16, group_size):
    weight, method, layer = _native_layer(group_size, "aiter")
    assert weight.group_size == group_size  # caller's config remains untouched
    assert method.weight_quant.group_size == method.group_size == -1
    assert type(method) is native_wna16.CompressedTensorsWNA16MoEMethod
    assert layer.w13_weight_packed.shape == (2, 4, 32)
    assert layer.w2_weight_packed.shape == (2, 2, 32)
    assert layer.w13_weight_scale.shape == layer.w2_weight_scale.shape == (2, 1, 32)
    assert layer.w13_qzeros.shape == layer.w2_qzeros.shape == (2, 16, 1)
    assert torch.all(layer.w13_qzeros == 0x88)
    assert torch.all(layer.w2_qzeros == 0x88)


@pytest.mark.parametrize("group_size", [None, -1])
@pytest.mark.parametrize("layout", ["aiter", "native_fallback", "packed_triton"])
def test_checkpoint_reloads_use_native_layerwise_lifecycle(
    monkeypatch, native_wna16, group_size, layout,
):
    from compressed_tensors.compressors.pack_quantized.helpers import pack_to_int32
    from vllm.model_executor.model_loader.reload.layerwise import (
        record_metadata_for_reloading, initialize_layerwise_reload,
        finalize_layerwise_reload,
    )
    from vllm_hcu.model_executor.layers.quantization import compressed_tensors_moe_runtime as kernels

    # Run the real SlimQuant delegate's post-load and native CT loader on CPU.
    # Accelerator stages use distinguishable layouts so reload must rebuild
    # kernel storage from checkpoint data, rather than silently skip conversion.
    prepares = []

    def prepare(method, layer):
        prepares.append(layout)
        if layout == "native_fallback":
            for prefix in ("w13", "w2"):
                packed = getattr(layer, prefix + "_weight").view(torch.uint8)
                high, low = packed >> 4, packed & 15
                signed = torch.stack((high, low), dim=-1).flatten(-2).to(torch.int8)
                signed = torch.where(signed >= 8, signed - 16, signed)
                setattr(layer, prefix + "_weight", torch.nn.Parameter(signed, requires_grad=False))

    monkeypatch.setattr(kernels, "prewarm_aiter_w4a8_moe",
                        lambda *_: object() if layout == "aiter" else None)
    monkeypatch.setattr(kernels, "install_aiter_moe_weight_layout",
                        lambda layer, config, **_: prepare(None, layer))
    monkeypatch.setattr(kernels, "install_aiter_moe_scale_layout", lambda *_, **__: None)
    monkeypatch.setattr(kernels, "mark_aiter_moe_native_layout", lambda _: None)
    monkeypatch.setattr(kernels, "prepare_vllm_w4a8_moe", prepare)
    monkeypatch.setattr(kernels, "prepare_boltops_w4a8_moe", prepare)
    _, method, layer = _native_layer(group_size, "triton" if layout == "packed_triton" else "aiter")
    record_metadata_for_reloading(layer)
    checkpoint = {name: torch.zeros_like(param) for name, param in layer.named_parameters()}
    kernel_parameters = None

    for generation in range(3):
        generator = torch.Generator().manual_seed(100 + generation)
        expected = {}
        for prefix, shape in (("w13", (2, 32, 32)), ("w2", (2, 32, 16))):
            q = torch.randint(-8, 8, shape, dtype=torch.int8, generator=generator)
            scale = torch.rand(shape[0], shape[1], 1, generator=generator).to(torch.bfloat16)
            checkpoint[prefix + "_weight_packed"] = pack_to_int32(q, 4).transpose(1, 2).contiguous()
            checkpoint[prefix + "_weight_scale"] = scale.transpose(1, 2).contiguous()
            expected[prefix] = (q, scale.float() / 16)

        if generation:
            # Same public helpers used by checkpoint-format reload_weights and
            # weight-transfer updates. The quant_method instance is retained.
            initialize_layerwise_reload(layer)
        for name, tensor in checkpoint.items():
            param = getattr(layer, name)
            loader = getattr(param, "weight_loader", None)
            if loader is None:
                param.data.copy_(tensor)
            else:
                loader(param, tensor)
        if generation:
            finalize_layerwise_reload(layer, None)
        else:
            method.process_weights_after_loading(layer)
            kernel_parameters = dict(layer.named_parameters(remove_duplicate=False))

        assert layer.quant_method is method
        assert method._hcu_ct_w4a8_loaded
        assert len(prepares) == generation + 1
        for prefix, (q, scale) in expected.items():
            packed = getattr(layer, prefix + "_weight_packed")
            assert packed is getattr(layer, prefix + "_weight")
            assert packed is kernel_parameters[prefix + "_weight"]
            expected_weight = q if layout == "native_fallback" else runtime.repack_ct_int4(
                checkpoint[prefix + "_weight_packed"], transposed=True)
            assert torch.equal(packed, expected_weight)
            assert torch.equal(getattr(layer, prefix + "_weight_scale"), scale)


@pytest.mark.parametrize("group_size", [None, -1])
@pytest.mark.parametrize("update", ["inplace", "replacement"])
def test_aiter_forward_uses_final_scales_after_checkpoint_reload(
    monkeypatch, native_wna16, group_size, update,
):
    from compressed_tensors.compressors.pack_quantized.helpers import pack_to_int32
    from vllm.model_executor.model_loader.reload.layerwise import (
        record_metadata_for_reloading, initialize_layerwise_reload,
        finalize_layerwise_reload,
    )
    from vllm_hcu.model_executor.layers.quantization import compressed_tensors_moe_runtime as kernels

    selected = SimpleNamespace(need_shuffle=False)
    monkeypatch.setattr(kernels, "prewarm_aiter_w4a8_moe", lambda *_: selected)
    monkeypatch.setattr(kernels, "install_aiter_moe_weight_layout", lambda *_, **__: None)
    monkeypatch.setattr(kernels, "install_aiter_moe_scale_layout", lambda *_, **__: None)
    monkeypatch.setattr(kernels, "select_aiter_moe_config", lambda *_, **__: selected)
    monkeypatch.setattr(kernels, "prepare_aiter_moe_weights", lambda w1, w2, *_, **__: (w1, w2))
    monkeypatch.setattr(kernels, "prepare_aiter_moe_scales", lambda s1, s2, *_, **__: (s1, s2))
    monkeypatch.setattr(kernels, "aiter_expert_map_for_solution", lambda *_, **__: None)
    aiter = ModuleType("aiter.moe")
    aiter.MoeQuantType = SimpleNamespace(W4A8="w4a8")
    monkeypatch.setitem(sys.modules, "aiter.moe", aiter)

    _, method, layer = _native_layer(group_size, "aiter")
    record_metadata_for_reloading(layer)
    checkpoint = {name: torch.zeros_like(param) for name, param in layer.named_parameters()}
    captured = []

    def execute(config, **kwargs):
        # Keep real adapter/delegate/runtime dispatch. Replace only the HCU
        # kernel boundary, where both GEMMs must see the current Parameters.
        assert kwargs["w1_scale"] is layer.w13_weight_scale
        assert kwargs["w2_scale"] is layer.w2_weight_scale
        captured.append((kwargs["w1_scale"].clone(), kwargs["w2_scale"].clone()))
        return kwargs["hidden_states"]

    monkeypatch.setattr(kernels, "execute_aiter_moe", execute)
    for generation in range(3):
        if generation:
            initialize_layerwise_reload(layer)
        for prefix, shape in (("w13", (2, 32, 32)), ("w2", (2, 32, 16))):
            q = torch.full(shape, generation, dtype=torch.int8)
            checkpoint[prefix + "_weight_packed"] = pack_to_int32(q, 4).transpose(1, 2).contiguous()
            checkpoint[prefix + "_weight_scale"] = torch.full(
                (shape[0], 1, shape[1]), generation + 1.0, dtype=torch.bfloat16,
            )
        for name, tensor in checkpoint.items():
            param = getattr(layer, name)
            loader = getattr(param, "weight_loader", None)
            if loader is None:
                param.data.copy_(tensor)
            else:
                loader(param, tensor)
        if generation:
            finalize_layerwise_reload(layer, None)
        else:
            method.process_weights_after_loading(layer)

        delegate = method._hcu_ct_w4a8_delegate
        config = delegate.moe_quant_config
        # A modular consumer may retain this config; rebinding must preserve
        # the config and descriptors, rather than replace them with new ones.
        descriptors = (config._w1, config._w2)
        x = torch.ones(2, 32, dtype=torch.bfloat16)
        ids = torch.zeros(2, 1, dtype=torch.int32)
        weights = torch.ones(2, 1)
        for multiplier in (2.0, 3.0):
            for prefix in ("w13", "w2"):
                scale = getattr(layer, prefix + "_weight_scale")
                if update == "inplace":
                    with torch.no_grad():
                        scale.mul_(multiplier)
                else:
                    setattr(layer, prefix + "_weight_scale", torch.nn.Parameter(
                        scale.detach() * multiplier, requires_grad=False,
                    ))
            method.apply(layer, x, weights, ids)
            assert method.moe_quant_config is delegate.moe_quant_config is config
            assert config._w1 is descriptors[0] and config._w2 is descriptors[1]
            torch.testing.assert_close(captured[-1][0], layer.w13_weight_scale)
            torch.testing.assert_close(captured[-1][1], layer.w2_weight_scale)
    assert len(captured) == 6


def test_scale_rebinding_supports_fullgraph_compile():
    from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig

    layer = torch.nn.Module()
    layer.w13_weight_scale = torch.nn.Parameter(torch.tensor([.25]), requires_grad=False)
    layer.w2_weight_scale = torch.nn.Parameter(torch.tensor([.5]), requires_grad=False)
    config = FusedMoEQuantConfig.make(
        torch.int8, w1_scale=torch.tensor([.125]), w2_scale=torch.tensor([.25]),
        per_act_token_quant=True, weight_dtype="int4",
    )

    class Delegate:
        moe_quant_config = config

        def apply(self, layer, x, *args, **kwargs):
            return x * self.moe_quant_config.w1_scale * self.moe_quant_config.w2_scale

    method = SimpleNamespace(_hcu_ct_w4a8_loaded=True, _hcu_ct_w4a8_delegate=Delegate())
    compiled = torch.compile(
        lambda x: runtime._apply(method, layer, x, None, None),
        backend="eager", fullgraph=True,
    )
    x = torch.ones(2)
    torch.testing.assert_close(compiled(x), x * .125)
    with torch.no_grad():
        layer.w13_weight_scale.mul_(2)
        layer.w2_weight_scale.mul_(2)
    torch.testing.assert_close(compiled(x), x * .5)


def test_mixed_checkpoint_storage_fails_before_conversion():
    layer = torch.nn.Module()
    layer.w13_weight_packed = torch.nn.Parameter(torch.zeros(2, 4, 32, dtype=torch.int32), requires_grad=False)
    layer.w2_weight_packed = torch.nn.Parameter(torch.zeros(2, 32, 8, dtype=torch.int8), requires_grad=False)
    original = layer.w13_weight_packed
    method = SimpleNamespace(_hcu_ct_w4a8_loaded=True)
    with pytest.raises(RuntimeError, match="w2_weight_packed"):
        runtime._process_weights_after_loading(method, layer)
    assert layer.w13_weight_packed is original
    assert not method._hcu_ct_w4a8_loaded
    with pytest.raises(RuntimeError, match="requires completed weight post-load"):
        runtime._apply(method, layer, None, None, None)
