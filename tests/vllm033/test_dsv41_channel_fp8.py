"""Focused CPU boundaries for the opt-in PCP + MegaMoE registration."""
from types import SimpleNamespace as NS

import pytest
import torch
from hcu_vllm_das import dsv41_activation as activation
from hcu_vllm_das.quantization import dsv41_channel_fp8 as quant
from vllm.model_executor.layers import quantization
from vllm.model_executor.models import ModelRegistry

SCHEMA = dict(quant_method="fp8", is_per_channel=True, activation_scheme="dynamic")


@pytest.fixture
def registered(monkeypatch):
    from vllm import platforms
    # Resolve lazy Core quantization imports on the real CPU platform first.
    quantization.get_quantization_config("fp8")
    platform = NS(is_rocm=lambda: True, device_name="hcu")
    monkeypatch.setattr(platforms, "current_platform", platform)
    monkeypatch.setattr(quant, "current_platform", platform)
    monkeypatch.setenv("VLLM_HCU_DSV41_PCP_ATTENTION", "1")
    monkeypatch.setattr(ModelRegistry, "models", dict(ModelRegistry.models))
    monkeypatch.setattr(quantization, "QUANTIZATION_METHODS", list(quantization.QUANTIZATION_METHODS))
    monkeypatch.setattr(quantization, "_CUSTOMIZED_METHOD_TO_QUANT_CONFIG",
                        dict(quantization._CUSTOMIZED_METHOD_TO_QUANT_CONFIG))
    return NS(vllm_version="0.33.0")


def test_registration_is_opt_in_and_reuses_public_registries(registered, monkeypatch):
    before = ModelRegistry.models[activation.ARCH]
    monkeypatch.delenv("VLLM_HCU_DSV41_PCP_ATTENTION")
    activation.activate_extension(registered)
    assert ModelRegistry.models[activation.ARCH] is before
    assert activation.QUANTIZATION_METHOD not in quantization.QUANTIZATION_METHODS
    monkeypatch.setenv("VLLM_HCU_DSV41_PCP_ATTENTION", "1")
    activation.activate_extension(registered)
    entry = ModelRegistry.models[activation.ARCH]
    assert (entry.module_name, entry.class_name) == activation.MODEL
    assert quantization.get_quantization_config(activation.QUANTIZATION_METHOD) is quant.HcuDSV41ChannelFP8Config
    activation.activate_extension(registered)
    assert ModelRegistry.models[activation.ARCH] is entry


def test_foreign_model_registration_is_preserved(registered):
    entry = NS(module_name="foreign", class_name="Foreign")
    ModelRegistry.models[activation.ARCH] = entry
    with pytest.raises(RuntimeError, match="foreign"):
        activation.activate_extension(registered)
    assert ModelRegistry.models[activation.ARCH] is entry


def test_channel_schema_is_scoped_to_hcu_v41_pcp(registered, monkeypatch):
    cls = quant.HcuDSV41ChannelFP8Config
    hf = NS(model_type="deepseek_v41")
    assert cls.override_quantization_method(SCHEMA, None, hf) == activation.QUANTIZATION_METHOD
    assert cls.override_quantization_method(SCHEMA, None, NS(model_type="llama")) is None
    assert cls.override_quantization_method(dict(SCHEMA, is_per_channel=False), None, hf) is None
    assert cls.override_quantization_method(SCHEMA, "quark", hf) is None
    monkeypatch.delenv("VLLM_HCU_DSV41_PCP_ATTENTION")
    assert cls.override_quantization_method(SCHEMA, None, hf) is None
    monkeypatch.setattr(quant, "current_platform", NS(is_rocm=lambda: False, device_name="cuda"))
    assert cls.override_quantization_method(SCHEMA, activation.QUANTIZATION_METHOD, hf) is None


def test_native_policy_and_mega_dispatch(registered, monkeypatch):
    from hcu_vllm_das.moe import v41_megamoe
    from vllm.model_executor.layers.fused_moe import RoutedExperts
    before = dict(SCHEMA)
    config = quant.HcuDSV41ChannelFP8Config.from_config(SCHEMA)
    assert isinstance(config, quant.DeepseekV4FP8Config)
    assert SCHEMA == before
    obj = RoutedExperts.__new__(RoutedExperts)
    torch.nn.Module.__init__(obj)
    obj.moe_config = NS(moe_backend="deep_gemm_mega_moe")
    calls = []
    def method(weights, inputs, moe, prefix):
        calls.append((weights, inputs, moe, prefix))
        return "mega"
    monkeypatch.setattr(v41_megamoe, "HcuV41MegaMoEMethod", method)
    assert config.get_quant_method(obj, "model.layers.0.ffn.experts") == "mega"
    assert calls[0][0].strategy == "channel"
    assert calls[0][1].strategy == "token"
    assert calls[0][2] is obj.moe_config
    obj.moe_config.moe_backend = "deep_gemm"
    with pytest.raises(ValueError, match="requires deep_gemm_mega_moe"):
        config.get_quant_method(obj, "model.layers.0.ffn.experts")


def test_empty_linear_only_and_native_nonempty(monkeypatch):
    from hcu_vllm_das.quantization.dsv41_channel_fp8_linear import HcuDSV41ChannelFP8LinearMethod
    method = object.__new__(HcuDSV41ChannelFP8LinearMethod)
    layer = NS(processed_weight_dims=(0, 1), weight=torch.empty(3, 5),
               scheme=NS(fp8_linear=NS(config=NS(out_dtype=None))))
    out = method.apply(layer, torch.empty(0, 3, dtype=torch.bfloat16))
    assert out.shape == (0, 5) and out.dtype == torch.bfloat16
    monkeypatch.setattr(quant.CompressedTensorsConfig, "get_quant_method", lambda *a: None)
    base = HcuDSV41ChannelFP8LinearMethod.__mro__[1]
    monkeypatch.setattr(base, "apply", lambda *a: "native")
    assert method.apply(layer, torch.empty(2, 3)) == "native"
    layer.processed_weight_dims = (1, 0)
    with pytest.raises(ValueError, match="finalized"):
        method.apply(layer, torch.empty(0, 3))
