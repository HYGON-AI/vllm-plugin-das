# SPDX-License-Identifier: Apache-2.0
"""Real official registry/config and compilation contracts, without a device."""

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize(
    "master,child,expected",
    [
        (None, None, False),
        (None, "1", False),
        ("0", "1", False),
        ("1", None, False),
        ("1", "1", True),
    ],
)
def test_effective_policy(monkeypatch, master, child, expected):
    from hcu_vllm_das.environment import lightop_awq_enabled

    for key, value in (
        ("VLLM_HCU_USE_CUSTOM_OPS", master),
        ("VLLM_HCU_USE_LIGHTOP_AWQ", child),
    ):
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)
    assert lightop_awq_enabled() is expected


@pytest.fixture
def activation(monkeypatch):
    from hcu_vllm_das import autoawq_activation as activation
    from vllm import envs
    from vllm.model_executor.layers import quantization

    monkeypatch.setattr(quantization, "_CUSTOMIZED_METHOD_TO_QUANT_CONFIG", {})
    monkeypatch.setattr(envs, "environment_variables", dict(envs.environment_variables))
    monkeypatch.setattr(activation, "_published", None)
    monkeypatch.setattr(activation, "_failure", False)
    monkeypatch.setattr(activation, "_activating", False)
    return activation, SimpleNamespace(
        vllm_version="0.33.0+test", requested_capabilities={"lightop-autoawq"}
    )


def test_real_registry_owns_all_official_aliases_without_patching_class(activation):
    from hcu_vllm_das.autoawq import HcuAutoAWQConfig
    from vllm.model_executor.layers.quantization import get_quantization_config
    from vllm.model_executor.layers.quantization.auto_awq import AutoAWQConfig

    module, context = activation
    original = AutoAWQConfig.get_quant_method
    module.activate(context)
    module.activate(context)
    module.check_health()
    assert all(
        get_quantization_config(name) is HcuAutoAWQConfig for name in module.METHODS
    )
    assert AutoAWQConfig.get_quant_method is original
    config = get_quantization_config("auto_awq").from_config(
        {"bits": 4, "group_size": 128, "zero_point": True}
    )
    assert type(config) is HcuAutoAWQConfig
    assert config.full_config["quant_method"] == "awq"


def test_publication_refuses_other_config_owner_before_mutation(
    activation, monkeypatch
):
    from vllm.model_executor.layers import quantization
    from vllm.model_executor.layers.quantization.auto_awq import AutoAWQConfig

    class OtherOwner(AutoAWQConfig):
        pass

    module, context = activation
    quantization.register_quantization_config("awq")(OtherOwner)
    before = dict(quantization._CUSTOMIZED_METHOD_TO_QUANT_CONFIG)
    with pytest.raises(RuntimeError, match="another owner"):
        module.activate(context)
    assert quantization._CUSTOMIZED_METHOD_TO_QUANT_CONFIG == before
    assert module._published is None and not module._failure


def test_partial_publication_is_poisoned_and_cannot_retry(activation, monkeypatch):
    from hcu_adapter.extensions import ExtensionCommitError
    from vllm.model_executor.layers import quantization

    module, context = activation
    original = quantization.register_quantization_config

    def fail(name):
        if name == "awq_marlin":
            raise RuntimeError("publication interrupted")
        return original(name)

    monkeypatch.setattr(quantization, "register_quantization_config", fail)
    with pytest.raises(ExtensionCommitError, match="publication failed"):
        module.activate(context)
    assert module._failure
    with pytest.raises(ExtensionCommitError, match="activation failed"):
        module.activate(context)


def test_explicit_capability_and_release_are_required(activation):
    module, context = activation
    context.requested_capabilities = set()
    with pytest.raises(RuntimeError, match="explicit"):
        module.activate(context)
    context.vllm_version = "0.32.0"
    with pytest.raises(RuntimeError, match="0.33"):
        module.activate(context)


def test_health_checks_registry_and_compile_policy_ownership(activation, monkeypatch):
    from hcu_adapter.extensions import ExtensionCommitError
    from vllm import envs
    from vllm.model_executor.layers import quantization
    from vllm.model_executor.layers.quantization.auto_awq import AutoAWQConfig

    module, context = activation
    module.activate(context)
    published = quantization._CUSTOMIZED_METHOD_TO_QUANT_CONFIG["awq"]
    quantization._CUSTOMIZED_METHOD_TO_QUANT_CONFIG["awq"] = AutoAWQConfig
    with pytest.raises(ExtensionCommitError, match="registry owner changed"):
        module.check_health()
    quantization._CUSTOMIZED_METHOD_TO_QUANT_CONFIG["awq"] = published
    monkeypatch.setitem(envs.environment_variables, module.FACTOR, lambda: False)
    with pytest.raises(ExtensionCommitError, match="compile policy owner changed"):
        module.check_health()


@pytest.mark.parametrize("m", [17, 128])
@pytest.mark.parametrize("prefill_layout", [False, True])
def test_custom_op_schema_and_fake_rule_are_nonmutating(m, prefill_layout):
    from hcu_vllm_das import autoawq_ops  # noqa: F401
    from torch._subclasses.fake_tensor import FakeTensorMode

    schema = torch.ops.vllm.hcu_lightop_autoawq.default._schema
    assert all(arg.alias_info is None for arg in schema.arguments)
    assert len(schema.returns) == 1 and schema.returns[0].alias_info is None
    with FakeTensorMode():
        x = torch.empty((m, 1024), dtype=torch.float16, device="cuda")
        w = torch.empty((4, 8192 * 32), dtype=torch.int32, device="cuda")
        sz = torch.empty((8192, 8), dtype=torch.int32, device="cuda")
        result = torch.ops.vllm.hcu_lightop_autoawq(x, w, sz, prefill_layout)
        assert result.shape == (m, 8192) and result.dtype == torch.float16
        assert result.device == x.device


@pytest.mark.parametrize(
    "m,prefill_layout,converted",
    [(64, True, False), (65, True, True), (65, False, False)],
)
def test_real_dispatch_uses_vendor_boundary_and_validated_layout(
    monkeypatch, m, prefill_layout, converted
):
    from hcu_vllm_das import autoawq_ops
    calls = []
    a = torch.arange(m * 8, dtype=torch.float16).reshape(m, 8) / 100
    expected_weight = torch.arange(24, dtype=torch.float16).reshape(8, 3) / 100
    packed = torch.empty((1, 3), dtype=torch.int32)
    parameters = torch.empty((3, 1), dtype=torch.int32)
    provider = ModuleType("lightop.gemm_ops")

    def vendor(x, weight, sz):
        assert x is a and weight is packed and sz is parameters
        calls.append("vendor")
        return torch.full((m, 3), 7, dtype=x.dtype)

    def reader(weight, sz, k):
        assert weight is packed and sz is parameters and k == 8
        calls.append("reader")
        return expected_weight

    provider.gemm_awq_w4a16_marlin = vendor
    reader_module = ModuleType("hcu_vllm_das.autoawq_prefill")
    reader_module.dequantize_for_matmul = reader
    monkeypatch.setitem(sys.modules, "lightop.gemm_ops", provider)
    monkeypatch.setitem(sys.modules, "hcu_vllm_das.autoawq_prefill", reader_module)
    output = autoawq_ops._gemm(a, packed, parameters, prefill_layout)
    expected = a @ expected_weight if converted else torch.full((m, 3), 7, dtype=a.dtype)
    torch.testing.assert_close(output, expected, atol=0, rtol=0)
    assert calls == ["reader" if converted else "vendor"]


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("batch_invariant", [False, True])
def test_official_method_selection_and_skip_are_preserved(
    monkeypatch, enabled, batch_invariant
):
    from hcu_vllm_das.autoawq import HcuAutoAWQConfig
    from hcu_vllm_das.autoawq_linear import LightOpAutoAWQLinearMethod
    from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
    from vllm.model_executor.layers.quantization.auto_awq import AutoAWQConfig

    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.setenv("VLLM_HCU_USE_LIGHTOP_AWQ", str(int(enabled)))
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", str(int(batch_invariant)))
    layer = object.__new__(LinearBase)
    torch.nn.Module.__init__(layer)
    config = HcuAutoAWQConfig(4, 128, True, False)
    selected = config.get_quant_method(layer, "model.linear")
    original = AutoAWQConfig.get_quant_method(config, layer, "model.linear")
    if enabled and not batch_invariant:
        assert isinstance(selected, LightOpAutoAWQLinearMethod)
        assert type(selected.delegate) is type(original)
    else:
        assert type(selected) is type(original)
    config.modules_to_not_convert = ["model.linear"]
    assert isinstance(
        config.get_quant_method(layer, "model.linear"), UnquantizedLinearMethod
    )
    assert config.get_quant_method(torch.nn.Module(), "other") is None
