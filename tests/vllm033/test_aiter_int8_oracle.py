from __future__ import annotations

import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


PLUGIN_SOURCE_ROOT = (
    Path(__file__).resolve().parents[2] / "adapter_extension" / "src"
)
sys.path.insert(0, os.fspath(PLUGIN_SOURCE_ROOT))

from hcu_vllm_das.moe import int8_oracle as oracle


def fake_oracle_module() -> ModuleType:
    module = ModuleType(oracle.TARGET_MODULE)
    exec(
        """
from enum import Enum

class Int8MoeBackend(Enum):
    TRITON = "TRITON"
    HUMMING = "HUMMING"
    CPU = "CPU"

def _get_priority_backends(config):
    return [Int8MoeBackend.TRITON, Int8MoeBackend.HUMMING, Int8MoeBackend.CPU]

def backend_to_kernel_cls(backend):
    if backend in (
        Int8MoeBackend.TRITON,
        Int8MoeBackend.HUMMING,
        Int8MoeBackend.CPU,
    ):
        return [backend.value.lower()]
    raise ValueError(f"Unknown Int8 MoE backend: {backend!r}")

def map_int8_backend(runner_backend):
    mapping = {"triton": Int8MoeBackend.TRITON, "humming": Int8MoeBackend.HUMMING}
    if runner_backend in mapping:
        return mapping[runner_backend]
    raise ValueError(f"unsupported: {runner_backend}")

def select_int8_moe_backend(config, weight_key="int8-channel", activation_key="int8-token"):
    backend = (
        _get_priority_backends(config)[0]
        if config.moe_backend == "auto"
        else map_int8_backend(config.moe_backend)
    )
    cls = backend_to_kernel_cls(backend)[0]
    if isinstance(cls, str):
        return backend, cls
    supported, reason = cls.is_supported_config(
        cls, config, weight_key, activation_key, "standard"
    )
    if not supported:
        raise ValueError(reason)
    return backend, cls

def convert_to_int8_moe_kernel_format(int8_backend, w13, w2, layer=None, w13_scale=None):
    if int8_backend not in (
        Int8MoeBackend.TRITON,
        Int8MoeBackend.HUMMING,
        Int8MoeBackend.CPU,
    ):
        raise ValueError(f"Unsupported Int8 MoE backend: {int8_backend!r}")
    return (f"converted:{int8_backend.value}:{w13}", f"converted:{int8_backend.value}:{w2}")

def make_int8_moe_quant_config(int8_backend, w1_scale, w2_scale, a1_scale=None, a2_scale=None, w1_bias=None, w2_bias=None, per_act_token_quant=False, layer=None):
    return ("official-quant", int8_backend, w1_scale, w2_scale, per_act_token_quant)

def int8_w8a8_moe_quant_config(*args, **kwargs):
    return ("official-w8a8", args, kwargs)

def make_int8_moe_kernel(int8_backend, moe_quant_config, moe_config, experts_cls, routing_tables=None):
    return ("official-kernel", int8_backend, moe_quant_config, experts_cls)
""",
        module.__dict__,
    )
    return module


class FakeAiterExperts:
    consumes_expert_mask = True

    def __init__(self, *, moe_config: object, quant_config: object, **kwargs: object):
        self.moe_config = moe_config
        self.quant_config = quant_config

    @staticmethod
    def _supports_quant_scheme(weight_key: object, activation_key: object) -> bool:
        return (weight_key, activation_key) == ("fp8-channel", "fp8-token")

    @staticmethod
    def is_supported_config(
        cls: type,
        moe_config: object,
        weight_key: object,
        activation_key: object,
        activation_format: object,
    ) -> tuple[bool, str]:
        supported = cls._supports_quant_scheme(weight_key, activation_key)
        return supported, "unsupported quant scheme"

    def apply(self, *args: object, **kwargs: object) -> str:
        return "official-parent"


def fake_imports(target: ModuleType) -> dict[str, object]:
    return {
        oracle.TARGET_MODULE: target,
        oracle.EXPERTS_MODULE: SimpleNamespace(AiterExperts=FakeAiterExperts),
        oracle.QUANT_UTILS_MODULE: SimpleNamespace(
            kInt8StaticChannelSym="int8-channel",
            kInt8DynamicTokenSym="int8-token",
        ),
    }


def standalone_fake(monkeypatch, target=None):
    target = fake_oracle_module() if target is None else target
    cls = oracle._make_experts_class(FakeAiterExperts, "int8-channel", "int8-token")
    monkeypatch.setattr(oracle, "HcuAiterInt8Experts", cls)
    return target


def test_plugin_expert_adds_only_int8_channel_token_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    standalone_fake(monkeypatch)
    experts = oracle.HcuAiterInt8Experts

    assert experts.consumes_expert_mask is False
    assert experts._supports_quant_scheme("int8-channel", "int8-token") is True
    assert experts._supports_quant_scheme("fp8-channel", "fp8-token") is False
    assert experts._supports_quant_scheme("int8-channel", "static-int8") is False


def test_explicit_aiter_rejects_expert_bias_before_weight_conversion(monkeypatch):
    target = standalone_fake(monkeypatch)
    experts = oracle.HcuAiterInt8Experts
    supported, reason = experts.is_supported_config(
        experts, SimpleNamespace(has_bias=True), "int8-channel", "int8-token", "standard"
    )
    assert not supported and "expert bias" in reason
    # The same configuration can still be delegated to the official backend.
    backend, experts = target.select_int8_moe_backend(
        SimpleNamespace(moe_backend="triton", has_bias=True)
    )
    assert backend is target.Int8MoeBackend.TRITON
    assert experts == "triton"


@pytest.mark.parametrize("activation", ("situ", "swigluoai", "swigluoai_uninterleave"))
def test_plugin_expert_does_not_inherit_unforwarded_activation_parameters(
    monkeypatch, activation
):
    standalone_fake(monkeypatch)
    experts = oracle.HcuAiterInt8Experts
    assert experts._supports_activation(SimpleNamespace(value=activation)) is False
    assert experts._supports_activation(SimpleNamespace(value="silu")) is True
    assert experts._supports_activation(SimpleNamespace(value="gelu")) is True


def test_plugin_expert_delegates_int8_execution_to_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    standalone_fake(monkeypatch)
    calls: list[dict[str, object]] = []
    result = object()

    monkeypatch.setattr(
        oracle,
        "apply_aiter_int8_moe",
        lambda **kwargs: calls.append(kwargs) or result,
    )

    class Output:
        dtype = "bf16"

        def __init__(self) -> None:
            self.value: object | None = None

        def copy_(self, value: object) -> None:
            self.value = value

    moe_config = SimpleNamespace(num_experts=8)
    quant_config = SimpleNamespace(use_int8_w8a8=True)
    instance = oracle.HcuAiterInt8Experts(
        moe_config=moe_config,
        quant_config=quant_config,
    )
    output = Output()
    instance.apply(
        output,
        "hidden",
        "w1",
        "w2",
        "topk-weights",
        "topk-ids",
        "silu",
        8,
        "expert-map",
        None,
        None,
        "workspace13",
        "workspace2",
        None,
        False,
    )

    assert output.value is result
    assert calls == [
        {
            "hidden_states": "hidden",
            "w1": "w1",
            "w2": "w2",
            "topk_weights": "topk-weights",
            "topk_ids": "topk-ids",
            "moe_config": moe_config,
            "quant_config": quant_config,
            "activation": "silu",
            "expert_map": "expert-map",
            "output_dtype": "bf16",
        }
    ]


@pytest.mark.parametrize("consumer", oracle.INT8_CONSUMER_MODULES)
def test_retired_install_does_not_modify_or_import_official_oracle(monkeypatch, consumer):
    target = fake_oracle_module()
    before = vars(target).copy()
    monkeypatch.setitem(sys.modules, oracle.TARGET_MODULE, target)
    monkeypatch.setitem(sys.modules, consumer, ModuleType(consumer))
    with pytest.raises(oracle.Int8OracleInstallError, match="public expert registration"):
        oracle.install()
    assert vars(target) == before
    assert not getattr(target, oracle.INSTALL_MARKER, False)
    with pytest.raises(ValueError, match="unsupported"):
        target.map_int8_backend("aiter")
