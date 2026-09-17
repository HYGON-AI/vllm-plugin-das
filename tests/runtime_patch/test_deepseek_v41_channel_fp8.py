# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

"""Contract tests for the DeepSeek V4.1 Channel-FP8 routing adapter.

These tests exercise the routing decision only.  They deliberately do not
require an HCU device or a real checkpoint: the adapter is a dispatch layer
whose contract is which quant method is selected for a given (layer, prefix)
pair and which HF ``quantization_config`` schema.
"""

from __future__ import annotations

from types import ModuleType, SimpleNamespace

import pytest
import torch

from vllm_hcu.patch.worker.core_fix import patch_deepseek_v41_channel_fp8
from vllm_hcu.patch.worker.core_fix._common import PatchCompatibilityError


def _module(name: str, **attributes: object) -> ModuleType:
    module = ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


CHANNEL_FP8_QUANT_CONFIG = {
    "activation_scheme": "dynamic",
    "is_checkpoint_fp8_serialized": True,
    "is_per_channel": True,
    "quant_method": "fp8",
}

BLOCK_FP8_QUANT_CONFIG = {
    "activation_scheme": "dynamic",
    "quant_method": "fp8",
    "weight_block_size": [128, 128],
}

TENSOR_FP8_QUANT_CONFIG = {
    "activation_scheme": "dynamic",
    "is_checkpoint_fp8_serialized": True,
    "quant_method": "fp8",
}

MXFP8_QUANT_CONFIG = {
    "activation_scheme": "dynamic",
    "quant_method": "fp8",
    "weight_block_size": [32, 32],
    "expert_dtype": "fp4",
}


class _FakeConfig:
    """Minimal stand-in for the official DeepseekV4FP8Config surface."""

    def __init__(self, config: object = None) -> None:
        self.config = config
        self.seen: list[tuple[object, str]] = []

    def get_quant_method(self, layer, prefix):
        self.seen.append((layer, prefix))
        return ("official", prefix)


def _vllm_config(hf_config: object) -> SimpleNamespace:
    return SimpleNamespace(model_config=SimpleNamespace(hf_config=hf_config))


def _apply(monkeypatch: pytest.MonkeyPatch, quant_config: object) -> _FakeConfig:
    """Install the adapter on a fake config class and return an instance."""

    config_class = type("DeepseekV4FP8Config", (_FakeConfig,), {})
    module = _module(
        patch_deepseek_v41_channel_fp8.TARGET_MODULE,
        DeepseekV4FP8Config=config_class,
    )
    assert patch_deepseek_v41_channel_fp8.apply_to_module(module) is True
    assert patch_deepseek_v41_channel_fp8.apply_to_module(module) is False

    monkeypatch.setattr(
        patch_deepseek_v41_channel_fp8,
        "_hf_quantization_config",
        lambda self: quant_config,
    )
    return config_class()


@pytest.mark.parametrize(
    "quant_config",
    [
        BLOCK_FP8_QUANT_CONFIG,
        TENSOR_FP8_QUANT_CONFIG,
        MXFP8_QUANT_CONFIG,
        {"quant_method": "compressed-tensors"},
        {"quant_method": "fp8", "is_per_channel": False, "activation_scheme": "dynamic"},
        {
            "quant_method": "fp8",
            "is_per_channel": True,
            "activation_scheme": "static",
        },
        {
            "quant_method": "fp8",
            "is_per_channel": True,
            "activation_scheme": "dynamic",
            "weight_block_size": [128, 128],
        },
        None,
    ],
)
def test_non_channel_schemas_keep_official_dispatch(
    monkeypatch: pytest.MonkeyPatch, quant_config: object
) -> None:
    config = _apply(monkeypatch, quant_config)
    layer = torch.nn.Module()
    assert config.get_quant_method(layer, "model.layers.0.attn.wq_a") == (
        "official",
        "model.layers.0.attn.wq_a",
    )
    assert config.seen == [(layer, "model.layers.0.attn.wq_a")]


def test_channel_fp8_schema_is_detected() -> None:
    assert patch_deepseek_v41_channel_fp8._is_channel_fp8_quant_config(
        CHANNEL_FP8_QUANT_CONFIG
    )
    assert not patch_deepseek_v41_channel_fp8._is_channel_fp8_quant_config(
        BLOCK_FP8_QUANT_CONFIG
    )
    assert not patch_deepseek_v41_channel_fp8._is_channel_fp8_quant_config(
        TENSOR_FP8_QUANT_CONFIG
    )


def test_channel_config_dict_is_channel_weights_and_token_activations() -> None:
    config = patch_deepseek_v41_channel_fp8._channel_fp8_config_dict()
    group = config["config_groups"]["group_0"]
    assert group["weights"]["strategy"] == "channel"
    assert group["weights"]["num_bits"] == 8
    assert group["weights"]["dynamic"] is False
    assert group["input_activations"]["strategy"] == "token"
    assert group["input_activations"]["dynamic"] is True
    assert group["targets"] == ["Linear"]
    assert config["ignore"] == ["re:.*attn\\.wo_a.*"]


def test_wo_a_stays_unquantized(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm.model_executor.layers.linear import (
        ColumnParallelLinear,
        UnquantizedLinearMethod,
    )

    config = _apply(monkeypatch, CHANNEL_FP8_QUANT_CONFIG)
    calls: list[str] = []
    monkeypatch.setattr(
        patch_deepseek_v41_channel_fp8,
        "_channel_quant_config",
        lambda self: SimpleNamespace(
            get_quant_method=lambda layer, prefix: calls.append(prefix)
        ),
    )

    class Layer(ColumnParallelLinear):
        pass

    method = config.get_quant_method(
        Layer.__new__(Layer), "model.layers.0.attn.wo_a"
    )
    assert isinstance(method, UnquantizedLinearMethod)
    assert calls == []


def test_routed_experts_use_channel_scheme(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm.model_executor.layers.fused_moe import RoutedExperts

    config = _apply(monkeypatch, CHANNEL_FP8_QUANT_CONFIG)
    sentinel = object()
    calls: list[str] = []

    def get_quant_method(layer, prefix):
        calls.append(prefix)
        return sentinel

    monkeypatch.setattr(
        patch_deepseek_v41_channel_fp8,
        "_channel_quant_config",
        lambda self: SimpleNamespace(get_quant_method=get_quant_method),
    )
    layer = RoutedExperts.__new__(RoutedExperts)
    assert config.get_quant_method(layer, "model.layers.0.ffn.experts") is sentinel
    assert calls == ["model.layers.0.ffn.experts"]


def test_linear_layers_use_channel_scheme(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm.model_executor.layers.linear import ColumnParallelLinear

    config = _apply(monkeypatch, CHANNEL_FP8_QUANT_CONFIG)
    sentinel = object()
    calls: list[str] = []
    monkeypatch.setattr(
        patch_deepseek_v41_channel_fp8,
        "_channel_quant_config",
        lambda self: SimpleNamespace(
            get_quant_method=lambda layer, prefix: (
                calls.append(prefix) or sentinel
            )
        ),
    )

    class Layer(ColumnParallelLinear):
        pass

    assert (
        config.get_quant_method(Layer.__new__(Layer), "model.layers.0.attn.wq_b")
        is sentinel
    )
    assert calls == ["model.layers.0.attn.wq_b"]


def test_channel_decision_is_latched(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _apply(monkeypatch, CHANNEL_FP8_QUANT_CONFIG)
    monkeypatch.setattr(
        patch_deepseek_v41_channel_fp8,
        "_channel_quant_config",
        lambda self: SimpleNamespace(get_quant_method=lambda layer, prefix: "ct"),
    )
    from vllm.model_executor.layers.linear import ColumnParallelLinear

    class Layer(ColumnParallelLinear):
        pass

    assert config.get_quant_method(Layer.__new__(Layer), "a.wq_a") == "ct"
    assert (
        getattr(config, patch_deepseek_v41_channel_fp8._IS_CHANNEL_ATTR) is True
    )


def test_missing_target_symbol_is_rejected() -> None:
    module = _module(patch_deepseek_v41_channel_fp8.TARGET_MODULE)
    with pytest.raises(PatchCompatibilityError, match="is missing"):
        patch_deepseek_v41_channel_fp8.apply_to_module(module)


def test_incompatible_signature_is_rejected() -> None:
    class DeepseekV4FP8Config:
        def get_quant_method(self, layer):
            return None

    module = _module(
        patch_deepseek_v41_channel_fp8.TARGET_MODULE,
        DeepseekV4FP8Config=DeepseekV4FP8Config,
    )
    with pytest.raises(PatchCompatibilityError, match="incompatible signature"):
        patch_deepseek_v41_channel_fp8.apply_to_module(module)


def test_hf_config_read_is_failure_tolerant(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _apply(monkeypatch, CHANNEL_FP8_QUANT_CONFIG)
    monkeypatch.setattr(
        patch_deepseek_v41_channel_fp8,
        "_hf_quantization_config",
        lambda self: None,
    )
    layer = torch.nn.Module()
    assert config.get_quant_method(layer, "model.layers.0.attn.wq_a") == (
        "official",
        "model.layers.0.attn.wq_a",
    )
