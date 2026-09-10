# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from types import SimpleNamespace

import torch

from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (
    compressed_tensors_moe_w8a8_fp8 as target_fp8,
)
from vllm_hcu.model_executor.layers.quantization.compressed_tensors import (
    compressed_tensors_marlin as marlin_config,
)
from vllm_hcu.model_executor.layers.quantization.compressed_tensors import (
    compressed_tensors_moe_marlin as marlin,
)


def test_explicit_aiter_bypasses_slimquant_marlin_selector(
    monkeypatch,
) -> None:
    from vllm.model_executor.layers.fused_moe import RoutedExperts
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe import (
        CompressedTensorsMoEMethod,
    )

    layer = RoutedExperts.__new__(RoutedExperts)
    torch.nn.Module.__init__(layer)
    layer.moe_config = SimpleNamespace(moe_backend="aiter")
    config = SimpleNamespace(ignore=[], packed_modules_mapping={})
    expected = object()
    calls = []

    def official_selector(quant_config, routed_experts, layer_name):
        calls.append((quant_config, routed_experts, layer_name))
        return expected

    monkeypatch.setattr(
        CompressedTensorsMoEMethod,
        "get_moe_method",
        staticmethod(official_selector),
    )
    monkeypatch.setattr(
        marlin.CompressedTensorsMarlinMoEMethod,
        "get_moe_method",
        staticmethod(
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("explicit AITER must bypass SlimQuant Marlin")
            )
        ),
    )

    method = marlin_config.SlimQuantCompressedTensorsMarlinConfig.get_quant_method(
        config,
        layer,
        "model.layers.0.mlp.experts",
    )

    assert method is expected
    assert calls == [(config, layer, "model.layers.0.mlp.experts")]


def test_explicit_aiter_precedes_slimquant_marlin(
    monkeypatch,
) -> None:
    weight_quant = object()
    input_quant = object()
    original_moe = SimpleNamespace(moe_backend="aiter")
    layer = SimpleNamespace(moe_config=original_moe)

    class QuantConfig:
        @staticmethod
        def _add_fused_moe_to_target_scheme_map():
            pass

        @staticmethod
        def get_scheme_dict(_layer, _layer_name):
            return {
                "weights": weight_quant,
                "input_activations": input_quant,
            }

        @staticmethod
        def _is_fp8_w8a8(weight, activation):
            return weight is weight_quant and activation is input_quant

        @staticmethod
        def _is_dynamic_token_w8a8(_weight, _activation):
            return False

    class TargetFp8Method:
        def __init__(self, weight, activation, moe, layer_name=None):
            self.weight = weight
            self.activation = activation
            self.moe = moe
            self.layer_name = layer_name

    monkeypatch.setattr(
        marlin,
        "is_lightop_marlin_moe_supported",
        lambda _moe: (_ for _ in ()).throw(
            AssertionError("explicit AITER must bypass LightOp config lookup")
        ),
    )
    monkeypatch.setattr(
        target_fp8,
        "CompressedTensorsW8A8Fp8MoEMethod",
        TargetFp8Method,
    )

    method = marlin.CompressedTensorsMarlinMoEMethod.get_moe_method(
        QuantConfig(),
        layer,
        "model.layers.0.mlp.experts",
    )

    assert isinstance(method, TargetFp8Method)
    assert method.weight is weight_quant
    assert method.activation is input_quant
    assert method.layer_name == "model.layers.0.mlp.experts"
    assert method.moe is original_moe
    assert original_moe.moe_backend == "aiter"


def test_slimquant_only_config_miss_selects_target_triton(
    monkeypatch,
) -> None:
    weight_quant = object()
    input_quant = object()
    original_moe = SimpleNamespace(moe_backend="auto")
    layer = SimpleNamespace(moe_config=original_moe)

    class QuantConfig:
        @staticmethod
        def _add_fused_moe_to_target_scheme_map():
            pass

        @staticmethod
        def get_scheme_dict(_layer, _layer_name):
            return {
                "weights": weight_quant,
                "input_activations": input_quant,
            }

        @staticmethod
        def _is_fp8_w8a8(weight, activation):
            return weight is weight_quant and activation is input_quant

        @staticmethod
        def _is_dynamic_token_w8a8(_weight, _activation):
            return False

    class TargetFp8Method:
        def __init__(self, weight, activation, moe, layer_name=None):
            self.weight = weight
            self.activation = activation
            self.moe = moe
            self.layer_name = layer_name

    monkeypatch.setattr(
        marlin,
        "is_lightop_marlin_moe_supported",
        lambda _moe: False,
    )
    monkeypatch.setattr(
        target_fp8,
        "CompressedTensorsW8A8Fp8MoEMethod",
        TargetFp8Method,
    )

    method = marlin.CompressedTensorsMarlinMoEMethod.get_moe_method(
        QuantConfig(),
        layer,
        "model.layers.0.mlp.experts",
    )

    assert isinstance(method, TargetFp8Method)
    assert method.weight is weight_quant
    assert method.activation is input_quant
    assert method.layer_name == "model.layers.0.mlp.experts"
    assert method.moe is not original_moe
    assert method.moe.moe_backend == "triton"
    assert original_moe.moe_backend == "auto"


def test_slimquant_adds_official_fused_target_before_scheme_lookup(
    monkeypatch,
) -> None:
    weight_quant = object()
    input_quant = object()

    class QuantConfig:
        fused_target_added = False

        @classmethod
        def _add_fused_moe_to_target_scheme_map(cls):
            cls.fused_target_added = True

        @classmethod
        def get_scheme_dict(cls, _layer, _layer_name):
            assert cls.fused_target_added
            return {
                "weights": weight_quant,
                "input_activations": input_quant,
            }

        @staticmethod
        def _is_fp8_w8a8(weight, activation):
            return weight is weight_quant and activation is input_quant

        @staticmethod
        def _is_dynamic_token_w8a8(_weight, _activation):
            return False

    class LightOpMethod:
        def __init__(self, quant_config, moe, scheme_dict):
            self.quant_config = quant_config
            self.moe = moe
            self.scheme_dict = scheme_dict

    layer = SimpleNamespace(moe_config=SimpleNamespace(moe_backend="auto"))
    monkeypatch.setattr(
        marlin,
        "is_lightop_marlin_moe_supported",
        lambda _moe: True,
    )
    monkeypatch.setattr(
        marlin,
        "CompressedTensorsW8A8FP8MarlinMoEMethod",
        LightOpMethod,
    )

    method = marlin.CompressedTensorsMarlinMoEMethod.get_moe_method(
        QuantConfig(),
        layer,
        "language_model.model.layers.3.mlp.experts",
    )

    assert isinstance(method, LightOpMethod)
    assert QuantConfig.fused_target_added
