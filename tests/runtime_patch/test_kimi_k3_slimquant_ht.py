# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm_hcu.model_executor.layers.quantization.slimquant_w4a8 import (
    SlimQuantW4A8Int8AiterMoEMethod,
)
from vllm_hcu.model_executor.layers.quantization import (
    compressed_tensors_moe_runtime as moe_runtime,
)


def _method(*, backend: str, kimi: bool = True):
    method = object.__new__(SlimQuantW4A8Int8AiterMoEMethod)
    method.moe = SimpleNamespace(
        _hcu_kimi_k3=kimi,
        moe_parallel_config=SimpleNamespace(all2all_backend=backend),
    )
    method.moe_quant_config = None
    return method


def _layer(dtype: torch.dtype = torch.float32) -> nn.Module:
    layer = nn.Module()
    layer.register_parameter(
        "w13_weight_scale",
        nn.Parameter(torch.tensor([[0.25, 0.5]], dtype=dtype), requires_grad=False),
    )
    layer.register_parameter(
        "w2_weight_scale",
        nn.Parameter(torch.tensor([[0.125, 0.375]], dtype=dtype), requires_grad=False),
    )
    layer.register_parameter("w13_input_scale", None)
    layer.register_parameter("w2_input_scale", None)
    return layer


def test_kimi_ht_uses_stable_derived_scale_parameters(monkeypatch):
    method = _method(backend="deepep_high_throughput")
    layer = _layer()
    captured = {}

    def capture_make(*args, **kwargs):
        captured.update(kwargs)
        return kwargs

    monkeypatch.setattr(FusedMoEQuantConfig, "make", capture_make)
    method._prepare_kimi_ht_scales(layer)
    w13_derived = layer._hcu_kimi_ht_w13_weight_scale
    w2_derived = layer._hcu_kimi_ht_w2_weight_scale

    assert isinstance(w13_derived, nn.Parameter)
    assert isinstance(w2_derived, nn.Parameter)
    assert not w13_derived.requires_grad and not w2_derived.requires_grad
    torch.testing.assert_close(w13_derived, layer.w13_weight_scale * 16.0)
    torch.testing.assert_close(w2_derived, layer.w2_weight_scale * 16.0)

    method.get_fused_moe_quant_config(layer)
    assert captured["w1_scale"] is w13_derived
    assert captured["w2_scale"] is w2_derived

    method._prepare_kimi_ht_scales(layer)
    assert layer._hcu_kimi_ht_w13_weight_scale is w13_derived
    assert layer._hcu_kimi_ht_w2_weight_scale is w2_derived
    torch.testing.assert_close(layer.w13_weight_scale, torch.tensor([[0.25, 0.5]]))


@pytest.mark.parametrize(
    ("backend", "kimi"),
    [("deepep_low_latency", True), ("deepep_high_throughput", False)],
)
def test_non_kimi_ht_routes_keep_checkpoint_scales(backend, kimi, monkeypatch):
    method = _method(backend=backend, kimi=kimi)
    layer = _layer()
    captured = {}

    def capture_make(*args, **kwargs):
        captured.update(kwargs)
        return kwargs

    monkeypatch.setattr(FusedMoEQuantConfig, "make", capture_make)
    method._prepare_kimi_ht_scales(layer)
    method.get_fused_moe_quant_config(layer)

    assert not hasattr(layer, "_hcu_kimi_ht_w13_weight_scale")
    assert captured["w1_scale"] is layer.w13_weight_scale
    assert captured["w2_scale"] is layer.w2_weight_scale


def test_kimi_ht_rejects_non_fp32_checkpoint_scales():
    method = _method(backend="deepep_high_throughput")
    with pytest.raises(TypeError, match="checkpoint scales must be FP32"):
        method._prepare_kimi_ht_scales(_layer(torch.float16))


def test_kimi_ht_rejects_replaced_scale_parameters():
    method = _method(backend="deepep_high_throughput")
    layer = _layer()
    method._prepare_kimi_ht_scales(layer)
    layer.w13_weight_scale = nn.Parameter(
        layer.w13_weight_scale.detach().clone(), requires_grad=False
    )

    with pytest.raises(RuntimeError, match="changed after binding"):
        method._prepare_kimi_ht_scales(layer)


def _situ_method(*, kimi: bool = True, beta=1.25, linear_beta=2.0):
    return SimpleNamespace(
        moe=SimpleNamespace(
            _hcu_kimi_k3=kimi,
            hidden_dim=4,
            activation=SimpleNamespace(value="situ"),
            activation_situ_beta=beta,
            activation_situ_linear_beta=linear_beta,
        ),
        moe_quant_config=None,
    )


def _packed_situ_layer():
    return SimpleNamespace(
        w13_weight=torch.zeros((1, 4, 2), dtype=torch.int8),
        w2_weight=torch.zeros((1, 4, 1), dtype=torch.int8),
        global_num_experts=1,
    )


def test_kimi_slimquant_w4a8_metadata_accepts_native_situ_only_for_kimi():
    w1, w2, logical_k, activation = moe_runtime._slimquant_w4a8_metadata(
        _situ_method(), _packed_situ_layer()
    )
    assert w1.shape == (1, 4, 2)
    assert w2.shape == (1, 4, 1)
    assert logical_k == 4
    assert activation == "situ"

    with pytest.raises(RuntimeError, match="only silu activation outside Kimi-K3"):
        moe_runtime._slimquant_w4a8_metadata(
            _situ_method(kimi=False), _packed_situ_layer()
        )


@pytest.mark.parametrize("beta", (0.0, float("inf"), float("nan")))
def test_kimi_situ_metadata_rejects_invalid_beta(beta):
    with pytest.raises(RuntimeError, match="SiTU beta must be finite and positive"):
        moe_runtime._slimquant_w4a8_metadata(
            _situ_method(beta=beta), _packed_situ_layer()
        )


def test_kimi_situ_aiter_call_receives_native_beta_parameters(monkeypatch):
    method = _situ_method()
    method.moe_quant_config = SimpleNamespace(
        w1_scale=torch.ones((1, 4, 1)),
        w2_scale=torch.ones((1, 4, 1)),
        a1_scale=None,
        a2_scale=None,
    )
    layer = _packed_situ_layer()
    config = SimpleNamespace(need_shuffle=False)
    hidden = torch.ones((2, 4))
    topk_weights = torch.ones((2, 1))
    topk_ids = torch.zeros((2, 1), dtype=torch.int32)
    captured = {}

    monkeypatch.setattr(
        moe_runtime,
        "_installed_weight_solution",
        lambda *args, **kwargs: "moe_c",
    )
    monkeypatch.setattr(
        moe_runtime,
        "select_aiter_moe_config",
        lambda *args, **kwargs: config,
    )
    monkeypatch.setattr(
        moe_runtime,
        "_weights_for_selected_config",
        lambda w1, w2, *args, **kwargs: (w1, w2),
    )
    monkeypatch.setattr(
        moe_runtime,
        "_scales_for_selected_config",
        lambda w1, w2, *args, **kwargs: (w1, w2),
    )
    monkeypatch.setattr(
        moe_runtime,
        "aiter_expert_map_for_solution",
        lambda *args, **kwargs: None,
    )

    def execute(config, **kwargs):
        captured.update(kwargs)
        return torch.zeros((2, 4))

    monkeypatch.setattr(moe_runtime, "execute_aiter_moe", execute)
    result = moe_runtime.apply_aiter_w4a8_moe(
        method, layer, hidden, topk_weights, topk_ids
    )

    assert result.shape == (2, 4)
    assert captured["activation"] == "situ"
    assert captured["gemm1_alpha"] == 1.25
    assert captured["gemm1_limit"] == 2.0
