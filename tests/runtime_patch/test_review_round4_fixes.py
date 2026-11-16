# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the fourth PR review round (CPU-only, kernels
mocked).

P2-1: swiglu_limit must be rejected (no execution path applies the clamp;
      running without it changes the math for clamped-SwiGLU models).
P2-2: ModelOptMixedPrecisionConfig overrides get_quant_method — the
      base-class wrapper does not reach its RoutedExperts branch; the
      mixed config now has its own W8A16 interception.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_hcu.model_executor.layers.quantization.modelopt_w8a16_hcu import (
    ModelOptW8A16MoEMethod,
)

_E, _N, _K = 4, 512, 2048


def _layer(**over):
    layer = SimpleNamespace(
        w13_weight=torch.zeros(_E, 2 * _N, _K, dtype=torch.int8),
        w2_weight=torch.zeros(_E, _K, _N, dtype=torch.int8),
        activation="silu",
        apply_router_weight_on_input=False,
        global_num_experts=_E,
        expert_map=None,
        moe_config=SimpleNamespace(dp_size=1),
    )
    for k, v in over.items():
        setattr(layer, k, v)
    return layer


def _self():
    return SimpleNamespace(moe_quant_config=None)


def _call(layer, monkeypatch, **env):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    x = torch.zeros(3, _K)
    return ModelOptW8A16MoEMethod.apply(
        _self(), layer, x, torch.ones(3, 8),
        torch.zeros(3, 8, dtype=torch.long), None, None)


def test_swiglu_limit_rejected_on_aiter_path(monkeypatch):
    pytest.importorskip("aiter")
    layer = _layer(moe_config=SimpleNamespace(dp_size=1, swiglu_limit=7.0))
    with pytest.raises(NotImplementedError, match="swiglu_limit"):
        _call(layer, monkeypatch, VLLM_ROCM_USE_AITER_MOE="1")


def test_swiglu_limit_rejected_on_vllm_fallback(monkeypatch):
    layer = _layer(moe_config=SimpleNamespace(dp_size=1, swiglu_limit=7.0))
    with pytest.raises(NotImplementedError, match="swiglu_limit"):
        _call(layer, monkeypatch, VLLM_ROCM_USE_AITER_MOE="0")


def test_no_swiglu_limit_still_works(monkeypatch):
    """swiglu_limit=None (default) must not trigger the rejection."""
    import vllm.model_executor.layers.fused_moe.fused_moe as fm
    monkeypatch.setattr(fm, "fused_experts",
                        lambda *a, **kw: torch.zeros_like(a[0]))
    layer = _layer()
    out = _call(layer, monkeypatch, VLLM_ROCM_USE_AITER_MOE="0")
    assert out.shape == (3, _K)


def test_mixed_precision_w8a16_moe_interception():
    pytest.importorskip("vllm")
    import vllm.model_executor.layers.quantization.modelopt as mo

    # Force-apply the patch (the import coordinator may not trigger for
    # already-imported modules in the pytest environment; apply_to_module
    # is idempotent via its marker).
    from vllm_hcu.patch.worker.op_opt.patch_modelopt_w8a16 import (
        apply_to_module,
    )
    apply_to_module(mo)

    mixed_cls = getattr(mo, "ModelOptMixedPrecisionConfig", None)
    if mixed_cls is None:
        pytest.skip("ModelOptMixedPrecisionConfig not in this build")

    cfg = mixed_cls.__new__(mixed_cls)
    cfg.is_layer_excluded = lambda prefix: False
    cfg._resolve_quant_algo = lambda prefix: "W8A16"

    class RoutedExperts:
        moe_config = SimpleNamespace(tp_rank=0, tp_size=1)

    result = cfg.get_quant_method(
        RoutedExperts(), "model.layers.0.mlp.experts")
    assert isinstance(result, ModelOptW8A16MoEMethod), (
        f"mixed W8A16 MoE expert must route to ModelOptW8A16MoEMethod, "
        f"got {type(result)}")

    # Excluded prefix must NOT intercept.
    cfg.is_layer_excluded = lambda prefix: True
    result2 = cfg.get_quant_method(
        RoutedExperts(), "model.layers.0.mlp.experts")
    assert not isinstance(result2, ModelOptW8A16MoEMethod)

    # Non-W8A16 algo must keep upstream dispatch.
    cfg.is_layer_excluded = lambda prefix: False
    cfg._resolve_quant_algo = lambda prefix: "FP8"
    cfg.fp8_config = None
    result3 = cfg.get_quant_method(
        RoutedExperts(), "model.layers.0.mlp.experts")
    assert not isinstance(result3, ModelOptW8A16MoEMethod)
