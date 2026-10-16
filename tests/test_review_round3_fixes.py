# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the third PR review round (CPU-only, kernels
mocked): the AITER MoE apply contract of ModelOptW8A16MoEMethod.

P1-A: no AITER backend / triton solution (need_shuffle=False) must raise
      — the expert weights are marlin-shuffled at load time and no raw-int8
      kernel may receive them.
P1-B: the vLLM fallback must forward expert_map / global_num_experts and
      apply_router_weight_on_input (EP correctness, Llama4 semantics).
P2:   the AITER path must reject apply_router_weight_on_input instead of
      silently changing the math.
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
        _self(), layer, x, torch.ones(3, 8), torch.zeros(3, 8, dtype=torch.long),
        None, None)


def test_aiter_no_backend_raises(monkeypatch):
    pytest.importorskip("aiter")
    import vllm_hcu.model_executor.layers.quantization.modelopt_w8a16_hcu as mo

    monkeypatch.setattr(
        mo, "get_aiter_moe_config", lambda **kw: (False, None))
    with pytest.raises(RuntimeError, match="no AITER backend"):
        _call(_layer(), monkeypatch, VLLM_ROCM_USE_AITER_MOE="1")


def test_aiter_triton_solution_raises(monkeypatch):
    pytest.importorskip("aiter")
    import vllm_hcu.model_executor.layers.quantization.modelopt_w8a16_hcu as mo

    monkeypatch.setattr(
        mo, "get_aiter_moe_config",
        lambda **kw: (True, SimpleNamespace(need_shuffle=False)))
    with pytest.raises(RuntimeError, match="triton solution"):
        _call(_layer(), monkeypatch, VLLM_ROCM_USE_AITER_MOE="1")


def test_vllm_fallback_forwards_ep_and_router_weight(monkeypatch):
    import vllm.model_executor.layers.fused_moe.fused_moe as fm

    captured = {}

    def recorder(x, w1, w2, **kw):
        captured.update(kw, x=x, w1=w1, w2=w2)
        return torch.zeros_like(x)

    monkeypatch.setattr(fm, "fused_experts", recorder)
    expert_map = torch.tensor([-1, -1, 0, 1], dtype=torch.int32)
    out = _call(
        _layer(apply_router_weight_on_input=True,
               global_num_experts=4, expert_map=expert_map),
        monkeypatch, VLLM_ROCM_USE_AITER_MOE="0")
    assert out.shape == (3, _K)
    assert captured["expert_map"] is expert_map
    assert captured["global_num_experts"] == 4
    assert captured["apply_router_weight_on_input"] is True


def test_aiter_rejects_router_weight_on_input(monkeypatch):
    pytest.importorskip("aiter")
    with pytest.raises(NotImplementedError,
                       match="apply_router_weight_on_input"):
        _call(_layer(apply_router_weight_on_input=True),
              monkeypatch, VLLM_ROCM_USE_AITER_MOE="1")
