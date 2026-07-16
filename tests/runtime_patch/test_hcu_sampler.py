# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import sys
from types import ModuleType, SimpleNamespace

os.environ.setdefault("VLLM_PLUGINS", "__disabled__")

import pytest
import torch
from vllm.v1.sample.ops.topk_topp_sampler import TopKTopPSampler

from vllm_hcu.ops import topk_topp_sample
from vllm_hcu.ops.topk_topp_sample import HcuSampler, HcuTopKTopPSampler


def test_hcu_sampler_keeps_official_topk_topp_when_custom_sampler_disabled(
    monkeypatch,
):
    monkeypatch.setattr(topk_topp_sample.henvs, "VLLM_HCU_USE_CUSTOM_OPS", True)
    monkeypatch.setattr(
        topk_topp_sample.henvs,
        "VLLM_HCU_USE_CUSTOM_TOPK_TOPP_SAMPLER",
        False,
    )

    sampler = HcuSampler()

    assert type(sampler.topk_topp_sampler) is TopKTopPSampler


def test_hcu_sampler_uses_hcu_topk_topp_only_when_explicitly_enabled(
    monkeypatch,
):
    monkeypatch.setattr(topk_topp_sample.henvs, "VLLM_HCU_USE_CUSTOM_OPS", True)
    monkeypatch.setattr(
        topk_topp_sample.henvs,
        "VLLM_HCU_USE_CUSTOM_TOPK_TOPP_SAMPLER",
        True,
    )

    sampler = HcuSampler()

    assert type(sampler.topk_topp_sampler) is HcuTopKTopPSampler


def test_hcu_topk_topp_forward_is_feature_gated_and_lazy(monkeypatch):
    sampler = HcuTopKTopPSampler()
    native_calls = []
    monkeypatch.setattr(
        sampler,
        "forward_native",
        lambda logits, generators, k, p: (
            native_calls.append((logits, generators, k, p)) or torch.tensor([7]),
            None,
        ),
    )
    logits = torch.zeros(1, 8)
    k = torch.tensor([2])

    monkeypatch.delitem(sys.modules, "lightop", raising=False)
    monkeypatch.setattr(topk_topp_sample.henvs, "VLLM_HCU_USE_CUSTOM_OPS", False)
    monkeypatch.setattr(
        topk_topp_sample.henvs,
        "VLLM_HCU_USE_CUSTOM_TOPK_TOPP_SAMPLER",
        True,
    )
    token_ids, _ = sampler(logits, {}, k, None)
    assert token_ids.tolist() == [7]
    assert len(native_calls) == 1
    assert "lightop" not in sys.modules

    lightop = ModuleType("lightop")
    custom_calls = []
    lightop.sampling = SimpleNamespace(
        top_k_top_p_sampling_from_probs=lambda probs, top_k, top_p, deterministic: (
            custom_calls.append((probs, top_k, top_p, deterministic))
            or torch.tensor([[3]])
        )
    )
    monkeypatch.setitem(sys.modules, "lightop", lightop)
    monkeypatch.setattr(topk_topp_sample.henvs, "VLLM_HCU_USE_CUSTOM_OPS", True)
    token_ids, _ = sampler(logits, {}, k, None)
    assert token_ids.tolist() == [3]
    assert len(custom_calls) == 1
    assert custom_calls[0][3] is True

    monkeypatch.setitem(sys.modules, "lightop", ModuleType("lightop"))
    with pytest.raises(RuntimeError, match="lightop is unavailable"):
        sampler(logits, {}, k, None)

    token_ids, _ = sampler(logits, {0: torch.Generator()}, k, None)
    assert token_ids.tolist() == [7]
    assert len(native_calls) == 2
