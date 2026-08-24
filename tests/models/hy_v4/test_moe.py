# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

import vllm_hcu.models.hy_v4.moe as moe


class _FakeGate(nn.Module):
    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, None]:
        logits = torch.arange(
            hidden_states.shape[0] * 16,
            device=hidden_states.device,
            dtype=torch.float32,
        ).reshape(hidden_states.shape[0], 16)
        return logits, None


class _FakeExperts(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.last_router_logits: torch.Tensor | None = None

    def forward(
        self,
        *,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor:
        self.last_router_logits = router_logits
        return hidden_states + 1


class _FakeSharedExperts(nn.Module):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states


def _hf_config() -> SimpleNamespace:
    return SimpleNamespace(
        hidden_size=32,
        hidden_act="silu",
        num_experts=16,
        num_experts_per_tok=4,
        expert_hidden_dim=24,
        num_shared_experts=1,
        route_norm=True,
        router_scaling_factor=2.827,
        swiglu_limit=10.0,
    )


def _vllm_config(backend: str) -> SimpleNamespace:
    return SimpleNamespace(
        kernel_config=SimpleNamespace(moe_backend=backend),
        parallel_config=SimpleNamespace(
            eplb_config=SimpleNamespace(num_redundant_experts=0)
        ),
    )


def test_hy_v4_rejects_non_triton_moe_backend() -> None:
    with pytest.raises(RuntimeError, match="--moe-backend triton"):
        moe.HYV4MoEFused(
            config=_hf_config(),
            vllm_config=_vllm_config("auto"),
            prefix="model.layers.1.mlp",
        )


def test_hy_v4_moe_preserves_router_and_clamp_contract(monkeypatch) -> None:
    fused_kwargs: dict[str, object] = {}
    fake_experts = _FakeExperts()
    fake_group = SimpleNamespace(size=lambda: 8)
    monkeypatch.setattr(moe, "get_tensor_model_parallel_world_size", lambda: 8)
    monkeypatch.setattr(
        moe,
        "get_ep_group",
        lambda: SimpleNamespace(device_group=fake_group, rank_in_group=2),
    )
    monkeypatch.setattr(moe, "GateLinear", lambda *args, **kwargs: _FakeGate())
    monkeypatch.setattr(
        moe,
        "HYV4FeedForward",
        lambda *args, **kwargs: _FakeSharedExperts(),
    )

    def fake_fused_moe(**kwargs):
        fused_kwargs.update(kwargs)
        return fake_experts

    monkeypatch.setattr(moe, "FusedMoE", fake_fused_moe)

    layer = moe.HYV4MoEFused(
        config=_hf_config(),
        vllm_config=_vllm_config("triton"),
        prefix="model.layers.1.mlp",
    )

    assert layer.expert_bias.dtype == torch.float32
    assert layer.physical_expert_start == 4
    assert layer.physical_expert_end == 6
    assert fused_kwargs["scoring_func"] == "sigmoid"
    assert fused_kwargs["renormalize"] is True
    assert fused_kwargs["routed_scaling_factor"] == 2.827
    assert fused_kwargs["swiglu_limit"] == 10.0
    assert fused_kwargs["use_grouped_topk"] is True
    assert fused_kwargs["num_expert_group"] == 1
    assert fused_kwargs["topk_group"] == 1
    assert fused_kwargs["e_score_correction_bias"] is layer.expert_bias
    assert fused_kwargs["shared_experts"] is layer.shared_experts

    hidden = torch.zeros(2, 3, 32)
    actual = layer(hidden)
    assert actual.shape == hidden.shape
    torch.testing.assert_close(actual, hidden + 1)
    assert fake_experts.last_router_logits is not None
    assert fake_experts.last_router_logits.dtype == torch.float32
