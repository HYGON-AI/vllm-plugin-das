# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Contracts for deterministic EPLB routing simulation."""

from __future__ import annotations

import importlib
from types import ModuleType

import pytest
import torch

from vllm_hcu.model_executor.layers.fused_moe import router_runtime
from vllm_hcu.platforms import envs as henvs


def _strategy(ep_size: int):
    factory = getattr(
        router_runtime,
        "make_hcu_eplb_balancedness_strategy",
        None,
    )
    assert callable(factory), "HCU EPLB balancedness strategy is not implemented"

    class RoutingStrategy:
        pass

    strategy_type = factory(RoutingStrategy)
    return strategy_type(ep_size_getter=lambda: ep_size)


def _rank_balancedness(ids: torch.Tensor, num_experts: int, ep_size: int) -> float:
    experts_per_rank = num_experts // ep_size
    ranks = torch.div(ids.long(), experts_per_rank, rounding_mode="floor")
    loads = torch.bincount(ranks.flatten(), minlength=ep_size).float()
    return float(loads.mean() / loads.max())


@pytest.mark.parametrize("target", [1.0, 0.8, 0.6, 0.4, 0.2])
def test_strategy_generates_requested_ep_rank_balancedness(
    monkeypatch: pytest.MonkeyPatch,
    target: float,
) -> None:
    monkeypatch.setattr(
        henvs,
        "VLLM_HCU_MOE_ROUTING_SIMULATION_BALANCEDNESS",
        target,
        raising=False,
    )
    strategy = _strategy(ep_size=8)
    hidden_states = torch.zeros((4096, 16))
    router_logits = torch.zeros((4096, 256))

    weights, ids = strategy.route_tokens(
        hidden_states,
        router_logits,
        top_k=2,
        indices_type=torch.int32,
    )

    assert weights.shape == (4096, 2)
    assert weights.dtype == torch.float32
    assert torch.equal(weights, torch.ones_like(weights))
    assert ids.shape == (4096, 2)
    assert ids.dtype == torch.int32
    assert ids.min().item() >= 0
    assert ids.max().item() < 256
    assert _rank_balancedness(ids, num_experts=256, ep_size=8) == pytest.approx(
        target,
        abs=1e-3,
    )


def test_strategy_uses_nearest_representable_small_batch_balancedness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        henvs,
        "VLLM_HCU_MOE_ROUTING_SIMULATION_BALANCEDNESS",
        0.8,
        raising=False,
    )
    strategy = _strategy(ep_size=8)

    _, ids = strategy.route_tokens(
        torch.zeros((8, 16)),
        torch.zeros((8, 256)),
        top_k=2,
    )

    # Sixteen assignments have mean rank load 2. The nearest possible
    # max load is 3 (2 / 3), not 2 (1.0).
    assert _rank_balancedness(ids, 256, 8) == pytest.approx(2 / 3)


@pytest.mark.parametrize("num_tokens", [1, 16, 256])
def test_strategy_is_reproducible_for_fake_prefill_and_fixed_mtp_shapes(
    monkeypatch: pytest.MonkeyPatch,
    num_tokens: int,
) -> None:
    monkeypatch.setattr(
        henvs,
        "VLLM_HCU_MOE_ROUTING_SIMULATION_BALANCEDNESS",
        0.6,
        raising=False,
    )
    strategy = _strategy(ep_size=8)
    hidden_states = torch.randn((num_tokens, 16))
    router_logits = torch.randn((num_tokens, 256))

    first = strategy.route_tokens(hidden_states, router_logits, top_k=8)
    second = strategy.route_tokens(
        torch.randn_like(hidden_states),
        torch.randn_like(router_logits),
        top_k=8,
    )

    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])
    assert all(len(set(row.tolist())) == 8 for row in first[1])


@pytest.mark.parametrize(
    ("ep_size", "target", "message"),
    [
        (4, 0.2, r"minimum.*0\.25"),
        (8, 0.0, r"minimum.*0\.125"),
        (8, 1.01, r"at most 1\.0"),
    ],
)
def test_strategy_rejects_unachievable_balancedness(
    monkeypatch: pytest.MonkeyPatch,
    ep_size: int,
    target: float,
    message: str,
) -> None:
    monkeypatch.setattr(
        henvs,
        "VLLM_HCU_MOE_ROUTING_SIMULATION_BALANCEDNESS",
        target,
        raising=False,
    )
    strategy = _strategy(ep_size=ep_size)

    with pytest.raises(ValueError, match=message):
        strategy.route_tokens(
            torch.zeros((32, 8)),
            torch.zeros((32, 64)),
            top_k=2,
        )


def test_strategy_rejects_topk_that_can_duplicate_experts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        henvs,
        "VLLM_HCU_MOE_ROUTING_SIMULATION_BALANCEDNESS",
        0.8,
        raising=False,
    )
    strategy = _strategy(ep_size=8)

    with pytest.raises(ValueError, match=r"top_k=2.*minimum experts per rank=1"):
        strategy.route_tokens(
            torch.zeros((8, 16)),
            torch.zeros((8, 8)),
            top_k=2,
        )


def test_balancedness_environment_is_lazy_and_defaults_to_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    name = "VLLM_HCU_MOE_ROUTING_SIMULATION_BALANCEDNESS"
    henvs.__dict__.pop(name, None)
    monkeypatch.delenv(name, raising=False)
    assert getattr(henvs, name) == 1.0

    monkeypatch.setenv(name, "0.4")
    assert getattr(henvs, name) == 0.4


def test_patch_registers_strategy_once() -> None:
    try:
        patch = importlib.import_module(
            "vllm_hcu.patch.worker.op_opt.moe.patch_routing_simulator"
        )
    except ModuleNotFoundError:
        pytest.fail("routing simulator patch is not implemented")

    class RoutingStrategy:
        pass

    class RoutingSimulator:
        strategies: dict[str, object] = {}

        @classmethod
        def register_strategy(cls, name: str, strategy: object) -> None:
            cls.strategies[name] = strategy

        @classmethod
        def get_available_strategies(cls) -> list[str]:
            return list(cls.strategies)

    module = ModuleType(patch.TARGET_MODULE)
    module.RoutingStrategy = RoutingStrategy
    module.RoutingSimulator = RoutingSimulator

    assert patch.apply_to_module(module) is True
    assert patch.apply_to_module(module) is False
    assert "hcu_eplb_balancedness" in RoutingSimulator.strategies
