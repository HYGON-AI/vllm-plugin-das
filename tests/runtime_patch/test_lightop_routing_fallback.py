# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm_hcu.model_executor.layers.fused_moe.lightop_routing import (
    lightop_moe_gate_kwargs,
)


def test_legacy_lightop_mode_uses_positional_abi() -> None:
    lightop_moe = SimpleNamespace()

    assert lightop_moe_gate_kwargs(lightop_moe, "sigmoid", True) == {}


@pytest.mark.parametrize(
    ("scoring_func", "renormalize"),
    (("softmax", False), ("sigmoid", False), ("softmax", True)),
)
def test_missing_capability_falls_back_for_unsupported_mode(
    scoring_func: str, renormalize: bool
) -> None:
    lightop_moe = SimpleNamespace()

    assert lightop_moe_gate_kwargs(
        lightop_moe, scoring_func, renormalize
    ) is None


def test_advertised_capability_forwards_routing_options() -> None:
    calls: list[dict[str, object]] = []

    def supports(*, scoring_func: str, renormalize: bool) -> bool:
        calls.append(
            {"scoring_func": scoring_func, "renormalize": renormalize}
        )
        return scoring_func == "softmax" and not renormalize

    lightop_moe = SimpleNamespace(
        supports_moe_fused_gate_routing=supports,
    )

    assert lightop_moe_gate_kwargs(lightop_moe, "softmax", False) == {
        "scoring_func": "softmax",
        "renormalize": False,
    }
    assert calls == [{"scoring_func": "softmax", "renormalize": False}]
    assert lightop_moe_gate_kwargs(lightop_moe, "sigmoid", False) is None


def test_malformed_capability_hook_fails_closed() -> None:
    lightop_moe = SimpleNamespace(
        supports_moe_fused_gate_routing=lambda value: value,
    )

    with pytest.raises(RuntimeError, match="must accept"):
        lightop_moe_gate_kwargs(lightop_moe, "softmax", False)
