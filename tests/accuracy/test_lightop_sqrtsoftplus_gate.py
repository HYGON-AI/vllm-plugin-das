# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

"""Live-HCU accuracy for LightOp DeepSeek V4 sqrt-softplus routing."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from vllm_hcu.model_executor.layers.fused_moe.sqrtsoftplus_routing import (
    run_lightop_sqrtsoftplus,
)


pytestmark = pytest.mark.hcu
_SEED = 20260906


def _reference(
    logits: torch.Tensor,
    bias: torch.Tensor,
    *,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = torch.sqrt(F.softplus(logits.float()))
    choice_scores = scores + bias.float().unsqueeze(0)
    ids = torch.topk(choice_scores, k=topk, dim=-1, sorted=True).indices
    weights = scores.gather(1, ids)
    if renormalize:
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-20)
    weights = weights * routed_scaling_factor
    return weights.float(), ids.to(torch.int32)


def _sort_by_id(
    weights: torch.Tensor,
    ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    sorted_ids, order = ids.sort(dim=-1)
    return weights.gather(1, order), sorted_ids


@pytest.mark.parametrize("num_tokens", (1, 33, 128))
@pytest.mark.parametrize("num_experts", (256, 384))
@pytest.mark.parametrize("topk", (6, 8, 16))
@pytest.mark.parametrize("renormalize", (True, False))
@pytest.mark.parametrize("routed_scaling_factor", (1.0, 1.5))
def test_lightop_sqrtsoftplus_matches_independent_fp32_reference(
    num_tokens: int,
    num_experts: int,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: float,
) -> None:
    generator = torch.Generator(device="cuda").manual_seed(_SEED)
    logits = torch.randn(
        (num_tokens, num_experts),
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    ).contiguous()
    bias = torch.randn(
        (num_experts,),
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    ).contiguous()
    logits_before = logits.clone()
    bias_before = bias.clone()

    expected_weights, expected_ids = _reference(
        logits,
        bias,
        topk=topk,
        renormalize=renormalize,
        routed_scaling_factor=routed_scaling_factor,
    )
    actual_weights, actual_ids = run_lightop_sqrtsoftplus(
        logits,
        bias,
        topk=topk,
        renormalize=renormalize,
        routed_scaling_factor=routed_scaling_factor,
        indices_dtype=torch.int32,
    )
    expected_weights, expected_ids = _sort_by_id(expected_weights, expected_ids)
    actual_weights, actual_ids = _sort_by_id(actual_weights, actual_ids)

    torch.testing.assert_close(actual_ids, expected_ids, rtol=0, atol=0)
    torch.testing.assert_close(
        actual_weights,
        expected_weights,
        rtol=1e-5,
        atol=1e-6,
    )
    assert torch.isfinite(actual_weights).all()
    torch.testing.assert_close(logits, logits_before, rtol=0, atol=0)
    torch.testing.assert_close(bias, bias_before, rtol=0, atol=0)


@pytest.mark.parametrize("value", (-80.0, 0.0, 80.0))
def test_lightop_sqrtsoftplus_constant_inputs_are_finite_and_normalized(
    value: float,
) -> None:
    logits = torch.full((4, 256), value, dtype=torch.float32, device="cuda")
    bias = torch.zeros((256,), dtype=torch.float32, device="cuda")

    weights, ids = run_lightop_sqrtsoftplus(
        logits,
        bias,
        topk=8,
        renormalize=True,
        routed_scaling_factor=1.5,
        indices_dtype=torch.int64,
    )

    assert ids.dtype is torch.int64
    assert torch.isfinite(weights).all()
    torch.testing.assert_close(
        weights.sum(dim=-1),
        torch.full((4,), 1.5, dtype=torch.float32, device="cuda"),
        rtol=1e-5,
        atol=1e-6,
    )
    assert ((ids >= 0) & (ids < 256)).all()
    assert all(row.unique().numel() == 8 for row in ids.cpu())
