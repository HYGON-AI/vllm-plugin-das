# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Live-HCU accuracy for the complete LightOp W16A16 expert pipeline."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from vllm_hcu.model_executor.layers.fused_moe.lightop_w16a16_runtime import (
    pack_lightop_w16a16_weights,
    run_lightop_w16a16,
)


pytestmark = pytest.mark.hcu
_SEED = 20260906


def _reference(
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    outputs: list[torch.Tensor] = []
    for token in range(hidden_states.shape[0]):
        result = torch.zeros(
            hidden_states.shape[1], device=hidden_states.device, dtype=torch.float32
        )
        for slot in range(topk_ids.shape[1]):
            expert = int(topk_ids[token, slot])
            gate_up = F.linear(
                hidden_states[token].float(), w13[expert].float()
            )
            gate, up = gate_up.chunk(2, dim=-1)
            expert_output = F.linear(F.silu(gate) * up, w2[expert].float())
            result.add_(expert_output, alpha=float(topk_weights[token, slot]))
        outputs.append(result)
    return torch.stack(outputs)


def _run_case(
    *,
    experts: int,
    tokens: int,
    hidden: int = 2048,
    intermediate: int = 512,
    topk: int = 8,
    zeros: bool = False,
) -> None:
    generator = torch.Generator(device="cuda").manual_seed(_SEED + experts + tokens)
    if zeros:
        hidden_states = torch.zeros(
            (tokens, hidden), dtype=torch.bfloat16, device="cuda"
        )
        w13 = torch.zeros(
            (experts, 2 * intermediate, hidden),
            dtype=torch.bfloat16,
            device="cuda",
        )
        w2 = torch.zeros(
            (experts, hidden, intermediate),
            dtype=torch.bfloat16,
            device="cuda",
        )
    else:
        hidden_states = torch.randn(
            (tokens, hidden),
            dtype=torch.bfloat16,
            device="cuda",
            generator=generator,
        ).contiguous()
        w13 = (
            torch.randn(
                (experts, 2 * intermediate, hidden),
                dtype=torch.bfloat16,
                device="cuda",
                generator=generator,
            )
            * 0.02
        ).contiguous()
        w2 = (
            torch.randn(
                (experts, hidden, intermediate),
                dtype=torch.bfloat16,
                device="cuda",
                generator=generator,
            )
            * 0.02
        ).contiguous()
    topk_ids = torch.stack(
        [
            torch.randperm(experts, device="cuda", generator=generator)[:topk]
            for _ in range(tokens)
        ]
    ).to(torch.int32)
    raw_weights = torch.rand(
        (tokens, topk), device="cuda", dtype=torch.float32, generator=generator
    )
    topk_weights = raw_weights / raw_weights.sum(dim=-1, keepdim=True)
    sample_indices13 = torch.tensor(
        (0, w13.numel() // 3, w13.numel() - 1), device="cuda"
    )
    sample_indices2 = torch.tensor(
        (0, w2.numel() // 3, w2.numel() - 1), device="cuda"
    )
    source_samples13 = w13.view(-1)[sample_indices13].clone()
    source_samples2 = w2.view(-1)[sample_indices2].clone()
    expected = _reference(hidden_states, w13, w2, topk_weights, topk_ids)

    packed13, packed2, _ = pack_lightop_w16a16_weights(w13, w2)
    workspace13 = torch.empty(
        (tokens * topk, max(2 * intermediate, hidden)),
        dtype=torch.bfloat16,
        device="cuda",
    )
    workspace2 = torch.empty(
        (tokens * topk, intermediate), dtype=torch.bfloat16, device="cuda"
    )
    output = torch.empty_like(hidden_states)
    run_lightop_w16a16(
        output=output,
        hidden_states=hidden_states,
        w13=packed13,
        w2=packed2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        workspace13=workspace13,
        workspace2=workspace2,
        global_num_experts=experts,
    )

    assert torch.isfinite(output).all()
    torch.testing.assert_close(output.float(), expected, rtol=0.03, atol=0.03)
    torch.testing.assert_close(
        w13.view(-1)[sample_indices13], source_samples13, rtol=0, atol=0
    )
    torch.testing.assert_close(
        w2.view(-1)[sample_indices2], source_samples2, rtol=0, atol=0
    )


@pytest.mark.parametrize("tokens", (1, 7, 33, 128))
def test_lightop_w16a16_matches_independent_fp32_reference(tokens: int) -> None:
    _run_case(experts=8, tokens=tokens)


def test_lightop_w16a16_matches_qwen36_expert_shape() -> None:
    _run_case(experts=256, tokens=4)


def test_lightop_w16a16_zero_inputs_are_exact_and_finite() -> None:
    _run_case(experts=8, tokens=4, zeros=True)
