# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

"""Live-HCU accuracy for categorized LightOp MLA decode concatenation."""

from __future__ import annotations

import pytest
import torch

from vllm_hcu.model_executor.layers.attention.lightop_concat_runtime import (
    is_lightop_mla_decode_concat_eligible,
    lightop_mla_decode_concat_impl,
)


pytestmark = pytest.mark.hcu


def _require_hcu() -> None:
    if not torch.cuda.is_available():
        pytest.skip("a live HCU/ROCm device is required")
    if not hasattr(torch.cuda.get_device_properties(0), "gcnArchName"):
        pytest.skip("the active device is not HCU/ROCm")


@pytest.mark.parametrize(("tokens", "heads"), [(1, 8), (33, 16), (768, 32)])
def test_lightop_mla_decode_concat_matches_torch(
    tokens: int,
    heads: int,
) -> None:
    _require_hcu()
    generator = torch.Generator(device="cuda").manual_seed(20260906 + tokens)
    left = torch.randn(
        (tokens, heads, 512),
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    right = torch.randn(
        (tokens, heads, 64),
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    left_before = left.clone()
    right_before = right.clone()

    actual = lightop_mla_decode_concat_impl(left, right)
    expected = torch.cat((left, right), dim=-1)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(left, left_before, rtol=0, atol=0)
    torch.testing.assert_close(right, right_before, rtol=0, atol=0)
    assert not is_lightop_mla_decode_concat_eligible(left, right)


def test_lightop_mla_decode_concat_matches_flashmla_strides() -> None:
    _require_hcu()
    tokens, heads = 128, 16
    left_data = torch.randn(
        tokens * heads * 512,
        dtype=torch.bfloat16,
        device="cuda",
    )
    left = torch.as_strided(
        left_data,
        size=(tokens, heads, 512),
        stride=(512, 512 * tokens, 1),
    )
    right_data = torch.randn(
        1536 * (heads // 8) * tokens,
        dtype=torch.bfloat16,
        device="cuda",
    )
    right = torch.as_strided(
        right_data,
        size=(tokens, heads, 64),
        stride=(1536 * (heads // 8), 192, 1),
    )

    torch.testing.assert_close(
        lightop_mla_decode_concat_impl(left, right),
        torch.cat((left, right), dim=-1),
        rtol=0,
        atol=0,
    )
    assert is_lightop_mla_decode_concat_eligible(left, right)
