# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_hcu.models.hy_v4.hc import (
    HYV4HCHeadLayer,
    HYV4HCPostLayer,
    HYV4HCPreLayer,
    HYV4HCLayer,
)


class _FixedLinear(nn.Module):
    def __init__(self, output: torch.Tensor) -> None:
        super().__init__()
        self.output = output

    def forward(self, _input: torch.Tensor) -> tuple[torch.Tensor, None]:
        return self.output, None


def test_hc_post_scatter_matches_fp32_reference() -> None:
    branch = torch.tensor([[0.5, -1.0]], dtype=torch.bfloat16)
    residual = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0]]],
        dtype=torch.bfloat16,
    )
    post = torch.tensor([[0.25, 0.75]], dtype=torch.float32)
    layer = HYV4HCPostLayer(SimpleNamespace())

    actual = layer(branch, residual, post)

    expected = torch.tensor(
        [[[1.125, 1.75], [3.375, 3.25]]],
        dtype=torch.bfloat16,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_hc_pre_uses_normalized_fp32_gate_math() -> None:
    layer = object.__new__(HYV4HCPreLayer)
    nn.Module.__init__(layer)
    layer.hidden_dim = 2
    layer.hc_mult = 2
    layer.magnitude = 2.0
    layer.hc_eps = 1e-6
    layer.layernorm_epsilon = 1e-5
    layer.hc_scale = nn.Parameter(torch.tensor([0.5, 0.25]))
    layer.hc_base = nn.Parameter(torch.tensor([0.0, 0.0, 0.0, 0.0]))
    mixes = torch.tensor([[1.0, -1.0, 0.5, -0.5]], dtype=torch.float32)
    layer.hc_fn = _FixedLinear(mixes)
    hidden = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0]]],
        dtype=torch.bfloat16,
    )

    reduced, post = layer(hidden)

    flat = hidden.flatten(1).float()
    rsqrt = torch.rsqrt(flat.square().mean(-1, keepdim=True) + 1e-5)
    normalized = mixes * rsqrt
    pre = torch.sigmoid(normalized[:, :2] * 0.5) + 1e-6
    expected_reduced = torch.sum(pre.unsqueeze(-1) * hidden.float(), dim=1)
    expected_post = 2.0 * torch.sigmoid(normalized[:, 2:] * 0.25) + 1e-6
    torch.testing.assert_close(reduced, expected_reduced.to(torch.bfloat16))
    torch.testing.assert_close(post, expected_post)


@pytest.mark.parametrize(
    ("input_shape", "expected_shape"),
    [
        ((3, 8), (3, 4, 8)),
        ((3, 32), (3, 4, 8)),
        ((3, 4, 8), (3, 4, 8)),
    ],
)
def test_hc_prepare_input_accepts_supported_layouts(
    input_shape: tuple[int, ...],
    expected_shape: tuple[int, ...],
) -> None:
    layer = object.__new__(HYV4HCLayer)
    nn.Module.__init__(layer)
    layer.config = SimpleNamespace(hidden_size=8, hc_mult=4)
    layer.enable_ihc = True

    actual = layer.prepare_input(torch.zeros(input_shape))

    assert actual.shape == expected_shape


def test_hc_prepare_input_rejects_wrong_flattened_width() -> None:
    layer = object.__new__(HYV4HCLayer)
    nn.Module.__init__(layer)
    layer.config = SimpleNamespace(hidden_size=8, hc_mult=4)
    layer.enable_ihc = True

    with pytest.raises(RuntimeError, match=r"hc_mult\*hidden_size \(32\)"):
        layer.prepare_input(torch.zeros(3, 31))


def test_hc_head_uses_fp32_gated_channel_reduction() -> None:
    layer = object.__new__(HYV4HCHeadLayer)
    nn.Module.__init__(layer)
    layer.config = SimpleNamespace(rms_norm_eps=1e-5)
    layer.hidden_size = 2
    layer.hc_mult = 2
    layer.hc_eps = 1e-6
    layer.hc_head_scale = nn.Parameter(torch.tensor([0.5]))
    layer.hc_head_base = nn.Parameter(torch.tensor([0.0, 0.0]))
    mixes = torch.tensor([[0.25, -0.75]], dtype=torch.float32)
    layer.hc_head_fn = _FixedLinear(mixes)
    hidden = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0]]],
        dtype=torch.bfloat16,
    )

    actual = layer(hidden)

    flat = hidden.flatten(1).float()
    rsqrt = torch.rsqrt(flat.square().mean(-1, keepdim=True) + 1e-5)
    gates = torch.sigmoid(mixes * rsqrt * 0.5) + 1e-6
    expected = torch.sum(gates.unsqueeze(-1) * hidden.float(), dim=1)
    torch.testing.assert_close(actual, expected.to(torch.bfloat16))
