# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

"""CPU tests for delayed mHC Pre split-K selection and forwarding.

The production selection helper must stay independent of AITER so these tests
can run without an HCU. The numeric cases model the staged projection's
split-K reduction with plain torch ops.
"""

from __future__ import annotations

import pytest
import torch

# Import the upstream reference before any test patches torch.ops.vllm, since
# importing the kernel package registers its custom ops.
from vllm.model_executor.kernels.mhc.torch import mhc_pre_delayed_torch

from vllm_hcu.model_executor.layers import mhc_runtime as runtime


def _fake_staged_runtime(hc: int, hidden: int) -> runtime.MhcRuntime:
    """Build a CPU model of the two staged AITER kernels."""

    def project(gemm, sqrsum, source, fn, tile_k, use_tf32):
        assert not use_tf32
        split_k = gemm.shape[0]
        flat = source.flatten(1).float()
        width = flat.shape[-1]
        split_size = width // split_k
        for split in range(split_k):
            columns = slice(split * split_size, (split + 1) * split_size)
            gemm[split].copy_(flat[:, columns] @ fn[:, columns].t())
            sqrsum[split].copy_(flat[:, columns].square().sum(-1))

    def big_fuse(post, comb, layer_input, gemm, sqrsum, hc_scale, hc_base,
                 residual, rms_eps, hc_pre_eps, hc_sinkhorn_eps,
                 hc_post_mult_value, sinkhorn_repeat):
        mixes = gemm.sum(0) * torch.rsqrt(
            sqrsum.sum(0) / (hc * hidden) + rms_eps
        )[:, None]
        post_value = torch.sigmoid(
            mixes[:, hc:2 * hc] * hc_scale[1] + hc_base[hc:2 * hc]
        ) * hc_post_mult_value
        comb_value = torch.softmax(
            (mixes[:, 2 * hc:].reshape(-1, hc, hc) * hc_scale[2]
             + hc_base[2 * hc:].reshape(1, hc, hc)),
            dim=-1,
        ) + hc_sinkhorn_eps
        comb_value = comb_value / (
            comb_value.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps
        )
        for _ in range(sinkhorn_repeat - 1):
            comb_value = comb_value / (
                comb_value.sum(dim=-1, keepdim=True) + hc_sinkhorn_eps
            )
            comb_value = comb_value / (
                comb_value.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps
            )
        post.copy_(post_value[..., None])
        comb.copy_(comb_value)
        layer_input.copy_(residual[:, 0])

    return runtime.MhcRuntime(project=project, big_fuse=big_fuse)


@pytest.fixture
def cpu_staged_ops(monkeypatch):
    """Patch torch.ops.vllm helpers used by the delayed path."""
    monkeypatch.setattr(
        torch.ops.vllm,
        "mhc_pre_mix_triton",
        lambda gemm, sqrsum, scale, base, hc, width, eps, pre_eps: (
            torch.sigmoid(
                gemm.sum(0)[:, :hc]
                * torch.rsqrt(sqrsum.sum(0) / width + eps)[:, None]
                * scale[0] + base[:hc]
            ) + pre_eps
        ),
        raising=False,
    )
    monkeypatch.setattr(
        torch.ops.vllm,
        "hc_collapse_triton",
        lambda residual, pre_mix: (
            residual.float() * pre_mix.unsqueeze(-1)
        ).sum(1).to(residual.dtype),
        raising=False,
    )


def test_delayed_pre_split_k_env_is_registered(monkeypatch):
    from vllm_hcu.platforms import envs as henvs

    assert "VLLM_HCU_MHC_PRE_SPLIT_K" in henvs.hcu_vllm_environment_variables
    monkeypatch.delenv("VLLM_HCU_MHC_PRE_SPLIT_K", raising=False)
    assert henvs.VLLM_HCU_MHC_PRE_SPLIT_K == "auto"
    monkeypatch.setenv("VLLM_HCU_MHC_PRE_SPLIT_K", "8")
    assert henvs.VLLM_HCU_MHC_PRE_SPLIT_K == "8"


@pytest.mark.parametrize(
    ("tokens", "hc", "hidden", "expected"),
    (
        (1, 4, 5120, 32),
        (64, 4, 5120, 32),
        (0, 4, 5120, 1),
        (65, 4, 5120, 1),
        (7, 8, 5120, 1),
        (7, 4, 7168, 1),
        (129, 4, 1024, 1),
    ),
)
def test_delayed_pre_auto_uses_only_the_validated_small_shape(
    tokens, hc, hidden, expected,
):
    assert runtime._delayed_pre_splits(tokens, hc, hidden, "auto") == expected


@pytest.mark.parametrize("split_k", (1, 2, 4, 8, 16, 32))
def test_delayed_pre_manual_override_is_honoured(split_k):
    assert runtime._delayed_pre_splits(
        7, 4, 5120, str(split_k)
    ) == split_k


@pytest.mark.parametrize("mode", ("3", "0", "64", "", "AUTO", "auto "))
def test_delayed_pre_rejects_invalid_override(mode):
    with pytest.raises(ValueError, match="Invalid VLLM_HCU_MHC_PRE_SPLIT_K"):
        runtime._delayed_pre_splits(7, 4, 5120, mode)


@pytest.mark.parametrize(
    ("hc", "hidden", "split_k", "valid"),
    (
        (4, 5120, 1, True),
        (4, 1024, 32, True),   # K / split == 128 is the lower bound.
        (4, 1024, 16, True),
        (4, 512, 32, False),   # K / split < 128.
        (4, 520, 32, False),   # K is not divisible by 64 * split.
        (4, 520, 1, False),    # K is not a multiple of the 64-wide tile.
    ),
)
def test_delayed_pre_manual_override_checks_k_constraints(
    hc, hidden, split_k, valid,
):
    if valid:
        assert runtime._delayed_pre_splits(
            7, hc, hidden, str(split_k)
        ) == split_k
    else:
        with pytest.raises(ValueError, match="K must be divisible"):
            runtime._delayed_pre_splits(7, hc, hidden, str(split_k))


def test_delayed_pre_auto_falls_back_when_selected_split_is_illegal(
    monkeypatch,
):
    # Auto only promotes the validated shape, which is always legal. Other
    # shapes keep the historical split of 1 even when K cannot use split-K.
    assert runtime._delayed_pre_splits(7, 4, 520, "auto") == 1
    assert runtime._delayed_pre_splits(7, 4, 512, "auto") == 1


def test_delayed_pre_forwards_selected_split_and_first_layer_x(
    monkeypatch, cpu_staged_ops,
):
    captured = {}

    def fake_staged_pre(residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps,
                        hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat,
                        n_splits=1, x=None):
        tokens, hc, hidden = residual.shape
        captured["n_splits"] = n_splits
        captured["x_is_input"] = x is first_layer_x
        return (
            torch.empty(tokens, hc, 1),
            torch.empty(tokens, hc, hc),
            torch.empty(tokens, hidden, dtype=residual.dtype),
            torch.empty(1, tokens, hc * (2 + hc)),
            torch.empty(1, tokens),
        )

    monkeypatch.setattr(runtime, "_staged_pre", fake_staged_pre)
    monkeypatch.setenv("VLLM_HCU_MHC_PRE_SPLIT_K", "auto")
    residual = torch.zeros(3, 4, 5120, dtype=torch.bfloat16)
    first_layer_x = torch.zeros(3, 5120, dtype=torch.bfloat16)
    actual = runtime.delayed_pre(
        residual,
        torch.zeros(24, 5120),
        torch.ones(3),
        torch.zeros(24),
        1e-6,
        1e-6,
        1e-6,
        2.0,
        20,
        x=first_layer_x,
    )
    assert captured == {"n_splits": 32, "x_is_input": True}
    assert actual[2].shape == (3, 5120)


def test_delayed_pre_numeric_first_layer_x_matches_reference(
    monkeypatch, cpu_staged_ops,
):
    torch.manual_seed(19)
    tokens, hc, hidden = 3, 4, 5120
    residual = torch.randn(tokens, hc, hidden, dtype=torch.bfloat16)
    first_layer_x = torch.randn(tokens, hidden, dtype=torch.bfloat16)
    fn = torch.randn(hc * (2 + hc), hidden) * 0.02
    scale = torch.randn(3)
    base = torch.randn(hc * (2 + hc)) * 0.1
    args = (residual, fn, scale, base, 1e-6, 1e-6, 1e-6, 2.0, 20)

    monkeypatch.setattr(runtime, "runtime", lambda: _fake_staged_runtime(hc, hidden))
    monkeypatch.setenv("VLLM_HCU_MHC_PRE_SPLIT_K", "auto")
    actual = runtime.delayed_pre(*args, x=first_layer_x)
    expected = mhc_pre_delayed_torch(*args, x=first_layer_x)

    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result, reference, rtol=1e-5, atol=1e-5)


def test_delayed_pre_numeric_split_k_uses_padded_reduction(
    monkeypatch, cpu_staged_ops,
):
    torch.manual_seed(23)
    tokens, hc, hidden = 6, 4, 5120
    residual = torch.randn(tokens, hc, hidden, dtype=torch.bfloat16)
    fn = torch.randn(hc * (2 + hc), hc * hidden) * 0.01
    scale = torch.randn(3)
    base = torch.randn(hc * (2 + hc)) * 0.1
    args = (residual, fn, scale, base, 1e-6, 1e-6, 1e-6, 2.0, 20)

    monkeypatch.setattr(runtime, "runtime", lambda: _fake_staged_runtime(hc, hidden))
    monkeypatch.setenv("VLLM_HCU_MHC_PRE_SPLIT_K", "32")
    actual = runtime.delayed_pre(*args)
    expected = mhc_pre_delayed_torch(*args)

    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result, reference, rtol=1e-5, atol=1e-5)


def test_delayed_pre_override_one_keeps_single_split(monkeypatch, cpu_staged_ops):
    torch.manual_seed(29)
    tokens, hc, hidden = 6, 4, 5120
    residual = torch.randn(tokens, hc, hidden, dtype=torch.bfloat16)
    fn = torch.randn(hc * (2 + hc), hc * hidden) * 0.01
    args = (residual, fn, torch.ones(3), torch.zeros(hc * (2 + hc)),
            1e-6, 1e-6, 1e-6, 2.0, 20)

    monkeypatch.setattr(runtime, "runtime", lambda: _fake_staged_runtime(hc, hidden))
    monkeypatch.setenv("VLLM_HCU_MHC_PRE_SPLIT_K", "1")
    actual = runtime.delayed_pre(*args)
    expected = mhc_pre_delayed_torch(*args)

    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result, reference, rtol=1e-5, atol=1e-5)


def test_ordinary_pre_keeps_its_default_split(monkeypatch):
    captured = []

    def fake_staged_pre(residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps,
                        hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat,
                        n_splits=1, x=None):
        captured.append(n_splits)
        tokens, hc, hidden = residual.shape
        return (
            torch.empty(tokens, hc, 1),
            torch.empty(tokens, hc, hc),
            torch.empty(tokens, hidden, dtype=residual.dtype),
            torch.empty(n_splits, tokens, hc * (2 + hc)),
            torch.empty(n_splits, tokens),
        )

    def unexpected_selector(*args, **kwargs):
        raise AssertionError("ordinary Pre must not consult delayed split-K")

    monkeypatch.setattr(runtime, "_staged_pre", fake_staged_pre)
    monkeypatch.setattr(runtime, "_delayed_pre_splits", unexpected_selector)
    monkeypatch.setattr(
        runtime,
        "runtime",
        lambda: runtime.MhcRuntime(
            project=lambda *args: None,
            big_fuse=lambda *args: None,
            tilelang_pre=lambda *args: None,
        ),
    )
    monkeypatch.setenv("VLLM_HCU_MHC_PRE_SPLIT_K", "32")

    residual = torch.zeros(2, 4, 5120, dtype=torch.bfloat16)
    runtime.pre(
        residual,
        torch.zeros(24, 20480),
        torch.ones(3),
        torch.zeros(24),
        1e-6,
        1e-6,
        1e-6,
        2.0,
        20,
    )
    assert captured == [1]


def test_empty_input_contract_skips_selection(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("empty input must not reach staged Pre")

    monkeypatch.setattr(runtime, "_staged_pre", unexpected)
    monkeypatch.setenv("VLLM_HCU_MHC_PRE_SPLIT_K", "not-a-split")
    residual = torch.empty(0, 4, 5120, dtype=torch.bfloat16)

    actual = runtime.delayed_pre(
        residual,
        torch.empty(24, 20480),
        torch.ones(3),
        torch.zeros(24),
        1e-6,
        1e-6,
        1e-6,
        2.0,
        20,
    )
    assert [tuple(t.shape) for t in actual] == [
        (0, 4, 1), (0, 4, 4), (0, 5120), (0, 4),
    ]
    assert [t.dtype for t in actual] == [
        torch.float32, torch.float32, torch.bfloat16, torch.float32,
    ]
