# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""CPU contract for the native ``relu^2`` replacement.

DTK2610 returns NaNs and values up to ~1e36 from ``torch.ops._C.relu_squared``.
``ReLUSquaredActivation`` is used by the Mamba2 MLP, the NemotronH dense MLP
and the shared experts, so the native path is mandatory for correctness.
"""

from __future__ import annotations

import pytest
import torch

from vllm.model_executor.custom_op import op_registry_oot

from vllm_hcu.ops.relu_squared import HcuReLUSquaredActivation


def _op() -> HcuReLUSquaredActivation:
    # forward_hip/forward_native read no module state; bypass __init__ so the
    # test does not need a cached CompilationConfig.
    return object.__new__(HcuReLUSquaredActivation)


def test_oot_registration_is_installed() -> None:
    assert op_registry_oot["ReLUSquaredActivation"] is HcuReLUSquaredActivation


def test_forward_hip_matches_native(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_HCU_USE_NATIVE_RELU_SQUARED", "1")
    x = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
    expected = torch.square(torch.relu(x))
    torch.testing.assert_close(
        _op().forward_hip(x), expected, rtol=0, atol=0
    )


def test_forward_hip_handles_nan_and_inf(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_HCU_USE_NATIVE_RELU_SQUARED", "1")
    x = torch.tensor([float("nan"), float("-inf"), float("inf")])
    expected = torch.square(torch.relu(x))
    torch.testing.assert_close(
        _op().forward_hip(x), expected, rtol=0, atol=0, equal_nan=True
    )


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16, torch.float32))
def test_forward_hip_preserves_dtype(
    dtype: torch.dtype, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VLLM_HCU_USE_NATIVE_RELU_SQUARED", "1")
    x = torch.tensor([-1.0, 0.0, 2.0], dtype=dtype)
    assert _op().forward_hip(x).dtype is dtype


def test_escape_hatch_falls_back_to_kernel(monkeypatch: pytest.MonkeyPatch) -> None:
    """With the flag off the upstream HIP path is used again."""

    monkeypatch.setenv("VLLM_HCU_USE_NATIVE_RELU_SQUARED", "0")
    calls: list[torch.Tensor] = []

    def fake_forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        calls.append(x)
        return x

    monkeypatch.setattr(
        HcuReLUSquaredActivation, "forward_cuda", fake_forward_cuda
    )
    x = torch.tensor([1.0, -1.0])
    assert torch.equal(_op().forward_hip(x), x)
    assert calls == [x]
