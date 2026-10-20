# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from __future__ import annotations

import torch

from vllm.model_executor.custom_op import op_registry_oot
from vllm_hcu.ops.relu_squared import HcuReLUSquaredActivation


def _uninitialized_op() -> HcuReLUSquaredActivation:
    # forward_hip/forward_native do not read module state; bypass __init__ so
    # the test does not need a cached CompilationConfig.
    return object.__new__(HcuReLUSquaredActivation)


def test_oot_registration() -> None:
    assert op_registry_oot["ReLUSquaredActivation"] is HcuReLUSquaredActivation


def test_forward_hip_matches_native() -> None:
    x = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
    expected = torch.square(torch.relu(x))
    actual = _uninitialized_op().forward_hip(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_forward_hip_handles_nan_and_inf() -> None:
    x = torch.tensor([float("nan"), float("-inf"), float("inf")])
    expected = torch.square(torch.relu(x))
    actual = _uninitialized_op().forward_hip(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)


def test_forward_hip_preserves_dtype() -> None:
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        x = torch.tensor([-1.0, 0.0, 2.0], dtype=dtype)
        assert _uninitialized_op().forward_hip(x).dtype is dtype