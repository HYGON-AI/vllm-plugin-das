# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from __future__ import annotations

import torch
import torch.nn.functional as functional
from vllm.config import VllmConfig, set_current_vllm_config

from vllm_hcu.ops import silu_and_mul


def test_hcu_silu_constructs_without_nvidia_extension(monkeypatch) -> None:
    monkeypatch.setattr(
        silu_and_mul.henvs,
        "VLLM_HCU_USE_CUSTOM_OPS",
        False,
    )
    with set_current_vllm_config(VllmConfig()):
        op = silu_and_mul.HcuSiluAndMul(compile_native=False)
    value = torch.randn(3, 16, dtype=torch.float32)

    actual = op.forward_hip(value)
    gate, up = value.chunk(2, dim=-1)

    torch.testing.assert_close(actual, functional.silu(gate) * up)
