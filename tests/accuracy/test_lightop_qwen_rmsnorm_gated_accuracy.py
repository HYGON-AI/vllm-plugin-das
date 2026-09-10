# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

"""Live-HCU accuracy and fallback checks for Qwen gated RMSNorm."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.config.compilation import CompilationConfig
from vllm_hcu.ops import rms_norm_gated
from vllm_hcu.ops.rms_norm_gated import (
    HcuRMSNormGated,
    _is_qwen_gated_rmsnorm_eligible,
)
from vllm_hcu.platforms import envs as henvs


pytestmark = pytest.mark.hcu
_SEED = 20260907


@pytest.fixture(scope="module")
def custom_op_vllm_config() -> VllmConfig:
    return VllmConfig(
        compilation_config=CompilationConfig(custom_ops=["all"])
    )


@pytest.mark.parametrize("width", (128, 256))
@pytest.mark.parametrize("rows", (1, 6, 32, 48, 144, 512, 1024))
def test_lightop_qwen_gated_rmsnorm_matches_fp32_reference(
    rows: int,
    width: int,
    custom_op_vllm_config: VllmConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(henvs, "VLLM_HCU_USE_CUSTOM_OPS", True)
    monkeypatch.setattr(
        henvs, "VLLM_HCU_USE_LIGHTOP_QWEN_RMSNORM_GATED", True
    )
    generator = torch.Generator(device="cuda").manual_seed(_SEED + rows)
    x = torch.randn(
        (rows, width),
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    ).contiguous()
    z = torch.randn_like(x, generator=generator).contiguous()
    weight = (
        1.0
        + 0.1
        * torch.randn(
            (width,),
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
    ).contiguous()
    x_before = x.clone()
    z_before = z.clone()
    weight_before = weight.clone()

    x_float = x.float()
    expected = (
        x_float
        * torch.rsqrt(x_float.square().mean(dim=-1, keepdim=True) + 1e-6)
        * weight.float()
        * F.silu(z.float())
    ).to(torch.bfloat16)
    with set_current_vllm_config(custom_op_vllm_config):
        layer = HcuRMSNormGated(
            hidden_size=width,
            eps=1e-6,
            group_size=None,
            norm_before_gate=True,
            activation="silu",
            device=torch.device("cuda"),
            dtype=torch.bfloat16,
        )
    with torch.no_grad():
        layer.weight.copy_(weight)
    assert _is_qwen_gated_rmsnorm_eligible(layer, x, z)
    actual = layer.forward_hip(x, z)

    assert actual.shape == x.shape
    assert actual.dtype is torch.bfloat16
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.05)
    torch.testing.assert_close(x, x_before, rtol=0, atol=0)
    torch.testing.assert_close(z, z_before, rtol=0, atol=0)
    torch.testing.assert_close(weight, weight_before, rtol=0, atol=0)


def test_qwen_gated_rmsnorm_fp32_weight_uses_vllm_fallback(
    custom_op_vllm_config: VllmConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(henvs, "VLLM_HCU_USE_CUSTOM_OPS", True)
    monkeypatch.setattr(
        henvs, "VLLM_HCU_USE_LIGHTOP_QWEN_RMSNORM_GATED", True
    )
    generator = torch.Generator(device="cuda").manual_seed(_SEED)
    x = torch.randn(
        (32, 128), device="cuda", dtype=torch.bfloat16, generator=generator
    )
    z = torch.randn_like(x, generator=generator)
    with set_current_vllm_config(custom_op_vllm_config):
        layer = HcuRMSNormGated(
            hidden_size=128,
            eps=1e-6,
            group_size=None,
            norm_before_gate=True,
            activation="silu",
            device=torch.device("cuda"),
            dtype=torch.float32,
        )
    with torch.no_grad():
        layer.weight.copy_(1.0 + 0.1 * torch.randn_like(layer.weight))
    assert not _is_qwen_gated_rmsnorm_eligible(layer, x, z)

    expected_x = x.float()
    expected = (
        expected_x
        * torch.rsqrt(expected_x.square().mean(dim=-1, keepdim=True) + 1e-6)
        * layer.weight
        * F.silu(z.float())
    ).to(torch.bfloat16)
    actual = layer.forward_hip(x, z)

    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.05)


def test_missing_lightop_export_uses_vllm_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = torch.Generator(device="cuda").manual_seed(_SEED + 1)
    x = torch.randn(
        (32, 128), device="cuda", dtype=torch.bfloat16, generator=generator
    )
    z = torch.randn_like(x, generator=generator)
    weight = torch.randn(
        (128,), device="cuda", dtype=torch.bfloat16, generator=generator
    )
    monkeypatch.setattr(
        rms_norm_gated,
        "_lightop_layer_norm_fwd_1pass_opt",
        lambda: None,
    )

    x_float = x.float()
    expected = (
        x_float
        * torch.rsqrt(x_float.square().mean(dim=-1, keepdim=True) + 1e-6)
        * weight.float()
        * F.silu(z.float())
    ).to(torch.bfloat16)
    actual = rms_norm_gated._hcu_lightop_qwen_rmsnorm_gated_impl(
        x, z, weight, 1e-6
    )

    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.05)
