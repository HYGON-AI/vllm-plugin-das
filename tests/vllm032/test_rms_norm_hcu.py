from __future__ import annotations

import os

import pytest


pytestmark = pytest.mark.hcu


def _live_requested() -> bool:
    return os.environ.get("HCU_VLLM032_RMSNORM_LIVE") == "1"


def _reference(value, weight, epsilon: float):
    import torch

    value_float = value.float()
    return (
        value_float
        * torch.rsqrt(
            value_float.square().mean(dim=-1, keepdim=True) + epsilon
        )
        * weight.float()
    )


@pytest.mark.skipif(
    not _live_requested(),
    reason=(
        "set HCU_VLLM032_RMSNORM_LIVE=1 for installed-wheel HCU "
        "RMSNorm validation"
    ),
)
def test_installed_lightop_rmsnorm_matches_references_and_fallbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.layernorm import RMSNorm
    from vllm.platforms import current_platform
    from vllm.plugins import load_general_plugins

    assert current_platform.__class__.__module__ == "hcu_vllm_adapter.platform.hcu"
    load_general_plugins()

    from hcu_vllm_das.rms_norm import HcuRMSNorm

    properties = torch.cuda.get_device_properties(0)
    architecture = properties.gcnArchName.split(":", 1)[0]
    assert architecture == "gfx938"

    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_RMS_NORM", "1")
    torch.manual_seed(20261026)

    with set_current_vllm_config(VllmConfig()):
        for dtype in (torch.bfloat16, torch.float16):
            layer = RMSNorm(256).to(device="cuda", dtype=dtype)
            assert isinstance(layer, HcuRMSNorm)
            layer.weight.data.normal_(mean=1.0, std=0.1)

            value = torch.randn((17, 256), device="cuda", dtype=dtype)
            expected = _reference(
                value,
                layer.weight.data,
                layer.variance_epsilon,
            )
            optimized = layer(value)
            torch.cuda.synchronize()
            torch.testing.assert_close(
                optimized.float(), expected, rtol=4e-2, atol=4e-2
            )

            value = torch.randn((11, 256), device="cuda", dtype=dtype)
            residual = torch.randn_like(value)
            summed = value.float() + residual.float()
            expected = _reference(
                summed,
                layer.weight.data,
                layer.variance_epsilon,
            )
            normalized, updated_residual = layer(value, residual)
            torch.cuda.synchronize()
            torch.testing.assert_close(
                normalized.float(), expected, rtol=4e-2, atol=4e-2
            )
            torch.testing.assert_close(
                updated_residual.float(), summed, rtol=2e-2, atol=2e-2
            )

            rollback_value = torch.randn(
                (7, 256), device="cuda", dtype=dtype
            )
            rollback_expected = _reference(
                rollback_value,
                layer.weight.data,
                layer.variance_epsilon,
            )
            monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "0")
            master_rollback = layer(rollback_value)
            torch.testing.assert_close(
                master_rollback.float(),
                rollback_expected,
                rtol=4e-2,
                atol=4e-2,
            )

            monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
            monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_RMS_NORM", "0")
            child_rollback = layer(rollback_value)
            torch.testing.assert_close(
                child_rollback.float(),
                rollback_expected,
                rtol=4e-2,
                atol=4e-2,
            )
            monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_RMS_NORM", "1")

        fallback_cases = (
            (RMSNorm(32), torch.randn((5, 32), device="cuda")),
            (
                RMSNorm(256, has_weight=False),
                torch.randn((5, 256), device="cuda"),
            ),
            (
                RMSNorm(256, var_hidden_size=128),
                torch.randn((5, 256), device="cuda"),
            ),
        )
        for layer, value in fallback_cases:
            layer = layer.to(device="cuda", dtype=torch.bfloat16)
            value = value.to(dtype=torch.bfloat16)
            assert isinstance(layer, HcuRMSNorm)
            expected = layer.forward_native(value.clone())
            actual = layer(value.clone())
            torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
