from __future__ import annotations

import math
import os
import warnings

import pytest


pytestmark = pytest.mark.hcu

SUPPORTED_LIVE_ARCHITECTURES = frozenset({"gfx936", "gfx938"})


def _live_requested() -> bool:
    return os.environ.get("HCU_VLLM033_RMSNORM_LIVE") == "1"


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
        "set HCU_VLLM033_RMSNORM_LIVE=1 for installed-wheel HCU "
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

    assert current_platform.__class__.__module__ == "hcu_adapter.bridges.vllm"
    assert current_platform.__class__.__name__ == "HcuPlatform"
    load_general_plugins()

    from hcu_vllm_das.rms_norm import HcuRMSNorm

    properties = torch.cuda.get_device_properties(0)
    architecture = properties.gcnArchName.split(":", 1)[0]
    assert architecture in SUPPORTED_LIVE_ARCHITECTURES

    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_RMS_NORM", "1")
    torch.manual_seed(20261026)

    with set_current_vllm_config(VllmConfig()):
        for dtype in (torch.bfloat16, torch.float16):
            layer = RMSNorm(256).to(device="cuda", dtype=dtype)
            assert isinstance(layer, HcuRMSNorm)
            assert layer._forward_method.__name__ == "forward_hip"
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


@pytest.mark.skipif(
    not _live_requested(),
    reason=(
        "set HCU_VLLM033_RMSNORM_LIVE=1 for installed-wheel HCU "
        "RMSNorm validation"
    ),
)
def test_installed_fused_rmsnorm_compile_replay_has_no_alias_warning() -> None:
    import torch

    from vllm.plugins import load_general_plugins

    load_general_plugins()

    torch.manual_seed(20261029)
    weight = torch.randn((256,), device="cuda", dtype=torch.bfloat16)

    def fused(value, residual):
        torch.ops.vllm.hcu_fused_add_rms_norm_lightop(
            value,
            residual,
            weight,
            1e-6,
        )
        return value, residual

    compiled = torch.compile(fused, dynamic=True)
    first_value = torch.randn((8, 256), device="cuda", dtype=torch.bfloat16)
    first_residual = torch.randn_like(first_value)
    second_value = first_value + 0.125
    second_residual = first_residual - 0.25
    expected_sum = second_value.float() + second_residual.float()
    expected = _reference(expected_sum, weight, 1e-6)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        compiled(first_value.clone(), first_residual.clone())
        actual, updated_residual = compiled(
            second_value.clone(),
            second_residual.clone(),
        )
        torch.cuda.synchronize()

    alias_warnings = [
        str(item.message)
        for item in caught
        if "may not alias any inputs" in str(item.message)
    ]
    assert alias_warnings == []
    torch.testing.assert_close(
        updated_residual.float(), expected_sum, rtol=2e-2, atol=2e-2
    )
    torch.testing.assert_close(actual.float(), expected, rtol=4e-2, atol=4e-2)


@pytest.mark.skipif(not _live_requested(), reason="explicit live RMSNorm validation")
@pytest.mark.parametrize("dtype_name", ("bfloat16", "float16"))
@pytest.mark.parametrize("fused", (False, True))
@pytest.mark.parametrize("width", (128, 4096))
def test_rmsnorm_preserves_official_round_before_weight_on_graph_replay(
    monkeypatch: pytest.MonkeyPatch,
    dtype_name: str,
    fused: bool,
    width: int,
) -> None:
    """A single final cast changes INT8 inputs even if a loose FP32 oracle passes."""
    import torch

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.layernorm import RMSNorm
    from vllm.plugins import load_general_plugins

    load_general_plugins()
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_RMS_NORM", "1")
    torch.manual_seed(20261115)
    dtype = getattr(torch, dtype_name)
    with set_current_vllm_config(VllmConfig()):
        layer = RMSNorm(width).to(device="cuda", dtype=dtype)
        layer.weight.data.normal_(mean=1.0, std=0.2)
        value = torch.randn((17, width), device="cuda", dtype=dtype)
        residual = torch.randn_like(value) if fused else None
        # Capture with persistent addresses; overwrite both inputs before replay.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                output = layer(value, residual)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = layer(value, residual)
        for shift in (0.125, -0.375):
            source = torch.randn_like(value) + shift
            source_residual = torch.randn_like(value) if fused else None
            summed = source.float()
            if fused:
                summed = summed + source_residual.float()
            # Independent formula preserves the target's intermediate dtype cast.
            normalized = summed * torch.rsqrt(
                summed.square().mean(-1, keepdim=True) + 1e-6
            )
            expected = normalized.to(dtype) * layer.weight
            value.copy_(source)
            if fused:
                residual.copy_(source_residual)
            graph.replay()
            actual = output[0] if fused else output
            torch.cuda.synchronize()
            torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)
            # Reduction-order ties can differ; a missing cast changes ~25% of values.
            assert (actual == expected).float().mean().item() >= 0.999
            if fused:
                torch.testing.assert_close(output[1], summed.to(dtype), rtol=0, atol=0)


@pytest.mark.skipif(not _live_requested(), reason="explicit live RMSNorm validation")
@pytest.mark.parametrize("width", (128, 2048))
@pytest.mark.parametrize("dtype_name,epsilon,amplitude", (
    ("bfloat16", 1e-6, 0.0006561279296875),
    ("bfloat16", 1e-5, 0.001953125),
    ("float16", 1e-6, 0.0006084442138671875),
    ("float16", 1e-5, 0.0018301010131835938),
))
def test_rmsnorm_keeps_epsilon_precision_on_small_input_graph_replay(
    monkeypatch: pytest.MonkeyPatch,
    width: int,
    dtype_name: str,
    epsilon: float,
    amplitude: float,
) -> None:
    """Constant rows distinguish epsilon narrowing from reduction-order ties."""
    import torch
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.layernorm import RMSNorm
    from vllm.plugins import load_general_plugins

    load_general_plugins()
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_RMS_NORM", "1")
    dtype = getattr(torch, dtype_name)
    with set_current_vllm_config(VllmConfig()):
        layer = RMSNorm(width, eps=epsilon).to(device="cuda", dtype=dtype)
        value = torch.full((17, width), amplitude, device="cuda", dtype=dtype)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                output = layer(value)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = layer(value)
        for sign in (1, -1):
            source = torch.full_like(value, sign * amplitude)
            # Every value equals the exact binary scalar: FP64 Python arithmetic
            # supplies an independent oracle without a GPU reduction.
            expected = torch.full_like(value, sign * amplitude / math.sqrt(
                amplitude * amplitude + epsilon))
            value.copy_(source)
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
            torch.testing.assert_close(value, source, rtol=0, atol=0)
