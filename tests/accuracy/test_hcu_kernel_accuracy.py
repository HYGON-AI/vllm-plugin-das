# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Numerical reference checks that execute real HCU extension kernels."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as functional


pytestmark = pytest.mark.hcu


def _hcu_device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("a live HCU/ROCm device is required")
    properties = torch.cuda.get_device_properties(0)
    if not hasattr(properties, "gcnArchName"):
        pytest.skip("the active torch device is not an HCU/ROCm device")
    return torch.device("cuda", 0)


@pytest.mark.parametrize("shape", [(1, 256), (17, 256), (4, 1024)])
def test_lightop_silu_and_mul_matches_float32_reference(
    shape: tuple[int, int],
) -> None:
    device = _hcu_device()
    try:
        from lightop.activation import silu_and_mul_opt
    except (ImportError, AttributeError) as exc:
        pytest.skip(f"lightop silu_and_mul kernel is unavailable: {exc}")

    generator = torch.Generator(device=device).manual_seed(20250724)
    value = torch.randn(
        shape,
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    actual = torch.empty(
        (shape[0], shape[1] // 2),
        device=device,
        dtype=value.dtype,
    )
    silu_and_mul_opt(actual, value)
    left, right = value.float().chunk(2, dim=-1)
    reference = functional.silu(left) * right

    torch.testing.assert_close(
        actual.float(),
        reference,
        rtol=3e-2,
        atol=3e-2,
    )


@pytest.mark.parametrize(
    ("shape", "dtype"),
    [
        ((1, 256), torch.bfloat16),
        ((11, 256), torch.bfloat16),
        ((3, 1024), torch.float16),
        ((2, 4096), torch.bfloat16),
    ],
)
def test_lightop_fused_add_rms_norm_matches_float32_reference(
    shape: tuple[int, int],
    dtype: torch.dtype,
) -> None:
    device = _hcu_device()
    try:
        from lightop.norm import fused_add_rms_norm
    except (ImportError, AttributeError) as exc:
        pytest.skip(f"lightop fused_add_rms_norm kernel is unavailable: {exc}")

    generator = torch.Generator(device=device).manual_seed(1219304)
    value = torch.randn(
        shape,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    residual = torch.randn(
        shape,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    weight = torch.randn(
        shape[-1],
        generator=generator,
        device=device,
        dtype=dtype,
    )
    original_value = value.float().clone()
    original_residual = residual.float().clone()
    epsilon = 1e-6
    summed = original_value + original_residual
    reference = (
        summed
        * torch.rsqrt(summed.square().mean(dim=-1, keepdim=True) + epsilon)
        * weight.float()
    )

    fused_add_rms_norm(value, residual, weight, epsilon)

    torch.testing.assert_close(
        value.float(),
        reference,
        rtol=4e-2,
        atol=4e-2,
    )
    torch.testing.assert_close(
        residual.float(),
        summed,
        rtol=2e-2,
        atol=2e-2,
    )


@pytest.mark.parametrize(
    ("shape", "valid_tokens"),
    [
        ((1, 7, 64), (7,)),
        ((4, 19, 127), (19, 11, 3, 0)),
        ((8, 33, 256), (33, 29, 23, 17, 11, 5, 1, 0)),
    ],
)
def test_hcu_expert_int8_quant_dequant_error_is_bounded(
    shape: tuple[int, int, int],
    valid_tokens: tuple[int, ...],
) -> None:
    device = _hcu_device()
    from vllm_hcu.model_executor.layers.fused_moe.int8_quant_runtime import (
        per_token_quant_int8,
    )

    generator = torch.Generator(device=device).manual_seed(136597310)
    value = torch.randn(
        shape,
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    counts = torch.tensor(valid_tokens, device=device, dtype=torch.int32)

    quantized, scales = per_token_quant_int8(value, counts)

    for expert, valid in enumerate(counts.tolist()):
        if valid == 0:
            continue
        source = value[expert, :valid].float()
        expected_scale = (
            source.abs().amax(dim=-1, keepdim=True).clamp_min(1e-10) / 127.0
        )
        dequantized = quantized[expert, :valid].float() * scales[expert, :valid]
        error = (dequantized - source).abs()
        assert torch.all(error <= expected_scale / 2 + 2e-2)
        torch.testing.assert_close(
            scales[expert, :valid],
            expected_scale,
            rtol=2e-2,
            atol=2e-3,
        )


@pytest.mark.parametrize(
    ("shape", "dtype"),
    [
        ((1, 128), torch.bfloat16),
        ((13, 256), torch.bfloat16),
        ((4, 1024), torch.float16),
        ((2, 4096), torch.bfloat16),
    ],
)
def test_lightop_rms_norm_matches_float32_reference(
    shape: tuple[int, int],
    dtype: torch.dtype,
) -> None:
    device = _hcu_device()
    try:
        from lightop.norm import rmsnorm_forward_autograd
    except (ImportError, AttributeError) as exc:
        pytest.skip(f"lightop RMSNorm kernel is unavailable: {exc}")

    generator = torch.Generator(device=device).manual_seed(20250725)
    value = torch.randn(shape, generator=generator, device=device, dtype=dtype)
    weight = torch.randn(
        shape[-1],
        generator=generator,
        device=device,
        dtype=dtype,
    )
    epsilon = 1e-6
    reference = (
        value.float()
        * torch.rsqrt(value.float().square().mean(dim=-1, keepdim=True) + epsilon)
        * weight.float()
    )

    actual = rmsnorm_forward_autograd(value, weight, epsilon, False)

    torch.testing.assert_close(
        actual.float(),
        reference,
        rtol=4e-2,
        atol=4e-2,
    )


@pytest.mark.parametrize("shape", [(3, 256), (2, 2048)])
def test_lightop_gemma_rms_norm_matches_float32_reference(
    shape: tuple[int, int],
) -> None:
    device = _hcu_device()
    try:
        from lightop.norm import gemma_rmsnorm
    except (ImportError, AttributeError) as exc:
        pytest.skip(f"lightop Gemma RMSNorm kernel is unavailable: {exc}")

    generator = torch.Generator(device=device).manual_seed(20250726)
    value = torch.randn(
        shape,
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    weight = torch.randn(
        shape[-1],
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    epsilon = 1e-6
    actual = torch.empty_like(value)
    gemma_rmsnorm(value, weight, epsilon, out=actual)
    normalized = value.float() * torch.rsqrt(
        value.float().square().mean(dim=-1, keepdim=True) + epsilon
    )
    reference = normalized * (1.0 + weight.float())

    torch.testing.assert_close(
        actual.float(),
        reference,
        rtol=4e-2,
        atol=4e-2,
    )


@pytest.mark.parametrize("shape", [(1, 256), (9, 512), (3, 2048)])
def test_lightop_fused_silu_quant_has_bounded_dequant_error(
    shape: tuple[int, int],
) -> None:
    device = _hcu_device()
    try:
        from lightop.activation import fuse_silu_mul_per_token_quant
    except (ImportError, AttributeError) as exc:
        pytest.skip(f"lightop fused SiluAndMul quant kernel is unavailable: {exc}")

    generator = torch.Generator(device=device).manual_seed(20250727)
    value = torch.randn(
        shape,
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    output = torch.empty(
        (shape[0], shape[1] // 2),
        device=device,
        dtype=torch.int8,
    )
    scales = torch.empty((shape[0], 1), device=device, dtype=torch.float32)
    fuse_silu_mul_per_token_quant(value, output=output, scales=scales)
    left, right = value.float().chunk(2, dim=-1)
    reference = functional.silu(left) * right
    expected_scale = (
        reference.abs().amax(dim=-1, keepdim=True).clamp_min(1e-10) / 127.0
    )
    dequantized = output.float() * scales

    torch.testing.assert_close(
        scales,
        expected_scale,
        rtol=2e-2,
        atol=2e-3,
    )
    assert torch.all(
        (dequantized - reference).abs() <= expected_scale / 2 + 3e-2
    )


@pytest.mark.parametrize("shape", [(5, 256), (2, 1024)])
def test_lightop_fused_rms_quant_has_bounded_dequant_error(
    shape: tuple[int, int],
) -> None:
    device = _hcu_device()
    try:
        from lightop.norm import rms_norm_dynamic_per_token_quant
    except (ImportError, AttributeError) as exc:
        pytest.skip(f"lightop fused RMS quant kernel is unavailable: {exc}")

    generator = torch.Generator(device=device).manual_seed(20250728)
    value = torch.randn(
        shape,
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    weight = torch.randn(
        shape[-1],
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    original_value = value.float().clone()
    epsilon = 1e-6
    output, scales = rms_norm_dynamic_per_token_quant(
        value,
        weight,
        epsilon,
        torch.int8,
        residual=None,
        update_input=True,
    )
    reference = (
        original_value
        * torch.rsqrt(original_value.square().mean(dim=-1, keepdim=True) + epsilon)
        * weight.float()
    )
    expected_scale = (
        reference.abs().amax(dim=-1, keepdim=True).clamp_min(1e-10) / 127.0
    )
    dequantized = output.float() * scales

    assert torch.all(torch.isfinite(scales))
    assert torch.all(scales > 0)
    tolerance = torch.maximum(scales, expected_scale) / 2 + 4e-2
    assert torch.all(
        (dequantized - reference).abs() <= tolerance
    )


@pytest.mark.parametrize("shape", [(3, 256), (2, 1024)])
def test_lightop_per_token_fp8_quant_matches_dequant_reference(
    shape: tuple[int, int],
) -> None:
    device = _hcu_device()
    try:
        from lightop.quant import per_token_quant_fp8
    except (ImportError, AttributeError) as exc:
        pytest.skip(f"lightop per-token FP8 kernel is unavailable: {exc}")
    fp8_dtype = getattr(torch, "float8_e4m3fn", None)
    if fp8_dtype is None:
        fp8_dtype = getattr(torch, "float8_e5m2", None)
    if fp8_dtype is None:
        pytest.skip("torch float8_e4m3fn/float8_e5m2 is unavailable")

    generator = torch.Generator(device=device).manual_seed(20250729)
    value = torch.randn(
        shape,
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    output = torch.empty_like(value, dtype=fp8_dtype)
    scales = torch.empty((shape[0], 1), device=device, dtype=torch.float32)
    per_token_quant_fp8(
        value,
        dtype=fp8_dtype,
        out_q=output,
        out_scale=scales,
    )
    dequantized = output.float() * scales

    torch.testing.assert_close(
        dequantized,
        value.float(),
        rtol=1.2e-1,
        atol=1.2e-1,
    )
