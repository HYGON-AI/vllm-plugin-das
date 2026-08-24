# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Numerical reference checks that execute real HCU extension kernels."""

from __future__ import annotations

from types import ModuleType, SimpleNamespace

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
        from lightop import op
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
    op.silu_and_mul_opt(actual, value)
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
        from lightop import fused_add_rms_norm
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
        from lightop.op import rmsnorm_forward_autograd
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
        from lightop import op
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
    op.gemma_rmsnorm(actual, value, weight, epsilon)
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
        from lightop import fuse_silu_mul_per_token_quant
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
        from lightop.op import rms_norm_dynamic_per_token_quant
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
    output = torch.empty_like(value, dtype=torch.int8)
    scales = torch.empty((shape[0], 1), device=device, dtype=torch.float32)
    epsilon = 1e-6
    rms_norm_dynamic_per_token_quant(
        output,
        value,
        weight,
        scales,
        epsilon,
        None,
        True,
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
        from lightop import op
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
    op.per_token_quant_fp8(output, value, scales)
    dequantized = output.float() * scales

    torch.testing.assert_close(
        dequantized,
        value.float(),
        rtol=1.2e-1,
        atol=1.2e-1,
    )


def test_hcu_interleaved_rotary_matches_float32_reference() -> None:
    device = _hcu_device()
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm_hcu.ops.rotary_embedding import HcuRotaryEmbedding

    head_size = 64
    base = 10_000
    with set_current_vllm_config(VllmConfig()):
        op = HcuRotaryEmbedding(
            head_size=head_size,
            rotary_dim=head_size,
            max_position_embeddings=512,
            base=base,
            is_neox_style=False,
            dtype=torch.bfloat16,
        )
    positions = torch.tensor([0, 1, 7, 127, 511], device=device)
    generator = torch.Generator(device=device).manual_seed(20260824)
    query = torch.randn(
        5, 4, head_size, generator=generator, device=device, dtype=torch.bfloat16
    )
    key = torch.randn(
        5, 1, head_size, generator=generator, device=device, dtype=torch.bfloat16
    )

    actual_query, actual_key = op.forward_hip(positions, query, key)

    inv_freq = 1.0 / (
        base
        ** (
            torch.arange(0, head_size, 2, device=device, dtype=torch.float32)
            / head_size
        )
    )
    angles = positions.float().unsqueeze(-1) * inv_freq.unsqueeze(0)
    cos = angles.cos().unsqueeze(-2)
    sin = angles.sin().unsqueeze(-2)

    def reference(value: torch.Tensor) -> torch.Tensor:
        value = value.float()
        even = value[..., ::2]
        odd = value[..., 1::2]
        return torch.stack(
            (even * cos - odd * sin, odd * cos + even * sin), dim=-1
        ).flatten(-2)

    torch.testing.assert_close(
        actual_query.float(), reference(query), rtol=2e-2, atol=2e-2
    )
    assert actual_key is not None
    torch.testing.assert_close(
        actual_key.float(), reference(key), rtol=2e-2, atol=2e-2
    )


def test_hcu_triton_group_fp8_ue8m0_matches_reference() -> None:
    device = _hcu_device()
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        get_fp8_min_max,
    )
    from vllm_hcu.model_executor.layers.quantization.group_fp8_runtime import (
        per_token_group_quant_fp8,
    )

    generator = torch.Generator(device=device).manual_seed(20260825)
    value = torch.randn(
        (7, 256), generator=generator, device=device, dtype=torch.bfloat16
    )
    value[0].zero_()

    quantized, scales = per_token_group_quant_fp8(
        value,
        group_size=128,
        use_ue8m0=True,
    )

    _, fp8_max = get_fp8_min_max()
    grouped = value.float().view(7, 2, 128)
    raw_scales = grouped.abs().amax(dim=-1).clamp_min(1e-10) / fp8_max
    expected_scales = torch.exp2(torch.ceil(torch.log2(raw_scales)))
    torch.testing.assert_close(scales, expected_scales, rtol=0, atol=0)
    torch.testing.assert_close(
        quantized.float().view_as(grouped) * scales.unsqueeze(-1),
        grouped,
        rtol=1.25e-1,
        atol=2e-2,
    )


def test_hcu_native_dynamic_per_token_fp8_matches_reference() -> None:
    device = _hcu_device()
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        get_fp8_min_max,
    )
    from vllm_hcu.model_executor.layers.quantization.native_fp8_runtime import (
        dynamic_per_token_quant_fp8,
    )

    generator = torch.Generator(device=device).manual_seed(20260826)
    value = torch.randn(
        (5, 6144), generator=generator, device=device, dtype=torch.bfloat16
    )
    value[0].zero_()

    quantized, scales = dynamic_per_token_quant_fp8(value)

    _, fp8_max = get_fp8_min_max()
    expected_scales = (value.abs().amax(dim=-1, keepdim=True).float() / fp8_max).clamp(
        min=1.0 / (fp8_max * 512.0)
    )
    torch.testing.assert_close(scales, expected_scales, rtol=0, atol=0)
    torch.testing.assert_close(
        quantized.float() * scales,
        value.float(),
        rtol=1.25e-1,
        atol=2e-2,
    )


@pytest.mark.parametrize("block_size", [1, 16])
def test_hcu_indexer_fp8_cache_roundtrip_matches_ue8m0_reference(
    block_size: int,
) -> None:
    device = _hcu_device()
    from vllm.platforms import current_platform
    from vllm_hcu.v1.attention.ops.rocm_aiter_mla_sparse import (
        cp_gather_indexer_k_quant_cache_triton,
        indexer_k_quant_and_cache_triton,
    )

    num_tokens = 17
    head_dim = 128
    num_blocks = (num_tokens + block_size - 1) // block_size
    generator = torch.Generator(device=device).manual_seed(20260827 + block_size)
    source = torch.randn(
        (num_tokens, head_dim),
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    source[0].zero_()
    cache = torch.zeros(
        (num_blocks, block_size, head_dim + 4),
        device=device,
        dtype=torch.uint8,
    )
    slots = torch.arange(num_tokens, device=device, dtype=torch.int64)

    indexer_k_quant_and_cache_triton(
        source,
        cache,
        slots,
        quant_block_size=head_dim,
        scale_fmt="ue8m0",
    )

    fp8_dtype = current_platform.fp8_dtype()
    gathered = torch.empty(
        (num_tokens, head_dim), device=device, dtype=fp8_dtype
    )
    gathered_scale_bytes = torch.empty(
        (num_tokens, 4), device=device, dtype=torch.uint8
    )
    block_table = torch.arange(
        num_blocks, device=device, dtype=torch.int32
    ).unsqueeze(0)
    cu_seqlen = torch.tensor(
        [0, num_tokens], device=device, dtype=torch.int32
    )
    token_to_seq = torch.zeros(num_tokens, device=device, dtype=torch.int32)
    cp_gather_indexer_k_quant_cache_triton(
        cache,
        gathered,
        gathered_scale_bytes,
        block_table,
        cu_seqlen,
        token_to_seq,
    )

    fp8_max = 224.0 if fp8_dtype == torch.float8_e4m3fnuz else 448.0
    raw_scale = source.float().abs().amax(dim=-1).clamp_min(1e-4) / fp8_max
    expected_scale = torch.exp2(torch.ceil(torch.log2(raw_scale)))
    actual_scale = gathered_scale_bytes.view(torch.float32).squeeze(-1)
    torch.testing.assert_close(actual_scale, expected_scale, rtol=0, atol=0)
    torch.testing.assert_close(
        gathered.float() * actual_scale.unsqueeze(-1),
        source.float(),
        rtol=1.25e-1,
        atol=2e-2,
    )


def test_lightop_moe_align_matches_reference_assignment() -> None:
    device = _hcu_device()
    try:
        from lightop import op
    except (ImportError, AttributeError) as exc:
        pytest.skip(f"lightop MoE align kernel is unavailable: {exc}")

    topk_ids = torch.tensor(
        [[0, 2], [1, 2], [0, 3]], device=device, dtype=torch.int32
    )
    padding_id = topk_ids.numel()
    sorted_ids = torch.full(
        (18,), padding_id, device=device, dtype=torch.int32
    )
    expert_ids = torch.empty((5,), device=device, dtype=torch.int32)
    num_tokens_post_pad = torch.empty((1,), device=device, dtype=torch.int32)

    op.moe_align_block_size(
        topk_ids,
        4,
        4,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
        None,
        None,
        None,
        False,
        False,
    )

    assert num_tokens_post_pad.item() == 16
    torch.testing.assert_close(
        expert_ids[:4],
        torch.tensor([0, 1, 2, 3], device=device, dtype=torch.int32),
    )
    expected_tokens = ({0, 4}, {2}, {1, 3}, {5})
    for expert, expected in enumerate(expected_tokens):
        block = sorted_ids[expert * 4 : (expert + 1) * 4]
        actual = {int(token) for token in block.tolist() if token != padding_id}
        assert actual == expected


def test_triton_moe_sum_uses_fp32_accumulation_on_hcu() -> None:
    device = _hcu_device()
    from vllm_hcu.patch.worker.op_opt.moe import patch_triton_moe

    class TritonExperts:
        @staticmethod
        def _supports_quant_scheme(weight_key, activation_key):
            del weight_key, activation_key
            return False

        def moe_sum(self, input, output):
            del self, input, output
            raise AssertionError("NVIDIA MoE sum must not run on HCU")

    weight_key = object()
    activation_key = object()
    module = ModuleType(patch_triton_moe.TARGET_MODULE)
    module.current_platform = SimpleNamespace(is_rocm=lambda: True)
    module.kInt8StaticChannelSym = weight_key
    module.kInt8DynamicTokenSym = activation_key
    module.TritonExperts = TritonExperts
    assert patch_triton_moe.apply_to_module(module) is True

    generator = torch.Generator(device=device).manual_seed(20260824)
    expert_output = torch.randn(
        (17, 8, 511),
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    reduced = torch.empty((17, 511), device=device, dtype=torch.bfloat16)

    TritonExperts().moe_sum(expert_output, reduced)

    expected = expert_output.float().sum(dim=1).to(torch.bfloat16)
    torch.testing.assert_close(reduced, expected, rtol=0, atol=0)
