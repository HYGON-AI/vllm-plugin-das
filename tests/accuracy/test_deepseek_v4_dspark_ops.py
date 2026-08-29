# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Live gfx938 numerical checks for the DeepSeek-V4 DSpark operator path."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as functional


pytestmark = pytest.mark.hcu


def _hcu_device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("a live HCU/ROCm device is required")
    properties = torch.cuda.get_device_properties(0)
    arch = str(getattr(properties, "gcnArchName", "")).split(":", 1)[0]
    if arch != "gfx938":
        pytest.skip(f"DeepSeek-V4 DeepGEMM checks require gfx938, got {arch!r}")
    return torch.device("cuda", 0)


def _channel_fp8_quantize(
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    fp8_dtype = torch.float8_e4m3fn
    fp8_max = 448.0
    scales = value.float().abs().amax(dim=-1).clamp_min(1e-8) / fp8_max
    quantized = (
        (value.float() / scales.unsqueeze(-1))
        .clamp(-fp8_max, fp8_max)
        .to(fp8_dtype)
    )
    return quantized, scales


def _channel_int8_quantize(
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    scales = value.float().abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 127
    quantized = (
        (value.float() / scales).round().clamp(-127, 127).to(torch.int8)
    )
    return quantized, scales.float()


def test_contiguous_channel_fp8_deepgemm_matches_dequantized_reference() -> None:
    from deepgemm import (
        marlin_fp8_contiguous_weight,
        m_grouped_fp8_gemm_nt_contiguous,
    )

    device = _hcu_device()
    experts, tokens, hidden, output_size = 2, 256, 128, 256
    generator = torch.Generator(device=device).manual_seed(731)
    activation, activation_scale = _channel_fp8_quantize(
        torch.randn(
            (tokens, hidden),
            generator=generator,
            device=device,
        )
        * 0.25
    )
    weight, weight_scale = _channel_fp8_quantize(
        torch.randn(
            (experts, output_size, hidden),
            generator=generator,
            device=device,
        )
        * 0.25
    )
    packed_weight = marlin_fp8_contiguous_weight(weight.clone())
    m_indices = torch.cat(
        (
            torch.zeros(tokens // 2, device=device, dtype=torch.int32),
            torch.ones(tokens // 2, device=device, dtype=torch.int32),
        )
    )
    output = torch.empty(
        (tokens, output_size),
        device=device,
        dtype=torch.bfloat16,
    )

    m_grouped_fp8_gemm_nt_contiguous(
        (activation, activation_scale),
        (packed_weight, weight_scale),
        output,
        m_indices,
    )

    references = []
    tokens_per_expert = tokens // experts
    for expert in range(experts):
        start = expert * tokens_per_expert
        stop = start + tokens_per_expert
        references.append(
            (
                activation[start:stop].float()
                * activation_scale[start:stop, None]
            )
            @ (weight[expert].float() * weight_scale[expert, :, None]).T
        )
    reference = torch.cat(references)
    assert output.dtype == torch.bfloat16
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output.float(), reference, rtol=3e-2, atol=1e-2)


@pytest.mark.parametrize(
    ("hidden", "output_size"),
    ((7168, 4096), (2048, 7168)),
)
def test_masked_channel_fp8_deepgemm_matches_dequantized_reference(
    hidden: int,
    output_size: int,
) -> None:
    from deepgemm import (
        marlin_fp8_masked_weight,
        m_grouped_fp8_gemm_nt_masked,
    )

    device = _hcu_device()
    experts, max_tokens = 1, 8
    generator = torch.Generator(device=device).manual_seed(hidden + output_size)
    activation, activation_scale = _channel_fp8_quantize(
        torch.randn(
            (experts, max_tokens, hidden),
            generator=generator,
            device=device,
        )
        * 0.1
    )
    weight, weight_scale = _channel_fp8_quantize(
        torch.randn(
            (experts, output_size, hidden),
            generator=generator,
            device=device,
        )
        * 0.1
    )
    packed_weight = marlin_fp8_masked_weight(weight.clone())
    tokens_per_expert = torch.tensor(
        [max_tokens],
        device=device,
        dtype=torch.int32,
    )
    output = torch.empty(
        (experts, max_tokens, output_size),
        device=device,
        dtype=torch.bfloat16,
    )

    m_grouped_fp8_gemm_nt_masked(
        (activation, activation_scale),
        (packed_weight, weight_scale),
        output,
        tokens_per_expert,
        max_tokens,
    )

    reference = (
        activation[0].float() * activation_scale[0, :, None]
    ) @ (weight[0].float() * weight_scale[0, :, None]).T
    assert output.dtype == torch.bfloat16
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output[0].float(), reference, rtol=3e-2, atol=1e-2)


def test_lightop_fp8_silu_quant_matches_float_reference_for_ht_and_ll() -> None:
    from lightop import fuse_silu_mul_fp8_quant, fuse_silu_mul_fp8_quant_ep

    device = _hcu_device()
    generator = torch.Generator(device=device).manual_seed(735)

    value = torch.randn(
        (8, 4096),
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    quantized, scales = fuse_silu_mul_fp8_quant(value, fp8type=0)
    reference = functional.silu(value[:, :2048].float()) * value[:, 2048:].float()
    error = (quantized.float() * scales - reference).abs()
    assert quantized.dtype == torch.float8_e4m3fn
    assert scales.dtype == torch.float32
    assert torch.isfinite(scales).all()
    assert error.mean() < 1e-2
    assert error.max() < 0.35

    expert_value = torch.randn(
        (2, 8, 4096),
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    tokens_per_expert = torch.tensor([8, 5], device=device, dtype=torch.int32)
    expert_quantized, expert_scales = fuse_silu_mul_fp8_quant_ep(
        expert_value,
        fp8type=0,
        tokens_per_expert=tokens_per_expert,
    )
    for expert, count in enumerate(tokens_per_expert.tolist()):
        reference = (
            functional.silu(expert_value[expert, :count, :2048].float())
            * expert_value[expert, :count, 2048:].float()
        )
        error = (
            expert_quantized[expert, :count].float()
            * expert_scales[expert, :count]
            - reference
        ).abs()
        assert error.mean() < 1e-2
        assert error.max() < 0.35


def test_contiguous_channel_int8_deepgemm_matches_dequantized_reference() -> None:
    from deepgemm import (
        marlin_i8_contiguous_weight,
        m_grouped_i8_gemm_nt_contiguous,
    )

    device = _hcu_device()
    experts, tokens, hidden, output_size = 2, 256, 128, 256
    generator = torch.Generator(device=device).manual_seed(738)
    activation, activation_scale = _channel_int8_quantize(
        torch.randn(
            (tokens, hidden),
            generator=generator,
            device=device,
        )
        * 0.25
    )
    weight, weight_scale = _channel_int8_quantize(
        torch.randn(
            (experts, output_size, hidden),
            generator=generator,
            device=device,
        )
        * 0.25
    )
    packed_weight = marlin_i8_contiguous_weight(weight.clone())
    m_indices = torch.cat(
        (
            torch.zeros(tokens // 2, device=device, dtype=torch.int32),
            torch.ones(tokens // 2, device=device, dtype=torch.int32),
        )
    )
    output = torch.empty(
        (tokens, output_size),
        device=device,
        dtype=torch.bfloat16,
    )

    m_grouped_i8_gemm_nt_contiguous(
        (activation, activation_scale),
        (packed_weight, weight_scale),
        output,
        m_indices,
    )

    references = []
    tokens_per_expert = tokens // experts
    for expert in range(experts):
        start = expert * tokens_per_expert
        stop = start + tokens_per_expert
        references.append(
            (activation[start:stop].float() * activation_scale[start:stop])
            @ (weight[expert].float() * weight_scale[expert]).T
        )
    reference = torch.cat(references)
    assert output.dtype == torch.bfloat16
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output.float(), reference, rtol=2e-2, atol=1e-2)


@pytest.mark.parametrize(
    ("hidden", "output_size"),
    ((7168, 4096), (2048, 7168)),
)
def test_masked_channel_int8_deepgemm_matches_dequantized_reference(
    hidden: int,
    output_size: int,
) -> None:
    from deepgemm import (
        marlin_i8_masked_weight,
        m_grouped_i8_gemm_nt_masked,
    )

    device = _hcu_device()
    experts, max_tokens = 1, 8
    generator = torch.Generator(device=device).manual_seed(hidden + output_size + 1)
    activation, activation_scale = _channel_int8_quantize(
        torch.randn(
            (experts, max_tokens, hidden),
            generator=generator,
            device=device,
        )
        * 0.1
    )
    weight, weight_scale = _channel_int8_quantize(
        torch.randn(
            (experts, output_size, hidden),
            generator=generator,
            device=device,
        )
        * 0.1
    )
    packed_weight = marlin_i8_masked_weight(weight.clone())
    tokens_per_expert = torch.tensor(
        [max_tokens],
        device=device,
        dtype=torch.int32,
    )
    output = torch.empty(
        (experts, max_tokens, output_size),
        device=device,
        dtype=torch.bfloat16,
    )

    m_grouped_i8_gemm_nt_masked(
        (activation, activation_scale),
        (packed_weight, weight_scale),
        output,
        tokens_per_expert,
        max_tokens,
    )

    reference = (activation[0].float() * activation_scale[0]) @ (
        weight[0].float() * weight_scale[0]
    ).T
    assert output.dtype == torch.bfloat16
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output[0].float(), reference, rtol=2e-2, atol=1e-2)


def test_lightop_int8_clamped_silu_quant_matches_vllm_for_ht_and_ll() -> None:
    from lightop import (
        fuse_silu_mul_clamp_quant,
        fuse_silu_mul_clamp_quant_ep,
    )

    device = _hcu_device()
    generator = torch.Generator(device=device).manual_seed(739)
    limit = 10.0

    def reference(value: torch.Tensor) -> torch.Tensor:
        gate, up = value.float().chunk(2, dim=-1)
        gate = gate.clamp(max=limit)
        up = up.clamp(min=-limit, max=limit)
        return functional.silu(gate) * up

    value = torch.randn(
        (8, 4096),
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    quantized, scales = fuse_silu_mul_clamp_quant(value, limit=limit)
    error = (quantized.float() * scales - reference(value)).abs()
    assert quantized.dtype == torch.int8
    assert scales.dtype == torch.float32
    assert torch.isfinite(scales).all()
    assert error.mean() < 1.5e-2
    assert error.max() < 0.07

    expert_value = torch.randn(
        (2, 8, 4096),
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    tokens_per_expert = torch.tensor([8, 5], device=device, dtype=torch.int32)
    expert_quantized, expert_scales = fuse_silu_mul_clamp_quant_ep(
        expert_value,
        limit=limit,
        mask_m=tokens_per_expert,
        expect_m=8,
    )
    for expert, count in enumerate(tokens_per_expert.tolist()):
        error = (
            expert_quantized[expert, :count].float()
            * expert_scales[expert, :count]
            - reference(expert_value[expert, :count])
        ).abs()
        assert error.mean() < 1.5e-2
        assert error.max() < 0.07


def test_dspark_non_pcp_lightop_context_insert_writes_fp8_cache() -> None:
    from vllm_hcu.models.deepseek_v4_dspark import _insert_context_kv

    device = _hcu_device()
    num_tokens, head_dim, block_size, num_blocks = 3, 512, 4, 2
    generator = torch.Generator(device=device).manual_seed(736)
    positions = torch.tensor([0, 1, 5], device=device, dtype=torch.int32)
    inverse_frequency = 1.0 / (
        10000.0
        ** (
            torch.arange(0, 64, 2, device=device, dtype=torch.float32)
            / 64.0
        )
    )
    frequency = torch.einsum(
        "i,j->ij",
        torch.arange(16, device=device, dtype=torch.float32),
        inverse_frequency,
    )
    cos_sin_cache = torch.cat((frequency.cos(), frequency.sin()), dim=-1)
    kv = torch.randn(
        (num_tokens, head_dim),
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    cache = torch.zeros(
        (num_blocks, block_size, 584),
        device=device,
        dtype=torch.uint8,
    )
    attention = SimpleNamespace(
        n_local_heads=1,
        head_dim=head_dim,
        eps=1e-6,
        rotary_emb=SimpleNamespace(cos_sin_cache=cos_sin_cache),
        swa_cache_layer=SimpleNamespace(kv_cache=cache, block_size=block_size),
    )

    _insert_context_kv(
        attention,
        kv,
        positions,
        torch.tensor([0, 3, 5], device=device, dtype=torch.int64),
    )
    torch.cuda.synchronize(device)

    assert cache.dtype == torch.uint8
    assert torch.count_nonzero(cache) > 0


def test_dspark_non_pcp_lightop_context_insert_matches_vllm_reference() -> None:
    """The LightOp cache bytes must follow vLLM's fp8_ds_mla contract."""
    from vllm.models.deepseek_v4.common.ops import (
        dequantize_and_gather_k_cache,
        quantize_and_insert_k_cache,
    )
    from vllm_hcu.models.deepseek_v4_dspark import _insert_context_kv

    device = _hcu_device()
    num_tokens, head_dim, block_size, num_blocks = 17, 512, 64, 2
    generator = torch.Generator(device=device).manual_seed(737)
    positions = torch.tensor(
        [0, 1, 2, 3, 7, 8, 15, 16, 31, 32, 63, 64, 127, 255, 511, 1023, 2047],
        device=device,
        dtype=torch.int64,
    )
    inverse_frequency = 1.0 / (
        10000.0
        ** (
            torch.arange(0, 64, 2, device=device, dtype=torch.float32)
            / 64.0
        )
    )
    frequency = torch.einsum(
        "i,j->ij",
        torch.arange(2048, device=device, dtype=torch.float32),
        inverse_frequency,
    )
    cos_sin_cache = torch.cat((frequency.cos(), frequency.sin()), dim=-1)
    kv = torch.randn(
        (num_tokens, head_dim),
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    slot_mapping = torch.arange(num_tokens, device=device, dtype=torch.int64)
    lightop_cache = torch.zeros(
        (num_blocks, block_size, 584),
        device=device,
        dtype=torch.uint8,
    )
    reference_cache = torch.zeros_like(lightop_cache)
    attention = SimpleNamespace(
        n_local_heads=1,
        head_dim=head_dim,
        eps=1e-6,
        rotary_emb=SimpleNamespace(cos_sin_cache=cos_sin_cache),
        swa_cache_layer=SimpleNamespace(
            kv_cache=lightop_cache,
            block_size=block_size,
        ),
    )

    _insert_context_kv(attention, kv, positions, slot_mapping)

    half_rope = 32
    nope_dim = head_dim - 2 * half_rope
    cos_sin = cos_sin_cache[positions].float()
    rope = kv[:, nope_dim:].float().view(num_tokens, half_rope, 2)
    even, odd = rope[..., 0], rope[..., 1]
    cos, sin = cos_sin[:, :half_rope], cos_sin[:, half_rope:]
    rotated = torch.stack(
        (
            torch.addcmul(-odd * sin, even, cos),
            torch.addcmul(odd * cos, even, sin),
        ),
        dim=-1,
    ).reshape(num_tokens, 2 * half_rope)
    kv_reference = kv.clone()
    kv_reference[:, nope_dim:] = rotated.to(torch.bfloat16)
    quantize_and_insert_k_cache(
        kv_reference,
        reference_cache.view(num_blocks, -1),
        slot_mapping,
        block_size=block_size,
        use_fnuz=False,
    )

    def dequantize(cache: torch.Tensor) -> torch.Tensor:
        output = torch.zeros(
            (1, num_tokens, head_dim),
            device=device,
            dtype=torch.bfloat16,
        )
        dequantize_and_gather_k_cache(
            output,
            cache,
            torch.tensor([num_tokens], device=device, dtype=torch.int32),
            None,
            torch.arange(num_blocks, device=device, dtype=torch.int32).unsqueeze(0),
            block_size,
            offset=0,
            use_fnuz=False,
        )
        return output[0]

    lightop_values = dequantize(lightop_cache)
    reference_values = dequantize(reference_cache)
    torch.testing.assert_close(
        lightop_values[:, :nope_dim],
        reference_values[:, :nope_dim],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        lightop_values[:, nope_dim:],
        reference_values[:, nope_dim:],
        rtol=0,
        atol=2**-7,
    )
