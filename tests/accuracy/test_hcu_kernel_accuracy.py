# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Numerical reference checks that execute real HCU extension kernels."""

from __future__ import annotations

import __future__
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn.functional as functional


pytestmark = pytest.mark.hcu

_REPOSITORY = Path(__file__).resolve().parents[2]


def _deepseek_v4_fused_insert_method() -> Any:
    """Compile the production wrapper method without importing its vLLM graph."""
    source_path = (
        _REPOSITORY
        / "vllm_hcu/model_executor/layers/deepseek_v4_attention.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    wrapper = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "DeepseekV4MultiHeadLatentAttentionWrapper"
    )
    method = next(
        node
        for node in wrapper.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_fused_qnorm_rope_kv_insert"
    )
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    namespace: dict[str, Any] = {
        "cast": lambda _type, value: value,
        "torch": torch,
    }
    exec(
        compile(
            module,
            str(source_path),
            "exec",
            __future__.annotations.compiler_flag,
        ),
        namespace,
    )
    return namespace[method.name]


def _deepseek_v4_cache_gather() -> Any:
    """Load the cache decoder without its vLLM-version-sensitive package init."""
    source_path = (
        _REPOSITORY
        / "vllm_hcu/v1/attention/ops/deepseek_v4_ops/cache_utils.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_hcu_test_deepseek_v4_cache_utils", source_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.dequantize_and_gather_k_cache


def _hcu_device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("a live HCU/ROCm device is required")
    properties = torch.cuda.get_device_properties(0)
    if not hasattr(properties, "gcnArchName"):
        pytest.skip("the active torch device is not an HCU/ROCm device")
    return torch.device("cuda", 0)


@pytest.mark.parametrize("layout", ["NHD", "HND"])
def test_hcu_reshape_and_cache_flash_fp8_matches_torch_quantization(
    layout: str,
) -> None:
    device = _hcu_device()
    import vllm_hcu.hcu_ops  # noqa: F401

    # Exercise the grid-stride loop beyond HCU's 256-thread launch bound.
    num_blocks, block_size, num_heads, head_size = 2, 4, 4, 128
    generator = torch.Generator(device=device).manual_seed(20260904)
    key = torch.randn(
        (4, num_heads, head_size),
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    value = torch.randn(
        key.shape,
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    if layout == "NHD":
        cache_stride = (
            block_size * num_heads * head_size + 64,
            num_heads * head_size,
            head_size,
            1,
        )
        key_cache = torch.empty_strided(
            (num_blocks, block_size, num_heads, head_size),
            cache_stride,
            device=device,
            dtype=torch.float8_e4m3fn,
        ).zero_()
        value_cache = torch.empty_strided(
            key_cache.shape,
            cache_stride,
            device=device,
            dtype=key_cache.dtype,
        ).zero_()
    else:
        cache_stride = (
            num_heads * block_size * head_size + 64,
            head_size,
            block_size * head_size,
            1,
        )
        key_cache = torch.empty_strided(
            (num_blocks, block_size, num_heads, head_size),
            cache_stride,
            device=device,
            dtype=torch.float8_e4m3fn,
        ).zero_()
        value_cache = torch.empty_strided(
            key_cache.shape,
            cache_stride,
            device=device,
            dtype=key_cache.dtype,
        ).zero_()

    slot_mapping = torch.tensor([1, 5, -1], device=device, dtype=torch.int64)
    k_scale = torch.linspace(
        0.25, 0.5, num_heads, device=device, dtype=torch.float32
    )
    v_scale = torch.linspace(
        0.5, 0.75, num_heads, device=device, dtype=torch.float32
    )

    torch.ops.hcu_ops.reshape_and_cache_flash(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        "fp8_e4m3",
        k_scale,
        v_scale,
    )
    torch.cuda.synchronize(device)

    expected_key = (key[:2] / k_scale.reshape(1, -1, 1)).to(
        torch.float8_e4m3fn
    )
    expected_value = (value[:2] / v_scale.reshape(1, -1, 1)).to(
        torch.float8_e4m3fn
    )
    torch.testing.assert_close(key_cache[0, 1].float(), expected_key[0].float())
    torch.testing.assert_close(key_cache[1, 1].float(), expected_key[1].float())
    torch.testing.assert_close(
        value_cache[0, 1].float(), expected_value[0].float()
    )
    torch.testing.assert_close(
        value_cache[1, 1].float(), expected_value[1].float()
    )
    assert torch.count_nonzero(key_cache[:, 0].float()) == 0
    assert torch.count_nonzero(value_cache[:, 0].float()) == 0


def test_hcu_reshape_and_cache_flash_replays_in_cuda_graph() -> None:
    device = _hcu_device()
    import vllm_hcu.hcu_ops  # noqa: F401

    num_heads, head_size, block_size = 2, 16, 4
    key = torch.zeros(
        (2, num_heads, head_size), device=device, dtype=torch.bfloat16
    )
    value = torch.zeros_like(key)
    key_cache = torch.zeros(
        (2, block_size, num_heads, head_size),
        device=device,
        dtype=torch.float8_e4m3fn,
    )
    value_cache = torch.zeros_like(key_cache)
    slot_mapping = torch.tensor([0, 5], device=device, dtype=torch.int64)
    scale = torch.ones((1,), device=device, dtype=torch.float32)

    def write_cache() -> None:
        torch.ops.hcu_ops.reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            "fp8_e4m3",
            scale,
            scale,
        )

    warmup_stream = torch.cuda.Stream(device=device)
    warmup_stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(warmup_stream):
        write_cache()
    torch.cuda.current_stream(device).wait_stream(warmup_stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        write_cache()

    replay_key = torch.full_like(key, 1.5)
    replay_value = torch.full_like(value, -2.0)
    key_cache.zero_()
    value_cache.zero_()
    key.copy_(replay_key)
    value.copy_(replay_value)
    graph.replay()
    torch.cuda.synchronize(device)

    torch.testing.assert_close(key_cache[0, 0].float(), replay_key[0].float())
    torch.testing.assert_close(key_cache[1, 1].float(), replay_key[1].float())
    torch.testing.assert_close(
        value_cache[0, 0].float(), replay_value[0].float()
    )
    torch.testing.assert_close(
        value_cache[1, 1].float(), replay_value[1].float()
    )


def test_hcu_reshape_and_cache_flash_saturates_fp8_e4m3_overflow() -> None:
    device = _hcu_device()
    import vllm_hcu.hcu_ops  # noqa: F401

    values = torch.tensor(
        [
            448.0,
            449.0,
            466.0,
            470.0,
            479.0,
            480.0,
            500.0,
            float("inf"),
            -float("inf"),
            float("nan"),
            -466.0,
            -470.0,
            -479.0,
            -500.0,
        ],
        device=device,
        dtype=torch.bfloat16,
    ).reshape(1, 1, 14)
    key_cache = torch.zeros(
        (1, 1, 1, 14), device=device, dtype=torch.float8_e4m3fn
    )
    value_cache = torch.zeros_like(key_cache)
    slot_mapping = torch.zeros((1,), device=device, dtype=torch.int64)
    scale = torch.ones((1,), device=device, dtype=torch.float32)

    torch.ops.hcu_ops.reshape_and_cache_flash(
        values,
        values,
        key_cache,
        value_cache,
        slot_mapping,
        "fp8_e4m3",
        scale,
        scale,
    )
    torch.cuda.synchronize(device)

    expected_bytes = torch.tensor(
        [[0x7E, 0x7E, 0x7E, 0x7E, 0x7E, 0x7E, 0x7E,
          0x7E, 0xFE, 0x7F, 0xFE, 0xFE, 0xFE, 0xFE]],
        device=device,
        dtype=torch.uint8,
    )
    expected_values = torch.tensor(
        [[448.0, 448.0, 448.0, 448.0, 448.0, 448.0, 448.0,
          448.0, -448.0, float("nan"), -448.0, -448.0, -448.0, -448.0]],
        device=device,
        dtype=torch.float32,
    )
    for cache in (key_cache, value_cache):
        assert torch.equal(cache[0, 0].view(torch.uint8), expected_bytes)
        torch.testing.assert_close(
            cache[0, 0].float(),
            expected_values,
            equal_nan=True,
        )


def test_hcu_reshape_and_cache_flash_is_opaque_to_torch_compile() -> None:
    device = _hcu_device()
    import vllm_hcu.hcu_ops  # noqa: F401

    key = torch.ones((1, 2, 128), device=device, dtype=torch.bfloat16)
    value = -key
    key_cache = torch.zeros(
        (1, 4, 2, 128), device=device, dtype=torch.float8_e4m3fn
    )
    value_cache = torch.zeros_like(key_cache)
    slot_mapping = torch.zeros((1,), device=device, dtype=torch.int64)
    scale = torch.ones((1,), device=device, dtype=torch.float32)

    def write_cache(
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        scale: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        torch.ops.hcu_ops.reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            "fp8_e4m3",
            scale,
            scale,
        )
        return key_cache, value_cache

    compiled = torch.compile(write_cache, backend="aot_eager", fullgraph=True)
    actual_key, actual_value = compiled(
        key, value, key_cache, value_cache, slot_mapping, scale
    )
    torch.cuda.synchronize(device)

    torch.testing.assert_close(actual_key[0, 0].float(), key[0].float())
    torch.testing.assert_close(actual_value[0, 0].float(), value[0].float())


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


def test_lightop_moe_align_out_kernel_produces_valid_expert_blocks() -> None:
    device = _hcu_device()
    try:
        from lightop.moe import moe_align_block_size_out
    except (ImportError, AttributeError) as exc:
        pytest.skip(f"lightop MoE align kernel is unavailable: {exc}")

    topk_ids = torch.tensor(
        [[0, 1], [1, 2], [3, 0], [2, 3]],
        device=device,
        dtype=torch.int32,
    )
    num_experts = 4
    block_size = 4
    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
    sorted_ids = torch.full(
        (max_num_tokens_padded,),
        topk_ids.numel(),
        device=device,
        dtype=torch.int32,
    )
    expert_ids = torch.empty(
        ((max_num_tokens_padded + block_size - 1) // block_size,),
        device=device,
        dtype=torch.int32,
    )
    num_tokens_post_pad = torch.empty((1,), device=device, dtype=torch.int32)

    moe_align_block_size_out(
        topk_ids,
        num_experts,
        block_size,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
        None,
        None,
        None,
        is_ep=False,
        is_fuse_fill=False,
    )
    torch.cuda.synchronize(device)

    valid_count = int(num_tokens_post_pad.item())
    assert valid_count == 16
    valid_sorted = sorted_ids[:valid_count].cpu()
    valid_experts = expert_ids[: valid_count // block_size].cpu()
    flattened_topk = topk_ids.flatten().cpu()
    routed = valid_sorted[valid_sorted < topk_ids.numel()]
    assert torch.equal(torch.sort(routed).values, torch.arange(topk_ids.numel()))
    for block_index, expert in enumerate(valid_experts.tolist()):
        token_indices = valid_sorted[
            block_index * block_size : (block_index + 1) * block_size
        ]
        token_indices = token_indices[token_indices < topk_ids.numel()]
        assert torch.all(flattened_topk[token_indices] == expert)


def test_lightop_sparse_mqa_matches_fp32_reference() -> None:
    device = _hcu_device()
    try:
        from lightop.attention import mqa_logits
    except (ImportError, AttributeError) as exc:
        pytest.skip(f"lightop sparse MQA kernel is unavailable: {exc}")
    from vllm.platforms import current_platform
    from vllm_hcu.platforms.hcu import on_gfx938

    use_fp8 = on_gfx938()
    kernel_dtype = current_platform.fp8_dtype() if use_fp8 else torch.bfloat16
    num_queries, num_keys, num_heads, head_dim = 4, 128, 8, 128
    generator = torch.Generator(device=device).manual_seed(20250825)
    query = torch.randn(
        (num_queries, num_heads, head_dim),
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    ).to(kernel_dtype)
    key = torch.randn(
        (num_keys, head_dim),
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    ).to(kernel_dtype)
    weights = torch.rand(
        (num_queries, num_heads),
        generator=generator,
        device=device,
        dtype=torch.float32,
    ).contiguous()
    row_starts = torch.zeros((num_queries,), device=device, dtype=torch.int32)
    row_ends = torch.full(
        (num_queries,), num_keys, device=device, dtype=torch.int32
    )
    key_scale = torch.ones((num_keys,), device=device, dtype=torch.float32)
    kernel_key_scale = key_scale if use_fp8 else None

    actual = mqa_logits(
        query,
        key,
        weights,
        row_starts,
        row_ends,
        kernel_key_scale,
    )
    score = (
        torch.einsum(
            "mhd,nd->hmn",
            query.to(torch.bfloat16),
            key.to(torch.bfloat16),
        ).float()
        * key_scale
    )
    reference = (score.relu() * weights.unsqueeze(-1).transpose(0, 1)).sum(dim=0)

    torch.testing.assert_close(actual, reference, rtol=2e-2, atol=2e-1)


def test_lightop_paged_sparse_mqa_matches_packed_cache_reference() -> None:
    device = _hcu_device()
    try:
        from lightop.attention import paged_mqa_logits
    except (ImportError, AttributeError) as exc:
        pytest.skip(f"lightop paged sparse MQA kernel is unavailable: {exc}")
    from vllm.platforms import current_platform
    from vllm_hcu.v1.attention.ops.rocm_aiter_mla_sparse import (
        fp8_paged_mqa_logits_torch,
        indexer_k_quant_and_cache_triton,
    )

    fp8_dtype = current_platform.fp8_dtype()
    block_size, num_heads, head_dim = 64, 8, 128
    generator = torch.Generator(device=device).manual_seed(20250826)
    query = torch.randn(
        (1, 1, num_heads, head_dim),
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    ).to(fp8_dtype)
    key = torch.randn(
        (block_size, head_dim),
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    kv_cache = torch.zeros(
        (1, block_size, 1, head_dim + 4),
        device=device,
        dtype=torch.uint8,
    )
    slot_mapping = torch.arange(block_size, device=device, dtype=torch.int64)
    indexer_k_quant_and_cache_triton(
        key,
        kv_cache,
        slot_mapping,
        quant_block_size=128,
        scale_fmt=None,
    )
    weights = torch.rand(
        (1, num_heads),
        generator=generator,
        device=device,
        dtype=torch.float32,
    ).contiguous()
    context_lens = torch.tensor([block_size], device=device, dtype=torch.int32)
    block_tables = torch.tensor([[0]], device=device, dtype=torch.int32)

    actual = paged_mqa_logits(
        query,
        kv_cache,
        weights,
        context_lens,
        block_tables,
        None,
        block_size,
        False,
    )
    reference = fp8_paged_mqa_logits_torch(
        query,
        kv_cache,
        weights,
        context_lens,
        block_tables,
        block_size,
    )

    torch.testing.assert_close(actual, reference, rtol=1e-4, atol=1e-4)


def test_lightop_deepseek_v4_fused_insert_updates_q_and_cache() -> None:
    device = _hcu_device()
    arch = torch.cuda.get_device_properties(device).gcnArchName.split(":")[0]
    if arch != "gfx938":
        pytest.skip("LightOp DeepSeek V4 fused insert requires gfx938")
    try:
        from lightop import attention as lightop_attention

        getattr(
            lightop_attention,
            "fused_deepseek_v4_qnorm_rope_kvnorm_rope_quant_insert_int32",
        )
    except (ImportError, AttributeError) as exc:
        pytest.skip(f"lightop DeepSeek V4 fused insert is unavailable: {exc}")
    fused_insert = _deepseek_v4_fused_insert_method()
    dequantize_and_gather_k_cache = _deepseek_v4_cache_gather()

    num_tokens, num_heads, head_dim = 2, 4, 512
    block_size = 64
    epsilon = 1e-6
    generator = torch.Generator(device=device).manual_seed(20250827)
    query = torch.randn(
        (num_tokens, num_heads, head_dim),
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    query_before = query.float().clone()
    key_value = torch.randn(
        (num_tokens, head_dim),
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    key_value_before = key_value.float().clone()
    kv_norm_weight = torch.randn(
        (head_dim,),
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    sentinel = 0xA5
    cache = torch.full(
        (1, block_size, 584), sentinel, device=device, dtype=torch.uint8
    )
    slot_mapping = torch.tensor(
        [1, 99, 3, 99], device=device, dtype=torch.int64
    )[::2]
    assert not slot_mapping.is_contiguous()
    positions = torch.tensor([1, 3], device=device, dtype=torch.int64)
    angles = torch.linspace(
        0.1, 1.2, steps=4 * 32, device=device, dtype=torch.float32
    ).view(4, 32)
    cos_sin_cache = torch.cat((angles.cos(), angles.sin()), dim=-1)
    metadata = SimpleNamespace(slot_mapping=slot_mapping, block_size=block_size)
    wrapper = SimpleNamespace(
        swa_cache_layer=SimpleNamespace(prefix="swa", kv_cache=cache),
        kv_norm=SimpleNamespace(weight=SimpleNamespace(data=kv_norm_weight)),
        rotary_emb=SimpleNamespace(cos_sin_cache=cos_sin_cache),
        eps=epsilon,
    )

    fused_insert(wrapper, query, key_value, positions, {"swa": metadata})
    torch.cuda.synchronize(device)

    def apply_rope(value: torch.Tensor) -> torch.Tensor:
        result = value.clone()
        rope = result[..., 448:].reshape(*result.shape[:-1], 32, 2)
        selected = cos_sin_cache[positions]
        broadcast_dims = (positions.shape[0],) + (1,) * (
            value.ndim - 2
        ) + (32,)
        cos = selected[:, :32].reshape(broadcast_dims)
        sin = selected[:, 32:].reshape(broadcast_dims)
        even = rope[..., 0].clone()
        odd = rope[..., 1].clone()
        rope[..., 0] = even * cos - odd * sin
        rope[..., 1] = odd * cos + even * sin
        return result

    reference_query = apply_rope(
        query_before
        * torch.rsqrt(
            query_before.square().mean(dim=-1, keepdim=True) + epsilon
        )
    )
    torch.testing.assert_close(
        query.float(), reference_query, rtol=1e-2, atol=1e-2
    )

    gathered_key = torch.empty(
        (1, 4, head_dim), device=device, dtype=torch.bfloat16
    )
    dequantize_and_gather_k_cache(
        gathered_key,
        cache,
        torch.tensor([4], device=device, dtype=torch.int32),
        None,
        torch.tensor([[0]], device=device, dtype=torch.int32),
        block_size,
        0,
    )
    torch.cuda.synchronize(device)
    reference_key = (
        key_value_before
        * torch.rsqrt(
            key_value_before.square().mean(dim=-1, keepdim=True) + epsilon
        )
        * kv_norm_weight.float()
    )
    reference_key = apply_rope(reference_key)
    gathered_targets = gathered_key[0, slot_mapping].float()
    torch.testing.assert_close(
        gathered_targets[:, :448],
        reference_key[:, :448],
        rtol=1.5e-1,
        atol=2.5e-1,
    )
    torch.testing.assert_close(
        gathered_targets[:, 448:],
        reference_key[:, 448:],
        rtol=1e-2,
        atol=1e-2,
    )

    raw_cache = cache.view(-1)
    scale_base = block_size * 576
    for untouched_slot in (0, 2):
        token_data = raw_cache[
            untouched_slot * 576 : (untouched_slot + 1) * 576
        ]
        scale_start = scale_base + untouched_slot * 8
        token_scales = raw_cache[scale_start : scale_start + 8]
        assert torch.all(token_data == sentinel)
        assert torch.all(token_scales == sentinel)
