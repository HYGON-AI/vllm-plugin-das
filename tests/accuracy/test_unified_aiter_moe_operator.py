# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Synthetic operator checks for unified AITER MoE routing."""

from __future__ import annotations

from contextlib import ExitStack

import pytest
import torch

from vllm_hcu.model_executor.layers.fused_moe.aiter_moe_dispatch import (
    AiterMoeProblem,
    execute_aiter_moe,
    prepare_aiter_moe_scales,
    prepare_aiter_moe_weights,
    select_aiter_moe_config,
)
from vllm_hcu.model_executor.layers.fused_moe.aiter_runtime import (
    aiter_asm_boltops_fp8_quant_context,
    aiter_asm_boltops_int8_quant_context,
)

TOLERANCES = {
    "w16a16": {"rtol": 2e-2, "atol": 2e-2},
    "int8_w8a8": {"rtol": 8e-2, "atol": 8e-2},
    "fp8_w8a8": {"rtol": 5e-2, "atol": 5e-2},
}

TOP_K = 2

SHAPES = {
    "w16a16": (8, 128, 64),
    "int8_w8a8": (256, 2048, 128),
    "fp8_w8a8": (256, 2048, 128),
}


def _hcu_device() -> torch.device:
    if not torch.cuda.is_available() or torch.version.hip is None:
        pytest.skip("ROCm/HCU device is unavailable")
    try:
        from aiter.moe import MoeQuantType  # noqa: F401
    except (ImportError, ModuleNotFoundError) as exc:
        pytest.skip(f"public AITER MoE API is unavailable: {exc}")
    return torch.device("cuda", 0)


def _inputs(
    m: int,
    device: torch.device,
    *,
    experts: int,
    hidden_size: int,
    intermediate_size: int,
):
    generator = torch.Generator(device=device).manual_seed(20260831 + m)
    hidden_states = torch.randn(
        (m, hidden_size),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    ) * 0.1
    topk_ids = (
        torch.arange(m * TOP_K, device=device).reshape(m, TOP_K) % experts
    ).to(torch.int32)
    topk_weights = torch.rand(
        (m, TOP_K),
        dtype=torch.float32,
        device=device,
        generator=generator,
    )
    topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
    w1 = torch.randn(
        (experts, 2 * intermediate_size, hidden_size),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    ) * 0.03
    w2 = torch.randn(
        (experts, hidden_size, intermediate_size),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    ) * 0.03
    return hidden_states, w1, w2, topk_weights, topk_ids


def _channel_int8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = weight.float().abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 127
    quantized = torch.round(weight.float() / scale).clamp(-127, 127).to(torch.int8)
    return quantized, scale


def _channel_fp8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    dtype = torch.float8_e4m3fn
    maximum = torch.finfo(dtype).max
    scale = weight.float().abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / maximum
    quantized = (weight.float() / scale).clamp(-maximum, maximum).to(dtype)
    return quantized, scale


def _solution_token(config: object) -> str:
    solution = getattr(config, "solution_type", None)
    solution = getattr(solution, "value", solution)
    return str(solution).rsplit(".", 1)[-1].upper()


@pytest.mark.parametrize("m", [1, 16, 64])
@pytest.mark.parametrize("quantization", ["w16a16", "int8_w8a8", "fp8_w8a8"])
def test_unified_aiter_moe_matches_vllm_triton(
    m: int,
    quantization: str,
    request: pytest.FixtureRequest,
):
    device = _hcu_device()
    from aiter.moe import MoeQuantType
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts_impl

    experts, hidden_size, intermediate_size = SHAPES[quantization]
    hidden_states, raw_w1, raw_w2, topk_weights, topk_ids = _inputs(
        m,
        device,
        experts=experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )
    w1_scale = None
    w2_scale = None
    native_kwargs = {
        "use_fp8_w8a8": False,
        "use_int8_w8a8": False,
        "per_channel_quant": False,
    }
    if quantization == "w16a16":
        quant_type = MoeQuantType.W16A16
        w1, w2 = raw_w1, raw_w2
    elif quantization == "int8_w8a8":
        quant_type = MoeQuantType.W8A8
        w1, w1_scale = _channel_int8(raw_w1)
        w2, w2_scale = _channel_int8(raw_w2)
        native_kwargs.update(use_int8_w8a8=True, per_channel_quant=True)
    else:
        quant_type = MoeQuantType.FP8_W8A8
        try:
            w1, w1_scale = _channel_fp8(raw_w1)
            w2, w2_scale = _channel_fp8(raw_w2)
        except AttributeError as exc:
            pytest.skip(f"FP8 weight quantization is unavailable: {exc}")
        native_kwargs.update(use_fp8_w8a8=True, per_channel_quant=True)

    reference = fused_experts_impl(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        activation="silu",
        global_num_experts=experts,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        **native_kwargs,
    )
    problem = AiterMoeProblem(
        M=m,
        E=experts,
        N1=2 * intermediate_size,
        N2=hidden_size,
        K=hidden_size,
        top_k=TOP_K,
        block_size=0,
        dtype=hidden_states.dtype,
        device=device,
        quant_type=quant_type,
        activation="silu",
        use_shuffle=True,
    )
    config = select_aiter_moe_config(problem, cache_owner=w1)
    if config is None:
        pytest.skip(
            "AITER has no "
            f"{quantization} config for M={m}, E={experts}, "
            f"K={hidden_size}, N={intermediate_size}"
        )
    route = _solution_token(config)
    if request.config.option.verbose:
        print(f"unified AITER route: quant={quantization}, M={m}, route={route}")

    prepared_w1, prepared_w2 = prepare_aiter_moe_weights(
        w1,
        w2,
        config,
        cache_owner=w1,
    )
    prepared_w1_scale, prepared_w2_scale = prepare_aiter_moe_scales(
        w1_scale,
        w2_scale,
        config,
        cache_owner=w1_scale if w1_scale is not None else w1,
    )
    with ExitStack() as stack:
        stack.enter_context(
            aiter_asm_boltops_int8_quant_context(
                enabled=quantization == "int8_w8a8" and route == "ASM"
            )
        )
        stack.enter_context(
            aiter_asm_boltops_fp8_quant_context(
                enabled=quantization == "fp8_w8a8" and route == "ASM"
            )
        )
        actual = execute_aiter_moe(
            config,
            hidden_states=hidden_states,
            w1=prepared_w1,
            w2=prepared_w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation="silu",
            w1_scale=prepared_w1_scale,
            w2_scale=prepared_w2_scale,
            global_num_experts=experts,
            use_weight_shuffle=bool(getattr(config, "need_shuffle", False)),
            output_dtype=hidden_states.dtype,
        )

    assert torch.isfinite(reference).all()
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, reference, **TOLERANCES[quantization])
