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
from tests.accuracy.deepseek_v4_dspark_ops_cases import (
    test_auto_w4a8_shared_storage_feeds_ht_and_ll_with_empty_expert,
    test_contiguous_channel_fp8_deepgemm_matches_dequantized_reference,
    test_contiguous_channel_int8_deepgemm_matches_dequantized_reference,
    test_dspark_non_pcp_lightop_context_insert_matches_vllm_reference,
    test_dspark_non_pcp_lightop_context_insert_writes_fp8_cache,
    test_lightop_fp8_silu_quant_matches_float_reference_for_ht_and_ll,
    test_lightop_int8_clamped_silu_quant_matches_vllm_for_ht_and_ll,
    test_masked_channel_fp8_deepgemm_matches_dequantized_reference,
    test_masked_channel_int8_deepgemm_matches_dequantized_reference,
)
from tests.accuracy.kimi_k3_ll_ops_cases import (
    test_kimi_ht_canonical_scales_match_live_hipc_gemms,
    test_kimi_ll_situ_masked_w4a8_matches_dequantized_reference,
)

pytestmark = pytest.mark.hcu


pytestmark = pytest.mark.hcu

NUMERICAL_LIMITS = {
    "w16a16": {
        "min_reference_rms": 5e-5,
        "max_nmae": 2e-2,
        "max_nrmse": 2e-2,
        "max_abs_error": 2e-5,
    },
    "int8_w8a8": {
        "min_reference_rms": 1e-3,
        "max_nmae": 3e-2,
        "max_nrmse": 3e-2,
        "max_abs_error": 1e-3,
    },
    "fp8_w8a8": {
        "min_reference_rms": 1e-3,
        "max_nmae": 4e-2,
        "max_nrmse": 4e-2,
        "max_abs_error": 2e-3,
    },
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


def _assert_moe_numerics(
    actual: torch.Tensor,
    reference: torch.Tensor,
    quantization: str,
) -> None:
    assert actual.shape == reference.shape
    assert actual.dtype == reference.dtype
    assert torch.isfinite(reference).all()
    assert torch.isfinite(actual).all()

    limits = NUMERICAL_LIMITS[quantization]
    reference_float = reference.float()
    error = actual.float() - reference_float
    reference_rms = reference_float.square().mean().sqrt()
    reference_mean_abs = reference_float.abs().mean()
    assert reference_rms.item() >= limits["min_reference_rms"], (
        f"{quantization} reference RMS {reference_rms.item():.6g} is below "
        f"the signal floor {limits['min_reference_rms']:.6g}"
    )

    nmae = error.abs().mean() / reference_mean_abs
    nrmse = error.square().mean().sqrt() / reference_rms
    max_abs_error = error.abs().max()
    assert nmae.item() <= limits["max_nmae"], (
        f"{quantization} NMAE {nmae.item():.6g} exceeds "
        f"{limits['max_nmae']:.6g}"
    )
    assert nrmse.item() <= limits["max_nrmse"], (
        f"{quantization} NRMSE {nrmse.item():.6g} exceeds "
        f"{limits['max_nrmse']:.6g}"
    )
    assert max_abs_error.item() <= limits["max_abs_error"], (
        f"{quantization} max absolute error {max_abs_error.item():.6g} "
        f"exceeds {limits['max_abs_error']:.6g}"
    )


@pytest.mark.parametrize(
    "quantization",
    ["w16a16", "int8_w8a8", "fp8_w8a8"],
)
def test_gfx938_has_at_least_one_aiter_route(quantization: str) -> None:
    device = _hcu_device()
    arch = str(
        getattr(torch.cuda.get_device_properties(device), "gcnArchName", "")
    ).lower()
    if "gfx938" not in arch:
        pytest.skip(f"fixed AITER route-presence gate requires gfx938, got {arch}")

    from aiter.moe import MoeQuantType

    quant_type = {
        "w16a16": MoeQuantType.W16A16,
        "int8_w8a8": MoeQuantType.W8A8,
        "fp8_w8a8": MoeQuantType.FP8_W8A8,
    }[quantization]
    experts, hidden_size, intermediate_size = SHAPES[quantization]
    cache_owner = torch.empty(1, device=device)
    routes = []
    for m in (1, 16, 64):
        config = select_aiter_moe_config(
            AiterMoeProblem(
                M=m,
                E=experts,
                N1=2 * intermediate_size,
                N2=hidden_size,
                K=hidden_size,
                top_k=TOP_K,
                block_size=0,
                dtype=torch.bfloat16,
                device=device,
                quant_type=quant_type,
                activation="silu",
                use_shuffle=True,
            ),
            cache_owner=cache_owner,
        )
        if config is not None:
            routes.append((m, _solution_token(config)))

    assert routes, (
        f"gfx938 AITER exposes no supported {quantization} route for "
        "M in (1, 16, 64)"
    )


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

    with pytest.raises(AssertionError):
        _assert_moe_numerics(
            torch.zeros_like(reference),
            reference,
            quantization,
        )
    with pytest.raises(AssertionError):
        _assert_moe_numerics(
            reference * 0.5,
            reference,
            quantization,
        )
    _assert_moe_numerics(actual, reference, quantization)


def test_glm5_tp8_fp8_aiter_prefill_batch_matches_triton():
    """Regress the MOE_C config selected by GLM-5 TP8 prefill batches."""
    device = _hcu_device()
    arch = str(
        getattr(torch.cuda.get_device_properties(device), "gcnArchName", "")
    ).lower()
    if "gfx938" not in arch:
        pytest.skip(f"GLM-5 AITER regression requires gfx938, got {arch}")

    from aiter.moe import MoeQuantType
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts_impl

    m = 512
    experts = 256
    hidden_size = 6144
    intermediate_size = 256
    top_k = 8
    generator = torch.Generator(device=device).manual_seed(20261024)

    hidden_states = torch.randn(
        (m, hidden_size),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    ) * 0.1
    topk_ids = (
        torch.arange(m * top_k, device=device).reshape(m, top_k) % experts
    ).to(torch.int32)
    topk_weights = torch.rand(
        (m, top_k),
        dtype=torch.float32,
        device=device,
        generator=generator,
    )
    topk_weights /= topk_weights.sum(dim=-1, keepdim=True)

    raw_w1 = torch.randn(
        (experts, 2 * intermediate_size, hidden_size),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    ) * 0.03
    w1, w1_scale = _channel_fp8(raw_w1)
    del raw_w1
    raw_w2 = torch.randn(
        (experts, hidden_size, intermediate_size),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    ) * 0.03
    w2, w2_scale = _channel_fp8(raw_w2)
    del raw_w2

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
        use_fp8_w8a8=True,
        per_channel_quant=True,
    )
    config = select_aiter_moe_config(
        AiterMoeProblem(
            M=m,
            E=experts,
            N1=2 * intermediate_size,
            N2=hidden_size,
            K=hidden_size,
            top_k=top_k,
            block_size=0,
            dtype=hidden_states.dtype,
            device=device,
            quant_type=MoeQuantType.FP8_W8A8,
            activation="silu",
            use_shuffle=True,
        ),
        cache_owner=w1,
    )
    assert config is not None
    assert _solution_token(config) == "MOE_C"
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
        cache_owner=w1_scale,
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
    torch.cuda.synchronize(device)

    _assert_moe_numerics(actual, reference, "fp8_w8a8")


def test_glm5_aiter_biased_grouped_topk_matches_torch_reference():
    """Cover GLM-5's AITER router ABI and routed scaling semantics."""
    device = _hcu_device()
    arch = str(
        getattr(torch.cuda.get_device_properties(device), "gcnArchName", "")
    ).lower()
    if "gfx938" not in arch:
        pytest.skip(f"GLM-5 AITER regression requires gfx938, got {arch}")

    from vllm_hcu.model_executor.layers.fused_moe.aiter_ops import (
        rocm_aiter_ops,
    )

    num_tokens = 8
    num_experts = 256
    top_k = 8
    routed_scaling_factor = 2.5
    generator = torch.Generator(device=device).manual_seed(20261024)
    router_logits = torch.randn(
        (num_tokens, num_experts),
        dtype=torch.float32,
        device=device,
        generator=generator,
    )
    correction_bias = torch.randn(
        (num_experts,),
        dtype=torch.float32,
        device=device,
        generator=generator,
    ) * 0.1
    actual_weights = torch.empty(
        (num_tokens, top_k), dtype=torch.float32, device=device
    )
    actual_ids = torch.empty(
        (num_tokens, top_k), dtype=torch.int32, device=device
    )

    rocm_aiter_ops.biased_grouped_topk(
        router_logits,
        correction_bias,
        actual_weights,
        actual_ids,
        1,
        1,
        True,
        routed_scaling_factor,
    )
    torch.cuda.synchronize(device)

    assert torch.all((actual_ids >= 0) & (actual_ids < num_experts)), (
        "GLM-5 routed top-k returned an out-of-range expert id; this usually "
        "means AITER interpreted need_renorm as num_fused_shared_experts"
    )

    scores = router_logits.sigmoid()
    reference_ids = torch.topk(
        scores + correction_bias.unsqueeze(0),
        k=top_k,
        dim=-1,
        sorted=False,
    ).indices
    reference_weights = scores.gather(1, reference_ids)
    reference_weights /= reference_weights.sum(dim=-1, keepdim=True)
    reference_weights *= routed_scaling_factor

    actual_order = torch.argsort(actual_ids, dim=-1)
    reference_order = torch.argsort(reference_ids, dim=-1)
    actual_ids = torch.gather(actual_ids, 1, actual_order)
    actual_weights = torch.gather(actual_weights, 1, actual_order)
    reference_ids = torch.gather(reference_ids, 1, reference_order).to(torch.int32)
    reference_weights = torch.gather(reference_weights, 1, reference_order)

    torch.testing.assert_close(actual_ids, reference_ids, rtol=0, atol=0)
    torch.testing.assert_close(
        actual_weights,
        reference_weights,
        rtol=1e-5,
        atol=1e-6,
    )
