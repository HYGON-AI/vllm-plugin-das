# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Kimi-K3 SiTU DeepEP HT/LL numerical checks on live HCU."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.hcu


def _device() -> torch.device:
    if not torch.cuda.is_available() or torch.version.hip is None:
        pytest.skip("a live HCU/ROCm device is required")
    return torch.device("cuda", 0)


def _pack_int4_high_low(weight: torch.Tensor) -> torch.Tensor:
    nibbles = weight.to(torch.int16) & 0xF
    pairs = nibbles.view(*weight.shape[:-1], -1, 2)
    return ((pairs[..., 0] << 4) | pairs[..., 1]).to(torch.int8)


def _unpack_int4_high_low(packed: torch.Tensor) -> torch.Tensor:
    values = packed.view(torch.uint8)
    high = ((values >> 4) & 0xF).to(torch.int16)
    low = (values & 0xF).to(torch.int16)
    high = torch.where(high >= 8, high - 16, high).to(torch.int8)
    low = torch.where(low >= 8, low - 16, low).to(torch.int8)
    return torch.stack((high, low), dim=-1).flatten(-2).contiguous()


def _situ(gateup: torch.Tensor, beta: float, linear_beta: float) -> torch.Tensor:
    gate, up = gateup.float().chunk(2, dim=-1)
    gate = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    up = linear_beta * torch.tanh(up / linear_beta)
    return gate * up


def test_kimi_ht_canonical_scales_match_live_hipc_gemms(monkeypatch):
    """Real HT apply/GEMMs/SiTU; one-expert identity routing is a test double."""
    from deepgemm import m_grouped_w4a8_gemm_nt_contiguous_hipc
    from lightop.activation import fuse_situ_mul_quant_contiguous
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm_hcu.model_executor.layers.quantization.slimquant_w4a8 import (
        SlimQuantW4A8Int8AiterMoEMethod,
    )
    from vllm_hcu.model_executor.layers.quantization import (
        slimquant_w4a8_deepgemm_runtime as runtime,
    )

    device = _device()
    tokens, hidden, intermediate = 256, 7168, 1536
    rng = torch.Generator(device=device).manual_seed(20261016)
    layer = torch.nn.Module()
    logical_weights = []
    for name, n, k in (("w13", intermediate * 2, hidden), ("w2", hidden, intermediate)):
        logical = torch.randint(-2, 3, (1, n, k), dtype=torch.int8, device=device, generator=rng)
        logical_weights.append(logical)
        layer.register_parameter(name + "_weight", torch.nn.Parameter(
            _pack_int4_high_low(logical), requires_grad=False))
        scale = torch.rand((1, n, 1), device=device, generator=rng).mul_(0.001).add_(0.001)
        layer.register_parameter(name + "_weight_scale", torch.nn.Parameter(scale, requires_grad=False))
    layer.w13_input_scale = layer.w2_input_scale = None
    method = object.__new__(SlimQuantW4A8Int8AiterMoEMethod)
    method.moe = SimpleNamespace(_hcu_kimi_k3=True, moe_parallel_config=SimpleNamespace(
        all2all_backend="deepep_high_throughput"))
    method._prepare_kimi_ht_scales(layer)
    experts = object.__new__(runtime.KimiK3HTExperts)
    experts.quant_config = method.get_fused_moe_quant_config(layer)
    experts.process_weights_after_loading(layer)
    experts.situ_beta, experts.situ_linear_beta = 1.25, 2.0
    activation = torch.randint(-8, 8, (tokens, hidden), device=device, dtype=torch.int8, generator=rng)
    activation_scale = torch.full((tokens, 1), 0.02, device=device)
    ids = torch.zeros(tokens, device=device, dtype=torch.int32)
    topk_ids = ids[:, None]
    topk_weights = torch.ones((tokens, 1), device=device)
    # These wrappers replace only distributed routing for this one local expert.
    monkeypatch.setattr(runtime, "compute_aligned_M", lambda **kwargs: tokens)
    monkeypatch.setattr(runtime, "deepgemm_moe_permute", lambda **kwargs: (
        kwargs["aq"], kwargs["aq_scale"], ids, ids, 256))
    monkeypatch.setattr(runtime, "deepgemm_unpermute_and_reduce",
                        lambda **kwargs: kwargs["output"].copy_(kwargs["a"]))
    gemms, quantized = [], []
    def gemm(a, w, out, indices):
        m_grouped_w4a8_gemm_nt_contiguous_hipc(a, w, out, indices)
        gemms.append((w[1], out.clone()))
    def situ(x, **kwargs):
        result = fuse_situ_mul_quant_contiguous(x, **kwargs)
        quantized.append(result)
        return result
    monkeypatch.setattr(runtime, "m_grouped_w4a8_gemm_nt_contiguous_hipc", gemm)
    experts._fuse_situ_mul_quant = situ
    workspace_shape = (tokens, max(hidden, 2 * intermediate))
    output = torch.empty((tokens, hidden), device=device, dtype=torch.bfloat16)
    experts.apply(
        output, activation, layer.w13_weight, layer.w2_weight,
        topk_weights, topk_ids, MoEActivation.SITU, 1, None,
        activation_scale, None,
        torch.empty(workspace_shape, device=device, dtype=torch.bfloat16),
        torch.empty(workspace_shape, device=device, dtype=torch.bfloat16),
        None, False,
    )
    assert gemms[0][0] is layer.w13_weight_scale
    assert gemms[1][0] is layer.w2_weight_scale
    reference_gateup = (activation.float() * activation_scale) @ (
        logical_weights[0][0].float() * layer.w13_weight_scale[0] * 16).T
    torch.testing.assert_close(gemms[0][1].float(), reference_gateup, rtol=3e-2, atol=0.1)
    q, scale = quantized[0]
    dequantized = q.float() * scale.reshape(tokens, 1)
    torch.testing.assert_close(dequantized, _situ(gemms[0][1], 1.25, 2.0), rtol=5e-2, atol=0.1)
    reference_down = dequantized @ (
        logical_weights[1][0].float() * layer.w2_weight_scale[0] * 16).T
    torch.testing.assert_close(output.float(), reference_down, rtol=4e-2, atol=0.1)
    assert torch.isfinite(output).all()


def test_kimi_ll_situ_masked_w4a8_matches_dequantized_reference() -> None:
    from deepgemm import m_grouped_w4a8_gemm_nt_masked_hipc
    from lightop.activation import fuse_situ_mul_quant_ep
    from vllm_hcu.model_executor.layers.quantization.slimquant_w4a8_deepgemm_runtime import (
        KimiK3LLExperts,
    )

    device = _device()
    experts_count, max_tokens, hidden, intermediate = 2, 32, 7168, 1536
    counts = torch.tensor([32, 16], device=device, dtype=torch.int32)
    generator = torch.Generator(device=device).manual_seed(20261011)
    activation = torch.randint(
        -8,
        8,
        (experts_count, max_tokens, hidden),
        generator=generator,
        device=device,
        dtype=torch.int8,
    )
    activation_scale = torch.rand(
        (experts_count, max_tokens, 1),
        generator=generator,
        device=device,
        dtype=torch.float32,
    ).add_(0.01)
    logical_w13 = torch.randint(
        -8,
        8,
        (experts_count, 2 * intermediate, hidden),
        generator=generator,
        device=device,
        dtype=torch.int8,
    )
    logical_w2 = torch.randint(
        -8,
        8,
        (experts_count, hidden, intermediate),
        generator=generator,
        device=device,
        dtype=torch.int8,
    )
    checkpoint_w13 = _pack_int4_high_low(logical_w13)
    checkpoint_w2 = _pack_int4_high_low(logical_w2)
    raw_w13 = checkpoint_w13.clone()
    raw_w2 = checkpoint_w2.clone()
    scale_w13 = torch.rand(
        (experts_count, 2 * intermediate, 1),
        generator=generator,
        device=device,
        dtype=torch.float32,
    ).add_(0.01)
    scale_w2 = torch.rand(
        (experts_count, hidden, 1),
        generator=generator,
        device=device,
        dtype=torch.float32,
    ).add_(0.01)

    layer = torch.nn.Module()
    layer.w13_weight = torch.nn.Parameter(checkpoint_w13, requires_grad=False)
    layer.w2_weight = torch.nn.Parameter(checkpoint_w2, requires_grad=False)
    layer.w13_weight_scale = torch.nn.Parameter(scale_w13, requires_grad=False)
    layer.w2_weight_scale = torch.nn.Parameter(scale_w2, requires_grad=False)
    experts = object.__new__(KimiK3LLExperts)
    experts._deepgemm_w13 = None
    experts._deepgemm_w2 = None
    experts.process_weights_after_loading(layer)

    assert experts._deepgemm_w13.untyped_storage().data_ptr() == (
        layer.w13_weight.untyped_storage().data_ptr()
    )
    assert experts._deepgemm_w2.untyped_storage().data_ptr() == (
        layer.w2_weight.untyped_storage().data_ptr()
    )

    workspace1 = torch.empty(
        (experts_count, max_tokens, 2 * intermediate),
        device=device,
        dtype=torch.bfloat16,
    )
    expected_m = 16
    scale_w13_hipc = layer.w13_weight_scale
    scale_w2_hipc = layer.w2_weight_scale
    m_grouped_w4a8_gemm_nt_masked_hipc(
        (activation, activation_scale),
        (experts._deepgemm_w13, scale_w13_hipc),
        workspace1,
        counts,
        expected_m,
    )
    quantized_situ, situ_scale = fuse_situ_mul_quant_ep(
        workspace1,
        counts,
        situ_beta=1.25,
        situ_linear_beta=2.0,
        expect_m=expected_m,
    )
    output = torch.empty(
        (experts_count, max_tokens, hidden),
        device=device,
        dtype=torch.bfloat16,
    )
    m_grouped_w4a8_gemm_nt_masked_hipc(
        (quantized_situ, situ_scale),
        (experts._deepgemm_w2, scale_w2_hipc),
        output,
        counts,
        expected_m,
    )

    unpacked_w13 = _unpack_int4_high_low(raw_w13)
    unpacked_w2 = _unpack_int4_high_low(raw_w2)
    for expert_id, count in enumerate(counts.tolist()):
        first_reference = (
            activation[expert_id, :count].float()
            * activation_scale[expert_id, :count]
        ) @ (
            unpacked_w13[expert_id].float()
            * (scale_w13_hipc[expert_id] * 16.0)
        ).T
        torch.testing.assert_close(
            workspace1[expert_id, :count].float(),
            first_reference,
            rtol=3e-2,
            atol=0.1,
        )

        situ_reference = _situ(
            workspace1[expert_id, :count], beta=1.25, linear_beta=2.0
        )
        quantized_reference = (
            quantized_situ[expert_id, :count].float()
            * situ_scale[expert_id, :count]
        )
        torch.testing.assert_close(
            quantized_reference,
            situ_reference,
            rtol=5e-2,
            atol=0.1,
        )
        output_reference = quantized_reference @ (
            unpacked_w2[expert_id].float()
            * (scale_w2_hipc[expert_id] * 16.0)
        ).T
        torch.testing.assert_close(
            output[expert_id, :count].float(),
            output_reference,
            rtol=4e-2,
            atol=0.25,
        )
        assert torch.isfinite(output[expert_id, :count]).all()
