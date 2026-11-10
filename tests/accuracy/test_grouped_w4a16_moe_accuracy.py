# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Packed GPTQ canonical layout versus an independent dequantized reference."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F


def _dequant(packed, scale):
    values = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2)
    return ((values.float() - 8) * scale.float().repeat_interleave(128, -1)).bfloat16()


@pytest.mark.hcu
@pytest.mark.parametrize(
    "num_tokens,intermediate_size",
    [(1, 512), (4, 512), (32, 512), (256, 512), (1, 256), (4, 256)],
)
def test_grouped_w4a16_matches_dequantized_reference(num_tokens, intermediate_size):
    if not torch.cuda.is_available():
        pytest.skip("HCU is unavailable")
    from vllm_hcu.model_executor.layers.quantization.grouped_w4a16_moe_runtime import (
        apply_grouped_w4a16_moe,
    )

    torch.manual_seed(42)
    e, h, n, top_k = 256, 2048, intermediate_size, 8
    legacy_group128_moec = False
    if n == 256:
        from aiter.moe import MoeQuantType, get_aiter_moe_config

        supported, config = get_aiter_moe_config(
            M=num_tokens, E=e, N1=2*n, N2=h, K=h, top_k=top_k,
            block_size=128, dtype=torch.bfloat16,
            quant_type=MoeQuantType.W4A16, activation="silu", use_shuffle=0,
        )
        assert supported and config.solution_type == "moe_c"
        legacy_group128_moec = getattr(config, "block_size", None) is None
    w1 = torch.randint(0, 256, (e, 2*n, h//2), dtype=torch.uint8, device="cuda")
    w2 = torch.randint(0, 256, (e, h, n//2), dtype=torch.uint8, device="cuda")
    s1 = torch.full((e, 2*n, h//128), 0.01, dtype=torch.bfloat16, device="cuda")
    s2 = torch.full((e, h, n//128), 0.01, dtype=torch.bfloat16, device="cuda")
    quant = SimpleNamespace(use_int4_w4a16=True, w1_zp=None, w2_zp=None)
    experts = SimpleNamespace(
        quant_config=quant, block_shape=[0, 128], w1_scale=s1, w2_scale=s2,
        moe_config=SimpleNamespace(moe_backend="aiter"),
    )
    x = torch.randn(num_tokens, h, dtype=torch.bfloat16, device="cuda") * 0.1
    weights, ids = torch.randn(num_tokens, e, device="cuda").softmax(-1).topk(top_k)
    weights = (weights / weights.sum(-1, keepdim=True)).contiguous()
    ids = ids.int()
    out = torch.empty_like(x)
    with torch.inference_mode():
        routed = apply_grouped_w4a16_moe(
            experts, out, x, w1, w2, weights, ids, "silu", e,
            None, None, None, None, False,
        )
        if legacy_group128_moec:
            assert not routed, "Legacy AITER group-32 shuffle must not run group-128"
            return
        assert routed, "No AITER solution available for QGPT group-128 shape"
        reference = torch.zeros_like(x, dtype=torch.float32)
        for expert in range(e):
            rows, slots = torch.where(ids == expert)
            if rows.numel() == 0:
                continue
            gate, up = F.linear(x[rows], _dequant(w1[expert], s1[expert])).chunk(2, -1)
            intermediate = (F.silu(gate.float()) * up.float()).bfloat16()
            y = F.linear(intermediate, _dequant(w2[expert], s2[expert]))
            reference.index_add_(0, rows, y.float() * weights[rows, slots, None])
        torch.cuda.synchronize()
        error = (out.float() - reference).abs().max().item()
        print(f"M={num_tokens}, max_abs_error={error:.8f}")
        torch.testing.assert_close(out.float(), reference, atol=0.002, rtol=0.03)
