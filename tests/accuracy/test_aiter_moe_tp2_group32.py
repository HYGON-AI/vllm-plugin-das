# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Probe TP=2 AITER MOE_C with lossless group-128 to group-32 scales."""

import pytest
import torch
import torch.nn.functional as F


def _dequant(packed, scale):
    values = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2)
    return ((values.float() - 8) * scale.float().repeat_interleave(128, -1)).bfloat16()


@pytest.mark.hcu
@pytest.mark.parametrize("num_tokens", [1, 4, 32, 256])
def test_tp2_moec_with_equivalent_group32_scales(num_tokens):
    if not torch.cuda.is_available():
        pytest.skip("HCU is unavailable")
    from aiter.moe import (
        MoeQuantType, get_aiter_moe_config, aiter_moe, aiter_moe_shfl_scale,
    )

    torch.manual_seed(42)
    e, h, n, top_k = 256, 2048, 256, 8
    w1 = torch.randint(0, 256, (e, 2*n, h//2), dtype=torch.uint8, device="cuda")
    w2 = torch.randint(0, 256, (e, h, n//2), dtype=torch.uint8, device="cuda")
    s1 = torch.empty((e, 2*n, h//128), dtype=torch.bfloat16, device="cuda").uniform_(0.005, 0.015)
    s2 = torch.empty((e, h, n//128), dtype=torch.bfloat16, device="cuda").uniform_(0.005, 0.015)
    x = torch.randn(num_tokens, h, dtype=torch.bfloat16, device="cuda") * 0.1
    weights, ids = torch.randn(num_tokens, e, device="cuda").softmax(-1).topk(top_k)
    weights = (weights / weights.sum(-1, keepdim=True)).contiguous()
    ids = ids.int()
    with torch.inference_mode():
        ok, config = get_aiter_moe_config(
            M=num_tokens, E=e, N1=2*n, N2=h, K=h, top_k=top_k,
            block_size=32, dtype=x.dtype, quant_type=MoeQuantType.W4A16,
            activation="silu", use_shuffle=0,
        )
        print(f"TP=2 M={num_tokens}, group32 backend={config.solution_type}, config={config.config}")
        assert ok and config.solution_type == "moe_c"
        assert not config.need_shuffle
        # Same INT4 weights, identical dequantized values: repeat each original
        # group-128 scale four times for the C kernel's required group size 32.
        c_s1, c_s2 = aiter_moe_shfl_scale(
            s1.repeat_interleave(4, -1), s2.repeat_interleave(4, -1), config,
        )
        out = aiter_moe(
            hidden_states=x, w1=w1, w2=w2, topk_weights=weights, topk_ids=ids,
            moe_config=config, w1_scale=c_s1, w2_scale=c_s2,
            w1_zp=torch.full((e, n, h//32), 0x88, dtype=torch.uint8, device="cuda"),
            w2_zp=torch.full((e, h//2, n//32), 0x88, dtype=torch.uint8, device="cuda"),
            block_shape=[0, 32], activation="silu", global_num_experts=e,
            output_dtype=x.dtype,
        )
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
