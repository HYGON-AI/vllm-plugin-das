# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Execute the real TP=2 group-128 AITER MOE_C/HIPC path."""

from types import SimpleNamespace

import pytest
import torch


@pytest.mark.hcu
@pytest.mark.parametrize("num_tokens", [1, 4])
def test_tp2_group128_moec_hipc(num_tokens):
    if not torch.cuda.is_available():
        pytest.skip("HCU is unavailable")
    from aiter.moe import (
        MoeQuantType,
        aiter_moe,
        aiter_moe_shfl_scale,
        get_aiter_moe_config,
    )

    torch.manual_seed(42)
    e, h, n, top_k = 256, 2048, 256, 8
    x = torch.randn(num_tokens, h, dtype=torch.bfloat16, device="cuda") * 0.1
    w1 = torch.randint(0, 256, (e, 2 * n, h // 2), dtype=torch.uint8, device="cuda")
    w2 = torch.randint(0, 256, (e, h, n // 2), dtype=torch.uint8, device="cuda")
    s1 = torch.empty((e, 2 * n, h // 128), dtype=torch.bfloat16, device="cuda").uniform_(0.005, 0.015)
    s2 = torch.empty((e, h, n // 128), dtype=torch.bfloat16, device="cuda").uniform_(0.005, 0.015)
    weights, ids = torch.randn(num_tokens, e, device="cuda").softmax(-1).topk(top_k)
    weights = (weights / weights.sum(-1, keepdim=True)).contiguous()
    ids = ids.int()
    z1 = torch.full((e, n, h // 128), 0x88, dtype=torch.uint8, device="cuda")
    z2 = torch.full((e, h // 2, n // 128), 0x88, dtype=torch.uint8, device="cuda")

    with torch.inference_mode():
        ok, config = get_aiter_moe_config(
            M=num_tokens, E=e, N1=2 * n, N2=h, K=h, top_k=top_k,
            block_size=128, dtype=x.dtype, quant_type=MoeQuantType.W4A16,
            activation="silu", use_shuffle=0,
        )
        print(
            f"TP=2 M={num_tokens}, group128 backend={config.solution_type}, "
            f"config={config.config}, need_shuffle={config.need_shuffle}, "
            f"need_shuffle_scale={config.need_shuffle_scale}"
        )
        assert ok and config.solution_type == "moe_c"
        if getattr(config, "block_size", None) is None:
            from vllm_hcu.model_executor.layers.quantization.grouped_w4a16_moe_runtime import (
                apply_grouped_w4a16_moe,
            )

            assert config.need_shuffle_scale and s2.shape[-1] == 2
            experts = SimpleNamespace(
                quant_config=SimpleNamespace(
                    use_int4_w4a16=True, w1_zp=None, w2_zp=None,
                ),
                block_shape=[0, 128], w1_scale=s1, w2_scale=s2,
                moe_config=SimpleNamespace(moe_backend="aiter"),
            )
            assert not apply_grouped_w4a16_moe(
                experts, torch.empty_like(x), x, w1, w2, weights, ids,
                "silu", e, None, None, None, None, False,
            )
            return
        assert config.block_size == 128
        c_s1, c_s2 = aiter_moe_shfl_scale(s1, s2, config)
        out = aiter_moe(
            hidden_states=x, w1=w1, w2=w2, topk_weights=weights, topk_ids=ids,
            moe_config=config, w1_scale=c_s1, w2_scale=c_s2,
            w1_zp=z1, w2_zp=z2, block_shape=[0, 128], activation="silu",
            global_num_experts=e, output_dtype=x.dtype,
        )
        torch.cuda.synchronize()
    assert out.shape == x.shape
    assert out.dtype == x.dtype
    assert torch.isfinite(out).all()
