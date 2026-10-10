# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Numeric contract of the gfx938 GLM-5.3 router projection."""

import pytest
import torch


@pytest.mark.parametrize("num_tokens", [1, 8, 16])
def test_glm53_router_gemv_matches_fp32_output(num_tokens):
    from vllm_hcu.model_executor.layers.fused_moe.glm53_router_gemv import (
        glm53_router_gemv,
    )
    from vllm_hcu.platforms.hcu import on_gfx938

    if not torch.cuda.is_available() or not on_gfx938():
        pytest.skip("GLM-5.3 router GEMV requires gfx938")

    torch.manual_seed(42 + num_tokens)
    x = torch.randn(num_tokens, 4096, dtype=torch.bfloat16, device="cuda")
    weight_nn = torch.randn(4096, 288, dtype=torch.bfloat16, device="cuda")
    expected = torch.mm(x, weight_nn, out_dtype=torch.float32)
    actual = glm53_router_gemv(x, weight_nn.T.contiguous())
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=5e-4)
    assert actual.dtype == torch.float32
    assert actual.shape == (num_tokens, 288)
