# SPDX-License-Identifier: Apache-2.0
"""Check the fused kernel against vLLM's native M-RoPE, including BF16 rounding."""

import pytest
import torch
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.rotary_embedding.mrope import MRotaryEmbedding

from vllm_hcu.models.hy_v4.mrope import forward_hcu_mrope


@pytest.mark.hcu
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("interleaved", [True, False])
@pytest.mark.parametrize("multimodal", [True, False])
@pytest.mark.parametrize("tokens", [1, 7, 1024])
@pytest.mark.parametrize("head_size", [64, 80])
def test_fused_mrope_matches_native_exactly(
    dtype, interleaved, multimodal, tokens, head_size
):
    if not torch.cuda.is_available():
        pytest.skip("a live HCU/ROCm device is required")
    with (
        set_current_vllm_config(VllmConfig()),
        torch.device("cuda"),
        torch.inference_mode(),
    ):
        rope = MRotaryEmbedding(
            head_size,
            64,
            4096,
            10000,
            False,
            dtype,
            mrope_section=[12, 10, 10],
            mrope_interleaved=interleaved,
        )
        positions = torch.randint(0, 4096, (3, tokens) if multimodal else (tokens,))
        # Strided views reproduce MLA and indexer input layouts.
        query = torch.randn(tokens, 16, head_size + 32, dtype=dtype)[..., 32:]
        key = torch.randn(tokens, 1, head_size + 16, dtype=dtype)[..., 16:]
        reference = rope.forward_native(positions, query, key)
        actual = forward_hcu_mrope(rope, positions, query, key)
        for output, expected in zip(actual, reference):
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
