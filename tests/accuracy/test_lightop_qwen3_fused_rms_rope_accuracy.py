# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from __future__ import annotations

import time

import pytest
import torch


def _require_hcu() -> None:
    if not torch.cuda.is_available():
        pytest.skip("HCU is unavailable")


def _cos_sin_cache(
    max_position: int, head_size: int, *, device: torch.device
) -> torch.Tensor:
    inverse_frequency = 1.0 / (
        1_000_000.0
        ** (
            torch.arange(0, head_size, 2, dtype=torch.float32, device=device)
            / head_size
        )
    )
    frequencies = torch.outer(
        torch.arange(max_position, dtype=torch.float32, device=device),
        inverse_frequency,
    )
    return torch.cat((frequencies.cos(), frequencies.sin()), dim=-1).to(
        torch.bfloat16
    )


@pytest.mark.hcu
@pytest.mark.parametrize("num_tokens", (1, 8, 128, 512))
def test_qwen3_fused_rms_rope_matches_current_hcu_path_and_is_faster(
    num_tokens: int,
) -> None:
    _require_hcu()
    from lightop.norm import rmsnorm_forward_autograd
    from vllm import _custom_ops as ops

    from vllm_hcu.ops.rms_rope import fused_rms_rotary_embedding

    torch.manual_seed(1)
    device = torch.device("cuda")
    head_size = 128
    num_query_heads = 32
    num_key_heads = 8
    epsilon = 1e-6
    positions = torch.arange(num_tokens, device=device, dtype=torch.long)
    query = torch.randn(
        num_tokens,
        num_query_heads * head_size,
        device=device,
        dtype=torch.bfloat16,
    )
    key = torch.randn(
        num_tokens,
        num_key_heads * head_size,
        device=device,
        dtype=torch.bfloat16,
    )
    weight_q = torch.randn(head_size, device=device, dtype=torch.bfloat16)
    weight_k = torch.randn(head_size, device=device, dtype=torch.bfloat16)
    cache = _cos_sin_cache(32768, head_size, device=device)

    def current_path() -> tuple[torch.Tensor, torch.Tensor]:
        output_q = rmsnorm_forward_autograd(
            query.view(num_tokens, num_query_heads, head_size),
            weight_q,
            epsilon,
            False,
        ).view_as(query)
        output_k = rmsnorm_forward_autograd(
            key.view(num_tokens, num_key_heads, head_size),
            weight_k,
            epsilon,
            False,
        ).view_as(key)
        ops.rotary_embedding(
            positions,
            output_q,
            output_k,
            head_size,
            cache,
            True,
        )
        return output_q, output_k

    def fused_path() -> tuple[torch.Tensor, torch.Tensor]:
        return fused_rms_rotary_embedding(
            positions,
            query.clone(),
            key.clone(),
            head_size,
            cache,
            True,
            weight_q,
            weight_k,
            epsilon,
        )

    with torch.inference_mode():
        reference_q, reference_k = current_path()
        actual_q, actual_k = fused_path()
        torch.cuda.synchronize()

        iterations = 200 if num_tokens < 128 else 100
        for _ in range(10):
            current_path()
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(iterations):
            current_path()
        torch.cuda.synchronize()
        current_ms = (time.perf_counter() - started) * 1000 / iterations

        for _ in range(10):
            fused_path()
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(iterations):
            fused_path()
        torch.cuda.synchronize()
        fused_ms = (time.perf_counter() - started) * 1000 / iterations

    max_query_error = float((actual_q - reference_q).abs().max())
    max_key_error = float((actual_k - reference_k).abs().max())
    mean_query_error = float((actual_q - reference_q).abs().float().mean())
    mean_key_error = float((actual_k - reference_k).abs().float().mean())
    print(
        f"M={num_tokens}: current={current_ms:.6f} ms, "
        f"fused={fused_ms:.6f} ms, speedup={current_ms / fused_ms:.3f}x, "
        f"max_q={max_query_error}, max_k={max_key_error}, "
        f"mean_q={mean_query_error:.8f}, mean_k={mean_key_error:.8f}"
    )
    assert max_query_error <= 0.0625
    assert max_key_error <= 0.0625
    assert mean_query_error <= 0.001
    assert mean_key_error <= 0.001
