# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from __future__ import annotations

import time

import pytest
import torch


def _require_hcu() -> None:
    if not torch.cuda.is_available():
        pytest.skip("HCU is unavailable")


def _mean_ms(call, iterations: int) -> float:
    for _ in range(5):
        call()
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(iterations):
        call()
    torch.cuda.synchronize()
    return (time.perf_counter() - started) * 1000 / iterations


@pytest.mark.hcu
def test_lightop_autoawq_matches_v0281_backend_and_dequantized_reference() -> None:
    _require_hcu()
    from lightop.gemm_ops import (
        awq_gemm_marlin_weight_repack,
        gemm_awq_w4a16_marlin,
    )
    from vllm.model_executor.layers.quantization.awq_triton import (
        awq_dequantize_triton,
        awq_gemm_triton,
    )

    from vllm_hcu.model_executor.layers.quantization.lightop_autoawq import (
        convert_awq_to_lightop_layout,
    )

    torch.manual_seed(7)
    device = torch.device("cuda")
    k, n, group_size = 4096, 8192, 128
    qweight = torch.randint(
        -(2**31), 2**31 - 1, (k, n // 8), device=device, dtype=torch.int32
    )
    qzeros = torch.randint(
        -(2**31),
        2**31 - 1,
        (k // group_size, n // 8),
        device=device,
        dtype=torch.int32,
    )
    scales = torch.empty(
        (k // group_size, n), device=device, dtype=torch.float16
    ).uniform_(0.0005, 0.002)

    weight_trans, scales_zeros = convert_awq_to_lightop_layout(
        qweight, qzeros, scales, group_size
    )
    repacked = awq_gemm_marlin_weight_repack(weight_trans, n, k)
    dequantized = awq_dequantize_triton(qweight, scales, qzeros)

    for num_tokens in (1, 2, 8, 16, 32, 64, 128):
        inputs = torch.randn(
            (num_tokens, k), device=device, dtype=torch.float16
        ).mul_(0.1)

        def current_backend() -> torch.Tensor:
            return awq_gemm_triton(inputs, qweight, scales, qzeros, 8)

        def lightop_backend() -> torch.Tensor:
            return gemm_awq_w4a16_marlin(inputs, repacked, scales_zeros)

        with torch.inference_mode():
            current = current_backend()
            actual = lightop_backend()
            reference = torch.matmul(inputs, dequantized)
            torch.cuda.synchronize()
            iterations = 100 if num_tokens <= 16 else 50
            current_ms = _mean_ms(current_backend, iterations)
            lightop_ms = _mean_ms(lightop_backend, iterations)

        current_error = float((actual - current).abs().max())
        reference_error = float((actual - reference).abs().max())
        print(
            f"M={num_tokens}: v0281={current_ms:.6f} ms, "
            f"lightop={lightop_ms:.6f} ms, "
            f"speedup={current_ms / lightop_ms:.3f}x, "
            f"max_vs_v0281={current_error}, "
            f"max_vs_dequant={reference_error}"
        )
        assert current_error <= 0.015625
        assert reference_error <= 0.015625
