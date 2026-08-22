# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Numerically compare AITER and vLLM gated-SiLU GPU operators.

Run as a focused pytest:

    HIP_VISIBLE_DEVICES=1 pytest -q -s \
        tests/accuracy/test_aiter_silu_and_mul.py

Run as a standalone diagnostic:

    HIP_VISIBLE_DEVICES=1 python \
        tests/accuracy/test_aiter_silu_and_mul.py
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from importlib import import_module
from typing import Callable

import pytest
import torch
import torch.nn.functional as F


pytestmark = pytest.mark.hcu


class OperatorDependencyUnavailable(RuntimeError):
    """A required AITER or vLLM operator is not installed."""


@dataclass(frozen=True)
class ErrorMetrics:
    max_abs: float
    mae: float
    nmae_percent: float
    max_row_nmae_percent: float


@dataclass(frozen=True)
class CaseResult:
    tokens: int
    rows: int
    dtype_name: str
    aiter_vs_fp32: ErrorMetrics
    vllm_vs_fp32: ErrorMetrics
    aiter_vs_vllm: ErrorMetrics
    mismatch_percent: float


def _load_operators() -> tuple[
    Callable[[torch.Tensor, torch.Tensor], None],
    Callable[[torch.Tensor, torch.Tensor], None],
]:
    try:
        aiter_activation = import_module("aiter.ops.triton.moe_activation")
    except ImportError as exc:
        raise OperatorDependencyUnavailable(
            "AITER Triton MoE activation is unavailable"
        ) from exc
    aiter_op = getattr(aiter_activation, "triton_silu_and_mul", None)
    if not callable(aiter_op):
        raise OperatorDependencyUnavailable(
            "AITER triton_silu_and_mul is unavailable"
        )

    try:
        import_module("vllm._C_stable_libtorch")
    except ImportError as exc:
        raise OperatorDependencyUnavailable(
            "vLLM stable extension is unavailable"
        ) from exc
    vllm_op = getattr(getattr(torch.ops, "_C", None), "silu_and_mul", None)
    if not callable(vllm_op):
        raise OperatorDependencyUnavailable(
            "vLLM _C.silu_and_mul is unavailable"
        )
    return aiter_op, vllm_op


def _error_metrics(
    actual: torch.Tensor,
    reference: torch.Tensor,
) -> ErrorMetrics:
    difference = (actual.float() - reference.float()).abs()
    reference_mean = reference.float().abs().mean().clamp_min(1e-12)
    row_reference_mean = reference.float().abs().mean(dim=-1).clamp_min(1e-12)
    row_nmae = difference.mean(dim=-1) / row_reference_mean
    return ErrorMetrics(
        max_abs=float(difference.max().item()),
        mae=float(difference.mean().item()),
        nmae_percent=float((difference.mean() / reference_mean).item() * 100),
        max_row_nmae_percent=float(row_nmae.max().item() * 100),
    )


def run_case(
    *,
    tokens: int,
    top_k: int = 8,
    intermediate_size: int = 512,
    seed: int = 20260821,
    input_scale: float = 2.0,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
) -> CaseResult:
    if tokens <= 0 or top_k <= 0 or intermediate_size <= 0:
        raise ValueError("tokens, top_k, and intermediate_size must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("a CUDA/ROCm device is required")
    properties = torch.cuda.get_device_properties(torch.device(device))
    if not hasattr(properties, "gcnArchName"):
        raise RuntimeError("the active device is not an HCU/ROCm device")
    if dtype not in {torch.bfloat16, torch.float16}:
        raise ValueError("only bfloat16 and float16 inputs are supported")

    aiter_op, vllm_op = _load_operators()
    rows = tokens * top_k
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + tokens)
    source_input = (
        torch.randn(
            (rows, intermediate_size * 2),
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
        * input_scale
    ).to(dtype)
    fp32_input = source_input.float()
    gate, up = fp32_input.chunk(2, dim=-1)
    fp32_reference = F.silu(gate) * up
    aiter_input = source_input.clone()
    vllm_input = source_input.clone()
    aiter_output = torch.full(
        (rows, intermediate_size),
        float("nan"),
        device=device,
        dtype=dtype,
    )
    vllm_output = torch.full_like(aiter_output, float("nan"))

    aiter_op(aiter_output, aiter_input)
    vllm_op(vllm_output, vllm_input)
    torch.cuda.synchronize()

    torch.testing.assert_close(aiter_input, source_input, rtol=0, atol=0)
    torch.testing.assert_close(vllm_input, source_input, rtol=0, atol=0)
    if not torch.isfinite(aiter_output).all():
        raise AssertionError("AITER output contains non-finite values")
    if not torch.isfinite(vllm_output).all():
        raise AssertionError("vLLM output contains non-finite values")
    if aiter_output.shape != fp32_reference.shape:
        raise AssertionError("AITER output shape does not match the reference")
    if vllm_output.shape != fp32_reference.shape:
        raise AssertionError("vLLM output shape does not match the reference")

    mismatch_percent = float(
        (aiter_output != vllm_output).float().mean().item() * 100
    )
    return CaseResult(
        tokens=tokens,
        rows=rows,
        dtype_name=str(dtype).removeprefix("torch."),
        aiter_vs_fp32=_error_metrics(aiter_output, fp32_reference),
        vllm_vs_fp32=_error_metrics(vllm_output, fp32_reference),
        aiter_vs_vllm=_error_metrics(aiter_output, vllm_output),
        mismatch_percent=mismatch_percent,
    )


def _assert_operators_match_reference(
    result: CaseResult,
    max_nmae_percent: float = 0.25,
    max_row_nmae_percent: float = 0.5,
    max_abs: float = 0.5,
) -> None:
    for name, metrics in (
        ("AITER vs FP32", result.aiter_vs_fp32),
        ("vLLM vs FP32", result.vllm_vs_fp32),
        ("AITER vs vLLM", result.aiter_vs_vllm),
    ):
        assert metrics.nmae_percent < max_nmae_percent, (
            f"{name} NMAE={metrics.nmae_percent:.5f}% exceeds "
            f"the {max_nmae_percent:.5f}% limit"
        )
        assert metrics.max_row_nmae_percent < max_row_nmae_percent, (
            f"{name} max-row NMAE={metrics.max_row_nmae_percent:.5f}% "
            f"exceeds the {max_row_nmae_percent:.5f}% limit"
        )
        assert metrics.max_abs < max_abs, (
            f"{name} max abs error={metrics.max_abs:.8f} exceeds "
            f"the {max_abs:.8f} limit"
        )


@pytest.mark.parametrize("tokens", [1, 16, 128])
def test_aiter_and_vllm_silu_and_mul_match_fp32_reference(tokens: int):
    if not torch.cuda.is_available():
        pytest.skip("a CUDA/ROCm device is required")
    properties = torch.cuda.get_device_properties(0)
    if not hasattr(properties, "gcnArchName"):
        pytest.skip("the active device is not an HCU/ROCm device")
    try:
        result = run_case(tokens=tokens)
    except OperatorDependencyUnavailable as exc:
        pytest.skip(str(exc))
    _assert_operators_match_reference(result)


def _format_metrics(metrics: ErrorMetrics) -> str:
    return (
        f"max={metrics.max_abs:.8f}, "
        f"mae={metrics.mae:.8f}, "
        f"nmae={metrics.nmae_percent:.5f}%, "
        f"max_row_nmae={metrics.max_row_nmae_percent:.5f}%"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare AITER and vLLM gated-SiLU GPU operators."
    )
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 16, 128])
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--intermediate-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--input-scale", type=float, default=2.0)
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16"),
        default="bfloat16",
    )
    args = parser.parse_args()
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    for tokens in args.tokens:
        result = run_case(
            tokens=tokens,
            top_k=args.top_k,
            intermediate_size=args.intermediate_size,
            seed=args.seed,
            input_scale=args.input_scale,
            dtype=dtype,
        )
        print(f"tokens={result.tokens}, rows={result.rows}")
        print(f"  AITER vs FP32: {_format_metrics(result.aiter_vs_fp32)}")
        print(f"  vLLM  vs FP32: {_format_metrics(result.vllm_vs_fp32)}")
        print(f"  AITER vs vLLM: {_format_metrics(result.aiter_vs_vllm)}")
        print(
            f"  mismatched {result.dtype_name} elements: "
            f"{result.mismatch_percent:.5f}%"
        )
        _assert_operators_match_reference(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
