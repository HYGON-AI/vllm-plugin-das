#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

"""Benchmark candidate LightOp/AITER routes against current vLLM paths."""

from __future__ import annotations

import argparse
import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class TimingSummary:
    """Robust summary of repeated device timings in microseconds."""

    samples_us: tuple[float, ...]
    median_us: float
    minimum_us: float
    maximum_us: float


def _positive_finite(value: float) -> bool:
    return math.isfinite(value) and value > 0.0


def summarize_timings(samples_us: Sequence[float]) -> TimingSummary:
    """Validate and summarize repeated timing samples."""

    samples = tuple(float(sample) for sample in samples_us)
    if not samples or not all(_positive_finite(sample) for sample in samples):
        raise ValueError("timing samples must be positive finite values")
    return TimingSummary(
        samples_us=samples,
        median_us=float(statistics.median(samples)),
        minimum_us=min(samples),
        maximum_us=max(samples),
    )


def speedup_percent(baseline_us: float, candidate_us: float) -> float:
    """Return candidate latency improvement relative to the baseline."""

    baseline = float(baseline_us)
    candidate = float(candidate_us)
    if not _positive_finite(baseline) or not _positive_finite(candidate):
        raise ValueError("baseline and candidate timings must be positive finite values")
    return (baseline - candidate) / baseline * 100.0


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--json-output", type=Path)


def build_parser() -> argparse.ArgumentParser:
    """Build the operator benchmark command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="operator", required=True)

    sqrtsoftplus = subparsers.add_parser(
        "sqrtsoftplus-gate",
        help="Compare LightOp sqrt-softplus routing with the vLLM path.",
    )
    _add_common_arguments(sqrtsoftplus)
    sqrtsoftplus.add_argument("--experts", type=int, nargs="+", default=[256, 384])
    sqrtsoftplus.add_argument("--top-k", type=int, nargs="+", default=[6, 8])
    sqrtsoftplus.add_argument(
        "--token-counts",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 1024],
    )

    silu = subparsers.add_parser(
        "silu-and-mul",
        help="Compare AITER, LightOp, and vLLM gated-SiLU paths.",
    )
    _add_common_arguments(silu)
    silu.add_argument("--hidden-sizes", type=int, nargs="+", default=[512, 2048])
    silu.add_argument(
        "--token-counts",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 1024],
    )

    w16a16 = subparsers.add_parser(
        "w16a16-moe",
        help="Compare LightOp W16A16 Marlin, AITER, and Triton MoE paths.",
    )
    _add_common_arguments(w16a16)
    w16a16.add_argument("--experts", type=int, default=256)
    w16a16.add_argument("--hidden-size", type=int, default=2048)
    w16a16.add_argument("--intermediate-size", type=int, default=512)
    w16a16.add_argument("--top-k", type=int, default=8)
    w16a16.add_argument(
        "--token-counts",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32, 64, 128, 256],
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    raise RuntimeError(
        f"benchmark runner for {args.operator!r} has not passed its "
        "candidate-specific accuracy gate"
    )


if __name__ == "__main__":
    raise SystemExit(main())
