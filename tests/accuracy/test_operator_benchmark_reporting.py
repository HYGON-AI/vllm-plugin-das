# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from __future__ import annotations

import math

import pytest

from tools.benchmark_sglang_operator_candidates import (
    build_comparison_record,
    build_parser,
    speedup_percent,
    summarize_timings,
)


def test_summarize_timings_uses_median_and_extrema() -> None:
    result = summarize_timings([7.0, 3.0, 5.0])

    assert result.samples_us == (7.0, 3.0, 5.0)
    assert result.median_us == 5.0
    assert result.minimum_us == 3.0
    assert result.maximum_us == 7.0


@pytest.mark.parametrize(
    "samples",
    (
        (),
        (0.0,),
        (-1.0,),
        (math.inf,),
        (math.nan,),
    ),
)
def test_summarize_timings_rejects_non_positive_or_non_finite_samples(
    samples: tuple[float, ...],
) -> None:
    with pytest.raises(ValueError, match="positive finite"):
        summarize_timings(samples)


def test_speedup_percent_is_relative_to_baseline() -> None:
    assert speedup_percent(10.0, 8.0) == pytest.approx(20.0)


@pytest.mark.parametrize(
    "baseline_us, candidate_us",
    ((0.0, 1.0), (-1.0, 1.0), (1.0, 0.0), (1.0, math.inf)),
)
def test_speedup_percent_rejects_invalid_timings(
    baseline_us: float,
    candidate_us: float,
) -> None:
    with pytest.raises(ValueError, match="positive finite"):
        speedup_percent(baseline_us, candidate_us)


@pytest.mark.parametrize(
    "operator",
    ("sqrtsoftplus-gate", "silu-and-mul", "w16a16-moe"),
)
def test_cli_exposes_each_screening_candidate(operator: str) -> None:
    args = build_parser().parse_args([operator])

    assert args.operator == operator
    assert args.warmup > 0
    assert args.iterations > 0
    assert args.repeats > 0


def test_comparison_record_applies_five_percent_acceptance_gate() -> None:
    record = build_comparison_record(
        shape={"tokens": 1, "experts": 256, "top_k": 6},
        baseline_name="vllm",
        baseline=summarize_timings([10.0, 11.0, 9.0]),
        candidate_name="lightop",
        candidate=summarize_timings([8.0, 8.5, 7.5]),
    )

    assert record["shape"] == {"tokens": 1, "experts": 256, "top_k": 6}
    assert record["baseline"]["median_us"] == 10.0
    assert record["candidate"]["median_us"] == 8.0
    assert record["speedup_percent"] == pytest.approx(20.0)
    assert record["meets_five_percent_gate"] is True
