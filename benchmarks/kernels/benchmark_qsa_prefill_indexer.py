# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Compare request-major QSA prefill scoring with HCU CUTLASS."""

from __future__ import annotations

import argparse
import math
import os
import statistics
import subprocess
from dataclasses import dataclass
from typing import Callable

import torch

from vllm_hcu.v1.attention.ops.qsa_prefill_indexer import (
    _prefill_chunks,
    qsa_mqa_paged_prefill,
)


@dataclass(frozen=True)
class BenchmarkCase:
    batch: int
    query_len: int
    prefix_len: int

    @property
    def label(self) -> str:
        return f"bs{self.batch}_q{self.query_len}_prefix{self.prefix_len}"


@dataclass
class Inputs:
    q: torch.Tensor
    k_cache: torch.Tensor
    page_table: torch.Tensor
    query_start_loc: torch.Tensor
    visible_blocks: torch.Tensor
    token_to_req: torch.Tensor
    query_positions: torch.Tensor
    sequence_lengths: torch.Tensor
    max_query_len: int
    compress_ratio: int


def required_cases() -> tuple[BenchmarkCase, ...]:
    return (
        BenchmarkCase(batch=4, query_len=4096, prefix_len=0),
        BenchmarkCase(batch=16, query_len=4096, prefix_len=0),
        BenchmarkCase(batch=4, query_len=4096, prefix_len=8192),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--max-logits-mb", type=int, default=128)
    parser.add_argument("--seed", type=int, default=248)
    return parser


def validate_args(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> None:
    if args.rounds < 3:
        parser.error("--rounds must be at least 3")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")
    if args.max_logits_mb < 1:
        parser.error("--max-logits-mb must be positive")


def make_inputs(case: BenchmarkCase, seed: int) -> Inputs:
    torch.manual_seed(seed)
    device = "cuda"
    heads, head_dim = 4, 128
    compress_ratio, page_size = 4, 4
    sequence_len = case.prefix_len + case.query_len
    compressed_len = sequence_len // compress_ratio
    pages_per_request = math.ceil(compressed_len / page_size)
    total_pages = case.batch * pages_per_request

    q = torch.randn(
        case.batch * case.query_len,
        heads,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    logical_cache = torch.randn(
        total_pages,
        page_size,
        1,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    permutation = torch.randperm(total_pages, device=device)
    k_cache = torch.empty_like(logical_cache)
    k_cache[permutation] = logical_cache
    del logical_cache
    page_table = permutation.reshape(case.batch, pages_per_request).to(torch.int32)

    query_start_loc = torch.arange(
        0,
        (case.batch + 1) * case.query_len,
        case.query_len,
        dtype=torch.int32,
        device=device,
    )
    token_to_req = torch.arange(
        case.batch, dtype=torch.int32, device=device
    ).repeat_interleave(case.query_len)
    local_positions = torch.arange(
        case.prefix_len,
        sequence_len,
        dtype=torch.int64,
        device=device,
    )
    query_positions = local_positions.repeat(case.batch)
    sequence_lengths = torch.full(
        (case.batch,),
        sequence_len,
        dtype=torch.int32,
        device=device,
    )
    visible_blocks = torch.minimum(
        (query_positions + 1) // compress_ratio,
        sequence_lengths.index_select(0, token_to_req.long()) // compress_ratio,
    ).to(torch.int32)
    return Inputs(
        q=q,
        k_cache=k_cache,
        page_table=page_table,
        query_start_loc=query_start_loc,
        visible_blocks=visible_blocks,
        token_to_req=token_to_req,
        query_positions=query_positions,
        sequence_lengths=sequence_lengths,
        max_query_len=case.query_len,
        compress_ratio=compress_ratio,
    )


def _baseline_chunk(inputs: Inputs, start: int, end: int):
    from flash_attn import mqa_paged_score_func

    return mqa_paged_score_func(
        inputs.q[start:end],
        inputs.k_cache,
        inputs.page_table,
        inputs.token_to_req[start:end],
        inputs.query_positions[start:end],
        inputs.sequence_lengths,
        inputs.compress_ratio,
    )


def _candidate_chunk(inputs: Inputs, start: int, end: int) -> torch.Tensor:
    return qsa_mqa_paged_prefill(
        inputs.q,
        inputs.k_cache,
        inputs.page_table,
        inputs.query_start_loc,
        inputs.visible_blocks,
        inputs.max_query_len,
        query_offset=start,
        num_queries=end - start,
    )


def _chunks(inputs: Inputs, max_logits_bytes: int):
    columns = inputs.page_table.shape[1] * inputs.k_cache.shape[1]
    return _prefill_chunks(inputs.q.shape[0], columns, max_logits_bytes)


def check_correctness(inputs: Inputs, max_logits_bytes: int) -> tuple[float, int]:
    max_error = 0.0
    checked_topk_rows = 0
    columns = inputs.page_table.shape[1] * inputs.k_cache.shape[1]
    column_ids = torch.arange(columns, device=inputs.q.device)
    sample_rows = {
        request * inputs.max_query_len + offset
        for request in range(inputs.page_table.shape[0])
        for offset in (0, inputs.max_query_len // 2, inputs.max_query_len - 1)
    }

    for start, end in _chunks(inputs, max_logits_bytes):
        expected, baseline_visible = _baseline_chunk(inputs, start, end)
        actual = _candidate_chunk(inputs, start, end)
        visible = inputs.visible_blocks[start:end]
        if not torch.equal(baseline_visible, visible):
            raise AssertionError("CUTLASS and metadata visible-block counts differ")
        valid = column_ids[None, :] < visible[:, None]
        torch.testing.assert_close(
            actual[valid],
            expected[valid],
            rtol=2e-2,
            atol=2e-2,
        )
        if valid.any():
            max_error = max(
                max_error,
                float((actual[valid] - expected[valid]).abs().max()),
            )

        for absolute_row in sorted(sample_rows):
            if not start <= absolute_row < end:
                continue
            local_row = absolute_row - start
            k = min(512, int(visible[local_row]))
            if not k:
                continue
            expected_values, expected_indices = expected[local_row, : int(visible[local_row])].topk(k)
            actual_indices = actual[local_row, : int(visible[local_row])].topk(k).indices
            if not torch.equal(
                expected_indices.sort().values,
                actual_indices.sort().values,
            ):
                cutoff = expected_values[-1]
                tolerance = 0.02 + 0.02 * cutoff.abs()
                actual_values = expected[local_row, actual_indices]
                if not bool((actual_values >= cutoff - 2 * tolerance).all()):
                    raise AssertionError(
                        f"top-k differs beyond tolerance at row {absolute_row}"
                    )
            checked_topk_rows += 1
        del actual, expected, baseline_visible
    return max_error, checked_topk_rows


def _run_baseline(inputs: Inputs, max_logits_bytes: int) -> torch.Tensor:
    last = None
    for start, end in _chunks(inputs, max_logits_bytes):
        last, _ = _baseline_chunk(inputs, start, end)
    assert last is not None
    return last


def _run_candidate(inputs: Inputs, max_logits_bytes: int) -> torch.Tensor:
    last = None
    for start, end in _chunks(inputs, max_logits_bytes):
        last = _candidate_chunk(inputs, start, end)
    assert last is not None
    return last


def measure(
    fn: Callable[[], torch.Tensor],
    *,
    warmup: int,
    repetitions: int,
    rounds: int,
) -> list[float]:
    for _ in range(warmup):
        result = fn()
        del result
    torch.cuda.synchronize()
    medians = []
    for _ in range(rounds):
        samples = []
        for _ in range(repetitions):
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            result = fn()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end))
            del result
        medians.append(statistics.median(samples))
    return medians


def peak_delta_mib(fn: Callable[[], torch.Tensor]) -> float:
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    result = fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    del result
    torch.cuda.synchronize()
    return (peak - before) / 2**20


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(args, parser)
    if not torch.cuda.is_available() or torch.version.hip is None:
        parser.error("this benchmark requires a ROCm/HCU device")

    max_logits_bytes = args.max_logits_mb * 1024 * 1024
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    print(
        f"GPU={torch.cuda.get_device_name()} torch={torch.__version__} "
        f"HIP={torch.version.hip} SHA={sha} "
        f"HIP_VISIBLE_DEVICES={os.environ.get('HIP_VISIBLE_DEVICES')}"
    )
    print(
        f"timer=synchronized HIP events rounds={args.rounds} "
        f"warmup={args.warmup} repetitions={args.repetitions} "
        f"max_logits_mb={args.max_logits_mb}"
    )

    for case_index, case in enumerate(required_cases()):
        inputs = make_inputs(case, args.seed + case_index)
        max_error, topk_rows = check_correctness(inputs, max_logits_bytes)
        baseline = lambda: _run_baseline(inputs, max_logits_bytes)
        candidate = lambda: _run_candidate(inputs, max_logits_bytes)
        baseline_peak = peak_delta_mib(baseline)
        candidate_peak = peak_delta_mib(candidate)
        baseline_rounds = measure(
            baseline,
            warmup=args.warmup,
            repetitions=args.repetitions,
            rounds=args.rounds,
        )
        candidate_rounds = measure(
            candidate,
            warmup=args.warmup,
            repetitions=args.repetitions,
            rounds=args.rounds,
        )
        baseline_median = statistics.median(baseline_rounds)
        candidate_median = statistics.median(candidate_rounds)
        reduction = (baseline_median - candidate_median) / baseline_median * 100
        print(
            f"case={case.label} rows={inputs.q.shape[0]} "
            f"columns={inputs.page_table.shape[1] * inputs.k_cache.shape[1]} "
            f"chunks={len(_chunks(inputs, max_logits_bytes))} "
            f"max_abs_error={max_error:.7g} topk_rows={topk_rows}"
        )
        print(
            f"baseline_ms={baseline_rounds} median={baseline_median:.6f} "
            f"peak_delta_mib={baseline_peak:.2f}"
        )
        print(
            f"candidate_ms={candidate_rounds} median={candidate_median:.6f} "
            f"peak_delta_mib={candidate_peak:.2f} reduction={reduction:.3f}%"
        )
        del inputs
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

