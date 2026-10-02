# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness-first BW1100 benchmark for the gfx938 FP8 indexer reader.

The comparison path reproduces the pre-direct-reader linearization and
LightOp call. Eager timing includes both public wrapper dispatch and output
allocation; graph replay timing is separately labelled and excludes Python.
"""

import argparse
import math
import os
import statistics
import subprocess
import time
from collections import Counter
from functools import partial

import torch
import triton

from vllm_hcu.v1.attention.ops.fp8_paged_mqa_gfx938 import _paged_logits
from vllm_hcu.v1.attention.ops.rocm_aiter_mla_sparse import (
    _get_lightop_attention,
    rocm_fp8_paged_mqa_logits,
)


def make_case(batch, page_size, length, heads, dim, seed):
    torch.manual_seed(seed)
    device = "cuda"
    pages = math.ceil(length / page_size)
    tokens = pages * page_size
    keys = torch.randn(batch, tokens, dim, device=device).to(torch.float8_e4m3fn)
    scales = torch.rand(batch, tokens, device=device) * 0.15 + 0.01
    query = torch.randn(batch, 1, heads, dim, device=device).to(torch.float8_e4m3fn)
    weights = torch.randn(batch, heads, device=device) * 0.05
    lengths = (length - torch.arange(batch, device=device) % 3).to(torch.int32)
    permutation = torch.randperm(batch * pages, device=device)
    table = permutation.reshape(batch, pages).to(torch.int32)
    packed = (
        keys.view(torch.uint8)
        .reshape(batch * pages, page_size // 16, 16, dim // 16, 16)
        .permute(0, 1, 3, 2, 4)
        .reshape(batch * pages, page_size * dim)
    )
    cache = torch.empty(
        (batch * pages, page_size * (dim + 4)),
        dtype=torch.uint8,
        device=device,
    )
    cache[permutation, : page_size * dim] = packed
    cache[permutation, page_size * dim :] = scales.reshape(
        batch * pages, page_size
    ).view(torch.uint8)
    cache = cache.view(batch * pages, page_size, 1, dim + 4)
    schedule = torch.empty(0, dtype=torch.int32, device=device)
    inputs = (query, cache, weights, lengths, table, schedule, length)
    return inputs, keys, scales


def oracle(inputs, keys, scales):
    query, _, weights, lengths, _, _, max_len = inputs
    batch, _, heads, dim = query.shape
    expected = torch.empty((batch, max_len), device=query.device)
    for row in range(batch):
        dots = query[row, 0].float().reshape(heads, dim) @ keys[row, :max_len].float().T
        scores = (dots.relu() * weights[row, :, None]).sum(0) * scales[row, :max_len]
        expected[row] = scores.masked_fill(
            torch.arange(max_len, device=query.device) >= lengths[row], -torch.inf
        )
    return expected


def check(actual, expected, label):
    if actual.shape != expected.shape or actual.dtype != torch.float32:
        raise AssertionError(f"{label}: unexpected output shape or dtype")
    if not torch.equal(torch.isneginf(actual), torch.isneginf(expected)):
        raise AssertionError(f"{label}: negative-infinity mask differs")
    finite = torch.isfinite(expected)
    torch.testing.assert_close(actual[finite], expected[finite], rtol=1e-3, atol=1e-2)
    max_error = (actual[finite] - expected[finite]).abs().max().item()
    checked_rows = 0
    exact_rows = 0
    near_tie_rows = 0
    near_tie_members = 0
    min_cutoff_gap = float("inf")
    max_row_error = 0.0
    for row in range(expected.shape[0]):
        row_finite = finite[row]
        finite_count = int(row_finite.sum())
        k = min(32, finite_count)
        if not k:
            continue
        checked_rows += 1
        row_error = (
            (actual[row, row_finite] - expected[row, row_finite]).abs().max().item()
        )
        max_row_error = max(max_row_error, row_error)
        expected_top = expected[row].topk(k).indices
        actual_top = actual[row].topk(k).indices
        if finite_count > k:
            cutoff = expected[row].topk(k + 1).values
            cutoff_gap = (cutoff[k - 1] - cutoff[k]).item()
            min_cutoff_gap = min(min_cutoff_gap, cutoff_gap)
        else:
            cutoff_gap = float("inf")
        if torch.equal(expected_top.sort().values, actual_top.sort().values):
            exact_rows += 1
            continue
        if finite_count <= k:
            raise AssertionError(f"{label}: top-k set differs without a cutoff")
        missing = expected_top[~torch.isin(expected_top, actual_top)]
        unexpected = actual_top[~torch.isin(actual_top, expected_top)]
        swapped_gap = (
            expected[row, missing].min() - expected[row, unexpected].max()
        ).item()
        missing_error = (
            (actual[row, missing] - expected[row, missing]).abs().max().item()
        )
        unexpected_error = (
            (actual[row, unexpected] - expected[row, unexpected]).abs().max().item()
        )
        error_bound = missing_error + unexpected_error
        if cutoff_gap > error_bound or swapped_gap > error_bound:
            raise AssertionError(
                f"{label}: row={row} top-k mismatch exceeds observed error: "
                f"cutoff_gap={cutoff_gap:.7g}, swapped_gap={swapped_gap:.7g}, "
                f"row_max_error={row_error:.7g}, bound={error_bound:.7g}"
            )
        near_tie_rows += 1
        near_tie_members += missing.numel()
        print(
            f"topk near tie {label}: row={row} members={missing.numel()} "
            f"cutoff_gap={cutoff_gap:.7g} swapped_gap={swapped_gap:.7g} "
            f"row_max_error={row_error:.7g} bound={error_bound:.7g}"
        )
    print(
        f"correctness {label}: max_abs_error={max_error:.7g}, "
        f"max_row_error={max_row_error:.7g}, "
        f"min_cutoff_gap={min_cutoff_gap:.7g}, "
        f"topk_exact_rows={exact_rows}/{checked_rows}, "
        f"near_tie_rows={near_tie_rows}, near_tie_members={near_tie_members}"
    )


def old_linearization(inputs):
    query, cache, weights, lengths, table, _, max_len = inputs
    batch, num_pages = table.shape
    page_size = cache.shape[1]
    dim = query.shape[-1]
    valid_pages = (
        torch.arange(num_pages, device=cache.device)[None, :] * page_size
        < (lengths[:, None])
    )
    page_ids = torch.where(valid_pages, table, 0).reshape(-1).long()
    selected = cache.index_select(0, page_ids)
    flat = selected.reshape(batch * num_pages, -1)
    keys = flat[:, : page_size * dim]
    keys = keys.reshape(-1, page_size // 16, dim // 16, 16, 16)
    keys = (
        keys.permute(0, 1, 3, 2, 4)
        .contiguous()
        .view(batch * num_pages, page_size * dim)
    )
    selected = torch.cat((keys, flat[:, page_size * dim :]), dim=-1)
    selected = selected.view(batch * num_pages, page_size, 1, dim + 4)
    linear_table = torch.arange(
        batch * num_pages, device=cache.device, dtype=table.dtype
    ).view(batch, num_pages)
    return _get_lightop_attention().paged_mqa_logits(
        query,
        selected,
        weights.float().contiguous(),
        lengths,
        linear_table,
        None,
        max_len,
        False,
    )


def direct_wrapper(inputs):
    return rocm_fp8_paged_mqa_logits(*inputs)


def kernel_only(inputs, output, block_tokens, warps):
    query, cache, weights, lengths, table, _, max_len = inputs
    batch, next_n, heads, dim = query.shape
    compiled = _paged_logits[(batch * next_n, triton.cdiv(max_len, block_tokens))](
        query,
        cache.view(query.dtype),
        weights,
        lengths,
        table,
        output,
        *query.stride()[:3],
        *weights.stride(),
        lengths.stride(0),
        0,
        *table.stride(),
        next_n,
        heads,
        dim,
        cache.shape[1],
        cache.shape[0],
        table.shape[1],
        max_len,
        False,
        max(16, triton.next_power_of_2(heads)),
        triton.next_power_of_2(dim),
        block_tokens,
        num_warps=warps,
    )
    return output, compiled


def measure(fn, warmup, repetitions, rounds):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    medians = []
    walls = []
    for _ in range(rounds):
        samples = []
        wall_samples = []
        for _ in range(repetitions):
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            wall_start = time.perf_counter()
            start.record()
            output = fn()
            end.record()
            end.synchronize()
            wall_samples.append((time.perf_counter() - wall_start) * 1000)
            samples.append(start.elapsed_time(end))
            del output
        medians.append(statistics.median(samples))
        walls.append(statistics.median(wall_samples))
    return medians, walls


def peak_extra(fn, output_bytes):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    result = fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    del result
    torch.cuda.synchronize()
    return (peak - before - output_bytes) / 2**20


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, nargs="+", default=[1, 8, 64])
    parser.add_argument("--page-size", type=int, default=32)
    parser.add_argument("--length", type=int, default=32768)
    parser.add_argument("--heads", type=int, default=64)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--mode", choices=("eager", "graph"), default="eager")
    parser.add_argument("--diagnose-tiles", action="store_true")
    args = parser.parse_args()
    if (
        args.rounds < 3
        or args.repetitions < 1
        or args.warmup < 0
        or any(batch < 1 for batch in args.batch)
        or args.page_size not in (16, 32, 64)
        or args.length < 3
        or args.dim < 16
        or args.dim % 16
        or args.heads not in (8, 16, 32, 64)
    ):
        parser.error("invalid shape or timing settings")
    if not torch.cuda.is_available() or torch.version.hip is None:
        parser.error("requires ROCm/HIP GPU")
    print(
        f"GPU={torch.cuda.get_device_name()} torch={torch.__version__} "
        f"HIP={torch.version.hip} triton={triton.__version__} "
        f"SHA={subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()}"
    )
    print(
        f"args={vars(args)} HIP_VISIBLE_DEVICES={os.environ.get('HIP_VISIBLE_DEVICES')} "
        "timer=synchronized HIP events; wall=perf_counter + sync"
    )
    for batch in args.batch:
        inputs, keys, scales = make_case(
            batch, args.page_size, args.length, args.heads, args.dim, args.seed
        )
        expected = oracle(inputs, keys, scales)
        del keys, scales
        direct = direct_wrapper(inputs)
        check(direct, expected, "direct")
        del direct
        baseline = old_linearization(inputs)
        check(baseline, expected, "linearization+LightOp")
        del baseline
        if args.diagnose_tiles:
            output = torch.empty_like(expected)
            for block_tokens, warps in ((64, 4), (128, 4), (256, 4), (256, 8)):
                _, compiled = kernel_only(inputs, output, block_tokens, warps)
                check(
                    output, expected, f"kernel-only tile={block_tokens} warps={warps}"
                )
                isa = compiled.asm.get("amdgcn", "")
                opcodes = Counter(
                    line.strip().split()[0]
                    for line in isa.splitlines()
                    if line.strip().startswith(("v_", "s_", "global_", "buffer_"))
                )
                print(
                    f"compiled tile={block_tokens} warps={warps} "
                    f"registers={compiled.n_regs} spills={compiled.n_spills} "
                    f"shared={compiled.metadata.shared} "
                    f"asm_keys={tuple(compiled.asm)} "
                    f"mfma={isa.count('v_mfma')} global_load={isa.count('global_load')} "
                    f"scratch={isa.count('scratch_')} top_opcodes={opcodes.most_common(12)}"
                )
                fn = partial(kernel_only, inputs, output, block_tokens, warps)
                medians, _ = measure(fn, args.warmup, args.repetitions, args.rounds)
                print(
                    f"B={batch} kernel-only tile={block_tokens} warps={warps} "
                    f"programs={batch * triton.cdiv(args.length, block_tokens)} "
                    f"HIP medians_ms={medians}"
                )
            del output
        torch.cuda.synchronize()
        output_bytes = batch * args.length * 4
        rows = []
        for name, fn in (
            ("direct full wrapper", partial(direct_wrapper, inputs)),
            ("linearization+LightOp", partial(old_linearization, inputs)),
        ):
            if args.mode == "graph":
                for _ in range(args.warmup):
                    fn()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    graph_output = fn()
                graph.replay()
                check(graph_output, expected, f"{name} graph replay")
                timed_fn = graph.replay
            else:
                timed_fn = fn
            medians, walls = measure(
                timed_fn, args.warmup, args.repetitions, args.rounds
            )
            extra = peak_extra(fn, output_bytes)
            rows.append((name, medians, walls, extra))
        del expected
        for name, medians, walls, extra in rows:
            gb = batch * args.length * (args.dim + 4 + 4) / 1e9
            bandwidth = gb / (statistics.median(medians) / 1000)
            print(
                f"B={batch} {name} mode={args.mode}: HIP medians_ms={medians} "
                f"median_ms={statistics.median(medians):.5f} "
                f"wall_medians_ms={walls} eager_extra_peak_MiB={extra:.2f} "
                f"ideal_input_output_GBps={bandwidth:.1f}"
            )
        if (
            args.mode == "eager"
            and (args.page_size, args.length, args.heads, args.dim)
            == (32, 32768, 64, 128)
            and batch == 64
        ):
            direct_median = statistics.median(rows[0][1])
            direct_extra = rows[0][3]
            if direct_median > 1.7 or direct_extra >= 128:
                raise RuntimeError(
                    f"B64 gate failed: {direct_median:.5f} ms, "
                    f"{direct_extra:.2f} MiB extra"
                )
            print("B64 eager gate passed: <=1.7 ms and <128 MiB extra")


if __name__ == "__main__":
    main()
