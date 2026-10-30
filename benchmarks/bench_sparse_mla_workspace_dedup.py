# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Measure host planning and gather/index/attention latency for shared prefixes.

Use --plan-only on a CPU host. GPU mode requires rebuilding the HCU plugin.
"""

import argparse
import json
import math
import statistics
import time

import numpy as np
import torch

from vllm_hcu.v1.attention.backends.mla.prefill_workspace import plan_prefill_workspace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=10)
    parser.add_argument("--seq-len", type=int, default=85000)
    parser.add_argument("--prefix-len", type=int, default=79050)
    parser.add_argument("--shared-prefix-len", type=int, default=None)
    parser.add_argument("--workspace-rows", type=int, default=90112 * 5)
    parser.add_argument("--pcp-size", type=int, default=16)
    parser.add_argument("--pcp-rank", type=int, default=0)
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--heads", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    shared_prefix_len = (
        args.prefix_len if args.shared_prefix_len is None else args.shared_prefix_len
    )
    if not (
        args.requests > 0
        and 0 <= args.prefix_len < args.seq_len <= args.workspace_rows
        and 0 <= shared_prefix_len <= args.prefix_len
        and 0 <= args.pcp_rank < args.pcp_size
        and args.topk > 0
        and args.topk % 128 == 0
        and args.heads > 0
        and args.heads % 64 == 0
        and args.repeats > 0
    ):
        parser.error("Invalid lengths, PCP rank/size, repeats, head count, or top-k")
    block_size = 64
    prefix_len = args.prefix_len // block_size * block_size
    shared_pages = shared_prefix_len // block_size
    suffix_pages = math.ceil(args.seq_len / block_size) - shared_pages
    query_len = args.seq_len - prefix_len
    segments = [(0, query_len)]
    if args.pcp_size > 1:
        width = math.ceil(query_len / (2 * args.pcp_size))
        segments = [
            (i * width, min(query_len, (i + 1) * width))
            for i in (args.pcp_rank, 2 * args.pcp_size - 1 - args.pcp_rank)
            if i * width < query_len
        ]
    tables, lengths, query_lengths = [], [], []
    for req in range(args.requests):
        pages = list(range(1, shared_pages + 1)) + list(
            range(
                1 + shared_pages + req * suffix_pages,
                1 + shared_pages + (req + 1) * suffix_pages,
            )
        )
        for start, end in segments:
            tables.append(pages)
            lengths.append(prefix_len + end)
            query_lengths.append(end - start)
    table = np.asarray(tables, dtype=np.int32)
    legacy_bounds = []
    start, used = 0, 0
    for row, length in enumerate(lengths):
        if used + length > args.workspace_rows:
            legacy_bounds.append((start, row))
            start, used = row, 0
        used += length
    legacy_bounds.append((start, len(lengths)))

    def plan():
        return plan_prefill_workspace(table, lengths, block_size, args.workspace_rows)

    plan()
    timings = []
    for _ in range(args.repeats):
        begin = time.perf_counter_ns()
        plans = plan()
        timings.append((time.perf_counter_ns() - begin) / 1000)
    result = dict(
        computed_prefix_tokens=prefix_len,
        shared_prefix_tokens=shared_pages * block_size,
        local_query_tokens=sum(query_lengths),
        virtual_rows=len(lengths),
        original_kv_rows=sum(lengths),
        dedup_kv_rows=sum(p.num_tokens for p in plans),
        original_groups=len(legacy_bounds),
        dedup_groups=len(plans),
        planning_us=dict(
            median=statistics.median(timings),
            minimum=min(timings),
            maximum=max(timings),
        ),
    )
    if not args.plan_only:
        if not torch.cuda.is_available():
            parser.error("GPU mode requires CUDA/HIP; use --plan-only on this host")
        from vllm import _custom_ops as ops
        from vllm.triton_utils import triton
        from vllm_hcu.v1.attention.backends.mla.workspace_indices import (
            triton_convert_req_index_to_global_index,
        )
        from vllm_hcu.v1.attention.ops.flashmla import flash_mla_sparse_fwd
        from vllm_hcu.v1.attention.backends.mla.workspace_dedup import (
            gather_workspace_pages,
        )

        torch.manual_seed(42)
        num_pages = 1 + shared_pages + args.requests * suffix_pages
        cache = torch.empty(
            (num_pages, block_size, 656), device="cuda", dtype=torch.uint8
        )
        values = torch.randn((num_pages, block_size, 512), device="cuda").to(
            torch.float8_e4m3fn
        )
        cache[..., :512] = values.view(torch.uint8)
        scales = torch.rand((num_pages, block_size, 4), device="cuda") + 0.01
        cache[..., 512:528] = scales.view(torch.uint8)
        rope = torch.randn(
            (num_pages, block_size, 64), device="cuda", dtype=torch.bfloat16
        )
        cache[..., 528:] = rope.view(torch.uint8)
        del values, scales, rope
        table_gpu = torch.tensor(table, device="cuda")
        lengths_gpu = torch.tensor(lengths, device="cuda", dtype=torch.int32)
        starts = np.cumsum([0] + lengths[:-1], dtype=np.int32)
        for begin, end in legacy_bounds:
            starts[begin:end] -= starts[begin]
        starts_gpu = torch.tensor(starts, device="cuda")
        row_map = np.full(table.shape, -1, dtype=np.int32)
        descriptors = []
        for p in plans:
            row_map[p.req_start : p.req_end, : p.row_block_starts.shape[1]] = (
                p.row_block_starts
            )
            descriptors.append(
                torch.tensor(
                    np.column_stack(
                        (p.physical_blocks, p.valid_tokens, p.workspace_starts)
                    ),
                    device="cuda",
                )
            )
        row_map_gpu = torch.tensor(row_map, device="cuda")
        query_starts = np.cumsum([0] + query_lengths)
        req_ids = torch.repeat_interleave(
            torch.arange(len(lengths), device="cuda", dtype=torch.int32),
            torch.tensor(query_lengths, device="cuda"),
        )
        first_positions = np.repeat(np.array(lengths) - query_lengths, query_lengths)
        local_positions = np.concatenate([np.arange(n) for n in query_lengths])
        limits = torch.tensor(first_positions + local_positions + 1, device="cuda")
        topk = (
            torch.rand((sum(query_lengths), args.topk), device="cuda") * limits[:, None]
        ).to(torch.int32)
        q = torch.randn(
            (sum(query_lengths), args.heads, 576), device="cuda", dtype=torch.bfloat16
        )
        workspace = torch.empty(
            (args.workspace_rows, 576), device="cuda", dtype=torch.bfloat16
        )
        output = torch.empty(
            (len(q), args.heads, 512), device="cuda", dtype=torch.bfloat16
        )

        def convert_indices(dedup):
            return triton_convert_req_index_to_global_index(
                req_ids,
                table_gpu,
                topk,
                NUM_TOPK_TOKENS=args.topk,
                HAS_PREFILL_WORKSPACE=True,
                prefill_workspace_request_ids=req_ids,
                prefill_workspace_starts=starts_gpu,
                prefill_workspace_block_starts=row_map_gpu if dedup else None,
            )

        def bounds_for(dedup):
            return [(p.req_start, p.req_end) for p in plans] if dedup else legacy_bounds

        def gather_chunk(dedup, i, begin, end):
            rows = plans[i].num_tokens if dedup else sum(lengths[begin:end])
            dst = workspace[:rows]
            if dedup:
                gather_workspace_pages(cache, dst, descriptors[i])
            else:
                ops.cp_gather_and_upconvert_fp8_kv_cache(
                    cache,
                    dst,
                    table_gpu[begin:end],
                    lengths_gpu[begin:end],
                    starts_gpu[begin:end],
                    end - begin,
                )
            return dst

        def gather_all(dedup):
            for i, (begin, end) in enumerate(bounds_for(dedup)):
                gather_chunk(dedup, i, begin, end)

        def run(dedup):
            mapped = convert_indices(dedup)
            for i, (begin, end) in enumerate(bounds_for(dedup)):
                dst = gather_chunk(dedup, i, begin, end)
                query_slice = slice(query_starts[begin], query_starts[end])
                output[query_slice] = flash_mla_sparse_fwd(
                    q[query_slice],
                    dst.view(-1, 1, 576),
                    mapped[query_slice].view(-1, 1, args.topk),
                    576**-0.5,
                )[0]
            return output

        expected = run(False).clone()
        torch.testing.assert_close(run(True), expected)
        for dedup in (False, True):
            prefix = "dedup" if dedup else "original"
            for component, function in (
                ("index", convert_indices),
                ("gather", gather_all),
                ("pipeline", run),
            ):
                latency = triton.testing.do_bench(
                    lambda function=function, dedup=dedup: function(dedup), rep=200
                )
                result[f"{prefix}_{component}_us"] = latency * 1000
        result["gpu"] = torch.cuda.get_device_name()
        result["torch"] = torch.__version__
        result["gpu_timings_exclude_host_planning_and_metadata_transfer"] = True
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
