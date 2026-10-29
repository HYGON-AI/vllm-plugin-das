# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU plans and an eager FP8 prefill MQA boundary.

The contiguous-partition cost model is adapted from sglang-das commit
d50074e6a6942df5c91da0076acf273c977738cc (mqa_request_split.py).
"""
from __future__ import annotations


def plan_prefill_mqa_chunks(
    q_lens, k_lens, query_start=0, query_end=None,
    *, min_saved_cells=2_000_000, max_logits_bytes=0,
):
    """Plan from CPU metadata once per batch; never inspect device lengths.

    Q offsets are relative to the selected query slice. K offsets retain all
    requests in the original chunk, including requests with no selected Q.
    An empty tuple keeps the original single-launch path.
    """
    if len(q_lens) != len(k_lens):
        raise ValueError("MQA request Q/K counts differ")
    if min_saved_cells < 0 or max_logits_bytes < 0:
        raise ValueError("MQA launch cost and memory budget must be nonnegative")
    if any(n < 0 for n in (*q_lens, *k_lens)):
        raise ValueError("MQA request lengths must be nonnegative")
    total_q = sum(q_lens)
    if query_end is None:
        query_end = total_q
    if not 0 <= query_start <= query_end <= total_q:
        raise ValueError("MQA query slice is outside the request rows")

    requests = []
    q_cursor = k_cursor = selected_q = 0
    for q_len, k_len in zip(q_lens, k_lens):
        count = max(0, min(query_end, q_cursor + q_len) - max(query_start, q_cursor))
        if count:
            requests.append((selected_q, selected_q + count, k_cursor, k_cursor + k_len))
            selected_q += count
        q_cursor += q_len
        k_cursor += k_len
    if not requests:
        return ()

    # Cost = invalid K-prefix cells + a fixed cost for each additional launch.
    # Exact ties prefer more groups, matching the source planner's threshold.
    count = len(requests)
    q_prefix, qk_prefix = [0], [0]
    for a, b, c, _ in requests:
        q_prefix.append(q_prefix[-1] + b - a)
        qk_prefix.append(qk_prefix[-1] + (b - a) * c)
    costs, group_counts, previous = [0], [0], [0]
    for end in range(1, count + 1):
        candidates = []
        for start in range(end):
            invalid = (qk_prefix[end] - qk_prefix[start]) - requests[start][2] * (
                q_prefix[end] - q_prefix[start]
            )
            cost = costs[start] + invalid + (min_saved_cells if start else 0)
            candidates.append((cost, -(group_counts[start] + 1), start))
        cost, negative_groups, start = min(candidates)
        costs.append(cost)
        group_counts.append(-negative_groups)
        previous.append(start)

    groups, end = [], count
    while end:
        start = previous[end]
        groups.append((requests[start][0], requests[end - 1][1],
                       requests[start][2], requests[end - 1][3]))
        end = start
    groups.reverse()
    chunks = []
    for a, b, c, d in groups:
        rows = max(1, max_logits_bytes // max((d - c) * 4, 1)) if max_logits_bytes else b - a
        for q_start in range(a, b, rows):
            chunks.append((q_start, min(q_start + rows, b), c, d))
    full = ((0, selected_q, 0, sum(k_lens)),)
    return () if tuple(chunks) == full else tuple(chunks)


def run_grouped_prefill_mqa(
    q, keys, scales, weights, row_starts, row_ends, out, chunks,
    *, mqa, topk,
):
    """Run a prevalidated DCP=1 plan, retaining request-relative indices.

    Caller guards eager execution and FP8/LightOp dispatch. No original logits
    matrix survives this function, so it must not be used for DCP score exchange.
    """
    if not chunks:
        return False
    previous_q = 0
    for a, b, c, d in chunks:
        if a != previous_q or not a < b <= q.shape[0] or not 0 <= c <= d <= keys.shape[0]:
            return False
        previous_q = b
    if previous_q != q.shape[0]:
        return False

    for a, b, c, d in chunks:
        if c == d:
            out[a:b].fill_(-1)
            continue
        starts = row_starts[a:b] - c if c else row_starts[a:b]
        ends = row_ends[a:b] - c if c else row_ends[a:b]
        scale = scales[c:d]
        # LightOp vector loads FP32 scales from a 16-byte aligned base.
        if scale.data_ptr() % 16:
            scale = scale.clone()
        logits = mqa(q[a:b], (keys[c:d], scale), weights[a:b], starts, ends)
        topk(logits, starts, ends, out[a:b], out.shape[1])
        del logits
    return True
