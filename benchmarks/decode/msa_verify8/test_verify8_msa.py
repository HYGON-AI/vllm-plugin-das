# SPDX-License-Identifier: Apache-2.0
"""Single-GPU correctness and CUDA graph A/B, without model weights."""
import argparse
import json
import statistics

import torch

from hcu_minimax_msa import _native as reference_native
from hcu_minimax_msa_verify8 import Verify8PageSelector
from triton_tmp_dir.minimax31_prefill_triton import force_local_pages


def compare(reference, candidate):
    torch.testing.assert_close(candidate, reference, atol=0, rtol=0)


def test_synthetic(batch, pages):
    queries = 8 * batch
    padded_pages = (pages + 127) // 128 * 128
    test_ends = (0, 1, 127, 128, 129, 15 * 128, 16 * 128, 16 * 128 + 1,
                 64 * 128, 147453, 147455, 147456, 147457, 147463,
                 pages * 128 - 1, pages * 128)
    ends = torch.tensor([min(test_ends[q % len(test_ends)], pages * 128) for q in range(queries)],
                        dtype=torch.int32, device="cuda")
    page_ids = torch.arange(batch * pages, dtype=torch.int32, device="cuda")
    scale = torch.ones(1, dtype=torch.float32, device="cuda")
    selector = Verify8PageSelector(ends, page_ids, scale, pages)
    page_numbers = torch.arange(padded_pages, device="cuda")[:, None]
    # Keep poison values outside visible pages to catch reading reused scratch.
    random_scores = torch.randint(-4, 5, (4, padded_pages, queries), device="cuda").float()
    all_tied = torch.full_like(random_scores, 5)
    for scores in (random_scores, all_tied, torch.full_like(random_scores, float("-inf"))):
        for _ in range(2):
            valid = page_numbers * 128 < ends
            expected_scores = scores.masked_fill(~valid, float("-inf"))
            force_local_pages(expected_scores, ends, 1)
            expected = reference_native.msa128_stage2_topk_select(
                expected_scores, 16, pages, 0, 0, None, None
            )
            compare(expected, selector.select_scores(scores))
            ends.copy_(torch.roll(ends, 1))


def test_scanner(batch, *, pages=1161, measure=False):
    queries = 8 * batch
    capacity = pages * 128
    q = torch.randn((queries, 4, 128), device="cuda").to(torch.float8_e4m3fn)
    cache = torch.empty(
        (1 + batch * pages, 1, 128, 128), device="cuda", dtype=torch.float8_e4m3fn
    )
    # Avoid a full FP32 cache temporary when running the single-card fixture.
    for chunk in cache.split(128):
        chunk.copy_(torch.randn(chunk.shape, device="cuda").to(cache.dtype))
    page_ids = (
        1 + torch.arange(batch, device="cuda")[:, None] * pages
        + (torch.arange(pages, device="cuda") * 37) % pages
    ).int().flatten()
    ends = torch.arange(queries, device="cuda", dtype=torch.int32).remainder_(8) + 147453
    lens_q = torch.full((batch,), 8, device="cuda", dtype=torch.int32)
    lens_kv = torch.full((batch,), capacity, device="cuda", dtype=torch.int32)
    scale = torch.ones(1, dtype=torch.float32, device="cuda")
    selectors = [Verify8PageSelector(ends, page_ids, scale, pages, mode) for mode in range(6)]

    def reference():
        scores = reference_native.msa128_stage1_page_max(
            q, cache, ends, True, -1, lens_q, lens_kv, None, page_ids, capacity,
            scale, scale, False, False, True,
        )
        force_local_pages(scores, ends, 1)
        return reference_native.msa128_stage2_topk_select(
            scores, 16, pages, 0, 0, None, None
        )

    operations = [reference, *[lambda selector=selector: selector.select(q, cache)
                              for selector in selectors]]
    expected = reference()
    for candidate in operations[1:]:
        compare(expected, candidate())
    graphs = []
    for operation in operations:
        operation()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = operation()
        graphs.append((graph, result))
    # Reuse the same workspaces, narrowing and widening the visible prefix.
    for delta in (-123, -145000, 145035, 10):
        q.copy_(torch.randn_like(q, dtype=torch.float32).to(torch.float8_e4m3fn))
        ends.add_(delta)
        page_ids.copy_(page_ids.view(batch, pages).roll(1, -1).flatten())
        for graph, _ in graphs:
            graph.replay()
        for _, selected in graphs[1:]:
            compare(graphs[0][1], selected)
    # All live scores tie for a zero query. Use the same live ends and
    # compare the vendor tie order through the actual candidate layouts.
    saved_query = q.clone()
    q.view(torch.uint8).zero_()
    tied = reference()
    for candidate in operations[1:]:
        compare(tied, candidate())
    q.copy_(saved_query)
    result = dict(batch_per_gpu=batch, exact_eager=True, exact_changed_graph=True,
                  exact_full_score_ties=True)
    if not measure:
        return result
    times = [[] for _ in graphs]
    for repeat in range(8):
        order = range(len(graphs))
        for index in (order if repeat % 2 == 0 else reversed(order)):
            graph = graphs[index][0]
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(30):
                graph.replay()
            end.record()
            end.synchronize()
            times[index].append(start.elapsed_time(end) / 30)
    return dict(result, median_ms=dict(zip(
        ("vendor", "verify8_v1", "full_page", "query_major",
         "query_quartet", "pages8", "pages32"),
        (statistics.median(t) for t in times), strict=True)), samples_ms=times)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measure", action="store_true")
    options = parser.parse_args()
    torch.manual_seed(447)
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Verify8 unit test requires exactly one visible GPU")
    torch.cuda.set_device(0)
    free, capacity = torch.cuda.mem_get_info()
    if free < 2 * 1024**3:
        raise RuntimeError("Verify8 unit test requires 2 GiB free on the selected GPU")
    torch.cuda.set_per_process_memory_fraction(1024**3 / capacity)
    test_synthetic(8, 1161)
    print(json.dumps([test_scanner(batch, measure=options.measure) for batch in (8, 16, 18)]),
          flush=True)
