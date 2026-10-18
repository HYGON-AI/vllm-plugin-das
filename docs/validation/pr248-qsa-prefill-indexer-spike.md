# PR 248 QSA prefill-indexer spike validation

## Scope

This experiment tests the request-major prefill score-kernel design from
upstream vLLM PR 54513 against the HCU baseline actually used by PR 248:
`flash_attn.mqa_paged_score_func`. It does not compare against upstream's
generic NVIDIA Triton baseline and it does not enable dense prefill.

The candidate changes only the BF16 QSA score calculation. Existing QSA
metadata, top-k semantics, index expansion, cache layout, cache writes, and
the FP8 sparse-attention reader are unchanged.

## Environment

- Device: BW1100, `HIP_VISIBLE_DEVICES=0`
- PyTorch: `2.11.0`
- HIP: `6.3.26113`
- vLLM: `0.28.1+dtk2604.torch2110.2609171627.g77acaf`
- Plugin branch: `codex/qsa-prefill-indexer-spike`
- Kernel commit: `525d241` (`feat(qsa): prototype request-major prefill scorer`)
- Timer: synchronized HIP events
- Temporary-logits budget: 128 MiB per chunk
- Timing: 5 warmups, 10 repetitions per round, 3 independent round medians

Command:

```bash
HIP_VISIBLE_DEVICES=0 python benchmarks/kernels/benchmark_qsa_prefill_indexer.py \
  --rounds 3 --warmup 5 --repetitions 10
```

## Correctness

Focused kernel tests compare valid logits and selected top-k sets against the
CUTLASS baseline for context-free, prefixed, permuted-page, and nonzero chunk
offset cases. Result: `7 passed`.

The benchmark also checks all valid scores and samples request start/middle/end
top-k sets using the production compressed top-k of 512. All three cases
passed. Maximum absolute score error was `3.814697e-06`.

## Performance

| Case | Rows × columns | Chunks | CUTLASS median | Candidate median | Reduction | Peak delta, both paths |
|---|---:|---:|---:|---:|---:|---:|
| BS4, query 4096, no prefix | 16384 × 1024 | 1 | 1.850718 ms | 1.161198 ms | 37.257% | 64 MiB |
| BS16, query 4096, no prefix | 65536 × 1024 | 2 | 7.327752 ms | 3.793676 ms | 48.229% | 256 MiB |
| BS4, prefix 8192 + query 4096 | 16384 × 3072 | 2 | 8.388951 ms | 6.662393 ms | 20.581% | 192 MiB |

Per-round medians in milliseconds:

- BS4/no-prefix CUTLASS: `1.850718, 1.850798, 1.850638`
- BS4/no-prefix candidate: `1.161198, 1.161999, 1.160478`
- BS16/no-prefix CUTLASS: `7.328871, 7.327752, 7.327431`
- BS16/no-prefix candidate: `3.797996, 3.793676, 3.792636`
- BS4/prefix CUTLASS: `8.388951, 8.389271, 8.387990`
- BS4/prefix candidate: `6.664792, 6.661833, 6.662393`

The peak metric includes two simultaneously live returned logits chunks while
the benchmark advances from one chunk to the next. It is identical between
baseline and candidate and remains independent of total batch logits beyond
the fixed chunk count. A serving integration must release or consume each
chunk before allocating the next one.

## Health check

All benchmark requests completed. The most recent HCU page-fault record in
`dmesg` remains at `2026-10-18 10:42:24`, before this experiment; no new
VMFault or invalid-address event appeared during the kernel tests or benchmark.

The repository-wide pytest command is not a green local gate in this image. It
was stopped at 17% after `666 passed, 11 failed, 63 skipped`; failures were due
to a missing `vllm_hcu.hcu_ops`, missing HumanEval data/isolation setup, and an
unrelated DeepGEMM numerical test. The QSA-focused suites passed.

## Decision

**Kernel gate: PASS.** The required BS16 4096 target improves by `48.229%`,
well above the 10% advancement threshold, with numerical/top-k parity and no
new HCU fault. Proceed to an opt-in prefill-only serving route. This result is
not yet evidence of end-to-end TTFT improvement; serving A/B remains required.

