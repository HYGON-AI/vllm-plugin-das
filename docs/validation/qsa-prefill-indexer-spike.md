# QSA prefill-indexer spike validation

## Scope

This experiment originated while analyzing PR 248. It tests the request-major
prefill score-kernel design from upstream vLLM PR 54513 against the HCU
baseline used by the standalone `v0.28.1-dev` feature:
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
- Source experiment branch: `codex/qsa-prefill-indexer-spike`
- Standalone integration branch: `feature/qsa-prefill-indexer`
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
new HCU fault. This justified testing an opt-in prefill-only serving route.

## Serving integration

The candidate was integrated behind both `VLLM_HCU_USE_CUSTOM_OPS=1` and the
new `VLLM_HCU_QSA_PREFILL_INDEXER=1` switch. Pure prefill QSA selection uses
the request-major scorer; decode, speculative decode, and mixed batches retain
the original selector. Dense prefill remained disabled.

The first ON startup exposed a configuration propagation defect: the resolved
`qsa_prefill_indexer=True` value was overwritten while `EngineArgs` rebuilt the
compile factors. Commit `7aebcc7` adds the field to the same preservation list
as the other platform-resolved policies and adds a regression assertion. The
relevant post-fix suite passed `345` tests.

The tested worktree was installed without changing vLLM or its dependencies:

```bash
pip install --force-reinstall --no-deps --no-build-isolation .
```

Serving configuration was identical on both sides except for the candidate
switch:

- Model: `/models/Qwen3.8-Flash-Next-w4a8-slimquant-ngram-int8-mtp`
- TP1, DP4, EP4 on HCU 0-3; AITER MoE and allgather/reducescatter
- MTP with 3 speculative tokens; FP8 e5m2 KV cache
- Maximum model length 32768; maximum batched tokens 16384; maximum sequences 64
- Prefix caching off, thinking off, dense prefill off
- `FULL_AND_PIECEWISE` compile mode
- Three warmup rounds followed by five measured seeds, each with 16 concurrent
  requests of exactly 4096 input and 1024 output tokens
- Temperature 0 and ignore-EOS enabled

ON logs explicitly reported `Qwen4Exp QSA routing: prefill-only selection uses
the HCU request-major indexer`. OFF and ON used distinct compile-cache hashes,
`90b5c642f5` and `d29b64389e`, respectively.

## Serving A/B

Each aggregate below is the median of the five run-level values:

| Metric | OFF | ON | ON vs OFF |
|---|---:|---:|---:|
| Mean TTFT | 2901.787 ms | 2884.335 ms | -0.601% |
| Median TTFT | 3042.263 ms | 2931.172 ms | -3.652% |
| Mean TPOT | 33.015 ms | 32.995 ms | -0.060% |
| Median TPOT | 29.136 ms | 30.941 ms | +6.196% |
| Output throughput | 288.428 token/s | 306.664 token/s | +6.323% |
| MTP acceptance | 60.840% | 59.550% | -1.290 percentage points |
| Completed requests | 80/80 | 80/80 | unchanged |

Mean-TTFT values by measured seed were:

- OFF: `3283.966, 2802.187, 2904.500, 2901.787, 2826.134` ms
- ON: `3713.418, 3814.879, 2854.862, 2773.788, 2884.335` ms
- Paired ON changes: `+13.077%, +36.139%, -1.709%, -4.411%, +2.059%`

The paired median is a `2.059%` regression and the paired mean is a `9.031%`
regression. The first two measured ON rounds still logged inference-time Triton
JIT compilation for selection expansion and MTP rejection/resampling kernels,
despite the three warmups. This explains some variance, but it also means the
current procedure does not demonstrate a reproducible end-to-end improvement.

Weighted over all draft tokens, MTP acceptance changed from `52,892 / 87,168 =
60.678%` OFF to `52,321 / 88,881 = 58.866%` ON. Because the generated token
trajectories differed, this is not a controlled measurement of an MTP change
caused by the scorer alone.

Raw results are under:

- `/models/pr248_qsa_prefill_indexer_ab_20261018/off`
- `/models/pr248_qsa_prefill_indexer_ab_20261018/on`

## Trace attribution

PyTorch/ROCTracer Chrome traces were captured after three kernel warmups for
the BS16, input-4096 score operation. Both paths launch two chunks under the
same 128 MiB logits budget:

| Trace | Kernel | Calls | GPU total |
|---|---|---:|---:|
| CUTLASS | `mqa_paged_score_mmac_kernel` | 2 | 7.296 ms |
| Candidate | `_qsa_mqa_paged_prefill_kernel` | 2 | 3.179 ms |

The trace therefore attributes a `56.43%` score-kernel reduction to the
request-major layout. This is consistent with the synchronized-event
microbenchmark. However, the saved scorer work is only about 4 ms against an
approximately 2.9-second serving TTFT, so the trace also explains why it does
not reproduce PR 247's seconds-scale TTFT change.

Trace files:

- `/models/pr248_qsa_prefill_indexer_trace_20261018/cutlass_bs16_in4096.json`
- `/models/pr248_qsa_prefill_indexer_trace_20261018/request_major_bs16_in4096.json`

## Accuracy and safety

Kernel-level numerical and top-k checks passed, and an ON serving smoke prompt
requesting exactly `OK` returned exactly `OK` with thinking disabled and
temperature 0. The response is saved as
`/models/pr248_qsa_prefill_indexer_ab_20261018/on/accuracy_smoke.json`.

The long random benchmark outputs were not byte-identical across separate OFF
and ON server runs (`0/80` exact matches; median common text prefix was zero).
This benchmark is not an accuracy evaluation, and separate DP/MoE runs can
also diverge, so it cannot isolate an accuracy regression. Conversely, the
smoke test alone cannot prove full HumanEval or MathVision parity. A real
accuracy claim would require rerunning the same evaluation set with the
candidate enabled.

After the serving runs and trace capture, the latest matching HCU fault in
`dmesg` was still the pre-existing `2026-10-18 10:42:24` event. No new VMFault,
invalid-address, page-fault, or failed serving request occurred.

## Final decision

**Default-enable gate: NO-GO.** Keep this implementation behind the default-off
experimental switch. The scorer itself is correct on the focused tests and
48-56% faster, but its absolute saving is only about 4 ms and the five-run
serving A/B does not show a reproducible TTFT gain. The standalone Draft MR
exists to continue isolated validation without coupling its lifecycle to PR
248.

## Standalone-head revalidation

The final revalidation below used standalone MR commit
`f4b02f078188ff108a78f6f2b1a87d071bbf43f9`, whose merge base with
`origin/v0.28.1-dev` is `49beac0a6a993ab441717abee66faa8176c02720`.
The installed plugin version was
`vllm_hcu-0.28.1rc1.dev491+das.f4b02f0.dtk26041`; vLLM remained
`0.28.1+dtk2604.torch2110.2609171627.g77acaf`.

The focused environment, kernel, config, routing, and worker suite completed
with `328 passed, 14 warnings` in 73.89 seconds. Re-running the synchronized
kernel benchmark on this exact commit produced:

| Case | Baseline | Candidate | Reduction | Maximum absolute error |
|---|---:|---:|---:|---:|
| BS4, query 4096, no prefix | 1.850319 ms | 1.152079 ms | 37.736% | 2.861023e-06 |
| BS16, query 4096, no prefix | 7.326634 ms | 3.778717 ms | 48.425% | 3.814697e-06 |
| BS4, prefix 8192 + query 4096 | 8.383914 ms | 6.626795 ms | 20.958% | 2.384186e-06 |

The exact-head serving A/B used the same configuration described above. There
were three warmups and five measured seeds on each side, with 16 concurrent
requests per measured seed and exactly 4096 input plus 1024 output tokens per
request. Both sides completed 80/80 measured requests without an error.

| Metric (median of five runs) | OFF | ON | ON vs OFF |
|---|---:|---:|---:|
| Mean TTFT | 2825.112 ms | 2897.326 ms | +2.556% |
| Median TTFT | 2976.844 ms | 3036.533 ms | +2.005% |
| Mean TPOT | 34.420 ms | 38.308 ms | +11.297% |
| Median TPOT | 30.036 ms | 36.883 ms | +22.797% |
| Output throughput | 278.136 token/s | 240.176 token/s | -13.648% |
| Weighted MTP acceptance | 57.786% | 57.687% | -0.099 percentage points |

Paired mean-TTFT changes by seed were `-1.566%, +0.341%, -24.555%,
+15.391%, -0.148%`; their median was `-0.148%`. This confirms that the
approximately 3.5 ms score-kernel saving is lost in run-to-run serving noise
and does not produce a repeatable end-to-end improvement. The run-level median
comparison is a regression, so the default-enable gate remains NO-GO.

Accuracy was then checked with the candidate route enabled, prefix caching and
dense prefill disabled, temperature 0, and thinking disabled:

- Full HumanEval: `158/164 = 96.34%`. The six failed tasks were
  `HumanEval/32`, `/93`, `/116`, `/140`, `/145`, and `/163`. A previous valid
  baseline run scored `159/164 = 96.95%`; the candidate run fixed `/129` but
  differed on `/116` and `/163`, a net one-sample change.
- MathVision sample: `14/20 = 70%` using four seeded samples from each of the
  five levels, batch size 16, `max_tokens=32768`, and server context length
  65536. Per-level scores were 25%, 100%, 75%, 100%, and 50%. This is a sampled
  check, not the full 3040-item benchmark. Failed sample IDs were level 1
  `4, 7, 9`; level 3 `58`; and level 5 `63, 126`. Level-1 sample 9 reached the
  32768-token output limit, and level-3 sample 58 generated 17038 tokens.
- Weighted MTP acceptance during MathVision was `66852 / 82077 = 81.45%`.
  All 20 requests completed; one finished by length and 19 by stop, with zero
  aborts and zero errors.

No `VMFault`, invalid-address, page-fault, segfault, fatal error, traceback, or
failed request appeared in the exact-head OFF/ON serving logs or the MathVision
service log. This validates the requested BS16, dense-prefill-off path; it does
not by itself prove that the earlier BS32 fault is fixed for every shape.

Exact-head artifacts are stored under:

- `/models/qsa_prefill_indexer_mr251_benchmark_20261018.log`
- `/models/qsa_prefill_indexer_mr251_ab_20261018/off`
- `/models/qsa_prefill_indexer_mr251_ab_20261018/on`
