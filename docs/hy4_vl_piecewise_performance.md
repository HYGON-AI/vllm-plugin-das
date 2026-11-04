# HYV4VL full/piecewise evaluation and optimization space

This follows the [decode-only evaluation](hy4_vl_graph_performance.md) on the
same checkpoint, eight BW1101 HCUs and TP8. The explicitly requested
`FULL_AND_PIECEWISE` mode is tested with breakable CUDA graphs and the vLLM
torch.compile pipeline disabled. The full 765-item concurrency-1 raw-prediction check **fails**;
all timing-suite differences are separately visible below. No production default
or model/kernel implementation changes are made in this round.

Use [the server/client commands](hy4_vl_serving.md) for standalone eager,
decode-only and full/piecewise launch, pinned dataset download, image checks,
streaming performance, complete accuracy and raw-answer comparison.

## Actual graph coverage

Source/client revision: `c17d77c`. Environment/build/checkpoint are unchanged
from the preceding report. Candidate settings:

```json
{"mode":0,"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[1,2,128,256,512,768,1024]}
```

`VLLM_USE_BREAKABLE_CUDAGRAPH=1` explicitly permits the PIECEWISE path without
Inductor in this installed DAS build. Decode dispatches FULL at one/two tokens;
eligible prefill/mixed batches dispatch PIECEWISE. Larger capture buckets are
necessary to cover image prefill; retaining only `[1,2]` would not exercise it
meaningfully. Sparse indexer/attention has an explicit eager break in PIECEWISE;
its FULL path does not take that break. ViT stays eager. Thus this does not test
an encoder graph, torch.compile or full-prefill FULL graphs.

The server captures seven PIECEWISE descriptors and two FULL descriptors.
All eight ranks retain matching runtime mode/shape counters, verified in
`dispatch-validation.json`. The initial image smoke already dispatches nine
PIECEWISE and 69 FULL forwards per rank. Successful capture alone is not used
as proof of real-request coverage.

## Matched streaming end-to-end results

The same process/weights are held fixed. The diagnostic worker can allow all
modes, restrict valid modes to FULL/NONE (decode-only control), or force NONE
(eager control). Captured buffers remain allocated in all modes. This controls
process/weights but does not replace an independent fresh launch of each mode.
The public commands start separate servers; historical independent eager and
decode-only results remain in their original reports.

| Suite | Requests | Decode-only → full/piecewise p50, ms | Decode-only → full/piecewise req/s | Throughput gain vs decode-only | Gain vs eager control | Changed raw answers vs decode-only |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RealWorldQA, c=1 | 64 | 817.5 → 720.2 | 1.285 → 1.457 | +13.4% | +59.3% | 4 |
| RealWorldQA, c=2 | 64 | 1424.7 → 1230.6 | 1.495 → 1.699 | +13.6% | +31.0% | 7 |
| Single image, 32 tokens | 4 | 2556.6 → 2453.3 | 0.415 → 0.427 | +3.0% | +225.0% | 4 |
| Two images, 32 tokens | 4 | 3316.6 → 3311.8 | 0.310 → 0.310 | -0.1% | +147.2% | 3 |

QA uses 32 fixed original RealWorldQA images, seed 338, two rounds and two
warmups. The captions have four requests each with min/max output 32 and EOS
ignored; these are indicative, not capacity measurements. Images are encoded
before the local streaming HTTP timer. Prefix and CPU processor caches remain
disabled; the default GPU encoder LRU is reset before each client sequence and
then permits repeated-image reuse. This is not a forced-cold vision benchmark.
The maximum image pixel setting stays 1,048,576, with max length 4096,
max sequences 2 and batched tokens 1024; channel FP8 LM / BF16 vision, prompts, temperature/seed and output
budgets remain unchanged. No concurrent GPU workload runs during measurements.

Initial timing order: full/piecewise, decode-only, eager. After both complete
accuracy runs, full/piecewise and decode-only are timed again. The table uses
those warmed repeats versus the earlier warmed eager control. All five timing
runs retain raw answers, usage, TTFT/p95 and manifests. An inference-time
`fused_moe_kernel` JIT warning occurred in the initial decode-only client
sequence; the final comparison uses warmed repeats. Neither run order nor
small caption counts support a statistical confidence claim.

## Full real-image accuracy pair

Both runs use the pinned original 765-item dataset/scorer, original question,
temperature 0, top-p 1, seed 0, no_think, max output 16, concurrency 1.
Full/piecewise runs first, then decode-only on the same loaded weights, with
CPU/GPU encoder caches reset before each full run. Image/question/reference
hashes and evaluator manifests match exactly.

| Mode | Task-correct | Strict exact match | HTTP errors | Empty | Truncated |
| --- | ---: | ---: | ---: | ---: | ---: |
| Decode-only | 561/765 (73.33%) | 546 | 0 | 0 | 0 |
| Full/piecewise | 552/765 (72.16%) | 536 | 0 | 0 | 0 |

- Raw changed: 58; correctness gains 9, regressions 18.
- Task accuracy difference: -1.18 percentage points.
- Predeclared acceptance (zero raw changes and no request/empty/truncation errors): `False`.

Examples of score-changing outputs (all 58 raw changes remain in the paired artifacts):

| Item | Reference | Decode-only | Full/piecewise | Result |
| --- | --- | --- | --- | --- |
| 94 | A | `A` | `B` | regression |
| 119 | 0 | `0` | `2` | regression |
| 126 | 2 | `2` | `3` | regression |
| 143 | No | `No` | `Yes` | regression |
| 155 | C | `C` | `A` | regression |
| 161 | A | `B` | `A` | gain |

This test combines prefill graph replay with capture-size padding and runtime
scheduling effects. Their individual contributions to changed predictions are
not isolated. No model/kernel source changes are made, and no root cause is
claimed for the observed variation.

The current decode-only run compared with the preceding complete decode-only
c1 run has 0 changed raw answers across 765 items
(561 → 561 task-correct),
retained in `historical-control-comparison.json`. This across-process observation
is distinct from the formal same-process pair above. The earlier decode-only/
eager full pair is documented separately and does not replace this comparison.
Aggregate-score parity is insufficient proof of prediction or numerical equality. These tests do not establish bitwise
logit/activation equivalence, reference-checkpoint parity or accuracy for
arbitrary image inputs, longer generation or higher concurrency. All seven
constrained API image checks pass both before and after the formal runs; the
optional photographic answer remains unscored manual review.

## Sequential real-image evaluation timing

The 765-item evaluation has 762 distinct PNG
hashes, with the encoder LRU reset before each full run. Unlike the repeated
32-image streaming suite, the timer includes client image decode/RGB/PNG encoding
and scoring and uses the non-streaming API. Full/piecewise runs first; the
decode-only control benefits from already warmed shape-dependent kernels.

| Mode | Full-split wall time | Request throughput | Median request latency |
| --- | ---: | ---: | ---: |
| Decode-only | 829.3 s | 0.922 req/s | 1117.9 ms |
| Full/piecewise | 764.9 s | 1.000 req/s | 1024.6 ms |

Completion throughput changes +8.4%.
This secondary measurement is not a concurrent capacity test. Its prediction
differences above remain material; speed does not establish accuracy acceptance.

## Remaining optimization space: measured MoE GEMM probe

The preceding eight-rank profiler analysis identified routed MoE and sparse
attention as substantial GPU work. The installed channel-FP8 Triton path has no
BW1101 E256/N256 tuning file. After stopping the serving process and releasing
its workers, an isolated HCU0 probe tests the **first gate/up GEMM only** using
actual layer-1 TP0 checkpoint weights/scales for all 256 experts. Input
activations and unique top-8 routes are synthetic with fixed seeds. The K tile
stays 128; candidates vary M/N tiles and warp count. Configs with any sampled
BF16 output difference are excluded from the fastest-exact column.

| Input M | Default GEMM, µs | Fastest sampled-exact, µs | Local speedup | M/N/K tiles, warps | Numerically differing candidates |
| --- | ---: | ---: | ---: | --- | ---: |
| 1 | 55.8 | 48.9 | 1.14× | 32/64/128, 4 | 0 |
| 2 | 60.4 | 54.6 | 1.11× | 32/64/128, 4 | 0 |
| 128 | 741.1 | 741.1 | 1.00× | 64/128/128, 4 | 0 |
| 256 | 765.0 | 757.8 | 1.01× | 32/64/128, 4 | 0 |
| 512 | 769.6 | 764.5 | 1.01× | 32/64/128, 4 | 0 |
| 768 | 807.3 | 777.6 | 1.04× | 64/128/128, 4 | 0 |
| 1024 | 811.3 | 782.6 | 1.04× | 64/128/128, 8 | 0 |

In this tested grid, local prefill-size gains are about 0–4%, and decode-size
gains about 11–14%. That does not establish a further 50% end-to-end gain.
Selection from one candidate sweep can favor timing noise for small differences.

Timing uses CUDA events, the median of five rounds, 64 kernel invocations per
round under graph replay, after compilation/warmup. This excludes quantization,
expert alignment/routing, activation and its clamp, down GEMM, TP communication
and HTTP. **These local gains are not whole-MoE or end-to-end speedups.**
Sample-exact output is not all-layer accuracy validation. The probe retains all
candidates/configs/errors, weight/scale hash, exactness/max-error and timing;
no tuned configuration is applied to serving. A deployment change requires both
GEMMs, real hidden-state/routing samples, every TP rank, SwiGLU clamp semantics,
full task accuracy and matched HTTP measurements. The installed config map can
select one setting for both expert GEMMs, so a first-GEMM winner must not be
installed blindly.

## Remaining optimization space: ViT and serving

- **ViT graph support requires implementation.** The adapter does not implement
  `SupportsEncoderCudaGraph`; the installed runner returns no encoder manager
  for it. Simply enabling `cudagraph_mm_encoder` would not graph this tower.
  A grid-aware implementation must preserve per-image TP GEMM shapes and
  collective order and compare complete embeddings, especially unequal-image
  batches. The preceding rejected TP8 packing experiment remains relevant.
- **Reduce TP vision copy/launch overhead.** The current per-layer cat/split
  preserves shapes. Retaining per-image tensors across layers could remove
  copies while keeping GEMM sizes, but allocator/alignment/collective behavior
  still needs actual TP8 numerical and timing checks. No untested rewrite is
  included here.
- **Clamped activation plus per-token FP8 quantization is another fusion lead.**
  Installed `TritonExperts.apply` uses its fused activation/quant path only for
  block FP8 with `[128,128]` scales. This channel-FP8 checkpoint instead runs
  its clamped activation and per-token quantization separately. A replacement
  must preserve the positive SwiGLU clamp, intermediate BF16 rounding and FP8
  scale/reduction semantics; no such fusion or gain is claimed here.
- **Sparse attention and communication remain candidates.** Reuse the existing
  all-eight-rank profile as diagnostic context only. Its short window, replay
  attribution limits and post-profiler VMFault do not establish steady-state
  rank imbalance or a fresh stage-3 bottleneck percentage. No live profiler is
  configured for this round. An AITER backend switch still fails the current
  SwiGLU clamp audit and is not used.
- **Cache/concurrency tuning is workload-specific.** CPU processor/prefix cache
  changes or higher batch capacity can help repeated-image serving, but they
  change the benchmark policy. They need separate accuracy/capacity controls
  and must not be called a code-only gain from the table above.

## Artifacts and skills

[`benchmarks/hyv4_vl_piecewise/`](benchmarks/hyv4_vl_piecewise/) contains full
paired predictions, all timing runs, per-rank dispatch counts/shapes, manifests,
smoke results, offline probe and driver/worker/launch snapshots. Full local logs
remain under `/root/hy4-perf-stage3`. Images, weights and credentials are excluded.
`SHA256SUMS.json` covers the compact artifacts.

The BBuf skills remain pinned to `371dee04ee30ed918b240823b2a12c139d1a67fa`:
[llm-torch-profiler-analysis](https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS/blob/371dee04ee30ed918b240823b2a12c139d1a67fa/skills/llm-torch-profiler-analysis/SKILL.md)
and
[model-pr-history-knowledge](https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS/blob/371dee04ee30ed918b240823b2a12c139d1a67fa/model-pr-optimization-history/SKILL.md).
Current installed source is authoritative; historical PR inventories are leads,
and no upstream PR code is copied. Decisions are recorded in the history note.
