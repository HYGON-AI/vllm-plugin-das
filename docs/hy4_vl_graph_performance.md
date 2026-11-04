# HYV4VL decode graph evaluation (2026-11-05)

This evaluates the installed breakable decode graph implementation on top of
`9eaf4d5`, after the [ViT/M-RoPE optimization](hy4_vl_performance.md).
The fixed-concurrency-1 RealWorldQA pair passes the predeclared raw-prediction acceptance check.
Two of four matched two-image captions change raw output. The configuration
is therefore kept experimental and is not accepted as generally preserving
predictions across the tested workloads. The previous eager path remains the
default. There are no additional model,
quantization, vision-resolution or kernel changes in this evaluation.

## Controlled end-to-end measurements

The formal pair uses one freshly loaded, profiler-free process on eight BW1101
HCUs, TP8, with the same checkpoint and limits as the preceding report. A
localhost-only diagnostic worker forces dispatcher `NONE` for the eager runtime
control, then restores graph dispatch. Captured graph buffers remain allocated
in both modes. This controls weights/process state but is not a separate fresh
`--enforce-eager` launch. The historical stage-1 eager results are also shown
separately, so the reference for the requested additional 50% is explicit.

| Suite | Requests | Eager → graph p50, ms | Eager → graph req/s | Gain vs paired eager | Gain vs stage-1 eager | Raw answers changed vs paired eager |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RealWorldQA, c=1 | 64 | 1134.9 → 822.6 | 0.909 → 1.292 | +42.0% | +42.9% | 0 |
| RealWorldQA, c=2 | 64 | 1598.6 → 1427.9 | 1.286 → 1.492 | +16.0% | +15.8% | 0 |
| Single image, 32 tokens | 4 | 7785.0 → 2562.8 | 0.129 → 0.414 | +219.7% | +216.9% | 0 |
| Two images, 32 tokens | 4 | 7990.8 → 3326.8 | 0.126 → 0.309 | +145.7% | +147.4% | 2 |

The same fixed 32 RealWorldQA images (seed 338), two QA rounds, two warmups and
four requests per caption suite are used. Image encoding occurs before timing.
HTTP timing includes serialization, preprocessing, encoder, prefill and decode.
Sampling, prompts and output budgets are unchanged; caption min/max are both
32 with EOS ignored. Four-request caption measurements are indicative, not a
capacity benchmark. A 50% throughput increase means 1.5× throughput, not a 50%
latency reduction. This evaluation does not establish that every workload gains
50%.

Prefix caching and CPU MM processor caching are disabled. The default GPU
encoder-embedding LRU remains enabled and is reset before each complete client
sequence. Repeated images can then reuse vision embeddings. These are the same
repeated-image workloads as stage 1; they do not measure a forced-cold encoder.
No concurrent GPU benchmark was run. The timing client and full-evaluation
client are unchanged from `9eaf4d5`.

`e2e-graph/` is an initial exploratory run, before the final paired control. Its
server had a profiler configured but inactive; a fused-MoE JIT warning appeared.
`e2e-eager-control/` and `e2e-graph-repeat/` are the formal, warmed measurements
from the new process with no profiler configured. Raw suite outputs, TTFT,
latency p95, usage and manifests are retained for all three runs.

The initial graph run compared with the historical stage-1 eager process has
zero raw changes in the 64+64 short-QA and four single-image caption requests,
but **three of four two-image captions change wording**. This across-process
variation is retained in `initial-output-comparison.json`; its root cause is
unresolved. The graph-enabled process forced to eager also changes three of
these historical captions, recorded in `runtime-control-output-comparison.json`.
The initial graph and this eager control differ in one of four two-image
captions; neither comparison isolates a root cause. The main table uses the
final same-process control and preserves this earlier observation.

## Paired real-image precision

The original pinned RealWorldQA test split has 765 image/question pairs. Full
original prompts, scorer/extractor revision and image hashes match the earlier
[dataset report](hy4_realworldqa_validation.md). Both runs use temperature 0,
seed 0, no_think, max 16 output tokens, max pixels 1,048,576 and concurrency 1.
Graph runs first, then eager on the same weights, with both caches reset.
Concurrency 1 holds batch formation fixed; this is not a concurrency-2
prediction-invariance claim.

- Eager: 561/765 task-correct; strict exact-match 546/765.
- Graph: 561/765 task-correct; strict exact-match 546/765.
- Raw predictions changed: 0; correctness gains 0, regressions 0.
- Eager/graph HTTP errors: 0/0; empty: 0/0; truncated: 0/0.
- Strict acceptance: `True`. Acceptance requires zero raw changes and zero request/empty/truncation errors; aggregate-score parity alone does not pass.

No raw predictions changed in this fixed-concurrency-1 pair.

The final profiler-free graph service also passes all seven constrained API
smoke checks after the full pair and timings (text, colors, count, OCR, image
order and repeated image). The additional photo answer remains manual review.

This measures sampled task behavior; it does not establish bitwise equality of
all language-model activations/logits or correctness on arbitrary inputs. Raw
answers for the fixed-length single/two-image captions and c=2 QA timing suite
are compared separately in the table. Historical stage-1 concurrency-2 run
variation is retained in its report and is not silently normalized away.

## Full-split sequential timing

The 765-item accuracy pair contains 762 distinct
PNG hashes, with the encoder LRU reset before each full run. Unlike the repeated
32-image streaming suite above, this secondary measurement includes client image
decode/RGB/PNG encoding and scoring and uses the non-streaming API. Graph runs
first; eager benefits from already warmed shape-dependent kernels.

- Eager: 969.9 s total, p50 1291.2 ms.
- Graph: 834.5 s total, p50 1121.5 ms.
- Sequential completion throughput: +16.2%; p50 reduction 13.1%.

This is the fixed accuracy-evaluation workload, not a concurrent capacity test
or a substitute for the separately timed streaming suites. Per-item seconds and
wall-time summaries are retained with the paired predictions.

## Explicit experimental launch

The candidate changes only these settings relative to the stage-1 eager launch:
remove `--enforce-eager`, set `VLLM_USE_BREAKABLE_CUDAGRAPH=1`, and set
`--compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2]}'`.
`mode=0` keeps torch.compile disabled. Prefill remains eager; captures cover only
one/two-token decode batches. Actual per-rank dispatcher counts are retained,
rather than treating successful capture as proof of replay.

```bash
VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_BREAKABLE_CUDAGRAPH=1 \
vllm serve /module/Hy4-testing-Channel-FP8-w8a8 \
  --served-model-name hy4-vl --tensor-parallel-size 8 --moe-backend triton \
  --gpu-memory-utilization 0.90 --max-model-len 4096 --max-num-seqs 2 \
  --max-num-batched-tokens 1024 \
  --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2]}' \
  --mm-processor-kwargs '{"max_pixels":1048576}' \
  --limit-mm-per-prompt '{"image":2}' --mm-processor-cache-gb 0 \
  --no-enable-prefix-caching \
  --default-chat-template-kwargs '{"reasoning_effort":"no_think"}' \
  --host 127.0.0.1 --port 8010
```

This is an explicit experimental configuration, not an automatic architecture
default change. Use the eager launch from the preceding report for the accepted
fallback. Diagnostic control additionally uses `VLLM_SERVER_DEV_MODE=1` and
`--worker-extension-cls stage2_worker.Stage2Worker` on localhost only; these are
excluded from the example above. Exact worker/driver snapshots are archived as
`.py.txt` beside the results. They use internal dispatcher/RPC APIs of the
installed build and are not a public compatibility interface.

## Profiler evidence and observed failure

The BBuf analyzer produced kernel/source, overlap and fusion tables for all
eight worker traces. Every trace contains GPU events: 14,608–14,688 kernel events
per rank. The three-iteration capture contains image/prefill work and one decode
forward, not three steady decode steps. Source attribution under replay often
ends at `hipGraphLaunch`; no separate eager mapping trace was used.

Rank-0 prefill-labelled fused MoE, sparse attention and NCCL durations are
147.47, 76.85 and 63.81 ms (28.3%, 14.8%, 12.3% of summed kernel durations).
Decode-labelled sparse attention is 16.26 ms (26.5%), MoE 5.83 ms (9.5%) and
custom all-reduce 5.07 ms (8.3%). All-rank summed kernel time ranges
575.37–833.23 ms; rank skew and profiler/stop instrumentation are unresolved.
These shares are diagnostic and are not additive removable wall time. No overlap
row cleared the analyzer threshold; that does not prove absence of overlap.
Fusion-table pattern labels do not establish that HCU compiler fusion ran.

After the profiler automatically stopped, the next decode failed with HSA
exception 0x1016 / KERNEL VMFault and HTTP 500. That process was discarded before
full accuracy testing; no source fix or root cause is claimed. Formal accuracy
and timing use a fresh process with no profiler configured. The failure remains
material for profiling this graph configuration; the trace is not a stability
pass. Original 143 MiB gzip traces remain at `/root/hy4-perf-stage2/profiles`;
per-file hashes, all eight reports and interpretation limits are archived in
`profile-analysis/`. Large binary traces and GPU fault dumps are not committed.

The installed runtime reports no tuned BW1101 E=256/N=256 channel-FP8 MoE
configuration. That is a future tuning lead, not a tested change. A switch to
AITER was excluded because the installed expert path does not preserve this
checkpoint's positive SwiGLU clamp contract. No lower precision, image token
reduction, speculative decoding or alternate MoE backend is used.

## Skills and artifacts

Applied from BBuf/AI-Infra-Auto-Driven-SKILLS at
`371dee04ee30ed918b240823b2a12c139d1a67fa`:
[llm-torch-profiler-analysis](https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS/blob/371dee04ee30ed918b240823b2a12c139d1a67fa/skills/llm-torch-profiler-analysis/SKILL.md)
and
[model-pr-history-knowledge](https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS/blob/371dee04ee30ed918b240823b2a12c139d1a67fa/model-pr-optimization-history/SKILL.md).
The Hunyuan4 history supplied graph-related leads; upstream #57526/#57811 are
discovery inventories, not manually audited HCU fixes, and no code was copied.
Decisions and precision contracts are recorded in the history note.

Compact results and paired predictions are in
[`benchmarks/hyv4_vl_graph/`](benchmarks/hyv4_vl_graph/), including all profiler
analysis tables and manifests. Full local requests/logs remain under
`/root/hy4-perf-stage2`. Original image/model/dataset files and credentials are
not included.
