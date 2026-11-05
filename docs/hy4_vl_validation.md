# HYV4-VL implementation and validation

[Server/client commands](hy4_vl_serving.md) cover eager, decode-only and guarded
full/piecewise execution, pinned dataset preparation and evaluation tools.
This summary records the final validated behavior; per-request benchmark
outputs and temporary diagnostic scripts are excluded from the source tree.

## Image support and optimization

The v0.25.1 baseline fails to load this checkpoint's `hy_v4_vl` model type.
The adapter registers its configuration/model and provides the checkpoint's
BF16 vision tower, RMSNorm/Conv2d projection, image-row separators and
interleaved three-axis position encoding. It retains the HYV4 language backbone
and strict language/vision weight completeness checks.

Vision patches use a 1×1 convolution over extracted patch channels, retaining
MIOpen accumulation/bias rounding and checkpoint parameter names/layout.
Attention boundaries come from CPU grid metadata. TP8 keeps per-image
projection/MLP shapes and collective order to preserve BF16 numerics; the TP1
path can pack independent images. Fused M-RoPE preserves eager BF16/FP16
product rounding, with native fallback for other layouts/dtypes.

Direct TP8 vision measurements with real checkpoint weights reduced the tested
single-image forward from 114.3 to 64.4 ms and two-image forward from 229.9 to
130.7 ms, with exact complete output tensors in those measured cases.
`tools/hyv4_vit_bench.py` provides a reusable real-weight comparison against a
reference checkout; the serving guide covers end-to-end measurements.

## Precision fixes

Two independent causes of changed answers were reproduced and fixed:

- **Prefill shape padding:** padded PIECEWISE inputs changed answers even
  without graph replay. The unguarded full/piecewise run scored 552/765 versus
  decode-only's 561/765, with 58 raw answer changes (9 gains, 18 losses).
  At DP=1, the HYV4-VL v1 runner now replays PIECEWISE only for exact-size token
  descriptors. Other chunks execute at their original eager size. FULL decode
  graphs remain available.
- **Sparse reduction order:** beyond the 2048-token Top-K boundary, native
  selection can return the same tokens in a different order. BF16 sparse
  attention rounds differently, and subsequent layers amplify the difference.
  The HCU HYV4-VL indexer now canonicalizes active index rows into ascending
  logical token order. Native scores, selected membership/multiplicity,
  invalid slots and the shared buffer are retained. Already ordered rows skip
  sorting/writeback; inactive backing-buffer rows are untouched.

The order diagnosis uses all eight ranks: the first producer's scores and
selected sets match across native repetitions while its order differs;
subsequent producer scores, then selected sets, diverge. Canonical ordering
makes all 21 producer calls identical in the traced repetitions. Four repeated
image pairs improve from 1/4 to 4/4 repeat-identical answers, including all 33
full-logit records of the traced pair.

An independent fixed-Q/K/V replay repeats exactly for each fixed order.
Reversing only the index order changes 160/4096 BF16 output elements, maximum
absolute difference 0.001953125. Both orders have about 0.00163 relative L2
error against the captured rank-0 CPU float64 attention reference. This
supports the rounding diagnosis for that input; it does not prove full-model
float64 parity or universal determinism for arbitrary Top-K ties.

## Final correctness validation

Fresh-process inference source: `47bbdcda999e2ca8c6099da72423fec2f6adc7b9`,
validated 2026-11-05. The same full-split and long-context gates were
independently rerun at
`a5b2eda`. The review follow-up below has separate focused validation.
The process loads committed source normally and captures new graphs. A Stage4
diagnostic worker and development RPC supply counters, mode controls and
traces; they are not required by the normal serving commands.

| Check | Result |
| --- | --- |
| Local model/runner regression tests | 296 passed after review fixes; originally 287 |
| Original RealWorldQA, concurrency 1 | 561/765 (73.33%); strict 546 |
| Raw answers versus preceding independent decode-only control | 765/765 identical |
| HTTP failures / empty answers / truncations | 0 / 0 / 0 |
| Four-mode two-image captions, 32 generated tokens | 32 requests; answers identical per pair across modes/repetitions |
| Longer captions, 128 generated tokens | 12 requests; eager/full-piecewise answers identical per pair |
| Traced complete logits | All 33 records in each of 8 short runs and all 129 in each of 4 long runs match eager reference |
| Modified long-prompt QA, 3072–3076 input tokens | 96 requests; fixed eager/full-piecewise have zero repeat differences; 32 paired answers identical |
| Constrained HTTP text/image/OCR smoke checks | 7/7 passed; optional real photograph unscored |

The long-prompt diagnostic uses 16 image pairs, asks about the first image and
adds ignored text filler. Each mode scores 12/16 in each repetition. It is
separate from the official original-question RealWorldQA score. Native eager
changes one raw answer across repetitions; both fixed modes have none.

During the official full-split run, actual wrapper execution counts on each
rank are NONE=764, FULL=1553 and PIECEWISE=703, after the padding guard.
All 21 indexers on each of eight ranks have canonicalization enabled.
Regression coverage includes vision, M-RoPE, registration, configuration,
weight loading, attention and runner contracts. The new order tests cover
multiplicity, sentinels, strided storage, inactive rows, idempotence and graph
replay with new inputs.

## Dataset and scoring

Use all 765 original questions (762 distinct images) from
[xai-org/RealworldQA](https://huggingface.co/datasets/xai-org/RealworldQA),
revision `17e7f75e092e47169732462ea3cdfebe911105dd`. The dataset is CC BY-ND
4.0; images are downloaded locally and are not redistributed in this PR.
The unmodified `mcq_extract.py` scorer comes from
[lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/7847ad8539889c87533b78c93e280ddfc117d16b/lmms_eval/tasks/_task_utils/mcq_extract.py),
revision `7847ad8539889c87533b78c93e280ddfc117d16b`.

The full-split client preserves original question text without an extra system
or answer prompt, using temperature 0, top-p 1, seed 0, no_think, maximum 16
output tokens and concurrency 1. It records image hashes, dataset/scorer/code
provenance, per-item predictions and all failures. Empty/truncated/error items
are retained. Compare raw predictions as well as aggregate accuracy.

## End-to-end performance

Both modes enable the fixes, use the same loaded weights/graph allocations and
32 fixed-seed real images over two rounds. Processor/prefix caches are disabled;
default encoder caching can reuse repeated images and is reset before each
mode. No profiler or competing inference process runs during measurement.

| Workload | Requests/mode | Eager p50 ms | Full/piecewise p50 ms | Throughput gain | Raw changes |
| --- | ---: | ---: | ---: | ---: | ---: |
| QA, concurrency 1 | 64 | 1129.1 | 812.4 | 43.2% | 0 |
| QA, concurrency 2 | 64 | 1597.2 | 1417.2 | 16.7% | 0 |
| Single-image caption, 32 tokens | 4 | 7720.4 | 2552.4 | 217.9% | 0 |
| Two-image caption, 32 tokens | 4 | 8119.7 | 3335.0 | 149.4% | 0 |

Caption timing has only four requests per mode. These gains compare fixed
graphs with fixed eager execution; they do not establish an additional 50%
gain over the preceding optimized graph version or isolate canonicalization
cost. This repeated-image workload is not a forced-cold encoder benchmark.

## Validated configuration and limits

The checkpoint is `/module/Hy4-testing-Channel-FP8-w8a8`, with FP8 linear/MoE
weights and BF16 model/attention KV. The indexer's normal packed FP8 K cache is
separate. Hardware is eight BW1101 HCUs (gfx938); v1 runner, TP8/DP1, Triton
MoE, 4096-token context, maximum two sequences/images, 1024 batched tokens and
1,048,576 image pixels. Software: DAS vLLM 0.25.1, PyTorch 2.11.0/DTK2604,
Transformers 5.20.1 and Triton 3.6.0/DTK2604.

Graph modes use the breakable-cudagraph implementation with compilation mode
0; ViT remains eager. Restart the service and capture new graphs after upgrading.
Other checkpoints, parallel configurations and context limits are not validated
by these measurements. Video, encoder-only mode, precomputed image embeddings
and speculative decoding are unsupported. The adapter remains experimental;
this score is not
an independent reference-model accuracy-parity result.

## Review follow-up

VL configuration now inherits HYV4's post-init normalization, so missing/null
layer lists follow the final layer count and explicit lists remain intact.
The real checkpoint's language/vision configuration is preserved; its
`use_cache=True` setting is now retained instead of being dropped by the generic
configuration constructor. The actual vLLM ModelConfig resolves the checkpoint
with 78 layer entries. Encoder-only mode is rejected before backbone allocation.

The performance client reports content-chunk intervals rather than dividing
visible-text timing by completion-token counts that may include EOS. Single
content events yield null; TTFT, latency and throughput calculations retain
their prior definitions. This changes the JSON metric names documented in the
serving guide. Nine added cases cover missing/null/explicit configuration,
round-trip serialization, early rejection and single/batched streaming deltas.
All 296 related regression tests pass. The full-split accuracy measurements
above precede these review fixes; subsequent public HTTP checks are below.

## ViT activation copies

At `00a5fa0`, the TP vision path splits image activations once and keeps each
image separate across all 40 layers. It removes the per-layer concatenation
and repeated splitting while retaining per-image GEMM shapes, image order,
collective order, parameters and BF16 computation. TP1 execution is unchanged.

Real checkpoint weights on TP8 are compared with `9481001`, using five warmups
per tower followed by 30 CUDA-event samples per case and version, alternating
which version executes first. Median encoder times:

| Input | Before ms | After ms | Latency reduction |
| --- | ---: | ---: | ---: |
| Small image, grid 16×24 | 19.160 | 18.051 | 5.8% |
| Large image, grid 50×78 | 64.377 | 63.503 | 1.4% |
| Two large images | 130.837 | 126.675 | 3.2% |

Six initial cases cover single/multiple images, mixed sizes and prior sensitive
inputs. A further check pairs all 765 original dataset image rows (762 unique
images) and compares complete encoder tensors bitwise on all eight ranks;
all comparisons pass. All 296 related regression tests pass.

A fresh normal server at `00a5fa0` captures full/piecewise graphs without a
diagnostic worker extension or development RPC. All 136 fixed real-image
performance requests match the preceding `a5b2eda` graph control's raw answers,
and seven constrained HTTP smoke checks pass. QA latency p50 is 812.0 ms at
concurrency 1 and 1415.4 ms at concurrency 2. These cross-run HTTP measurements
are a compatibility check, not an isolated end-to-end speedup measurement.
The encoder timing gains above do not establish a comparable end-to-end gain;
the full 765-question QA evaluation was not rerun for this copy-only change.
