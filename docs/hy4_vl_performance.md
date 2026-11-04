# HYV4VL ViT and end-to-end performance (2026-11-05)

For the later explicit decode-only graph configuration, full concurrency-1
accuracy pair and performance controls, see the
[graph follow-up](hy4_vl_graph_performance.md).

This compares the working image implementation at `bd1545a` against the ViT and
M-RoPE optimization at `edeb432`, using the same real
`/module/Hy4-testing-Channel-FP8-w8a8` checkpoint. Checkpoint parameters, image
resolution, language-model precision and generation settings are unchanged.

## Changes

- Collapse each already extracted patch into the input channels of a 1×1
  convolution. This avoids the expensive spatial patch convolution while
  retaining MIOpen accumulation/bias rounding and the original Conv2d checkpoint
  parameter names/layout. The merger retains its original convolutions,
  RMSNorm, GELU and newline embedding after each image row.
- Prepare vision attention boundaries once from CPU grid metadata, avoiding
  repeated allocations and GPU-to-CPU length synchronization. Under TP, retain
  per-image projection/MLP GEMM shapes and collective order to preserve BF16
  numerics. The TP1 path can pack independent images using variable-length
  attention; TP8 intentionally keeps separate image execution.
- Fuse language attention/indexer M-RoPE into one Triton kernel. It implements
  GPT-J even/odd pairs with three position axes and explicitly preserves eager
  BF16/FP16 product rounding. Other layouts/dtypes retain the native fallback.

## End-to-end results

Hardware/software: eight BW1101 HCUs, TP8, vLLM 0.25.1 DAS build, PyTorch
2.11.0+DTK2604, Transformers 5.20.1. Both servers use eager execution, Triton FP8
MoE, max model length 4096, max sequences 2, max batched tokens 1024 and max image
pixels 1,048,576. Prefix caching and the CPU MM processor cache are disabled.
The built-in GPU encoder-embedding LRU remains enabled: repeated images can reuse
vision features. Both runs use exactly the same image/request sequence, including
warmups. These are measurements of that repeated-image workload, not a forced
cold-encoder benchmark. The direct vision forward measurements below separately
measure encoder computation without an embedding cache.
No other model or GPU benchmark runs concurrently with either measurement.

Inputs are the original RealWorldQA images encoded losslessly as PNG data URLs.
The client prepares images before timing; its timer includes HTTP POST/JSON
serialization, server image processing, vision encoding, language prefill and
decoding. TTFT ends at the first nonempty streamed answer content. This is a
local HTTP measurement, not an isolated GPU kernel time.

The QA suites use 32 fixed random items (seed 338), two rounds, after two warmup
requests; temperature 0, seed 0, no_think and max output 16 tokens. Each caption
suite has four requests and fixes min/max output length to 32, ignoring EOS.
Those small caption suites are indicative measurements, not a capacity benchmark.

| Suite | Requests / concurrency | TTFT p50 before → after | End-to-end p50 before → after | Request throughput before → after |
| --- | --- | --- | --- | --- |
| RealWorldQA | 64 / 1 | 1009.4 → 903.1 ms | 1254.4 → 1127.3 ms (**−10.1%**) | 0.790 → 0.904 req/s (**+14.4%**) |
| RealWorldQA | 64 / 2 | 1302.9 → 1156.3 ms | 1789.7 → 1589.9 ms (**−11.2%**) | 1.143 → 1.288 req/s (**+12.7%**) |
| Single image, 32 output tokens | 4 / 1 | 839.8 → 750.2 ms | 8483.5 → 7707.6 ms (**−9.1%**) | 0.119 → 0.131 req/s (**+9.9%**) |
| Two images, 32 output tokens | 4 / 1 | 1250.0 → 1111.3 ms | 8912.4 → 8022.2 ms (**−10.0%**) | 0.113 → 0.125 req/s (**+10.6%**) |

QA TTFT p95 changes from 1106.1 to
955.8 ms at concurrency 1, and from
1747.8 to 1552.4 ms at
concurrency 2. For the fixed-length caption suites, median decode time per
output token changes from about 247 to
225 ms. Short QA answers are unsuitable for that
metric because the usage count can include EOS without another visible content
chunk; use their TTFT, total latency and throughput.

The isolated ViT gains below do not translate into an equal end-to-end gain:
language prefill/decoding and CPU image processing remain in every request.
No CUDA graph mode or larger serving batch is used in this comparison.

## Isolated ViT and numerical checks

The standalone vision comparison loads every real vision weight into both
implementations on one HCU (TP1), uses BF16, five warmups and the median of 15
CUDA-event measurements. This configuration uses the native RMSNorm path and is
separate from the TP8 server measurements above. Cases include a small fixture
and two real-world photo fixtures with unequal aspect ratios.

| Input | Patch grid(s) | Before | After | Speedup |
| --- | --- | --- | --- | --- |
| small | 16×24 | 22.70 ms | 16.97 ms | 1.34× |
| large | 50×78 | 162.99 ms | 112.35 ms | 1.45× |
| two | 50×78 + 72×54 | 326.94 ms | 207.35 ms | 1.58× |

The baseline profiler attributes about 60.4 ms (37.2%) of the large-image
forward to convolution, with about 58.8 ms in patch projection. In the three
measured cases, all complete vision embeddings match the original implementation
**exactly** (maximum absolute difference and relative L2 error both zero).
The packed two-image outputs also exactly match encoding each image separately.
The HCU BF16 projection regression checks equality at 384, 3900 and 7788 patches;
these measurements do not claim bitwise identity for every possible shape or
software/hardware version.

The additional **TP8** vision-only check uses the original images from dataset
items 552 and 571, each with a 50×78 patch grid, and native RMSNorm. The final implementation
matches all reference embeddings exactly, both singly and jointly:

| TP8 input | Before | After | Speedup |
| --- | --- | --- | --- |
| Item 552 | 114.30 ms | 64.36 ms | 1.78× |
| Items 552 + 571 | 229.89 ms | 130.75 ms | 1.76× |

The per-image TP strategy matters: packing these two images into larger TP8
GEMMs produced relative L2 differences up to 0.179 across the complete tower,
even though the analogous TP1 check matched exactly. That TP8 packing is
excluded from the final implementation. The diagnostic and corrected TP8
measurements are retained so this precision tradeoff can be reviewed.

The fused M-RoPE is checked against native vLLM on 72 combinations of
BF16/FP16/FP32, 1-D/3-axis positions, sectioned/interleaved axes, 1/7/1024 tokens,
strided inputs, unequal Q/K head counts and full/partial rotary head dimensions.
All outputs match exactly (`rtol=atol=0`). A single-HCU wall-time microbenchmark
measures native → fused as 0.197 → 0.065 ms for one token and 0.210 → 0.088 ms
for 1024 tokens; this is not the full model decode time.

## Accuracy regression

The optimized server was evaluated on all 765 original RealWorldQA questions
after completing the performance suites, with the unchanged prompt/scorer from
[the original report](hy4_realworldqa_validation.md). Original image hashes,
questions and reference answers match the original evaluation for all 765 items.

| Metric | Earlier default-cache report | Original operators, matched caches | Optimized, matched caches |
| --- | --- | --- | --- |
| Reference-scored accuracy | 560/765 (73.20%) | 558/765 (72.94%) | 558/765 (72.94%) |
| Multiple choice | 344/438 (78.54%) | 344/438 (78.54%) | 344/438 (78.54%) |
| Open-ended | 216/327 (66.06%) | 214/327 (65.44%) | 214/327 (65.44%) |
| Raw strict matches | 545/765 | 543/765 | 542/765 |
| HTTP errors / empty answers / truncations | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |

The matched-cache control uses the same loaded checkpoint weights and restores
the original Conv2d, per-image vision block/tower forwards and native M-RoPE.
Before its run the encoder cache is cleared, then the identical 32-image
performance/warmup sequence is replayed. All generation/scoring settings match.
The temporary worker harness and its mode metadata are retained in the artifacts;
its model source revision denotes the loaded optimized model, while the recorded
runtime mode explicitly identifies the original forward functions.

Against this matched control there are **2 changed
raw answers**, with 1 gained and 1 regressed
correct items. These are the directly comparable accuracy results. The paired
per-item predictions and image hashes are retained as `matched-predictions.jsonl`.
A nonzero difference remains a measured correctness variation: the operator
audit below does not erase it or establish its cause. Changes in runtime batch
formation are one possible explanation, not a demonstrated root cause. This
report therefore does not claim prediction invariance. The aggregate score is
unchanged in this matched comparison; raw strict matching decreases by one item.

Against the earlier default-cache report there are 4 changed raw answers and
4 changed normalized answers: 1 previously
incorrect items become correct and 3 previously correct items become
incorrect. The complete change list is retained in the artifacts. This is a
comparison with the working plugin baseline; reference/unquantized-checkpoint
parity is still unmeasured. The model retains its existing real-world errors.
Accuracy-run latency is not used for the performance claims above: the original
accuracy run used normal processor/prefix caches, while the optimized run
disables those. The default GPU encoder LRU remains enabled in both.

## Actual-runtime operator audit

After the full evaluation, the same loaded TP8 model was audited on the four
historically changed items (368, 374, 552, 571). Canonical PNG hashes match the
full-dataset inputs. Original and optimized vision forwards match exactly on
all eight ranks, both singly and in two-image pairs. In live language forwards,
**1782 M-RoPE calls per rank** (14,256 across eight ranks) match the native
implementation exactly, with zero differing elements and zero maximum error.

Clearing the encoder cache before each individual request, all four operator
modes (original, projection-only, optimized vision/native RoPE, fully optimized)
produce the same answers: `Yes`, `pedestrian`, `3`, `Red`. This audit validates
the sampled real operator inputs; it does not establish global batch invariance
or reference-checkpoint parity. The full matched-cache control above separately
checks the resulting dataset predictions.

## Reproduction and artifacts

Use the server command in [the RealWorldQA report](hy4_realworldqa_validation.md),
adding `--mm-processor-cache-gb 0 --no-enable-prefix-caching` for both baseline
and optimized performance runs. Start each server from its corresponding
isolated checkout. Run the client from the optimized checkout against each
server (the baseline revision predates this client), then run:

```bash
python tools/hyv4_vl_perf.py \
  --data-dir /tmp/realworldqa/data \
  --output-dir /tmp/hy4-perf-results
python -m pytest -q \
  tests/models/hy_v4/test_mrope.py tests/models/hy_v4/test_vl.py \
  tests/models/hy_v4/test_registration.py tests/models/hy_v4/test_runtime_config.py \
  tests/models/hy_v4/test_weight_loading.py
```

For isolated vision measurements, stop the server and use image fixtures with
`tools/hyv4_vit_bench.py`:

```bash
python tools/hyv4_vit_bench.py \
  --model /module/Hy4-testing-Channel-FP8-w8a8 \
  --reference-checkout /tmp/vllm-plugin-das-perf-baseline \
  --case large /path/to/photo1.png \
  --case two /path/to/photo1.png /path/to/photo2.png \
  --output-dir /tmp/hy4-vit-results
```

For TP8 vision-only validation, prefix the command with
`torchrun --nproc-per-node=8 --master-port=29942` and add `--tp-size 8`.

The tested configuration passes **193 tests**. Ruff and `git diff --check` pass.
[Raw timings, answers, manifests, source/dataset hashes, profiler and numerical
checks](benchmarks/hyv4_vl_perf/) are retained for review. Results apply to this
checkpoint, hardware, resolution and concurrency; non-eager serving and higher
concurrency have not been measured here.
