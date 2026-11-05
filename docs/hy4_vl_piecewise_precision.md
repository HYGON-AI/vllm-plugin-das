# HYV4VL prefill padding precision investigation

The earlier FULL_AND_PIECEWISE regression is reproducible without graph replay:
keeping the padded prefill shapes while executing eagerly gives the same changed
answers. Removing that padding restores the decode-only answers. The fix keeps
PIECEWISE for exact-size prefill chunks and uses unpadded eager execution for
other chunks in the validated single-DP HYV4-VL serving path. FULL decode graphs
remain available.

This follows the [original full/piecewise experiment](hy4_vl_piecewise_performance.md),
which measured 552/765 versus decode-only's 561/765. Its 58 changed raw answers,
9 gains and 18 regressions remain preserved as historical results. This report
uses the same checkpoint, TP8, BW1101 hardware, software, prompts, pinned dataset,
image cap and unmodified scorer. No weight, image-resolution, quantization,
attention algorithm or scoring changes are made.

## Causal ablation

A fresh diagnostic server captures FULL_AND_PIECEWISE with sizes
`[1,2,128,256,512,768,1024]`. All comparisons hold its weights and captured
allocations fixed. Three runtime modes are each repeated twice over 15 selected
items: unchanged controls, a formatting change and 13 historical regressions.
This deliberately selected subset is a regression reproducer, not an accuracy
estimate.

| Runtime | Prefill tensor shape | PIECEWISE replay | Correct / selected 15 | Changed raw answers versus decode-only |
| --- | --- | --- | ---: | ---: |
| Decode-only control | Actual token count | No | 15 | 0 |
| Padded eager ablation | Captured bucket size | Bypassed | 2 | 14 |
| Original full/piecewise | Captured bucket size | Yes | 2 | 14 |

The padded eager ablation leaves the dispatch descriptor, input preparation,
positions and attention metadata as in PIECEWISE, but calls the wrapper's
underlying runnable directly. FULL decode is retained in all three modes.
Padded eager and original PIECEWISE produce identical raw answers for all 15
items in both repetitions. Per-rank bypass counters verify that the eager
ablation actually bypasses every PIECEWISE model invocation.

All 58 changed answers in the preceding complete evaluation have a 1024-token
first prefill chunk and a short final chunk. For example, item 119 has 1046
prompt tokens: its last 22 tokens are expanded to 128 by the graph dispatcher.
The original decode-only answer is `0`, while both padded eager and original
PIECEWISE answer `2`. Item 94 has a 43-token tail and changes `A` to `B`.
Correlation across the full split is supporting evidence; the eager-padding
ablation is the direct intervention.

## Where the numbers diverge

For items 119 and 94, rank 0 records effective rows 0, middle and last at every
decoder-layer boundary, with more detailed hooks in layer 0. The same-mode
repetitions are exact across all 1592 recorded tensors per item/mode. The sampled
1024-token chunk outputs match at every decoder layer across padded and unpadded
modes. The short final chunk diverges.

The first observed difference is in layer 0's iHC pre-layer FP32 post gates:
maximum absolute differences are `8.94e-8` (22-token tail) and `5.96e-8`
(43-token tail). The iHC launch policy depends on physical M:

| M | Token block | Split-K |
| ---: | ---: | ---: |
| 22 | 32 | 64 |
| 43 | 32 | 32 |
| 128 | 64 | 128 |
| 1024 | 64 | 16 |

A separate all-eight-rank probe holds effective input rows and actual layer
weights fixed while changing M from 22/43 to 128. This local probe tiles the
three recorded effective rows to construct the two input sizes, with the entire
effective prefix identical across sizes. It reproduces these iHC post-gate
differences; the reduced BF16 iHC outputs remain exact in those local probes.
The same-input layer-0 attention `linear_gate` projection also changes for the
22-token probe: its BF16 output maximum difference reaches `0.0078125` across
ranks. That projection is excluded from FP8 quantization in the checkpoint.
For the 43-token probe its output remains exact. Thus a single iHC operator is
not asserted to explain every subsequent difference.

A same-input shape probe of layer 0's dense MLP and layer 1's MoE returns exact
valid-row outputs for M=22 versus M=128 on all eight ranks, including equal
layer-1 router logits and selected expert sets. This excludes a direct
shape-dependent error in those particular MLP/MoE probes; it does not prove
all layers/operators invariant for every input.

In the actual model forward, the early differences grow across the remaining
layers. On the three sampled effective rows of item 119's last chunk, relative
L2 error is approximately `3.0e-6` after layer 0, `0.0049` after layer 1, `0.063`
after layer 40 and `0.44` after final normalization. Item 94 reaches `0.31` after
final normalization. These are sampled hidden-state differences, not logits or
full-tensor error bounds. BF16 rounding, dynamic FP8 quantization and routed
layers provide mechanisms for downstream amplification; the contribution of
each is not separately apportioned by this experiment.

The demonstrated trigger is therefore the padded computation shape, which
changes numerical execution inside the language model. The graph-replay
ablation does not require a replay error to reproduce these regressions.

## Precision guard

`GPUModelRunner._determine_batch_execution_and_padding` checks the descriptor
returned by the existing dispatcher. For HYV4VLForConditionalGeneration with
DP=1 on the v1 runner, a PIECEWISE descriptor whose token count exceeds its requested count is
replaced by NONE with the requested count. Exact descriptors, FULL decode,
other architectures and the existing DP coordination path retain their
previous dispatch behavior. The guard does not undo padding independently
required by sequence/context parallelism.

For the tested workload this keeps 1024-token chunks graphed and runs short
prefill tails at their original M. It preserves the reference's computation
shapes rather than changing all kernels to a new numerical reference. This is
a conservative precision fix; invariant-kernel implementations could later
recover padded-graph coverage if they pass the same paired checks. Another
candidate is capturing frequent exact tail lengths such as 22/43 directly,
with startup/memory costs and a new complete precision pair measured before
use. Those extra capture sizes are not installed by this fix.

The source guard (`9245a63`) is installed into the already loaded diagnostic process by
compiling its exact method from the modified source file. Decode-only controls
then use that same method and process. This avoids changing weights or graph
allocations midway through the comparison. The artifact manifest records the
measured source SHA-256. The evaluators started before the source-only commit,
so their `plugin_revision` field records base `ba44dc2` plus that modified working
tree; the recorded runner hash matches the committed guard at `9245a63`.
A fresh-process image smoke is checked separately.

## Complete paired validation

Guarded full/piecewise and the same-process decode-only control both score **561/765 (73.33%)**, with strict exact-answer scores 546 and 546. **0 original predictions differ**. HTTP failures, empty answers and truncations are zero in both runs.

The guard result also matches all 765 original predictions from the previous
independent decode-only process. Equal original answers establish the tested
request-level precision gate; full logits and every intermediate value are not
claimed bitwise equal. The complete test uses original questions, temperature 0,
seed 0, no_think, max output 16 and concurrency 1.

Actual wrapper dispatch on each of the eight ranks:

| Mode | Actual counts per rank |
| --- | --- |
| Guarded full/piecewise | NONE=764, FULL=1553, PIECEWISE=703 |
| Decode-only control | NONE=1467, FULL=1553 |

These counters are collected at the model wrapper after the source guard.
Dispatcher-attempt logs also retain the padded proposals that the guard rejects;
they are not treated as executed PIECEWISE graphs.

## Performance after the precision guard

The same fixed 32-image, two-round streaming client is timed after both full
accuracy runs, first guarded full/piecewise and then decode-only. Both modes reset
multimodal caches before the sequence; the default GPU encoder LRU can then
reuse repeated images. Prefix and CPU processor caches remain disabled. There
is no concurrent GPU workload or configured profiler.

| Suite | Requests | Decode-only → guarded p50, ms | Throughput gain | Changed raw answers |
| --- | ---: | ---: | ---: | ---: |
| RealWorldQA c=1 | 64 | 821.6 → 816.3 | +0.7% | 0 |
| RealWorldQA c=2 | 64 | 1429.7 → 1420.9 | +0.7% | 0 |
| Single image, 32 tokens | 4 | 2573.2 → 2562.6 | +0.3% | 0 |
| Two images, 32 tokens | 4 | 3332.2 → 3329.6 | -0.1% | 2 |

All measured throughput changes versus decode-only are below 1%; this round
establishes no meaningful additional speedup. The original 13% QA gain is not
retained after protecting the padded tails.

**Two-image 32-token captions still change 2/4 raw answers versus decode-only.**
This guard fixes the complete single-image QA regression, but does not establish
unrestricted graph precision. No reference-scored caption accuracy is claimed.
Both changed cases have 2039 prompt tokens; retokenizing the returned text puts
the first differences around context positions 2049 and 2065. The checkpoint's
sparse indexer top-k is 2048. This is a lead for a separate long-context
investigation, not proof that the threshold causes the differences. The
raw responses and prefix analysis remain preserved.

Caption suites contain four requests each and are indicative. The original
unguarded speedups are not substituted for this guarded result. The additional
50% end-to-end target is not established by this fix. FULL_AND_PIECEWISE remains
an explicit option, with eager as the documented fallback. Concurrency-1 QA
invariance does not by itself establish concurrency-2 or caption invariance.


## Fresh-process checks and unit validation

A new normal server uses the committed guard with `VLLM_USE_V2_MODEL_RUNNER=0`,
FULL_AND_PIECEWISE and the same public serving settings. It has no diagnostic
worker extension or developer RPC mode. All seven constrained smoke checks (including the text-only control) pass;
the real-photo response describes the cherry blossoms and tower, with its raw
response retained for manual review. All 15 selected questions match the
reference answers exactly. Both diagnostic and fresh servers are stopped after
validation.

Ten focused runner/dispatch tests pass, including the new fallback, exact-size
PIECEWISE retention, FULL decode retention, unrelated architectures and existing
DP coordination behavior. The new test file passes Ruff; the runner passes the
focused syntax/undefined-name checks, and the full diff passes whitespace checks.

## Reproduction and artifacts

Use [the server and client commands](hy4_vl_serving.md) at the PR revision
containing the guard. `HY4_EXECUTION=full-piecewise` runs the guarded mode;
`HY4_EXECUTION=decode` runs the reference. The original unguarded experiment is
reproducible at `ba44dc2` with the historical scripts and settings.

The diagnostic scripts and compact results are in
[`benchmarks/hyv4_vl_precision/`](benchmarks/hyv4_vl_precision/). Raw sampled
snapshot tensors remain under `/root/hy4-perf-stage4/` on the test host; their
hashes and filenames are recorded so reviewers can distinguish retained raw
evidence from the smaller JSON metrics committed to the PR. Images and weights
are not copied into the PR. Dataset/scorer attribution and revisions are in the
[RealWorldQA protocol](hy4_realworldqa_validation.md).
