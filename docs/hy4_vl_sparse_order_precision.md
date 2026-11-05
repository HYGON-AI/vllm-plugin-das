# HYV4VL sparse-index order precision fix

The long-context variation reported in the
[ready verification](hy4_vl_ready_validation.md) comes from an unstable sparse
reduction order in the reproduced cases. Native Top-K can return the same
selected tokens in different orders. BF16 sparse attention rounds differently
for those orders, and the language model amplifies the difference. The fix in
`47bbdcd` orders active HYV4-VL index rows by logical token position before
attention, retaining native scores, selected tokens, invalid-slot padding and
the shared output buffer.

## Controlled reproduction

The baseline implementation is `7b5b973`, using the same checkpoint, TP8 BW1101,
BF16 model/KV cache, FP8 linear/MoE weights, image resolution and serving limits
as the preceding reports. The indexer's own K cache remains its normal packed
FP8 format. This distinction matters: the observed attention rounding is in
the BF16 sparse MLA consumer, not a new FP8 KV-cache experiment.

Four fixed pairs of RealWorldQA images receive the same 32-token caption
requests, temperature 0, seed 0 and no_think. Each intervention runs all four
pairs twice in eager mode. The installed selector wrapper is replaced on every
RPC invocation, and all eight ranks report its version and selector. Thus the
canonical intervention here actually executes; it is separate from the
explicitly excluded source-only edit in the previous report.

| Decode selector intervention | Pairs with identical repeated text | Identical complete repeated logits records for traced pair |
| --- | ---: | ---: |
| Native selector | 1/4 | 11/33 |
| Native selector + ascending logical index order | 4/4 | 33/33 |
| Stable descending-score PyTorch argsort | 4/4 | 33/33 |

The traced pair is dataset rows 478 + 618, with 2039 prompt tokens. The first
changed native logits are at record 11, corresponding to context length 2049.
The shorter control pair 422 + 204 has 1585 prompt tokens. These captions are
reproducibility probes, not the original reference-scored QA task.

At context length 2049, all eight ranks record every producer call's complete
score/index tensors. The 78-layer checkpoint has full indexers at layers
0, 1, 5, 9, ..., 77, with shared consumers between producers. Comparing two
native repetitions:

1. Layer 0 input scores and selected sets match exactly; their index order
   differs.
2. Layer 1 indexer input scores differ.
3. Layer 5 selected token sets first differ.

With canonical logical order, all 21 producer calls have identical valid
scores, index sets and index order across repetitions on all eight ranks.
This resolves the observed sequence of first divergence; it does not claim
that arbitrary native ties always select the same membership for every input.

The stable-score argsort intervention is supporting evidence only. It produces
a different BF16 reduction order and is not installed in the PR. The fix
retains the native Top-K membership rather than choosing a new score/tie policy.

## Fixed-input attention replay

A separate operator probe holds one captured query, its selected K/V vectors,
the attention sink and scale fixed. It replays the same BF16 sparse attention
ten times for each of two index orders. Each fixed order repeats bitwise
exactly. Reversing only the indices changes 160 of 4096 output elements, with
maximum absolute difference `0.001953125`.

The CPU float64 reference computes
`softmax(concat(Q K^T * scale, sink))[:, :num_keys] @ V`, with V taken from
the first 512 KV components. The two BF16 orders have relative L2 errors
`0.0016361` and `0.0016339`, and cosine similarity about `0.99999869` to that
reference. These are rounding differences within the existing attention
implementation, not evidence that one permutation changes the mathematical
attention set. The full-model experiment demonstrates their practical
amplification. This operator reference covers the captured rank-0 input,
not every head/layer/input of the complete checkpoint.

## Implementation

`Indexer` enables canonicalization for the HCU
`HYV4VLForConditionalGeneration` architecture. After the native indexer fills
its shared output buffer, `canonicalize_topk_indices` processes only the
active `hidden_states.shape[0]` rows in place. The backing buffer can have
1024 rows during one-token decode; processing its inactive tail would add
unnecessary work and change unrelated storage.

The Triton kernel sorts nonnegative logical positions ascending and keeps -1
slots last. It first checks for inversions on device: already ordered rows,
including native short-context rows, need no sorting or writeback. Tensor
values never synchronize to the host. Native score computation, candidate
membership, attention implementation, quantization, checkpoint weights,
image processing and sampling parameters are retained. Shared indexer
consumers see the same canonical producer buffer.

The source method installed into the already loaded eager diagnostic process
matches all four repeated captions and all 33 traced logits records from the
canonical-order reference. The fresh-process checks below separately validate
normal source loading and newly captured graph execution.

## Validation

Local regression checks pass: **287 tests**, covering vision/M-RoPE,
registration, runtime configuration, weight loading, HYV4 attention and the
focused runner contracts. New tests check selected-token multiplicity,
sentinel placement, strided buffers, inactive rows, idempotence, active-row
integration and graph replay with changed inputs.

A fresh process loads `47bbdcd` normally and captures new FULL/PIECEWISE
graphs. All 21 indexers on each of the eight ranks report canonicalization
enabled, with recorded source hashes. The complete original RealWorldQA split
scores **561/765 (73.33%)**, strict exact-answer count **546**. All **765 raw
answers match** the preceding independent decode-only control. HTTP errors,
empty answers and truncations are zero. Original questions, pinned images and
the unmodified pinned scorer are retained; concurrency is 1, output limit 16.

Actual model-wrapper dispatch on every rank is NONE=764, FULL=1553 and
PIECEWISE=703. These are execution counts after the padding guard, not rejected
dispatcher proposals.

The fresh four-mode repeated double-image caption suite completes all 32
requests with identical answers for each pair, across both repetitions and
all modes. All 33 complete logits records of the traced pair match the eager
reference across modes and repetitions. This validates new FULL graph replay
in addition to the earlier eager source installation.

The 128-token suite adds 12 requests: three long pairs, eager and guarded
full/piecewise, each repeated twice. Answers are identical per pair, and all
129 full-logit records match in each of the four traced runs. Traced active
indices are canonical on all eight ranks in both caption suites.

A separate diagnostic uses 16 fixed image pairs and the original question
about the first image, with ignored text filler to reach 3072–3076 prompt
tokens. These modified prompts are **not the official RealWorldQA score**.
Across two repetitions in native eager, fixed eager and fixed full/piecewise
(96 requests), each mode scores 12/16 per repetition. Native eager changes one
raw answer between repetitions; both fixed modes have zero repeat differences,
and all 32 fixed graph/eager comparisons are identical.

Seven constrained HTTP smoke checks pass (text, color, counting, OCR and image
order/repeat). An eighth real photograph response is retained for manual
inspection and has no scored correctness label.

## End-to-end performance

Both modes use the fix, the same loaded weights and graph allocations, and the
same 32 fixed-seed images over two rounds. Processor/prefix caches are disabled;
the default encoder cache can reuse images and is reset before each mode.
No profiler or competing inference process runs during measurement.

| Workload | Requests per mode | Eager p50 ms | Full/piecewise p50 ms | Throughput improvement | Changed raw answers |
| --- | ---: | ---: | ---: | ---: | ---: |
| QA, concurrency 1 | 64 | 1129.1 | 812.4 | 43.2% | 0 |
| QA, concurrency 2 | 64 | 1597.2 | 1417.2 | 16.7% | 0 |
| One-image 32-token caption | 4 | 7720.4 | 2552.4 | 217.9% | 0 |
| Two-image 32-token caption | 4 | 8119.7 | 3335.0 | 149.4% | 0 |

Caption timings have only four requests per mode. These gains compare fixed
graph execution with fixed eager execution; they do not establish a further
50% gain over the preceding optimized graph implementation or isolate the
canonicalization kernel's cost.

## Reproduction and scope

Use the [server/client commands](hy4_vl_serving.md) from the current checkout.
Restart the service and capture new graphs after upgrading: existing captured
graphs do not acquire a Python source change. This fix is automatic for the
HYV4-VL architecture on HCU; no additional serving flag is required.

The validated configuration is v1 runner, TP8/DP1, BW1101, this checkpoint,
4096-token context, two images at most, and the recorded software stack. The
fresh verification process loads committed source normally, with the Stage4
worker extension and development RPC enabled for counters/mode controls and
traces. HTTP inference is exercised throughout. No source hot-patch is used in
this fresh process. Other configurations have not been established by this run.

[Archived evidence](benchmarks/hyv4_vl_sparse_order/README.md) includes full
predictions, comparison summaries, executed scripts, source flags, test logs
and manifests for host-retained tensor traces. All release gates pass; the
previous draft blocker is resolved for these reproduced cases.
