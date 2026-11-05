# HYV4VL ready-for-review verification

**Historical result at `125d5e5`; superseded by the
[sparse-index order fix and passing fresh verification](hy4_vl_sparse_order_precision.md)
at `47bbdcd`.** The evidence below records the original blocker.

**Original result: keep PR #338 in draft.** Verification of head `125d5e5` found a
remaining long-context reproducibility problem. The earlier prefill-padding
guard passes its single-image QA gate, but it does not resolve this issue.
This round changes no inference implementation or serving configuration.

## Repeated real-image requests

The same TP8 checkpoint and hardware/settings as the
[padding investigation](hy4_vl_piecewise_precision.md) are used. A fresh
diagnostic process loads the committed guard normally. Four pairs of original
RealWorldQA images are captioned with temperature 0, seed 0, no_think and exactly
32 output tokens (`min_tokens=max_tokens=32`, `ignore_eos=True`). These are
caption reproducibility probes, not reference-scored RealWorldQA questions.
The `length` finish reason is intentional. Prefix/processor caches are disabled;
encoder and multimodal caches are reset when changing execution mode.

Each mode processes all four pairs, then the sequence is repeated: 32 completed
requests, with no HTTP failure or empty answer. All eight ranks switch together.

| RealWorldQA row indices, zero-based | Prompt tokens | Guarded full/piecewise repeats identical | Padded-eager repeats identical | Decode-only repeats identical | Pure eager repeats identical |
| --- | ---: | --- | --- | --- | --- |
| 112 + 507 | 2039 | No | No | No | No |
| 422 + 204 | 1585 | Yes | Yes | Yes | Yes |
| 287 + 719 | 2039 | No | No | No | No |
| 478 + 618 | 2039 | No | No | No | No |

Padded-eager retains the PIECEWISE dispatch proposal but bypasses replay; the
source guard still rejects padded proposals. Pure eager dispatches NONE for
both prefill and decode. Thus enabling graph replay is not required to
reproduce the long-context variation. This does not establish that every graph
configuration is otherwise numerically invariant.

## Logits and sparse-indexer boundary

Rank 0 records full pre-sampler logits and corresponding final hidden states
for pair 478 + 618. Each request has 33 records: an ignored sampling result
after the first partial prefill, the completed prompt, and 31 decode steps.
Its prefill chunks are 1020 + 1019 tokens and execute as NONE in all four
modes. There is no PIECEWISE replay in this particular request.

Within each mode, the two repetitions have bitwise-identical logits for
records 0 through 10. The first changed record is 11, whose decode context
length is 2049. Maximum absolute logit differences at that record are:

| Mode | Maximum absolute difference |
| --- | ---: |
| Guarded full/piecewise | 2.115 |
| Padded eager | 1.974 |
| Decode-only | 1.868 |
| Pure eager | 2.018 |

The checkpoint's sparse indexer selects 2048 tokens. Up to that length, the
selector returns all valid positions in sequence order; above it, actual Top-K
selection starts. Additional eager-only ablations capture the first indexer
call at context lengths 2048 and 2049, and replace the decode selector on all
eight ranks. The first-indexer valid input scores repeat exactly in these
probes. At 2049, the native fast selector and legacy selector return the same
2048-element set across repetitions but in different orders; both correctly
exclude the minimum-score token, position 44, in this captured first call.
There are no duplicate or out-of-range indices in those captured outputs.

| Eager-only decode selector | Repetitions | Pair 478 + 618 text repeats identical | First changed logits record, repetitions 0 vs 1 |
| --- | ---: | --- | ---: |
| Native fast selector | 2 | No | 11 |
| Legacy native selector | 2 | Yes | 11 |
| Existing masked PyTorch fallback | 2 | Yes | 24 |

Equal final text does not imply equal logits. The PyTorch replacement still
diverges later. These
interventions implicate the sparse-selection/attention path but do not prove
that index ordering alone explains the full-model variation. The first-call
capture does not establish index-set equality at every layer, rank or later
token. No selector replacement is installed in the PR and no correctness or
performance claim is made for these diagnostic alternatives. Their changes
apply to eager Python execution, not previously captured FULL graphs.

An attempted canonical-index sorting ablation is **excluded**: the probe
closure had already been installed, and editing its source did not refresh
that closure. The three requests therefore used the original selector. Their
responses/tensor hashes are retained with an explicit invalid-intervention
note; no result is claimed for canonical sorting.

## Other verification and review status

At the tested head, 193 model tests pass across M-RoPE, vision, registration,
runtime configuration and weight loading. Ten focused runner tests also pass.
The earlier complete guarded/decode-only single-image QA pair remains
561/765 for both, with all 765 raw answers identical. That complete evaluation
is not rerun in this round; it is not a long-context caption quality gate.

The GitHub checks inspected before this documentation update have four
completed quality/security/compliance checks passing; the self-hosted HCU
static/test-selection check is queued. Local passing tests do not substitute
for completion of that queued check.

The diagnostic server is stopped before the model tests. A subsequent
environment-information RPC had a serialization error because it returned a
`TorchVersion` object instead of a string; it occurred after all 41 inference
requests and is recorded separately from their results. No workload result
comes from that failed RPC.

Ready conversion remains blocked by the unresolved long-context behavior.
Resolve and validate it with matched long-context reference checks and
repeated eager/graph runs before promoting unrestricted image serving. Eager
alone is not a demonstrated fix for contexts above 2048. No additional 50%
performance gain is established.

## Evidence and reproduction

Compact evidence is in
[`benchmarks/hyv4_vl_ready_validation/`](benchmarks/hyv4_vl_ready_validation/):
all raw response text, per-step comparison metrics, dispatch-attempt counts,
test logs and diagnostic scripts. `verification-summary.json` contains the
within-mode full-logit comparisons and first-indexer score/index checks.
`external-tensor-manifest.json` hashes the 26 retained tensor files under
`/root/hy4-pr-ready-validation/`; tensors, images and model weights are not
distributed with the PR. `SHA256SUMS` covers the committed evidence.

The scripts preserve the actual test-host paths. For reproduction, check out
the tested head, prepare the pinned dataset from the
[serving guide](hy4_vl_serving.md), and copy these scripts to
`/root/hy4-pr-ready-validation/`. Copy
`benchmarks/hyv4_vl_precision/scripts/stage4_worker.py` to
`/root/hy4-perf-stage4/`. Adjust checkout/checkpoint/dataset paths if necessary.
Start `serve.sh`, then execute `run.py` after readiness. The script installs
the tracing hooks and performs the four-mode repeated suite. With that same
diagnostic server still loaded, `topk_probe.py` performs the eager-only
ablations. `canonical_probe.py` records the excluded attempted intervention;
it is not a validated sorting experiment. `analyze.py` and `verification_report.py` generate
comparison metrics from the retained tensor files. These diagnostic commands
use developer RPCs and are separate from normal serving commands.
