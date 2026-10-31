# Optional BoltOPs Qwen GDN decode — vLLM 0.32

This candidate adds two decode providers at the imported aliases of official
`qwen_gdn_linear_attn`. It requires the RMS cache-factor change in PR #323;
the review branch is based on its commit `057bf829`. It does not enable RMS.
The provider defaults OFF and has not passed a performance acceptance gate.

## Publication-failure revision

The follow-up based on published PR325 `19db5e0` additionally requires Adapter's
public `ExtensionCommitError`, `is_fatal_extension_error(error)` and read-only
`require_runtime_healthy()` ABI. The minimal Adapter candidate is `60efffb`
on PR17; these interfaces call no model-load API and do not require PR19.
Compatibility is checked by public callable shapes, without an exact wheel pin.

Before publication, the owner validates compile-factor ownership/policy, the
oracle install callable, and requested FLA provider/alias compatibility. Missing
required FLA fails cleanly before factor or alias publication; a clean optional
preparation error retains the official baseline and later owners may run.
Preparation is not a reservation: installation rechecks current state.

Preparation may initialize the official GDN consumer, which registers classes
and Torch dispatcher ops at import. This is not a pure side-effect-free phase:
a failed consumer import explicitly signals fatal because partial official
registration may survive. A successful complete official import does not mount
Plugin aliases. The owner also tracks the entire preparation lifetime across
fork; a child inheriting preparation or publication in progress refuses direct
activation. Provider locks are reconstructed without clearing owner failure.

The first compile-factor registration is a shared-state commit boundary. Every
exception thereafter, including interrupted activation, raises the public fatal
signal. Adapter stops later owners, preserves actual reports and latches its
runtime. Plugin also keeps its own failure across direct retries and fork.
Removing the original error, changing flags or resetting reports cannot heal
this process. No generic rollback of Torch ops, classes, aliases or factors is
claimed; use a fresh process. Health is queried before acquiring the owner lock.

This follow-up changes preparation and failure control flow; its historical
GPU/model/performance results below belong to the frozen artifacts listed there.
CPU fixture mount markers are not evidence of native FLA execution. New source,
wheel, installed discovery and affected hardware results must be recorded for
the new artifacts separately before promotion.

## Ownership and behavior

Adapter retains the platform and general-plugin lifecycle. The extension
registers `boltops-fla-decode`; it neither registers another platform nor copies
models, runners, scheduling, allocation, prefill, or causal convolution.
The target has no per-function decode registration seam, so the extension checks
official/vendor signatures and alias ownership before mounting the two aliases.
Remove this narrow mount when upstream provides a provider seam or native HCU
implementation. Other owners and target ABI drift fail before alias mutation.

Supported metadata: detected gfx936/gfx938, BF16 activations, FP32 state/A_log,
BF16 or FP32 dt_bias, head dimensions 128, compatible grouped value heads,
indexed in-place GDN, and N*Hv <= 65535. Slot-pitched state views retain their
pointer, storage offset, inner packing and padding. No state normalization/copy
is added. The public vendor sigmoid entry may normalize activation tensors.
Unsupported metadata, disabled flags and batch-invariant mode call the captured
official entry. A selected vendor execution error is not caught as fallback.
Numerical results below apply to gfx936; gfx938 model acceptance is outstanding.

Set both `VLLM_HCU_USE_CUSTOM_OPS=1` and
`VLLM_HCU_USE_CUSTOM_AITER_FLA=1` before compilation. Provider source/config,
architecture, envelope and actual mount ownership enter the derived factor in
both ordinary compilation and second-level model AOT keys. Equivalent disabled
policies are equivalent. Do not change flags or uninstall while serving an
already compiled/captured model; restarting is required.

## Frozen independent artifact

Validation date: 2026-11-01. Exact versions identify evidence; runtime acceptance
uses release-line and required callable capability checks, not exact wheel pins.

| Component | Built source | Wheel SHA256 |
| --- | --- | --- |
| OpenDAS vLLM | a198423b7be0ccf9c9ac3fb01c02ac03ab145e93 | b2c8c31a554cd8c71c7f83c68d2cc9c5ae1835036336585e63404cf4c2d073ac |
| Adapter | e9746cb3580d26f501506d7bf5eb93134290a87a | 57a69a87160a5d756998140cf8588a199c90debc332a96edb0ed7a737816024b |
| This FLA candidate | 7db30bf (reviewed production code) | e9af2b50d5715438a3105e3f62d26355836343202ec5b432d307936ff61d41ce |

The isolated environment contains 13 Plugin and 197 Adapter Python files,
verified against built source, wheel bytes and installed RECORD. Later review
changes to documentation, test scratch preflight and CI registration do not
change production Python. The wheel has no SlimQuant or non-unit E5M2 changes.

## Validation results

| Evidence | Observed result | Boundary |
| --- | --- | --- |
| Portable source-phase suite | 276 passed, 20 skipped, 114 deselected | Explicit source provenance, not installed-wheel numerical proof; skips remain skips. |
| Installed-wheel FLA suite | 24 passed, 110 deselected; 48 changed-input Graph waves | Hk=Hv16/D128, BF16/FP32 bias, NULL/accepted/MTP, original alias/slot pitch/offset/padding checks. Performance excluded. |
| Direct vendor/official pairs | 66 comparisons; 27 output bitwise exact, no state bitwise exact | Worst output NRMSE 1.4002736e-4; updated-state NRMSE 3.5132264e-8. |
| Actual Qwen3.5-0.8B service | HumanEval/0–31 raw 1/32, format-normalized 15/32 | MTP3, Graph, prefix_match_unit16 and shared prefix checkpoint retained; isolated scorer, no semantic repairs. |
| Fine-prefix/MTP mechanism | 56 requests passed; accepted2648/drafted2832 | Actual manager page560, hash unit16, cold/warm output equality and a sibling fine-prefix hit; no independent semantic accuracy claim. |
| Actual worker calls | 846 BoltOPs successful sigmoid launches; 1 official warmup launch | Read-only monitoring of actual JIT completion; replay bypasses Python, so these are not per-request counts. |

The independent artifact's same-GPU1 sequence is ON initial load, OFF control,
then a fresh ON process loading the original target/draft AOT artifacts. All
original ON compile factors match; ON/OFF ordinary and AOT keys are separate.
The repeat reproduced raw1/normalized15 and all32 normalized completion strings
exactly. OFF was raw1/normalized14. This is not per-task dominance: ON gains
2/18 and loses0 relative to this OFF. Earlier integration OFF repeats changed
their0/18 pass vector at unchanged14count; do not infer numerical equivalence
from the aggregate score.

The same-wheel, same-GPU1 performance control used64 input/128 forced output
tokens, max_num_seqs8, two warmups plus five measured rounds per concurrency.
Median ON/OFF tokens/s were360.025/348.967 atc1,1095.292/1076.675 atc4, and
2009.317/1977.725 atc8: +3.16885%/+1.72914%/+1.59741%. Sequential time/order
confounding remains and no measured gain reaches5%; defaultOFF remains.
The initial OFF timing attempt possibly overlapped a read-only snapshot RPC and
is retained but excluded. Only its fresh exclusive rerun is the control; the
matched exclusive phases include per-round timestamps and no other API traffic.
Evidence: `fla-review-same-gpu-aot-performance-summary.json`.

The integration wheel also passed
124 HCU cases/240 Graph waves across five head geometries and a 35B combined
SlimQuant+FLA HumanEval32 run, but those are separate artifacts and not counted
as this independent wheel's model/config coverage.

## Reproduction and retained failures

Build `adapter_extension`, install the three audited local wheels with
`--no-deps` into a fresh environment, and clear PYTHONPATH. On HCU launches,
unset ROCR_VISIBLE_DEVICES rather than setting it empty. Create absolute,
writable TMPDIR and all compiler cache directories before starting Python.
The test fixture checks this before any Torch GPU operation.

The independent installed-wheel run uses `tests/vllm032/test_fla_provider_hcu.py`
with `-k 'hk16-hv16 and not performance'`, HCU_VLLM032_FLA_LIVE=1,
HCU_FLA_TEST_MODE=installed-wheel, HCU_FLA_WHEEL_VENV pointing to the fresh venv,
and HCU_FLA_EXPECT_PLUGIN_RECORD_SHA256 set to the built wheel's RECORD digest.
Use `VLLM_PLUGINS=hcu_adapter`, both FLA flags and the GPU selected exclusively
for this run. The audited wheel RECORD SHA256 is
`3ee2e14290ba6cac836ae83ff225ca82bdeee03a62e9a867aa40eecdbc0908f9`.
The full executable launch and raw records live under the local evidence root
`/data/models/vllm032-upgrade-20261031`, including `run-fla-review-hcu-final.sh`,
`fla-review-hcu-final.log` and `fla-review-hcu-records.jsonl`.
The final records were extracted from that run's log because earlier scratch
experiments used the same raw evidence directory; the summary records this
provenance instead of presenting appended records as a fresh run.

Two initial HCU attempts failed in native Torch random/fill before any FLA call
with `invalid device function`. The configured TMPDIR did not exist. Creating
only that directory made the same fixture/wheels pass; failed logs, the
scratch-only diagnostic and the final run remain separate. This is an
environment diagnosis, not a vendor numerical pass or a VMFault.

## Legacy regression and CI limits

The requested full `tests/runtime_patch` source suite was run against the exact
v0.32 target: **3070 passed, 348 failed, 73 skipped, 167 errors**. It exercises
the retained legacy `vllm_hcu` tree, which is not shipped in this wheel.
There is no diff to that tree or its tests between MR321 and this review branch.
One representative node from each of 39 affected files was run at MR321 on the
same target: 35 failed, 3 errored, 1 passed. The passing QSA case also passed alone
on this review branch; the full-suite failure was duplicate custom-op schema
registration. No whole-suite MR321 result or complete failure classification is
inferred from those representatives.

Examples include removed KpoolTailSpec (159 occurrences), new
InputBatch.decode_graph_eligible arguments (122), removed indexer helper (56),
old exact-v0.28 release checks, changed EPLB/DP/runner signatures and stale PLE
source hashes. These are open migration/test-isolation gaps in the full upgrade;
they are not repaired by this decode provider and not counted as passed.
Evidence: `fla-review-legacy-runtime-patch-summary.json`,
`fla-review-legacy-mr321-baseline-summary.json`, and the preserved full logs.

The new HCU test has a literal registry entry. It is explicitly disabled in the
existing v0.28.1 contract image, with its reason in the registry; running v0.32
code in that image would give false coverage. An audited v0.32 CI environment,
gfx938 validation, larger-model coverage
and performance acceptance remain required before broad release/default enable.
