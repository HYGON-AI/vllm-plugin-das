# Native packed integration validation

Latest review (2026-11-21): the experimental fused-padding candidate improves
packed serving throughput but still fails performance, full-model output
equivalence and clean-lifecycle gates. PR367 remains Draft; see the
[fused follow-up](#fused-padding-preparation-follow-up). Earlier results below
retain their original dependencies and scope.

The native packed candidate was merged normally with current v0.33.0
`2cdc04242cee17fb24f7e758e62cb5c5ea0a3222`, producing runtime source
`01dee6db8abfe689d0aa796de2edfd2361bdef07`. The integration preserves the
0.33 package version, current Adapter minimum, AITER entry points, root-build
guard and concurrent upstream additions. It does not reintroduce the legacy
platform plugin. Unrelated DeepSeek V4.1 testing was deferred by the user.

The normally resolved Plugin wheel has SHA256
`2ae2e1a21b984502921743e235106b8dad4290bdbaeb49d6ace70c71cec935ee`.
Core `ae5ebff332de0bd467f285e1df0127b5e7d6b3ac` HIP modules were rebuilt;
the installed wheel, Python sources and parent/child origins were checked.
Portable target-source contracts passed 622 tests with 6 HCU cases deselected;
the final installed packed CPU contracts passed 64 tests. The source-overlay
and fully installed checks are distinct evidence scopes.

On gfx936, the four existing TP1 native packed tests passed for BF16/FP16 and
padding/no-padding, with two official raw reload generations and two changed
input/routing states each: 16 replay states. The tests use four experts,
hidden size 256, independent signed INT4 CPU references, stable raw/scale/output
addresses and wrong-nibble/signedness negative controls. These are component
results, not TP2/TP4 or full-checkpoint/model quality/performance acceptance.

The retained `resume-20261114/packed-device-summary.json` indexes the native
receipts. Interpreter teardown still reports the installed BoltOps
duplicate-qualified schema registration/finalizer error. An isolated vendor
source reproduction and candidate correction passed on CPU; the installed
vendor package was left unchanged. Therefore clean native lifecycle is not
accepted, despite the numerical assertions passing and VRAM returning to its
baseline. This vendor issue must not be hidden by ignoring teardown errors.

Current project image/device CI remains a separate gate: the configured
`HARBOR_SITE` and approved immutable `HCU_CI_IMAGE_V033` must resolve before
device tests can run. The local component results do not waive that gate.
Full packed checkpoint HumanEval 0-31, required TP2/TP4 coverage and clean
shutdown remain open for this combined artifact.

## Current-main component follow-up, 2026-11-21

The separate validation tree normally merges PR367 head `a4af6441389c` with
Plugin v0.33.0 `32a1a6c05d6d`, without conflicts, at `4222b3d94640`.
It uses Core v0.33.0 `c440c089fe96` and Adapter main `9c20b33d8ebf`.
Adapter and Plugin Python wheels were built into two new physical prefixes;
the compatible native dependencies were reused, not rebuilt. This is a
development pairing, not a newly published release image. FlashAttention
was neither modified nor rebuilt, and no DeepSeek V4.1 checkpoint was tested.

Both prefixes pass full owned RECORD/wheel/source checks; shared Torch,
Triton and LightOp are verified at their physical owners. The installed
BoltOps distribution passes its complete 912-file RECORD check. The current
baseline passes 34 selected SlimQuant CPU cases. The candidate passes the
same 34 plus 64 packed contracts, with zero failures/errors/skips; ten
hardware cases are deliberately deselected in each CPU run. Current CI
selection tests also pass: 15 tests and 14 subtests.

On an actual gfx938/BW1100 device, all four unchanged packed native cases
pass with the current installed candidate: BF16/FP16, padding/no-padding,
two official reload generations and 16 changed-input Graph replay states.
The independent signed INT4 reference and stable-pointer assertions remain
unchanged. This component process exits without weakref, nanobind or traceback
diagnostics. The first toolchain preflight selected the Ninja console script
instead of its ELF and was refused before any device process started; the
separate corrected run uses the shipped `ninja/data/bin/ninja`, verified against
its RECORD. No installed Ninja file was changed.

These component results do not qualify a new full-model score, serving
performance, remote project device CI or full-model lifecycle. The completed
historical TP2/TP4 expanded-versus-packed HE32 pairs retain their original
dependencies and raw/normalized scores; they are not rerun to improve a score
or transferred to this new combination. Full-model lifecycle failures and the
earlier TP8 timeout remain recorded.

### Paired serving performance and shutdown

The same sealed `Qwen3.8-Flash-Next-Channel-INT4-w4a8` checkpoint was run on
the same gfx938/BW1100 devices 0/1, TP2 and BF16. Both arms use the official
default FULL_AND_PIECEWISE Graph policy, prefix caching ON, context 4096,
max sequences 16, max batched tokens 1024, fixed KV allocation 1 GiB/rank,
and device PLE (`engram_config.cpu_offload=false`). Vendor configuration
and installed native packages are unchanged. This shared node was not
exclusively reserved.

The official `vllm bench serve` random workload uses 32 requests, concurrency
4, 256 input tokens and 128 forced output tokens, temperature 0 and
ignore-EOS. After warmup, seeds 0/1/2 provide three measured repetitions.
All 192 measured requests succeed, each arm producing 4096 output tokens
per repetition. Every final server-tokenizer-aligned model-input field
matches between arms. The official client's random per-run request IDs are
retained separately; complete request receipts are not byte-identical.

| Measurement | Expanded INT8 | Packed BoltOps |
| --- | ---: | ---: |
| Output token/s, repetition 1 | 151.731 | 88.307 |
| Output token/s, repetition 2 | 151.517 | 87.958 |
| Output token/s, repetition 3 | 151.681 | 86.655 |
| Median output token/s | 151.681 | 87.958 |
| Median mean TTFT, ms | 338.649 | 382.525 |
| Median mean TPOT, ms | 23.901 | 42.800 |
| Median mean end-to-end latency, ms | 3373.949 | 5818.064 |

**Packed throughput regresses 42.01%; mean TPOT increases 79.07%. This is
a failed performance gate, and PR367 remains Draft.** This synthetic
workload establishes neither accuracy nor a general SLA. Prefix hits are
zero in both arms; input length 256 is below the observed cache block 1568,
so this is not a prefix-hit performance qualification.

Independent post-measurement profiler sessions preserve both TP-rank traces.
For rank 0 the expanded/packed traces contain 14485/19573 kernel events.
Packed adds route-preparation sort/scan/copy work and uses distinct packed
GEMM kernels. The vendor warns that a dedicated gfx938 INT4 configuration
for this geometry is absent and selects its default. Neither observation
alone establishes the dominant cause of the serving regression; profiler
times and Graph annotations are diagnostic, not additional timed benchmark
repetitions.

Both server frontends exit 0 after one controller SIGINT, within the original
15-second cleanup deadline. Both executors nevertheless escalate to worker
SIGTERM after their internal grace period, and both servers and all benchmark
clients report nanobind leaks. The candidate additionally reports
`AsyncLLM output_handler failed` / `EngineDeadError` during shutdown, after
all requests completed. Profiler teardown is present in these server runs;
this does not establish a packed-only or unprofiled-production root cause.
Neither arm passes clean lifecycle. No external forced kill was needed;
owned process trees are gone, ports are bindable and VRAM returns to baseline.

Double-read checkpoint seals match before/after (144 assets,
127135690592 bytes). Owned RECORD/source/wheel and shared dependency checks
remain equal before/after; both arms have identical Core/Adapter/AITER/
sgl-kernel payloads, and BoltOps' complete 912-file RECORD remains unchanged.
All-rank inspection confirms 48 routed layers per rank, stable storage and
the expected expanded versus packed dimensions. No HumanEval requests were
generated. Local evidence: `evidence/pr367-current-performance-20261121/`
(`performance-audit.json`, original results, requests, traces and exit logs).

### Fused padding preparation follow-up

Production tree `b53f900a553cf3165b6f377ece81592dbcbdf48b` replaces packed
padding preparation with one HCU launch for contiguous hidden rows of width
1–8192 and top-k 1–32. CPU, larger geometry and other layouts retain the
original Torch implementation. Active route IDs/weights are preserved; each
padding slot receives a distinct unused expert, zero weight, and wholly
padded NaN/Inf hidden rows become zero. The derived-weight, reload, admission,
Core routing/shared-expert/reduction and vendor GEMM owners are unchanged.

A new physical installed development prefix uses the same Core c440 and
Adapter main9c20 with the new Plugin Python wheel. Full owned RECORD/wheel/
source and shared dependency checks pass. The default expanded provider is
unchanged. No FlashAttention or vendor source/package rebuild is included;
ordinary private-cache AITER/Triton runtime JIT is recorded separately.

Installed CPU contracts pass 98 cases, zero failures/errors/skips, with ten
HCU cases intentionally deselected. Official source CI passes eight cases,
including packaging boundaries; selector contracts pass 15 tests plus 14
subtests. On real gfx938/BW1100, all 26 HCU cases pass: the four original
full-MoE numerical/reload/Graph test bodies are unchanged and rerun, plus
22 preparation cases for BF16/FP16, non-power-of-two top-k, strided route
metadata, empty batches, bounded/fallback shapes and independent CPU route
oracles. Changed-input Graph replay alternates padding/NaN rows and live
rows while retaining output pointers. The component process exits without
nanobind, weakref or traceback diagnostics. These are component results;
full serving performance and lifecycle are assessed separately below.

The follow-up retains the same benchmark protocol and sealed checkpoint. The
three earlier expanded-control and old-packed rounds are reused with their
original receipts, not regenerated. The new candidate runs its own warmup and
three repetitions; all 96 measured requests succeed with 4096 output tokens
per round. All aligned model-input fields match the controls; opaque request
IDs remain separate.

| Measurement | Expanded control | Old packed | Fused-padding packed |
| --- | ---: | ---: | ---: |
| Median output token/s | 151.681 | 87.958 | 103.756 |
| Median mean TTFT, ms | 338.649 | 382.525 | 359.885 |
| Median mean TPOT, ms | 23.901 | 42.800 | 36.002 |
| Median mean E2E, ms | 3373.949 | 5818.064 | 4932.158 |

New candidate output token/s by round: 103.756413, 103.289037, 103.830901.

Packed throughput improves 17.96% relative to old packed, but remains
31.60% below expanded; mean TPOT remains
50.63% higher. **Performance acceptance remains false.**
Independent traces show the actual fused preparation on both ranks (96 visible
calls/rank in this observation), with 14773 kernel events/rank versus 19573
for old packed and 14485 for expanded. These profiler totals are diagnostic
and do not measure the timed concurrent workload's critical path.

**A separate exact-output diagnostic fails:** decoded text matches old packed
32/32 in round1, 32/32 in round2 and 0/32 in round3, despite equal model inputs.
The original strict audit script and failure are retained. A separate auditor
completes provenance/performance checks while explicitly preserving the failed
output gate and false functional acceptance; it does not convert this finding
into a pass. These are synthetic decoded-text comparisons, not token-ID/logit
comparisons or a new HumanEval score. No request was regenerated for a better
result. Baseline-versus-packed texts already differ under the original pair;
that does not explain or waive this new within-packed observation.

A read-only preparation probe covers all1024 candidate-occupancy patterns for
top-k10 at hidden2560, BF16/FP16, low/high expert IDs and reversed slots. Hidden
and route-weight payloads match exactly. Padding-ID ordering differs in968/1024
low-expert and1012/1024 high-expert rows per dtype. Distinct zero-weight padding
IDs are mathematically valid, but actual vendor invariance is not established;
this is a possible mechanism, not a proven cause of generation divergence.
Do not mark this experimental candidate Ready without resolving the finding.

The profiled server exits0 after one controller SIGINT within15seconds; no
external forced kill is required, owned processes are gone, ports are bindable
and VRAM returns to baseline. Its executor still sends internal worker SIGTERM,
and the server and every benchmark client report nanobind leaks. No EngineDead
message is observed in this follow-up, but the prior occurrence remains open;
this is not a lifecycle fix. **Clean lifecycle remains false.**

Checkpoint pre/post double-read seals match all144 assets/127135690592bytes.
Owned/shared attestations, identical common Core/Adapter/AITER/sgl-kernel payloads,
all-rank storage/pointers and all912 BoltOps RECORD files pass verification.
This does not waive the failed functional, performance or lifecycle gates.
Evidence: `evidence/pr367-fused-padding-20261121/`, including
`performance-audit.json`, `synthetic-text-divergence.json`,
`prepare-exhaustive-comparison.json`, `trace-comparison.json` and the original
strict audit failure. Submitted-head project CI, original topology/PLE/update
and release-quality qualification remain separate. PR367 stays Draft.

### Output-divergence component diagnosis

The next diagnostic uses the unchanged installed production tree `b53f900`,
Core `c440c089` and Adapter `9c20b33d` on a real gfx938 device. The public
`boltops.fused_moe.fused_experts` receives **the same** packed weights and
FP32 kernel scales through the original Torch preparation and fused preparation.
Geometry matches the measured checkpoint's TP2 experts:
E512/H2560/I320/top-k10. This is a vendor component diagnostic, without
checkpoint loading, TP collectives or model-generation acceptance.

| Diagnostic | Cases | Dtypes | Batch sizes | Result |
| --- | ---: | --- | --- | --- |
| Decode geometry | 36 | BF16/FP16 | 1/2/4/8/16/32 | All eight comparisons/case exact and finite |
| Larger batches and CPU oracle | 24 | BF16/FP16 | 64/128/256/1024 | All eight comparisons/case exact and finite; independent INT4 oracle passes |

The eight comparisons cover old/old and new/new eager repeats, old/new eager,
old/new Graph, each Graph's repeat, and eager/Graph for each arm. Replay changes
hidden values, active IDs and padding. Mixed/all-padding rows include NaN/Inf
hidden values and NaN padding weights. The 1024-row low-ID input covers every
candidate-occupancy bit pattern. Padding-ID order differs in 40/60 cases, yet
all 480 output comparisons are exact. The 24 independently checked nonzero
valid rows pass the signed-INT4 CPU reference at rtol 0.015 / atol 0.001; maximum
absolute discrepancy is 0.00048828125. These references are sampled rows,
not an independent oracle for every output element in the matrix.

All packed weight/scale hashes remain unchanged. Both diagnostic processes
exit 0 without nanobind/weakref/traceback diagnostics, owned descendants are
gone and GPU7 returns to 2 MiB. Native JIT compilation uses private caches;
owned vendor payloads stay unchanged. FlashAttention is unchanged and is not
rebuilt.

This matrix does **not** reproduce an output change caused by padding-ID
permutation. It narrows that hypothesis for the tested weights/shapes; it does
not prove full-model invariance or explain the original third-round divergence across all 32
texts. The original strict output, serving-performance and model
lifecycle gates stay failed. Do not copy an unspecified Torch argsort tie
order into production to conceal the unresolved observation.

Evidence: `evidence/pr367-output-diagnosis-20261121/`, including both
`vendor-r*/results.json`, launch/process/exit receipts and
`production-binding.json`. Existing CPU98/HCU26/source-CI8/selector15+14
subtest results retain their original source and scope; this diagnostic adds
no HumanEval generation and does not qualify submitted-head project CI.

### Short model token/logprob diagnosis

The same installed old/fused packed pair is then exercised on the original
failing round's first four random prompts. Each arm receives one single-prompt
request and one explicitly grouped list of four prompts, with temperature 0,
ignore_eos, eight output tokens, top 5 logprobs and token-ID reporting. The
actual serialized HTTP bodies match across arms and raw responses are kept.
There are 10 new short diagnostic completions; the 96-request benchmark and
HumanEval cohorts are not regenerated. The diagnostics use TP2/BF16, device
PLE and the original configured FULL_AND_PIECEWISE Graph/prefix policies.
They omit the original separate profiler session and measure no serving
performance. Actual per-step scheduler grouping is not instrumented.

| Comparison | Matching eight-token sequences | First divergent token (zero based) |
| --- | ---: | --- |
| Old/fused single | 1/1 | None |
| Old/fused grouped4 | 0/4 | [2, 1, 7, 1] |
| Old single/grouped4, first prompt | 0/1 | 2 |
| Fused single/grouped4, first prompt | 1/1 | None |

The old packed source itself produces a different third token between its
single and grouped requests. This establishes a full-model difference
without the fusion change; it does not isolate grouping, warmup/order,
Graph shape, state/cache behavior or a particular operator as the cause.
The short requests do not recreate the original concurrent 32-request round.
Preserve the failed original exact-output gate and investigate step inputs
and logits under controlled execution rather than changing a sort order.

The first controller exits 1 at its immediate post-old-service bare port bind
with EADDRINUSE. It had not persisted the old child exit code, which remains
**unknown**, not 0. Completed old requests/storage receipts are reused;
recovery sends only the missing five fused completions. A later bare bind
succeeds, owned processes are gone and VRAM is released; the port refusal's
cause is not established. The original controller failure is retained.
The recovery saves child shutdown receipts before port checks.
The recovered fused arm also has initial bare-bind refusals before its
bounded later bind succeeds. This is separate from the saved process-exit
deadline; the refusal does not establish a remaining listener or worker.
Server nanobind and internal termination findings remain failed gates.
**Controller, full-model lifecycle and release qualification remain false.**

Both arms' pre/post installed payload attestations pass. Checkpoint
pre/post double-read seals match all 144 assets / 127135690592 bytes. Both ranks
each retain the 48 expected packed expert methods and storage addresses. All
observed owned processes are terminal and the allocated cards are released.
See `diagnostic-audit.json`, raw `model-r1/*/*.request-body` and response
files, `controller-closeout-failure.json`, the recovery source and all-rank
storage/launch/process receipts. No production fix or new acceptance is
claimed; PR367 remains Draft.

### Official scheduler-step observation and order controls

A separate logged diagnostic keeps the same installed old/fused pair, checkpoint,
physical devices 0/1, TP2/BF16, device PLE and default Graph/prefix policies.
Only Core's official `--enable-logging-iteration-details` observation is added.
Each arm receives single-before, grouped4, single-after with prefix reset before
each phase; all completions contain eight tokens with actual token IDs and top-5
logprobs retained. The 12 new completions are diagnostic, not a benchmark or a
regeneration of HumanEval. Cross-arm request bytes and before/after single bodies
match. Raw log byte windows are independently reparsed and account for every
context token (256 per prompt).

| Comparison | Matching eight-token sequences | Aggregate step signatures equal | First divergent token (zero based) |
| --- | ---: | --- | --- |
| Old/fused single-before | 1/1 | True | [None] |
| Old/fused batch4 | 0/4 | False | [2, 1, 0, 2] |
| Old/fused single-after | 1/1 | True | [None] |
| old single-before/after | 1/1 | True | None |
| new single-before/after | 1/1 | True | None |

Per-phase signatures below are ordered
`[context requests, context tokens, generation requests, generation tokens]`:

- old single-before: `[[1, 256, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]]`.
- old batch4: `[[3, 768, 0, 0], [1, 256, 3, 3], [0, 0, 4, 4], [0, 0, 4, 4], [0, 0, 4, 4], [0, 0, 4, 4], [0, 0, 4, 4], [0, 0, 4, 4], [0, 0, 1, 1]]`.
- old single-after: `[[1, 256, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]]`.
- new single-before: `[[1, 256, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]]`.
- new batch4: `[[1, 256, 0, 0], [1, 256, 1, 1], [2, 512, 2, 2], [0, 0, 4, 4], [0, 0, 4, 4], [0, 0, 4, 4], [0, 0, 4, 4], [0, 0, 4, 4], [0, 0, 3, 3], [0, 0, 2, 2]]`.
- new single-after: `[[1, 256, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]]`.

Grouped4 context admissions are old `[3, 1]` and fused `[1, 1, 2]`.
A grouped HTTP prompt list is not a fixed execution batch. Equal aggregate
signatures also do not establish matching per-request identities, actual GPU
padding, routes or layer values. Differences under unequal steps cannot isolate
a preparation-kernel defect; matching steps alone cannot clear that defect.
The original strict-output failure remains open.

The immutable Core `c440c089` reproducibility documentation does not guarantee
default online reproducibility. Its beta `VLLM_BATCH_INVARIANT=1` is separate,
defaults off, and lists AMD gfx90a/gfx942/gfx950; this does not establish support
for gfx938 or this checkpoint. That mode is not enabled here and the original
validation criteria are not relaxed.

| Compiled-cache comparison | Exact Python multiset | JIT-body AST multiset | Cached launch-option multiset |
| --- | --- | --- | --- |
| previous_old / previous_new | False | True | False |
| logged_old / logged_new | False | True | False |
| previous_old / logged_old | False | True | False |
| previous_new / logged_new | False | True | False |

Generated files are parsed without importing or executing them. JIT-body AST
comparison excludes decorators while preserving function names, arguments,
annotations and bodies. Cached launch options exclude cache-identity/timing
metadata, whose originals remain in evidence. Caches can contain benchmark
candidates; this comparison does not identify the native binary executed or a
numerical root cause.

The old server exits 0 with controller signals `['SIGINT']` and no owned descendants. The original process-exit deadline passes: True. Clean lifecycle: False; teardown findings `{"engine_dead": true, "internal_worker_sigterm": true, "nanobind": true, "weakref": false}`. Immediate bare bind passes: False; 73 total refusals are retained before bounded release. task-port TCP/TCP6 state snapshots contain `['06']`.

The new server exits 0 with controller signals `['SIGINT']` and no owned descendants. The original process-exit deadline passes: True. Clean lifecycle: False; teardown findings `{"engine_dead": true, "internal_worker_sigterm": true, "nanobind": true, "weakref": false}`. Immediate bare bind passes: False; 83 total refusals are retained before bounded release. task-port TCP/TCP6 state snapshots contain `['06']`.

TCP state `06` means TIME_WAIT, while `0A` means LISTEN. These snapshots describe
the observed socket state and do not turn the first refusal into an immediate
bind pass. Child exit/identity receipts are saved before the port check. Earlier
controller exit 1 and its unsaved/unknown old child exit code remain historical
failures; this new controller does not retroactively repair them.

The checkpoint double-read seals still match 144 assets / 127135690592 bytes.
Both arms retain pre/post installed-payload verification and all-rank storage
checks. Production/tests/tools/CI remain the previously tested `b53f900` trees;
no Core, Adapter, vendor or FlashAttention modification/rebuild is made.
Original throughput -31.60%, strict-output and full-model lifecycle failures,
submitted-head device CI and remaining quality/topology/PLE/update gates stay
open. Model qualification remains false and PR367 stays Draft.

Evidence: `evidence/pr367-step-layout-20261122/step-layout-audit.json`,
`compiled-cache-comparison.json`, official source hashes, raw requests/responses,
log windows, package/checkpoint attestations and process/port receipts.
