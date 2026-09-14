# HYV4 v0.25.1 Clean Integration Design

## Status

- Date: 2026-09-12
- Target repository: `HYGON-AI/vllm-plugin-das`
- Target branch: `v0.25.1`
- Frozen target commit: `bc9329029ac14216b671793d49f5b0f47a3b5c7f`
- Historical source: pull request #34 at
  `c25d6f87381ba8385083a22e9cdab3a2abfba032`
- Validation model: `/models/Hy4-preview-Channel-FP8-w8a8`
- Validation vLLM:
  `0.25.1+das185.dtk2604.torch2110.2608171710.g7b108a`

## Problem

Pull request #34 contains the complete HYV4 bring-up history, but it is no
longer a reviewable integration unit. Relative to the current target it has 79
commits, 158 changed files, about 31,000 added lines, and 36 textual conflict
files. It also carries historical versions of common HCU infrastructure that
has since been merged into `v0.25.1` through independently reviewed work:
unified AITER MoE selection, SlimQuant, categorized LightOp APIs, native FP8 KV
cache, bounded MoE layout ownership, PCP+EP support, and DeepEP fixes.

Resolving the old branch as an ordinary merge would make the older copies win
in some conflict resolutions and would obscure which layer owns AITER,
SlimQuant, LightOp, DeepEP, and PCP behavior. The new change must preserve the
HYV4 capabilities while rebuilding their integration against the current
branch contracts.

## Goals

1. Create one clean pull request based on the frozen current `v0.25.1` commit.
2. Preserve the useful HYV4 capability set represented by pull request #34:
   target inference, native MTP, reasoning/tool parsers, Channel-FP8 and
   block-wise FP8, explicitly gated custom/native W4A8 formats, PCP/EP,
   PP+PCP, DeepEP/DeepGEMM, EPLB/offline maps, and Mooncake P/D integration.
3. Treat current `v0.25.1` implementations of AITER, SlimQuant, LightOp,
   native FP8 KV, PCP/EP, and DeepEP as authoritative.
4. Add only HYV4-specific behavior and the narrowest missing common extension
   points. Do not restore old generic implementations from pull request #34.
5. Validate the available Channel-FP8 model on real HCU hardware, including
   target-only, MTP, FP8 KV, representative parallel layouts, accuracy, and
   teardown.
6. Complete a full code review of the final target-to-head diff before the new
   pull request is opened.

## Non-goals

- Rewriting or patching the installed vLLM source.
- Replacing current `v0.25.1` AITER or SlimQuant policy with a HYV4-private
  route.
- Claiming hardware validation for checkpoint formats whose model files are
  unavailable.
- Claiming two-node Mooncake P/D validation from a single-node test.
- Preserving historical implementation-plan documents or failed intermediate
  approaches from pull request #34.
- Expanding support beyond the exact fail-closed topology and backend
  contracts covered by tests or runtime evidence.

## Source-of-truth hierarchy

For every conflict or duplicated behavior, ownership is resolved in this
order:

1. The installed pinned vLLM wheel defines the framework ABI.
2. Current `v0.25.1` defines common HCU behavior and plugin lifecycle.
3. Pull request #34 defines HYV4 architecture intent and model-specific
   behavior.
4. Earlier PR #34 commits and documents are diagnostic history only.

This means a historical HYV4 change is not copied merely because it exists.
It is retained only when the current target lacks equivalent behavior and a
test demonstrates the remaining requirement.

## Chosen approach

Reconstruct the feature on a clean branch rather than merging or rebasing the
old branch. Add the HYV4-owned modules first, then adapt each integration seam
to the current target API through test-driven slices. The final pull request
will contain a staged commit series, but all stages remain in one MR as
requested.

The rejected alternatives are:

- Merge and resolve: quickest mechanically, but retains unrelated history,
  broad generic rewrites, and ambiguous ownership.
- Rebase all 79 commits: preserves chronology rather than the desired final
  architecture and repeats conflicts many times.
- Multiple dependent MRs: easier to review individually, but does not meet the
  requested single replacement MR.

## Architecture and ownership

### 1. HYV4 model, configuration, and registration

The plugin owns `HYV4Config`, `HYV4ForCausalLM`, and `HYV4MTPModel` under
`vllm_hcu.models.hy_v4`. Registration is idempotent and follows the current
exact-import callback lifecycle. The plugin preserves independent
Hyper-Connections, HYV4 projection layouts, sparse MLA/indexer behavior,
learnable sinks, shared experts, pipeline-parallel interfaces, and strict
checkpoint loading.

Core configuration adapters may classify HYV4 for Model Runner V2, MLA, MTP,
head dtype, and graph behavior only at audited extension points. Each adapter
must validate the exact pinned vLLM owner and fail closed on incompatible
source.

### 2. AITER, SlimQuant, LightOp, and MoE

Current `v0.25.1` remains the sole owner of:

- AITER backend selection, tuned-config lookup, layout lifecycle, and Triton
  fallback;
- SlimQuant registration, weight methods, and W4A8 execution;
- categorized LightOp symbol resolution and environment policy;
- router scaling ownership, MoE alignment, shared-expert combination, and
  DeepEP integration.

HYV4 constructs the standard vLLM/HCU MoE abstractions and supplies only its
model attributes: sigmoid routing, correction bias, normalized top-k,
`routed_scaling_factor=2.827`, eight routed experts, one shared expert, and
the routed SwiGLU clamp. It must not copy an old AITER dispatcher or
SlimQuant facade into the model.

Custom and native HYV4 W4A8 checkpoint formats remain explicit adapters over
the current SlimQuant implementation. They are selected only by their audited
`checkpoint_format`; ordinary SlimQuant and Channel/Block FP8 behavior remains
unchanged for all other models.

### 3. Sparse MLA, indexer, and KV cache

HYV4 owns the model-specific attention projections, gate, sink, full/shared
indexer pattern, and MTP buffer sharing. Existing HCU sparse MLA, LightOp/Torch
indexer fallback, and native HIPC KV writers remain common owners.

Channel-FP8 and block-wise FP8 scale loading must follow current
compressed-tensors contracts. FP8 KV cache selection is independent of weight
quantization. Cache layout, block size, dtype, slot mappings, and writer/reader
views must be checked end to end; successful allocation alone is insufficient.

### 4. MTP and pipeline parallelism

The plugin registers HYV4's native draft model and preserves stable target/draft
buffer ownership. Current vLLM sampling, PP collectives, and scheduler rules
remain authoritative. HYV4 adapters may add only missing architecture
classification, model-specific weight loading, and the already audited PP
state needed by the model.

The standard acceptance topology is TP8 with MTP3 and default CUDA Graph.
PP2+PCP4 remains a separately constrained topology and does not imply that all
PP/PCP/MTP cross-products are supported.

### 5. PCP, EP, PP2+PCP4, and DeepEP

Current `v0.25.1` PCP+EP communication and batch ownership are reused. HYV4
adds model capability declarations and exact configuration gates without
constructing another model runner.

The retained PP+PCP topology is:

```text
PP=2, TP=1, PCP=4, DP=1, DCP=1, EP=4
VLLM_PP_LAYER_PARTITION=41,37
eager execution
```

It uses the current DeepEP high-throughput and DeepGEMM implementations.
Nearby unsupported layouts fail during configuration, before model weights are
loaded. Any broader topology is accepted only if it has a distinct test and
runtime gate.

### 6. EPLB and offline expert maps

The retained offline-map feature is expressed through current
`MixtureOfExperts`/`RoutedExperts` ownership. Generic direct-load behavior may
be added only if the current target truly lacks it and the implementation is
model-neutral. HYV4-specific fused checkpoint layouts use thin hooks over the
common validated map representation.

Static-map mode validates the map before weight loading, loads physical slots
from the requested logical experts, commits matching routing metadata, and
does not perform a post-load rearrangement. Dynamic EPLB and record mode retain
current behavior. All ranks must agree on the map fingerprint before serving.

### 7. Mooncake P/D

Mooncake support remains an integration adapter over the current connector.
HYV4 may declare the producer-only PCP behavior and model-specific ownership
needed by the historical feature, but must not fork generic connector
scheduling. Unsupported PP+PCP+P/D combinations remain fail closed.

### 8. Parsers

The HYV4 reasoning parser and tool parser are plugin-owned and registered via
the current parser registry lifecycle. Streaming must be prefix-stable across
split structural tokens, incremental JSON arguments, multiple tool calls, and
ordinary text containing `<`. Parser tests are CPU-safe and do not depend on
the model checkpoint.

## Migration structure

The implementation is divided into reviewable vertical commits:

1. HYV4 config, registry, parser, and static contracts.
2. Core target model, iHC, sparse MLA/indexer, and strict weight loading.
3. Channel/Block FP8 and current AITER/LightOp integration.
4. Native MTP and graph/PP integration.
5. PCP/EP and PP2+PCP4 fail-closed support.
6. DeepEP/DeepGEMM, EPLB/offline-map, and Mooncake adapters.
7. Explicit HYV4 W4A8 formats over current SlimQuant.
8. Final validation documentation and CI inventory.

Commit boundaries may combine adjacent items when they cannot be tested
independently, but unrelated common refactoring is not permitted.

## Test-driven implementation

Every production behavior change starts with a failing test against the clean
`v0.25.1` branch. The failure must demonstrate a missing HYV4 requirement,
not an incorrect test fixture or an ABI mismatch caused by the wrong vLLM.

For each slice:

1. Add or port the smallest current-contract regression test.
2. Run it and record the expected failure.
3. Add the minimum production change.
4. Run the focused test and neighboring target tests.
5. Review the complete uncommitted diff.
6. Commit the independently valid slice.
7. Review the committed diff again.

Tests copied from pull request #34 are rewritten when they stub old
`v0.25.1` APIs or encode a historical implementation rather than behavior.

## Isolated environment

The requested vLLM distribution is installed into a new target directory with
`--no-deps --target`, using the provided index. Validation sets
`PYTHONNOUSERSITE=1` and places the pinned vLLM target before the clean plugin
worktree on `PYTHONPATH`. Before any test or serve claim, diagnostics record:

- Python, torch, DTK, AITER, LightOp, DeepEP, and plugin versions;
- imported `vllm` and `vllm_hcu` module paths;
- target and plugin Git SHAs;
- model config and checkpoint-index hashes;
- installed wheel filename and SHA-256 when discoverable.

An import that resolves vLLM or the plugin from another checkout invalidates
the result.

## Validation matrix

Validation proceeds from cheapest to most expensive and uses proxy-free local
HTTP requests.

### Static and contract gates

- All tests changed by the full target-to-head diff.
- HYV4 config, registration, parsers, iHC, attention, MoE, MTP, weight loading,
  quantization, PCP/EP, PP, EPLB, Mooncake, and W4A8 focused suites.
- Existing AITER, SlimQuant, LightOp, sparse-indexer, FP8 KV, DeepEP, and PCP
  regression suites touched by integration seams.
- Full repository contract suite, Python compile checks, patch coverage,
  production-boundary checks, and `git diff --check`.

### Real model gates

All runs use `/models/Hy4-preview-Channel-FP8-w8a8`, the pinned vLLM wheel,
the clean plugin branch, and all eight available HCUs.

1. TP8, AITER MoE, target-only, default CUDA Graph:
   load all weights, verify the selected Channel-FP8/AITER implementations,
   serve health, short chat, and long prefill requests.
2. TP8, AITER MoE, native MTP3, default CUDA Graph:
   repeat the functional requests and require nonzero per-position drafted and
   accepted-token metrics.
3. TP8, AITER MoE, MTP3, native FP8 KV cache:
   verify cache dtype/layout, repeat a long prefix, require coherent outputs
   and nonzero later prefix-cache hits.
4. PP2+PCP4+EP4, DeepEP high-throughput, DeepGEMM, FP8 KV, eager:
   require all eight workers, both PP stages, stage-local PCP/EP groups, a
   short request, and a long partitioned prefill.
5. DP8+EP8, explicit DeepEP mode and DeepGEMM:
   run a representative service request and exercise available EPLB/offline
   map behavior without post-load expert movement.

Each service is launched as a foreground PTY process, tracked by its exact
session/process group, stopped through that session, and followed by a worker
and device-memory teardown check. Unrelated services are never killed.

### Accuracy gate

Run HumanEval first 32 numeric tasks for both the TP8 target-only baseline and
TP8 MTP3 candidate with identical settings:

- temperature 0 and sampling disabled;
- `reasoning_effort=no_think` in request `extra_body`;
- one sample per task;
- fixed maximum output length and stop semantics;
- fresh result directory for each run;
- proxy variables cleared and localhost in both `NO_PROXY` forms.

Record prediction/review counts, Accuracy, Pass@1, API/runtime errors,
per-sample finish reason, output-token count, pass/fail flips, and MTP
acceptance. A lower score remains a blocker until the output protocol,
truncation, and repeatability have been investigated.

### Explicit validation gaps

The supplied Channel-FP8 checkpoint cannot validate custom/native GPTQ W4A8
weight bytes or model quality. Those adapters require numerical kernel tests,
loader/isolation tests, and fail-closed selection tests, but remain documented
as hardware-unvalidated unless a compatible checkpoint becomes available.

Mooncake P/D receives complete configuration and connector contract tests.
Single-node tests may cover process construction where safe, but two-node
runtime success is not claimed without a second endpoint.

## Failure semantics

- Missing or incompatible framework owners raise a compatibility error before
  mutation.
- Unsupported backend, topology, quantization format, or cache layout fails
  before loading weights where possible.
- Missing, duplicated, unexpected, or shape-incompatible required checkpoint
  tensors are fatal.
- A requested AITER path records config lookup and final expert
  implementation; selector intent alone is not success.
- A model request, graph capture, or cache-hit counter alone does not establish
  numerical correctness.
- Runtime failures are investigated at their owning boundary before any fix;
  fixes require a reproducing regression test.

## Code-review and MR gate

Before push, review the complete `origin/v0.25.1..HEAD` diff for correctness,
scope, current-API alignment, model interactions, performance regressions,
test quality, and unsupported claims. Critical and Important findings block
the MR. Fixes are followed by focused and full relevant verification.

After the final commit, repeat review against the exact committed diff and
recheck the remote commit range after push. The new MR targets `v0.25.1`,
contains the exact commands and evidence paths, distinguishes unit coverage
from hardware validation, and links PR #34 as historical provenance rather
than as a merge dependency.

## Acceptance criteria

The work is complete only when:

1. The clean MR contains the retained HYV4 feature set without restoring old
   generic AITER, SlimQuant, LightOp, FP8 KV, PCP/EP, or DeepEP ownership.
2. Every production change has red-green regression evidence.
3. All changed-test modules and the full relevant contract suite pass against
   the pinned wheel from isolated import roots.
4. TP8 target-only, TP8 MTP3, FP8 KV/prefix reuse, PP2+PCP4+EP4, and the
   representative DP8+EP8 path pass their stated real-model gates.
5. HumanEval-32 baseline and MTP3 results are complete and compared without
   hidden protocol changes.
6. No test service or HCU allocation owned by this work remains after
   validation.
7. Final code review has no unresolved Critical or Important findings.
8. The remote branch matches the reviewed commit, and the new MR is open,
   targets `v0.25.1`, and reports all validation limits honestly.
