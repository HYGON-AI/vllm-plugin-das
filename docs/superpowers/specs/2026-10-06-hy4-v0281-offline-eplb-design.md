# HY4 Offline EPLB on vLLM 0.28.1

## Outcome and frozen inputs

Add complete, fail-closed offline expert-load balancing for the supplied HY4
Channel-FP8 checkpoint on the existing v0.28.1 plugin branch and PR. The
accepted topology is one node with eight HCU devices, data parallel size 8,
tensor parallel size 1, expert parallel size 8, eight redundant routed experts,
MTP with three speculative tokens, explicit DeepEP low-latency all-to-all,
DeepGEMM MoE, E4M3 sparse-MLA KV cache, prefix caching, Model Runner V2, and
the default `FULL_AND_PIECEWISE` CUDA Graph policy.

The implementation boundary is frozen at:

- Plugin base: `origin/v0.28.1-dev@9788dc633fb3d0a0a0a124b5886f62c07b3de765`.
- Existing validation branch and PR head:
  `codex/validate-v0281-gfx938-models@f5612f62c7d140052d52a06ff2d4bd7362de5f5c`.
- OpenDAS vLLM source lineage: `77acaf633d4b1bfdc31390a012cfc470441df88f`.
- Requested wheel:
  `vllm==0.28.1+dtk2604.torch2110.2609171627.g77acaf` from the DTK nightly
  index. The installed package currently identifies the paired vendor lineage
  as `0.28.1+das.77acaf6.dtk2604`; validation must record the actual package,
  import root, Python, Torch, DTK, plugin SHA, and provider versions rather than
  assuming equivalence from the version string.
- Model: `/models/Hy4-preview-Channel-FP8-w8a8`.
- Behavioral reference only: v0.25.1 commits
  `c0899322c5fb63cae0c0f3eece00bf9bfea081d5`,
  `3f5cdf5fd5c5eea6b68ee91cf39e8beacb9da19e`, and
  `5e68a650586527396934204a0d2f6d89029e5fdd`.

Refresh the target branch before the final push. If it advances, record the new
SHA and re-audit every patched owner. The old commits are not to be cherry-picked
wholesale: current vLLM interfaces and current plugin owners are authoritative.

## Scope and non-goals

The feature has two explicit phases:

1. A calibration service records target and MTP expert loads and atomically
   publishes a versioned placement file for 256 logical plus 8 redundant
   experts.
2. A fresh static service validates that file before checkpoint loading,
   directly loads the 33 physical experts owned by each EP rank, installs a
   rank-local locality-aware dispatch order, and serves without any live expert
   transfer or rearrangement.

This work does not claim dynamic HY4 expert migration, asynchronous EPLB,
elastic EP, pipeline parallelism, PCP, DCP, multi-node placement, SlimQuant or
W4A8 static loading, DeepEP auto selection, or high-throughput DeepEP Graph
support. Those combinations remain rejected unless separately designed and
validated. Ordinary current online EPLB behavior for other models must remain
unchanged when neither offline path is configured.

## Configuration contract

HCU-only EPLB controls remain in `additional_config['hcu']`, transported from
the public `--eplb-config` object by the existing EngineArgs compatibility
adapter. The supported HCU keys are:

- `expert_map_record_path`: calibration output path.
- `expert_map_path`: immutable static input path.
- `disable_rearrange`: suppress non-profile live rearrangement when explicitly
  requested; static load mode always suppresses it.
- `static_dispatch_policy`: `nearest` or `locality_fair`. HY4 acceptance uses
  `locality_fair`.

The adapter removes these keys before constructing official `EPLBConfig` and
persists their normalized values in the HCU sidecar. Worker deserialization
binds them to private attributes on `parallel_config`; no new field is added to
official vLLM dataclasses. Record and load paths are mutually exclusive and
must be non-empty strings. Unknown policies and malformed values fail before
workers load weights.

Both phases require official EPLB and expert parallelism. The accepted config
sets `num_redundant_experts=8`, yielding 264 physical experts. With EP8 each
rank owns exactly 33 physical experts. The configuration must disable async
EPLB because recording and static serving have deterministic, synchronous
lifecycle boundaries. The accepted calibration profile sets `window_size=16`
and `step_interval=16` and must observe at least one non-profile proposal before
its output is accepted.

HY4 continues to reject ordinary dynamic online EPLB. It permits EPLB only
when exactly one validated offline path is configured. The explicit runtime
all-to-all backend is `deepep_low_latency`; `deepep_auto` is not enabled or
used as a fallback.

## Placement file and model identity

The placement file is JSON with top-level `version: 2` and a `model_maps`
object. Each model entry contains:

- `model_name` and `model_class`;
- `num_moe_layers`, `num_logical_experts`, `num_physical_experts`, and
  `num_redundant_experts`;
- `physical_to_logical_map`, shaped
  `[num_moe_layers][num_physical_experts]`.

Target and MTP are separate entries in the same file. For the current HY4
owners their keys resolve from the runtime class names `HYV4ForCausalLM` and
`HYV4MTP`. The key resolver remains generic and can add a pipeline-rank suffix,
but this feature rejects PP before that path can be used.

Every map value is an integer logical expert ID in `[0, 256)`. Every layer row
has exactly 264 entries, contains every logical expert at least once, and
therefore contains exactly eight redundant placements. Declared metadata, the
runtime owner, and tensor shapes must agree. The immutable in-memory plan keeps
the canonical absolute path and SHA-256 of the exact file bytes. EP ranks use a
CPU-group all-gather to compare plan fingerprints before publishing any runtime
state.

Only EP rank zero writes the file. Writers use an adjacent lock, merge the
target or MTP entry with any valid existing entries, write and `fsync` a
temporary file in the destination directory, then atomically replace the
destination. A malformed existing file, conflicting metadata, incomplete row,
missing target/MTP entry, or rank fingerprint mismatch aborts startup. A
partially written file is never considered valid.

## Calibration data flow

The calibration service constructs the target and drafter through normal
Model Runner V2 loading, registers both with the current `EplbState`, and
records their initial maps. Real, non-dummy requests populate official expert
load counters. When the configured window produces a proposed map, record mode
captures the proposal but does not transfer weights and does not commit the
proposal into the running model. This preserves the checkpoint-loaded layout
during calibration and avoids a half-static, half-dynamic HY4 state.

Target and MTP proposals are keyed and merged independently. The final
calibration artifact is valid only after both entries have received a
non-profile proposal with the requested 256+8 dimensions. Profile and dummy
steps may exercise official initialization but cannot satisfy this gate.
Elastic rank mappings are rejected. Record mode forces synchronous behavior
and cannot start the async EPLB loop.

The calibration request set must be documented and replayable. It must include
enough concurrent and varied prompts to reach the configured window on every
DP rank. A successful HTTP response or a file containing only initial identity
maps is not calibration evidence.

## Static direct-load data flow

Static startup binds plans after model construction and before checkpoint
weight loading. The binding seam wraps the current `initialize_model` owner and
all audited imported aliases, with an exact signature and identity guard. The
wrapper validates the current loader format before it constructs the model;
accepted formats are only those covered by focused tests. Unsupported custom
or streaming loaders fail before any partial weight mutation.

For each target and MTP MoE layer, the binder:

1. validates the current `RoutedExperts` owner, EPLB capability, physical and
   logical counts, parameter loader ownership, and layer order;
2. attaches that layer's immutable physical-to-logical row;
3. changes only the instance-owned expert loader so one checkpoint logical
   expert is copied into every rank-local physical replica assigned by the
   row;
4. keeps shared experts, router bias, global scales, and non-rearrangeable
   state on their existing loaders; and
5. exposes logical checkpoint mappings so the upstream EP weight filter cannot
   discard a replicated logical expert before the static loader sees it.

Static mode rejects `enable_ep_weight_filter` rather than silently mixing its
filtering policy with direct load. It does not replace quantization methods,
weight layouts, tensors, or global classes. Binding is idempotent only for an
identical immutable plan; changing the file after binding fails closed.

After loading, `EplbState.add_model` verifies the bound plan again, commits the
map into official state without calling the expert-transfer function, and
orders replicas with `locality_fair`. This policy gives a source EP rank a
same-GPU replica when available, otherwise a same-node replica, while balancing
primary choices deterministically across source ranks. Runtime DeepEP
low-latency dispatch and DeepGEMM consume the resulting expert map normally.

The static EPLB communicator is constructed through the audited Gloo owner so
it does not eagerly register all expert buffers with NIXL; the user's DeepEP
low-latency data-plane backend is unchanged. Static `step` and `rearrange`
return without mutation, the async loop is disabled, and both transfer and
rearrangement counters must remain zero for the complete service run.

## HY4 target and MTP integration

The current HY4 target and draft already expose the common MoE metadata,
`moe_layers`, checkpoint mapping, and EPLB state methods, but deliberately
reject EPLB and contain static-plan guards. The implementation replaces those
rejections only for the validated offline modes.

Both target and MTP create 264 physical experts and 33 local physical experts
per rank before weights are allocated. Their metadata update and state-binding
methods accept only the already-bound plan dimensions; a request to resize or
rearrange them after construction fails. Target and draft maps are not assumed
identical merely because their expert counts match. Each owner loads and
fingerprints its own entry.

The implementation must preserve the existing Channel-FP8 checkpoint loader,
sparse-MLA indexer, HCU FP8 cache writer, DeepEP low-latency dispatch,
DeepGEMM masked experts, shared-expert synchronization, and MTP metadata
rebuild paths. Static EPLB is a placement and loading feature, not permission
to change these operator contracts.

## Graph, cache, and topology invariants

The accepted launch keeps all of the following enabled together:

- `VLLM_USE_V2_MODEL_RUNNER=1` and actual `HcuGPUModelRunnerV2` construction;
- DP8, TP1, EP8;
- explicit `--all2all-backend deepep_low_latency` and
  `--moe-backend deep_gemm`;
- `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'`;
- `--kv-cache-dtype fp8_e4m3`, resolved by sparse MLA to `fp8_ds_mla`;
- `FLASHMLA_SPARSE`, prefix caching, and HY4 `reasoning_effort=no_think`;
- default graph configuration, resolving to `FULL_AND_PIECEWISE`, with target
  and speculator PIECEWISE and FULL captures.

No acceptance command may add `--enforce-eager`, force PIECEWISE-only mode,
disable MTP or prefix caching, reduce EP below eight, replace DeepEP
low-latency, or silently fall back to a different MoE backend. If graph capture
or the requested route fails, the feature remains unverified.

## Error handling and observability

Compatibility wrappers guard exact current signatures and alias identity. A
target interface drift, stale patch marker, unsupported loader, absent model
entry, changed file, malformed map, unsupported quantization method, wrong
expert counts, or cross-rank disagreement raises a targeted error before
serving. The implementation must never fall back to identity placement when a
static file was requested.

Logs and retained artifacts must identify record versus static mode, canonical
map path and SHA-256, model key, map shape, redundancy count, dispatch policy,
per-rank local expert count, actual runner, attention/MoE/all-to-all providers,
resolved KV dtype, and Graph capture mode. They must not print credentials or
the complete expert map at routine log levels.

Calibration evidence records proposal events for target and MTP. Static
evidence records one fingerprint per rank and explicit zero transfer and zero
rearrangement totals. Existing official EPLB balancedness metrics remain
available but are not a substitute for those lifecycle markers.

## Tests and hardware acceptance

Implementation follows test-driven development. Focused tests must first fail
for and then cover:

- EngineArgs extraction, sidecar serialization, worker rebinding, mutual
  exclusion, and policy validation;
- version-2 file parsing, metadata and coverage checks, target/MTP merge,
  atomic replacement, immutable fingerprints, and cross-rank mismatch;
- record-only suppression of transfer and commit while publishing proposed
  target and MTP maps;
- direct loading of duplicated logical experts into rank-local physical slots,
  shared expert handling, loader-format gating, idempotence, and post-bind file
  replacement rejection;
- locality-fair deterministic ordering and malformed replica maps;
- HY4 target and MTP construction with 256+8 experts, 33 local experts at EP8,
  plan-required state binding, and rejection of unsupported online/elastic
  modes;
- static lifecycle suppression of step, rearrangement, async execution, and
  transfer without affecting ordinary online EPLB when no offline path is set;
- DeepEP low-latency and DeepGEMM consumption of the installed expert map;
- current module registration, patch ordering, and real installed-vLLM import
  smoke.

Run every changed test file, the complete HY4 suite, the relevant EPLB and
DeepEP runtime-patch tests, and the full `tests/runtime_patch` suite before the
hardware gate. Review the full branch diff against its target, not only the
newest commit.

Hardware validation is sequential because calibration and static loading use
the same eight devices:

1. Confirm all eight devices are idle and start the calibration service in a
   foreground PTY. Send the documented calibration workload, require target
   and MTP proposal records, stop the complete process group, and verify device
   memory is released.
2. Inspect the placement file offline: version, both keys, exact shapes,
   256 logical experts, 264 physical experts, eight redundant placements per
   layer, and stable SHA-256.
3. Start a fresh static service with the exact accepted topology and no eager
   or manual Graph override. Require eight matching fingerprints, 33 local
   experts per rank, intended providers, resolved `fp8_ds_mla`, and successful
   target plus speculator FULL and PIECEWISE capture.
4. Send at least nine identical prefix probes for DP8 before requiring an
   aggregate prefix hit, then run HumanEval 16 at temperature zero with
   `reasoning_effort=no_think` inside the declared isolated EvalScope boundary.
   Require raw and independently normalized 16/16, 16 unique predictions and
   reviews, HTTP success, nonzero drafted and accepted tokens, and no ERROR or
   Traceback.
5. Require zero live transfers and rearrangements throughout the static run.
   Stop the owned PTY process group, verify listeners and workers exit, and
   require HCU memory to return to idle.

All localhost probes and evaluation clients clear proxy variables and set both
`NO_PROXY` and `no_proxy`. HumanEval generated code is never executed directly
on the credentialed host; the secure evaluator keeps its API key out of argv
and removes host credential channels.

## Delivery and review

The implementation, tests, validation documents, exact server and client
commands, results, self-review, and skill updates are added as commits on the
existing branch and pushed to the existing PR only. No second PR is opened.
Before each commit, review the complete intended diff; after each commit,
review the exact committed diff and rerun any gate affected by findings.

The PR receives the calibration and static commands, placement-file checksum,
test totals, target/draft Graph evidence, provider evidence, HumanEval result,
MTP and prefix metrics, zero-transfer evidence, teardown result, and remaining
non-goals. `/models/upgrading-vllm-hcu` is updated only after those claims are
backed by retained artifacts. If any required gate fails, preserve the failure
and report the feature as unverified rather than weakening the requested
topology or runtime features.
