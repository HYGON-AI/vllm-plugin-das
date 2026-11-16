# PR187 EPLB Review Closure Design

## Goal

Close the six confirmed P2 findings against PR #187 without broadening the
supported offline EPLB topology. Preserve ordinary online EPLB behavior and
validate the final DP8/TP1/EP8 HY4 record-to-static workflow on eight devices.

This design amends
`docs/superpowers/specs/2026-10-06-hy4-v0281-offline-eplb-design.md` where the
review found the implementation or original wording incomplete.

## Calibration publication

Record-only calibration must publish the output of a completed EPLB policy
calculation independently of whether online vLLM would migrate weights. The
ROCm five-percent online migration threshold is not an acceptance threshold
for an offline calibration artifact.

During record-only `EplbState.rearrange`, a per-call policy proxy captures each
model's `rebalance_experts` output and whether its already-all-reduced logical
load tensor contains any load. The normal upstream rearrange implementation
still owns topology calculation and policy invocation. Transfer remains
suppressed and the live model maps remain unchanged. After upstream returns,
EP rank zero publishes every captured candidate whose global load is nonzero,
including a candidate equal to the current layout and a candidate rejected by
the ROCm online-gain threshold.

Local `is_dummy` is not a publication veto. A locally idle EP0 rank may publish
when the collective result contains real load from another rank. An all-zero
global load, profile rearrangement, or missing policy result cannot publish a
proposal. Target and MTP are evaluated independently, so an unused MTP model
remains `initial` and the complete-artifact validator rejects it.

## Session reset and malformed artifacts

Starting a new calibration session invalidates stale proposals only after an
existing artifact has been parsed and fully validated while holding its file
lock. Valid prior content is atomically replaced by an empty version-2 payload;
malformed JSON or schema is rejected without changing the file. Subsequent
initial target/MTP entries merge into this fresh session as before.

## Static primary routing

`nearest` and `locality_fair` choose a rank-local primary; column order alone
does not change vLLM's hashed replica selector. Static add-model therefore
marks each bound `EplbLayerState` with an immutable, device-local primary-only
replica-count tensor containing ones. The true `logical_replica_count` remains
unchanged for metadata and validation.

The HCU BaseRouter adapter explicitly consumes this marker when applying EPLB
mapping and passes the primary-only count to the installed map-and-record
function. Both the HCU Torch implementation and the delegated upstream Triton
implementation therefore select column zero. Online and non-static states have
no marker and preserve hashed selection across all replicas.

Tests must assert returned physical IDs for repeated tokens, not merely map
ordering. A logical expert with replicas on GPU0 and GPU4 must route every
rank-0 token to its rank-0 primary in static mode, while the same unmarked map
continues to distribute tokens by the upstream hash.

## Feature activation and DBO

The offline loading callbacks remain gated only by record/load paths. The
EPLB-state lifecycle callback gets a separate feature state enabled by a
record path, a load path, or `disable_rearrange=true`. Thus the standalone
disable flag suppresses non-profile rearrangement without arming unrelated
static model-loader behavior.

The fixed DeepEP low-latency GPU-DP wrapper is removed from the callback
inventory and terminal requirements. vLLM's native
`sync_cudagraph_and_dp_padding` remains installed for fixed and auto backends,
preserving its collective DP state and DBO `num_ubatches` selection. This
change addresses the confirmed DBO defect; it does not claim that every
previous unsynchronized configuration necessarily hangs.

## Verification

Every product change follows a witnessed RED/GREEN test cycle. Focused tests
cover ROCm low-gain publication, EP0-local-dummy/global-load publication,
all-zero rejection, malformed reset preservation, actual primary physical IDs
through both mapping backends, standalone disable activation, and fixed
DeepEP-LL DBO delegation.

After the relevant software suites pass, hardware acceptance uses
`/models/Hy4-preview-Channel-FP8-w8a8-v2`, explicit DeepEP low latency,
DeepGEMM, DP8/TP1/EP8, MTP3, E4M3 KV, prefix caching, and the default graph
policy. It runs record-only calibration, validates proposal-complete target
and MTP entries, starts a fresh static service, verifies primary routing and
zero transfer/rearrangement evidence, exercises low-latency fake prefill, and
runs deterministic HumanEval/0-31 with a required normalized 32/32 result
before the branch is pushed and review threads are resolved.
