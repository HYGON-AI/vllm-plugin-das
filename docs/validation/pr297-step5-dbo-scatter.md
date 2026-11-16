# PR297: DBO split and fused-scatter contracts

Reviewed PR head: `44b15d96de990184287e9adf016645d003209429`.
Declared OpenDAS vLLM pin: `462fdb097c66b487ef4826e8009431c10fe88fb8`.

## DBO eligibility

The total-token gate accepted `[4]*512+[2048]` at 4096 tokens although the
2048-token split leaves its first microbatch decode-only. Eligibility now uses
the current ordered BatchReqState and initialized DSA builder threshold, checks
the clipped queries at the globally shared split point, and takes a CPU DP MIN
vote. One unsafe rank vetoes the split on every rank before microbatch threads
start. Safe mixed and all-prefill batches still split.

A scoped ContextVar supplies this dispatch's metadata and resets on exceptions
and nested calls. Unavailable/invalid metadata and PCP/adaptive-verification
layouts fall back because their eventual query boundaries are not established.
The existing common count/tail gate precedes the additional collective; native
DBO configurations without the Step5 marker retain their original threshold.

## Fused-scatter layout

The receiver binds the exact aligned expert counts and batch shape to the
scatter tensors. The consumer validates that layout and uses its row count,
independently of COMPACT_ROWS and HT_EXACT_LAYOUT. This fixes source 256 versus
expected 7680(E22) / 7936(E23) before GEMM. Ordinary permutation sizing and
conservative startup workspace planning remain unchanged.

## Verification

- 99 CPU tests and 32 subtests passed, including real pinned DP dispatch/sync,
 the actual Step5 execute entry, and real CPU Torch tensors. Only collective
 communication was simulated with two-thread barriers; CUDA initialization
 was forbidden. The isolated test container had no GPU devices or network.
- 7 additional real Torch meta-tensor checks passed for E22/E23, multiple slices,
 empty layouts and noncontiguous rejection.
- The reported mixed batch and unequal per-rank layouts changed from two
 microbatches on both ranks to one on both; safe cases remained at two.

The test environment used the exact pinned Python sources with image Torch and
native extensions providing import dependencies. These checks do not constitute
GPU kernel numerics, a hardware distributed run, or a model-performance rerun.


## Native DBO ownership follow-up

At head `fea967ee2e5fa8e1915d5df14a8b5c0c53088cd8`, an already enabled
native DBO configuration skipped HCU initialization but still acquired the HCU
runner and DSA hooks when the Step5 environment threshold was positive.
`install_runner` now requires both enabled DBO and the exact HCU owner marker.
Explicit native DBO retains its own threshold and methods; the environment
threshold does not partially take over that runner. HCU auto-enable and
idempotent hook installation remain available.

The new26-case CPU ownership suite reproduces9 unwanted-mutation failures on
that reviewed head and passes after the guard. The combined portable regression
selection passes119 tests. These cases execute actual enable/install/DSA bodies
with runtime dependencies doubled; no GPU or thread-hang experiment was run.
This follow-up adds no per-step collective or model operator.
