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
