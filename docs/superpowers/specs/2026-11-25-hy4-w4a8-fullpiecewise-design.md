# HY4 W4A8 FULL_AND_PIECEWISE

User requests W4A8 eager precision alignment and FULL_AND_PIECEWISE support,
prioritizing existing Adapter/Plugin operators. Retain raw channel INT4 packed
weights, dynamic token INT8 activation quantization, BF16 rounding boundaries,
SwiGLU clipping and Core loader/router/reduction ownership. No Core changes.

## Approach

Reuse current routed_reference and native integer kernels, plus existing Core
opaque MoE/fake boundary and scoped Flash Graph-bank admission. A new arithmetic
backend would add numerical drift; removing guards without qualification would
admit unsupported execution. Only the shared qualified Graph configuration is
admitted; ordinary TP eager remains available. Explicit reference config stays
required during qualification. No silent eager fallback.

## Acceptance

CPU admission regressions and existing related suites; actual method and Core
capture/replay; fullmodel variable batch and graph-bank smoke; HumanEval164 C64
within3percentagepoints against W4A8 eager; profiler-off throughput and TPOT.
Component capture evidence alone does not satisfy wholemodel acceptance.


## Explicit AITER W4A8 candidate admission

An explicit standard `--moe-backend aiter` admits the original Flash channel
W4A8 checkpoint for both routed experts and dense/DFly construction, without
requiring a `flash_w4a8_backend` metadata override. Dense/DFly CT methods remain
Core-owned; routed experts use `HYV4FlashAiterW4A8Config`. Auto/reference and
FP8 behavior stay unchanged. All existing serialized schema, HCU, ordinary TP
and restricted Graph admission checks still apply.

The candidate preserves CT packed loading and INT32 Parameter ABI, performs
one lossless CT-to-HIPC packed conversion and calls public AITER configuration,
prepare and fused execution APIs. Preparation owns layout/shuffle selection
over the configured capacity; a legacy get-config recommendation is not used
to override the public fused router's backend restrictions. Device padding
masks retain the semantics of `-1` routes while replacing their native IDs
with distinct unused experts. Invalid weights and zero-contribution NaN inputs
are zeroed on the device; valid route IDs and nonzero weights are retained.

AITER `/127` FP32 activation rounding differs from the exact CT `/127.5` BF16
reference. Full-model HumanEval C64 acceptance and actual Graph/candidate
performance remain pending; partial operator tests cannot establish them.
