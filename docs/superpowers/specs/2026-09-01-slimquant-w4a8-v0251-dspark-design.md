# SlimQuant W4A8 v0.25.1 and DSpark Design

## Context

MR #38 was developed against `v0.25.1@8615729`. The current remote base is
`v0.25.1@85a4ad5`, which now owns the public SlimQuant registry, unified AITER
MoE dispatch, DeepSeek-V4 DSpark integration, and `deepep_auto` runtime policy.
MR #38 overlaps those implementations and must not retain competing LightOp,
AITER, DeepSeek-V4, sampler, or fixed DeepEP routing paths.

The target checkpoint is:

```text
/models/DeepSeek-V4-Pro-0813-INT4-Channel
```

Its configuration exposes the DSpark metadata, target layers, Markov rank,
noise token, and MTP projection quantization needed for a live feasibility
test. Support is accepted only after the checkpoint starts and produces valid
speculative output on HCU.

## Goals

1. Rebuild MR #38 on the latest remote `v0.25.1` instead of mechanically
   retaining obsolete conflict resolutions.
2. Use the base branch's SlimQuant W4A8 AITER implementation unchanged for
   pure TP: canonical packed weights, runtime config selection, solution-aware
   derived layouts, and vLLM Triton fallback only on an explicit capability
   miss.
3. Add only the missing SlimQuant W4A8 DP+EP integration for DeepEP and
   DeepGEMM.
4. Use `deepep_auto` as the single DP+EP public contract. Runtime batch state
   chooses contiguous high-throughput or masked low-latency experts.
5. Validate DSpark on TP8+AITER and DP8+EP8+`deepep_auto` using the target
   checkpoint.
6. Keep the MR free of `.github`, workflow, and script changes.

## Non-goals

- Restoring the MR's previous default-LightOp versus explicit-AITER split.
- Forcing AITER ASM, HIPC, Triton, MOE_C, or CK from plugin call sites.
- Adding public fixed `deepep_high_throughput` or `deepep_low_latency` recipes.
- Replacing the latest base's DeepSeek-V4 DSpark, attention, weight-loading,
  ROCm layout, or sampler patches.
- Claiming DSpark support from configuration or unit tests without live model
  evidence.

## Selected Architecture

### Branch reconstruction

Create the updated MR branch from `origin/v0.25.1@85a4ad5` and port only code
that remains absent from the base. Preserve the existing MR number by updating
its remote head after verification. The resulting diff must be reviewed as a
new delta against the current base, rather than as conflict resolutions on top
of the obsolete base.

### Pure TP ownership

The latest base remains the sole owner of TP SlimQuant W4A8:

```text
--quantization slimquant_w4a8 --moe-backend aiter
    -> unified AITER problem/config dispatcher
    -> AITER-selected solution and derived layout
    -> vLLM Triton only when AITER reports status=False
```

MR #38 must not introduce a second AITER runtime, mutate canonical packed
parameters, or route `auto` to LightOp.

### DP+EP ownership

For DP8+EP8, `--all2all-backend deepep_auto` owns dispatch and combine. The
SlimQuant method supplies W4A8 DeepGEMM experts compatible with both layouts:

```text
deepep_auto forward snapshot
    |-- high throughput -> contiguous HIPC W4A8 DeepGEMM
    `-- low latency     -> masked N32 HIPC W4A8 DeepGEMM
```

The implementation must preserve raw checkpoint parameters as canonical
owners and cache any packed/view layout separately when required by the base
branch's expert lifecycle. Scale conversion remains exactly once at the
quantization boundary. Expert maps, empty-rank behavior, shared experts, and
prepare/expert/finalize selection must follow the existing `deepep_auto`
snapshot contract.

Unsupported topologies or quantization metadata fail with an explicit error;
they must not silently enter AITER, LightOp, or an incompatible W8A8 kernel.

### DSpark integration

No new DSpark model class is added. The latest base's HCU
`DSparkDeepseekV4ForCausalLM`, target auxiliary-state patch, weight aliases,
and non-PCP cache insertion remain authoritative. The SlimQuant delta must
work for both target and draft MoE layers through the same quantization and
backend contracts.

The supported speculative configuration is:

```json
{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}
```

## Validation

### Static and unit gates

- Rebased diff contains no obsolete duplicate patches and no `.github`,
  workflow, or script changes.
- Existing latest-base SlimQuant W4A8 and unified AITER suites remain green.
- New tests cover `deepep_auto` expert selection, contiguous and masked W4A8
  layouts, scale propagation, empty dispatch, expert-map behavior, and
  fail-closed unsupported metadata.
- Existing DSpark target/draft and DeepEP-auto synchronization tests remain
  green.

### Live TP8 gate

Start the checkpoint with TP8, explicit AITER, and DSpark. Acceptance requires:

- all eight workers complete model and draft-model loading;
- logs show unified AITER W4A8 selection rather than a duplicate MR runtime;
- a deterministic OpenAI-compatible request returns non-empty output;
- speculative decoding reports proposed and accepted tokens without runtime
  or weight-loading errors.

### Live DP8+EP8 gate

Start one service with DP8, EP enabled, `deepep_auto`, and DSpark. Acceptance
requires:

- all ranks complete target and draft loading;
- DeepEP dispatch/combine and W4A8 DeepGEMM execute successfully;
- synchronized runtime evidence includes the expected auto-selected layout;
- a client request returns non-empty output and DSpark acceptance evidence.

After both runtime gates pass, run HumanEval-32 where time and environment
permit. Report raw predictions/reviews, pass@1, and artifact paths. A startup
or request failure is reported as evidence and debugged before support is
claimed.

## MR and Review Requirements

- Update MR #38 rather than opening a competing MR.
- Rewrite its Summary around the latest-base ownership boundaries, exact
  server/client commands, observed DSpark evidence, and test results.
- Preserve commit authorship as `zhangzbb <1414695739@qq.com>`.
- Perform an independent code review of the complete final diff. Critical and
  Important findings must be fixed and reverified before the MR is declared
  ready.

## Acceptance Criteria

- MR #38 is based on current `origin/v0.25.1` and contains only the missing
  SlimQuant W4A8 DP+EP delta plus its tests and documentation.
- Pure TP uses the base branch's unified AITER dispatcher without a competing
  LightOp/AITER routing implementation.
- DP8+EP8 uses one `deepep_auto` public configuration and correct HT/LL W4A8
  DeepGEMM layouts.
- TP8+AITER+DSpark and DP8+EP8+DSpark have reproducible live evidence, or the
  exact blocking root cause is documented without claiming support.
- The final test suite passes and independent review has no unresolved
  Critical or Important findings.
