# Qwen4Exp SlimQuant W4A8 DeepEP Auto and Phase-Aware CUDA Graph Design

## Status

Approved in chat on 2026-10-18. This document defines the functional-support
change stacked on `feature/int8-allgatherv` (MR #247). The implementation will
remain opt-in through the existing explicit CLI combination:

```text
--enable-expert-parallel
--all2all-backend deepep_auto
--moe-backend deep_gemm
```

The default AITER plus all-gather/reduce-scatter route is unchanged.

## Goal

Support Qwen4Exp SlimQuant W4A8 on DP4/EP4 with one `deepep_auto` service that
uses DeepEP high-throughput (HT) communication and contiguous DeepGEMM experts
for prefill or mixed batches, and DeepEP low-latency (LL) communication and
masked DeepGEMM experts for pure decode. The service must remain live when DP
ranks become idle at different times, and it must retain decode CUDA Graphs
without attempting to capture graph-incompatible HT communication.

Success requires functional correctness and stability, not a performance win.
Performance and accuracy results are nevertheless required MR evidence.

## Observed Failure and Root Cause

The initial architecture-gate experiment proved that both real Qwen4Exp paths
can be constructed and executed:

- prefill selected contiguous HT experts;
- decode selected masked LL experts;
- both used the SlimQuant W4A8 HIPC DeepGEMM operators.

However, the first implementation only added the Qwen4Exp architecture names
to the existing DeepEP-auto allow-list. It exposed two independent runtime
problems.

First, the existing compilation patch globally changes
`FULL_AND_PIECEWISE` to `NONE` whenever DP `deepep_auto` is selected. This is
safe for HT but unnecessarily removes the decode graph even though LL is graph
compatible. The measured decode TPOT regressed from about 13.6 ms to 41.3 ms.

Second, Model Runner V2 propagates request phase from `prepare_inputs()` for a
real batch, but an idle DP rank executes `_dummy_run(..., uniform_decode=True)`
without taking that path. The invocation-local phase holder is therefore empty
for the dummy rank. The existing DP phase all-reduce is present, but the dummy
rank contributes padded work as an active non-decode rank and forces HT while
another rank is still performing decode. In the DP4 workload this transition
was followed by all engines waiting indefinitely after three ranks became
idle. The missing dummy phase, rather than a missing all-reduce, is the contract
violation to fix.

At `--gpu-memory-utilization 0.90`, a separate 6.08 GiB contiguous DeepGEMM HT
workspace allocation also failed because only 4.82 GiB was free. Hardware
validation will use a memory limit that reserves this dynamic workspace. That
capacity issue is not part of the phase fix.

## Execution Contract

`deepep_auto` will bind its DeepEP/DeepGEMM delegate to the runtime graph mode.
This makes the selected Python delegate part of the graph cache contract rather
than mutable state hidden from capture.

| Runtime mode | Workload contract | DeepEP mode | DeepGEMM layout |
|---|---|---|---|
| `FULL` | Uniform pure decode | LL | masked |
| `PIECEWISE` | Prefill or mixed batch | HT | contiguous |
| `NONE` | Uncaptured shape or eager fallback | DP-synchronized phase | LL for all-active decode, otherwise HT |

For `FULL` and `PIECEWISE`, the DP graph-mode synchronization performed by the
vLLM runner is authoritative. No additional phase collective is needed: all
ranks already agreed on the same runtime mode before model execution.

For `NONE`, the current conservative DP phase collective remains. Empty ranks
are neutral, every active rank must report pure decode before LL is selected,
and any prefill, mixed, or missing phase evidence selects HT.

## CUDA Graph Policy

The configured `FULL_AND_PIECEWISE` mode will no longer be globally disabled
for `deepep_auto`.

- FULL capture is decode-only and must capture the LL delegate.
- PIECEWISE capture must capture the HT delegate. DeepEP HT communication
  remains outside captured graph segments; surrounding compilable computation
  may use the piecewise graph.
- Replay must select the same delegate captured for that graph mode.
- An uncaptured decode shape uses eager `NONE` plus synchronized LL rather than
  replaying a PIECEWISE HT graph under LL mutable state.

The first hardware gate will test PIECEWISE HT capture explicitly. If the
installed vLLM/ROCm stack still exhibits the upstream compile-cache limitation
for HT piecewise graphs, the implementation must fail closed to phase-aware
`prefill/mixed=NONE` and `decode=FULL`. It must not fall back to disabling
decode graphs globally, and it must not claim PIECEWISE prefill support.

## Model Runner V2 Phase Propagation

The HCU V2 runner will preserve the `_dummy_run()` `uniform_decode` decision
until its nested `execute_model()` invocation enters
`deepep_auto_request_phase_scope()`:

- runtime DP dummy decode contributes explicit non-prefill evidence;
- profile and non-uniform dummy runs contribute prefill/HT evidence;
- the state is invocation-local and restored in `finally`;
- ordinary real batches continue to use `InputBatch.is_prefilling_np`;
- the behavior is enabled only for `deepep_auto`.

This change must not infer phase from token count. With MTP3, a decode request
can schedule up to four target tokens, while short prefills can have the same
shape.

## Target and MTP Behavior

The target model and native MTP draft share the same graph-mode contract.
During capture, FULL binds both target and draft MoE layers to LL, while
PIECEWISE binds both to HT. During eager execution, both observe the same
invocation phase and DP synchronization order.

No change is made to the number of speculative tokens, rejection sampling,
KV-cache dtype, or Qwen attention selection.

## Scope and Compatibility

In scope:

- Qwen4Exp causal and conditional-generation architecture wrappers;
- SlimQuant W4A8 DeepEP-auto factory selection;
- Model Runner V2 real, dummy, target, and MTP phase propagation;
- phase-aware FULL/PIECEWISE/NONE graph selection;
- DP4/EP4 functional, accuracy, and performance validation.

Out of scope:

- changing the default MoE or all-to-all backend;
- enabling fixed `deepep_high_throughput` or `deepep_low_latency` for this
  checkpoint;
- changing AITER/AGRS behavior from MR #247;
- solving DeepGEMM workspace fragmentation or changing KV-cache allocation
  policy;
- enabling DBO/ubatching for `deepep_auto`;
- claiming that DeepEP outperforms AITER/AGRS for this model.

Existing validated DeepSeek-V4 and GLM behavior must be preserved.

## Failure Handling and Observability

The implementation will fail closed:

- missing or mixed phase evidence in `NONE` selects HT;
- an unsupported graph-mode/delegate combination raises during capture or
  startup rather than silently replaying the wrong layout;
- the existing HIPC operator capability checks remain mandatory;
- logs identify the effective graph policy and report the first HT and LL
  selection.

Optional diagnostic logging may record rank, dummy/real state, runtime graph
mode, and selected delegate, but normal serving logs must remain one-shot and
low volume.

## Verification

Software tests will use red-green TDD and cover:

1. Qwen4Exp factory admission for both wrapper architecture names.
2. V2 runtime dummy decode propagation inside the invocation scope.
3. Restoration of dummy phase state after success and exceptions.
4. `FULL -> LL`, `PIECEWISE -> HT`, and `NONE -> synchronized dynamic mode`.
5. Empty DP ranks being neutral in eager phase synchronization.
6. Target and MTP capture using the same phase contract.
7. Feature-off and existing DeepSeek/GLM behavior remaining unchanged.

Hardware validation will run on DP4/EP4 with sufficient free memory for the HT
workspace:

- repeated requests where only one rank remains active, proving no collective
  hang;
- BS16, input 4096, output 1024, three warmups and five measured rounds;
- logs proving prefill HT, decode LL, and the effective CUDA Graph modes;
- HumanEval first 32 samples at concurrency 32;
- focused trace inspection for prefill and decode graph replay.

The MR will be stacked on MR #247 and include the exact server command,
benchmark command, accuracy result, relevant log paths, performance comparison,
and any PIECEWISE-to-NONE fallback disclosed explicitly.

