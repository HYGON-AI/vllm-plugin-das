# HCU Custom-Op Master Fallback Design

## Goal

Make `VLLM_HCU_USE_CUSTOM_OPS` a reliable safety switch for optional HCU
operator optimizations.  When it is disabled, managed LightOp and AITER
sub-operators must not execute: the runtime must select an audited Triton,
BoltOPs, upstream vLLM, or vllm_hcu-native implementation.  Operators without
an audited replacement remain explicitly outside the managed set until a
fallback is available; the master must not make an otherwise working model
unstartable merely to enforce provider purity.

## User-visible contract

`VLLM_HCU_USE_CUSTOM_OPS` defaults to enabled and has higher priority than
the child switches for managed optional operators.

With `VLLM_HCU_USE_CUSTOM_OPS=1`, existing child switches and backend
selection continue to work as they do today.

With `VLLM_HCU_USE_CUSTOM_OPS=0`:

- managed LightOp and AITER optimization sub-operators are ineligible even
  when their child switch is enabled;
- sparse MLA uses the BoltOPs implementation;
- channel-FP8 and channel-INT8 dense GEMM use the target vLLM Triton
  implementations;
- routing/gate, normalization, activation, sampling, FLA, MHC, cache helper,
  and similar optional replacements use their upstream vLLM or vllm_hcu
  reference path;
- fallback discovery prefers an existing upstream/native implementation,
  then BoltOPs or Triton, and finally a small maintainable reference
  implementation owned by vllm_hcu;
- an operator for which none of those choices is safe remains unchanged and
  is documented as an explicit unmanaged exception rather than being disabled
  without a working replacement;
- selection emits a one-time message identifying the fallback provider when
  that is not already evident from existing backend logs.

The master switch is a startup contract.  Changing it after model
construction is unsupported because some providers select or transform
weight and cache layouts during initialization.

## Boundaries and exceptions

The switch does not choose the MoE expert backend.  An explicit
`--moe-backend aiter`, `triton`, or `deep_gemm` remains authoritative.  The
master may disable optional auxiliary operators used around that backend,
including a LightOp gate/router replacement, but it must not rewrite the MoE
backend itself.  Users can select Triton explicitly when they want the MoE
expert implementation to change.

FP8 QSA reader and cache-writer operators for E4M3/E5M2 remain independent of
the generic master switch, matching the previously accepted QSA cache ABI
contract.  Tests must name this exception so a future broad gate does not
disable it accidentally.

Necessary HCU platform primitives that are not optional optimizations are not
disabled merely because their implementation is registered as a custom
operator.  Such a path must be documented as a platform primitive or receive
an audited fallback before it is added to the managed set.

Likewise, an optional operator with no safe fallback is not added to the
managed set yet.  This is a temporary compatibility exception, not permission
to silently claim that the operator fell back.  The audit records why the
existing provider remains necessary and which replacement options were
checked.

## Architecture

### Effective policy

`vllm_hcu.platforms.envs` owns one small effective-policy helper.  Callers
pass the child feature value and receive `master && child`; this avoids each
dispatcher independently reimplementing precedence.  Raw environment getters
remain raw so diagnostics can distinguish "requested" from "effective".

Provider/backend selectors use the helper before importing or registering an
optional LightOp/AITER operator.  Runtime wrappers retain their native
fallback callable and delegate to it when the policy is off.  Layout-changing
providers must decide before weight/cache conversion; they may not convert to
a proprietary layout and then attempt to fall back.

### Provider scope

The audited managed set covers:

| Area | Optimized provider | Master-off behavior |
| --- | --- | --- |
| Sparse MLA | HCU/LightOp optimized path | BoltOPs sparse MLA |
| Dense FP8/INT8 GEMM | LightOp/hipBLASLt/AITER selection | target vLLM Triton GEMM |
| MoE gate/router auxiliary | LightOp | upstream vLLM router |
| RMSNorm/Gemma RMSNorm/gated RMSNorm | LightOp | upstream/native implementation |
| SiLU-and-mul and fused quant helpers | LightOp | upstream/native composition |
| sampler/top-k helpers | LightOp | upstream/native implementation |
| optional FLA and MHC kernels | AITER | BoltOPs/Triton/native implementation |
| optional MLA concat/top-k/cache helpers | LightOp/AITER | native, BoltOPs, Triton, or a vllm_hcu reference implementation; otherwise documented unchanged exception |
| PLE prefetch stream | HCU optimization | synchronous existing path |

MoE expert kernels, DeepEP/DeepGEMM provider selection, and AITER MoE weight
shuffle/configuration belong to the selected MoE backend and are outside this
master policy.

### Failure behavior

Fallback is capability-based, not exception-based.  The runtime decides the
provider before execution.  It must not execute a custom kernel, catch a
precision or runtime failure, and retry a different provider inside a graph.

Missing optional packages select the fallback when the layout is still
portable.  ABI mismatches, corrupted installations, and failures after a
provider-specific layout conversion remain visible errors.  Before declaring
that no fallback exists, the implementation audit must check upstream vLLM,
vllm_hcu native code, BoltOPs, and Triton, then assess whether a small portable
reference implementation can be maintained locally.  If none is safe, the
operator remains unchanged and outside the managed set with an explicit
documented reason.

## Validation

Unit tests cover all four master/child combinations, assert the selected
provider, and use call spies to prove that LightOp/AITER entry points are not
reached when the master is off.  Separate tests prove that explicit AITER MoE
selection is not rewritten and that FP8 QSA reader/writer selection remains
independent.

Regression tests cover the Hy4 grouped-top-k gate because that is the known
precision-risk path.  The master-off result must match the upstream router's
top-k ids and weights for the same logits and bias.

Hardware validation on the eight-card host uses the existing MR commands and
records exact branch/commit, wheel, topology, graph mode, environment, server
readiness, route evidence, and accuracy result.  At minimum it reruns the
previous master-off GLM sparse-MLA case and a Hy4 DP+EP+MTP3 case with the
managed LightOp gate disabled.  HumanEval uses the established eight-prompt
gate.  The final server commands and evidence are posted to MR #163.

## Documentation

Update `/models/upgrading-vllm-hcu` with the effective-policy contract, MoE
backend boundary, QSA exception, fallback matrix, and validation evidence.
The implementation and documentation remain in the existing MR #163 rather
than opening another MR.

## Validation record

Candidate `c7250ed333f26bb4c50b8437b3919e5ed6e6926f` was validated against
installed vLLM `0.28.1+dtk2604.torch2110.2609171627.g77acaf`:

- focused attention/GEMM/QSA suites: 388 passed;
- v0.28.1 repository suite: 2294 passed, 63 skipped, one explicitly deselected
  v0.25.1-source test; the two other v0.25.1-only files were ignored;
- GLM-5.3 DP8/EP8/MTP3 and Hy4 DP8/EP8/MTP3 both reached HTTP health 200,
  executed BoltOPs sparse MLA on all eight workers, captured target and MTP
  PIECEWISE/FULL graphs, passed a prompt longer than the 2048 sparse threshold,
  and scored HumanEval/0-7 at 8/8;
- a same-source master-off probe reported the Channel-FP8 adapter backend as
  `target-triton` with its HCU patch marker installed;
- both owned server process groups stopped cleanly and all eight cards returned
  to zero reported memory use.

The hardware run used the candidate Python source through `PYTHONPATH` and the
installed `237e559` native extension. There are no changes from `237e559` to
the candidate in `vllm_hcu/csrc`, `setup.py`, or `pyproject.toml`; this is
therefore exact Python-source validation, not an exact candidate-wheel claim.
The only retained no-fallback exception found by the provider audit is the
DeepSeek-V4/DSpark `fp8_ds_mla` fused qnorm+RoPE+KVNorm+UE8M0 paged cache
writer. FP8 QSA reader/writer remains the intentional E4M3/E5M2 exception, and
explicit MoE expert selection remains outside the generic master.
