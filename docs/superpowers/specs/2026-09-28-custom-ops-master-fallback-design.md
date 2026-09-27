# HCU Custom-Op Master Fallback Design

## Goal

Make `VLLM_HCU_USE_CUSTOM_OPS` a reliable safety switch for optional HCU
operator optimizations.  When it is disabled, managed LightOp and AITER
sub-operators must not execute: the runtime must select an audited Triton,
BoltOPs, upstream vLLM, or vllm_hcu-native implementation, or reject the
configuration before serving if no safe equivalent exists.

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
- a configuration whose cache/layout ABI has no audited non-custom
  implementation is rejected early with an error naming the operator and
  required setting;
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
| optional MLA concat/top-k/cache helpers | LightOp/AITER | native implementation, or early rejection if the cache ABI has no fallback |
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
provider-specific layout conversion remain visible errors.  Unsupported
master-off cache formats fail during configuration validation when possible.

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
