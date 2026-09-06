# sglang-das Operator Adaptation Design

Date: 2026-09-06

## Status

Approved for implementation. The user approved an evidence-driven adaptation
of generally useful LightOp and AITER operators from `sglang-das` into the
vLLM HCU plugin. Model coverage is desirable but is not an acceptance gate for
an operator when an independent live-HCU accuracy test proves correctness.

## Objective

Improve vLLM-HCU serving performance by adapting LightOp and AITER operator
paths that are used by `sglang-das`, are relevant to general LLM inference,
and are not already implemented by this plugin.

The work must:

- preserve upstream vLLM source and canonical `vllm.*` ownership;
- follow the plugin runtime patch architecture and explicit registration rules;
- remain globally controllable by `VLLM_HCU_USE_CUSTOM_OPS`;
- provide leaf switches for new routes;
- prove live-HCU numerical correctness before integration;
- retain the existing plugin/vLLM path when a route is disabled or ineligible;
- demonstrate operator-level performance benefit before enabling a route; and
- validate the two requested models and HumanEval-32 after integration.

## Non-goals

- Copying all `sglang-das` code or mirroring its framework-specific dataflow.
- Editing `/models/zb/vllm_025` or installed vLLM files.
- Claiming model accuracy from a startup or generation smoke test.
- Adding an operator only because its symbol exists in LightOp or AITER.
- Treating unavailable operators in the installed dependency versions as
  implemented.
- Replacing an existing plugin route without numerical and performance evidence.

## Source and Dependency Baselines

- Plugin base: latest `origin/v0.25.1` at implementation start.
- vLLM source reference: `/models/zb/vllm_025`.
- Runtime vLLM: installed vLLM 0.25.1-compatible environment.
- Reference implementation: current `HYGON-AI/sglang-das` main checkout.
- Hardware: local HCU devices, checked for ownership immediately before tests.
- Installed optional backends: LightOp and AITER as reported in the validation
  environment fingerprint.

No access token, private endpoint, or credential may be recorded in source,
Git configuration, test artifacts, logs, or documentation.

## Approaches Considered

### Evidence-driven allowlist (selected)

Audit every LightOp/AITER use in `sglang-das`, map it to the corresponding
vLLM dataflow, and adapt only operators with a compatible seam, installed ABI,
correct numerical behavior, and measurable benefit. This gives a complete
decision record without importing framework-specific assumptions.

### Wholesale framework mirroring (rejected)

Copying SGLang routes would be faster initially but would couple vLLM to
SGLang layouts, scheduling, custom-op schemas, and model implementations. It
would violate plugin ownership boundaries and make fallbacks unreliable.

### Two-model-only optimization (rejected)

Special-casing the two requested validation models would miss reusable serving
paths and would not satisfy the requirement to adapt generally useful
operators.

## Operator Audit

The repository will contain a reviewable matrix for every relevant LightOp and
AITER call site found in `sglang-das`. Each row records:

- source operator and SGLang call site;
- installed symbol and callable ABI;
- closest vLLM/plugin execution seam;
- current plugin coverage;
- dtype, layout, shape, and mutation semantics;
- fallback owner;
- numerical-test status;
- benchmark status; and
- disposition: already covered, adapt, dependency unavailable, no compatible
  seam, numerical rejection, or performance rejection.

The initial high-confidence missing candidates are:

1. LightOp `moe_fused_gate_sqrtsoftplus` for non-hash DeepSeek V4 routing.
2. LightOp W16A16 Marlin MoE (`get_moe_cuda_marlin_config_w16a16`,
   `moe_gemm_marlin_w16a16`, activation, and reduction) for compatible BF16
   MoE layers.
3. AITER SiLU-and-multiply as a benchmark candidate against the current
   LightOp and vLLM implementations.
4. LightOp `ep_build_m_indices`, RMSNorm/quant variants, and generic KV-store
   fusions when an equivalent vLLM seam exists.

Existing plugin integrations such as AITER MoE selection/shuffle, quantization,
MHC, GDN and top-k; and LightOp EP scatter/gather, MoE alignment, INT8/FP8
quantization, Marlin quantized MoE, sparse MLA, and DeepSeek V4 attention are
classified as already covered rather than duplicated.

Symbols absent from the installed dependency (currently including AITER
`topk_gating`, `greedy_sample`, and selected fused QK/RoPE APIs) are recorded as
dependency unavailable. They are not hidden behind a path that can never run.

## Plugin Architecture

### Registration and ownership

- Patch adapters declare stable patch IDs, exact target modules and symbols.
- Registration is explicit and ordered; no directory scanning is introduced.
- Adapters validate required callable signatures before replacing behavior.
- Target vLLM modules are imported only at patch application time.
- Application remains idempotent using `_vllm_hcu_*` markers.
- First patch failure is propagated and latched by the existing coordinator.
- Substantial HCU behavior lives in `vllm_hcu`-owned runtime modules.
- Small wrappers or callbacks only select a route and preserve the original
  callable for fallback.

### Configuration

Every new route uses:

```text
VLLM_HCU_USE_CUSTOM_OPS && VLLM_HCU_USE_<BACKEND>_<OPERATOR>
```

The master switch disables every new route. Leaf switches allow isolation and
bisecting. New layout-mutating routes start opt-in. A stateless route may become
default-on only after live-HCU accuracy, performance, model integration, and
master-off regression tests pass.

Environment variables are registered in `vllm_hcu/platforms/envs.py` using the
existing lazy getter convention. Worker-dependent configuration remains bound
after deserialization through existing sidecar/config mechanisms; no upstream
vLLM schema field is repurposed.

### Fallback semantics

For stateless operators, route selection is:

```text
master + leaf + installed ABI + semantic eligibility
    -> candidate LightOp/AITER operator
otherwise
    -> pre-existing plugin/vLLM implementation
```

An explicit enabled route with a missing or incompatible required backend fails
with an actionable error. An ineligible shape or semantic mode uses the
pre-existing implementation.

For weight-layout-changing MoE operators, backend choice is made before weight
packing. Once a LightOp layout is installed, runtime execution may not silently
fall back to a kernel expecting the original layout. Unsupported configurations
select the existing AITER/Triton path before mutation; failures after mutation
fail closed. Dual copies of expert weights are not retained.

### Compile and custom-op boundaries

- Dynamic dimensions remain tensor or integer metadata supported by the target
  custom-op schema.
- No data-dependent Python branch or `SymBool` escapes a compiled graph.
- Fake implementations only describe output metadata and never execute real
  computation.
- Dtype, device, shape, stride, aliasing, and mutation contracts are explicit.
- Direct registration follows the installed vLLM 0.25.1 API rather than a
  copied helper from SGLang.

## Candidate-specific Design

### DeepSeek V4 sqrt-softplus routing

The LightOp route applies only when its ABI and the vLLM router semantics match:

- scoring mode is sqrt-softplus;
- correction bias is present;
- inputs have the required device, dtype, dimensionality and contiguous layout;
- expert count and top-k are supported by LightOp;
- shared-expert count and scaling semantics are representable; and
- the layer is not a hash-routing layer.

Hash layers and all unsupported cases call the original vLLM router. Returned
indices are normalized to vLLM's requested index dtype. Route evidence is
exposed through focused tests/logging without per-token log spam.

### LightOp W16A16 Marlin MoE

This route is model-agnostic and selected from layer metadata, not model names.
It requires BF16 activations and weights, supported expert/intermediate shapes,
valid LightOp configuration, supported activation semantics, and compatible
parallel/expert mapping.

The runtime owns:

- configuration lookup;
- one-time expert weight packing;
- layout generation markers;
- first and second expert GEMMs;
- fused activation; and
- expert reduction and router-weight application.

Unsupported cases preserve the existing AITER/Triton backend before weight
packing. Expert parallelism, shared experts, EPLB remapping, and CUDA Graph are
not claimed until directly tested.

### SiLU-and-multiply

Compare current LightOp, candidate AITER, and existing vLLM/Triton behavior on
model-derived BF16 shapes and any clamp/limit semantics. AITER is integrated
only if it is numerically equivalent for the supported contract and beats the
current selected implementation by the accepted threshold. Otherwise the audit
records a performance rejection and code behavior remains unchanged.

### Remaining candidates

`ep_build_m_indices`, fused norm/quant, KV-store, metadata, concatenate, and
sampling candidates are adapted only when vLLM exposes an equivalent ownership
and data-layout seam. Framework-specific scheduling or metadata kernels are
documented as not applicable instead of being forced through a brittle patch.

## Test Strategy

### L0/L1: inventory and contracts

- New patch modules satisfy the adapter inventory contract.
- Registration order, exact-target checks, idempotence, late import, master
  switch, leaf switch, missing dependency, ineligible input, and failure latch
  behavior are covered in clean processes where necessary.
- Production-boundary and patch-coverage checks remain clean.

### L2/L3: numerical accuracy

Each candidate is tested against an implementation independent from the
candidate kernel, preferably float32 PyTorch or the official vLLM path. Tests
record seed, shapes, input/output dtype, tolerances, maximum absolute/relative
error, NaN/Inf, contiguity, and input mutation.

Representative shapes come from the model configurations and include decode,
small batch, prefill, boundary and unsupported cases. Quantized routes also
check scales, saturation, zero/extreme values, and dequantized error.

An operator without model coverage can pass acceptance through these live-HCU
accuracy tests. A model smoke alone cannot pass numerical acceptance.

### Performance

- Warm up kernels before measuring.
- Synchronize the device around timed regions.
- Run repeated samples and report a robust statistic plus dispersion.
- Compare on the same idle device, process, dtype, shape and input distribution.
- Require at least 5% stable operator-level improvement on target shapes.
- Separate performance reports from correctness assertions.
- Reject or leave disabled candidates that do not meet the threshold.

### Model integration

Mandatory primary models:

- `/models/DeepSeek-V4-Flash-Channel-INT8-w8a8`
- `/models/Qwen3.6-35B-A3B`

Supplemental models are selected by operator coverage from `/models`, including
DeepSeek V4 FP8, Qwen3.5 W8A8, HY4 BF16, or dense Qwen/Gemma when relevant.
Lack of a supplemental model is not a blocker for an independently verified
operator.

For each primary model, run the same deterministic server configuration with
the relevant routes disabled and enabled. Record startup, route evidence,
generation result, throughput and peak device memory. End-to-end throughput may
not regress by more than 2% relative to the feature-off baseline.

### HumanEval-32

Run exactly 32 HumanEval samples for both primary models when their supported
server configurations start successfully. Feature-off and feature-on runs use
the same prompts, tokenizer, generation parameters, parallel topology and
dataset revision. The feature-on Pass@1 may not be lower than its same-model
feature-off baseline. Reports must prove that exactly 32 predictions were
produced and reviewed.

## Delivery

The implementation is developed on a standalone branch from `origin/v0.25.1`.
The final change contains:

- operator audit matrix;
- runtime and patch adapters for accepted candidates;
- environment-variable documentation;
- contract and live-HCU accuracy tests;
- reproducible benchmark tooling and recorded results;
- model and HumanEval-32 validation artifacts or summarized report paths; and
- one focused merge request with no unrelated history or credentials.

Before the merge request, run relevant focused tests followed by the repository
production-boundary, patch-coverage, inventory, contract, accuracy-HCU and model
acceptance commands. Any skipped layer is explicitly reported with the reason.
