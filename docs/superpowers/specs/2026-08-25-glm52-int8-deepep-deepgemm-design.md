# GLM-5.2 INT8 DeepEP and DeepGEMM Design

## Context

The target is `/models/GLM-5.2-Channel-INT8-w8a8`, a
`GlmMoeDsaForCausalLM` checkpoint with 256 routed experts, top-k 8, SiLU
activation, channel-wise symmetric INT8 weights, and dynamic per-token INT8
activations. The implementation must live in the `vllm-plugin-das` repository
and be based directly on remote branch `v0.25.1`.

The v0.25.1 plugin already contains the required building blocks:

- DeepEP high-throughput and low-latency prepare/finalize adapters;
- INT8 dispatch and quantization handling for both DeepEP modes;
- `DeepGemmExperts` for standard/contiguous activation layout;
- `BatchedDeepGemmExperts` for batched/masked activation layout; and
- HCU DeepGEMM and LightOP INT8 grouped-GEMM calls.

The missing connection is the v0.25.1 INT8 MoE oracle. Its HCU adapter only
registers AITER. Consequently, an explicitly requested
`moe_backend=dpsk_deep_gemm` cannot select or construct the existing INT8
DeepGEMM experts for a compressed-tensors W8A8 checkpoint.

## Goals

1. Make explicit HCU `dpsk_deep_gemm` selection work for channel/token INT8
   W8A8 MoE under DeepEP high-throughput mode.
2. Make the same explicit selection work under DeepEP low-latency mode.
3. Preserve the existing default, Triton, Humming, CPU, and AITER INT8 backend
   behavior.
4. Fail closed when `dpsk_deep_gemm` is explicitly selected for an unsupported
   quantization, activation, device, or parallel configuration.
5. Verify correctness and performance with the real GLM-5.2 checkpoint on
   eight HCUs.

## Non-goals

- No changes to the DeepEP or DeepGEMM binary wheels.
- No GLM-specific model implementation or model-name conditional.
- No automatic selection of the HCU DeepGEMM path when `moe_backend=auto`.
- No DeepEP auto-mode, DCP, PCP, MTP, multi-node, or disaggregated-serving work.
- No guaranteed performance-improvement percentage. Measurements will be
  reported against a default-backend reference using the same model and
  request shape.

## Selected Approach

Extend the HCU adapter for `vllm.model_executor.layers.fused_moe.oracle.int8`
instead of backporting the v0.21 compressed-tensors method override.

The adapter will add a distinct HCU INT8 backend enum member for
`dpsk_deep_gemm`, map the sidecar request to it, and participate in the target
oracle's normal support-checking flow. The backend exposes two existing expert
implementations:

- `DeepGemmExperts`: standard activation format used by DeepEP
  high-throughput;
- `BatchedDeepGemmExperts`: batched activation format used by DeepEP
  low-latency.

The target oracle will still own prepare/finalize construction and kernel
assembly. HCU supplies backend registration, selection, its W8A8 quant-config
adapter, and the DeepGEMM architecture-aware canonical INT8 weight packing.
This keeps the implementation aligned with v0.25.1's modular MoE architecture.

The alternatives were rejected:

1. Backporting the v0.21 compressed-tensors class replacement would bypass
   the v0.25.1 oracle and duplicate selection logic.
2. Adding a GLM model-specific branch would couple a generic INT8 MoE backend
   to one architecture and leave other compatible channel-W8A8 models broken.

## Runtime Configuration

Both deployments use eight data-parallel ranks with expert parallelism. Dense
weights remain data-parallel while routed experts are partitioned across the
EP group.

High-throughput mode:

```text
--data-parallel-size 8
--enable-expert-parallel
--all2all-backend deepep_high_throughput
--moe-backend dpsk_deep_gemm
```

Low-latency mode:

```text
--data-parallel-size 8
--enable-expert-parallel
--all2all-backend deepep_low_latency
--moe-backend dpsk_deep_gemm
```

Memory utilization, maximum model length, maximum sequences, DeepEP SM count,
and eager/compiled execution may be tuned during real-model diagnosis. Those
operational values will be recorded in the merge request, not hard-coded into
the plugin.

## Backend Selection and Data Flow

1. The EngineArgs adapter normalizes `dpsk_deep_gemm` into the HCU sidecar
   while leaving vLLM's official `moe_backend` value as `auto`.
2. The compressed-tensors INT8 method asks the INT8 oracle to select a backend
   for `kInt8StaticChannelSym` weights and `kInt8DynamicTokenSym` activations.
3. The HCU oracle adapter sees the explicit sidecar value and evaluates the
   existing DeepGEMM expert classes with the target oracle's
   `is_supported_config` contract.
4. DeepEP high-throughput yields standard/contiguous expert activations and
   selects `DeepGemmExperts`.
5. DeepEP low-latency yields batched expert activations and selects
   `BatchedDeepGemmExperts`.
6. The existing expert classes invoke HCU LightOP grouped W8A8 GEMM and fused
   SiLU/multiply/requantization kernels, then the existing DeepEP finalizer
   returns routed outputs to their originating ranks.

## Compatibility and Failure Handling

- Explicit `dpsk_deep_gemm` selection must never silently fall back to AITER,
  Triton, or another backend.
- Unsupported configurations must raise a message containing the rejected
  backend and the expert support reason.
- Existing AITER registration and format conversion remain unchanged.
- The patch remains idempotent and validates every wrapped target symbol and
  signature before mutation, following the plugin's runtime-patch contract.
- Optional DeepGEMM/LightOP imports remain lazy so unrelated deployments do
  not require these packages at plugin import time.

## Test Strategy

### Automated contract tests

Tests will be written before production changes and will cover:

- enum extension and `dpsk_deep_gemm` mapping;
- exact preservation of existing AITER mapping;
- explicit selection of `DeepGemmExperts` for standard activation format;
- explicit selection of `BatchedDeepGemmExperts` for batched activation
  format;
- rejection of unsupported quantization or activation formats;
- canonical INT8 weight conversion for the HCU backend;
- construction of a normal v0.25.1 modular MoE kernel; and
- idempotence and target-signature compatibility.

Focused runtime-patch, configuration, worker-dispatch, and module-exchange
tests will run after each change. The full plugin test suite will run before
the merge request is created.

### Real-model correctness

For each DeepEP mode:

1. Start the model using eight HCUs and confirm logs identify DeepEP plus the
   intended INT8 DeepGEMM expert class.
2. Run a deterministic completion and compare text, token IDs, and available
   log probabilities with the default-backend reference.
3. Run at least four concurrent requests and require every request to complete
   successfully without worker or collective errors.
4. Run the first 32 HumanEval examples with temperature 0, sampling disabled,
   thinking disabled, and Pass@1 evaluation.

### Performance characterization

Using identical prompts and generation lengths, measure:

- aggregate output-token throughput;
- request throughput;
- time to first token (TTFT); and
- time per output token (TPOT).

The default backend, DeepEP high-throughput plus DeepGEMM, and DeepEP
low-latency plus DeepGEMM will be measured after warmup. High-throughput will
be exercised at concurrent load; low-latency will be exercised at concurrency
one and a small concurrent load. Results, command lines, package versions, and
known cold-start/JIT effects will be included in the merge request.

## Delivery and Review

The work is isolated on branch `feat/glm52-deepep-deepgemm-v0251`, forked from
remote `v0.25.1`. Only INT8 DeepGEMM/DeepEP integration, its tests, and concise
validation documentation belong in the diff. After fresh verification, every
changed line will be reviewed for correctness, compatibility, unnecessary
scope, and security before a single merge request is opened against
`v0.25.1`.
