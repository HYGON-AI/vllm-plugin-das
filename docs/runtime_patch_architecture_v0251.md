# vLLM-HCU v0.25.1 Runtime Patch Architecture

This document describes the production runtime integration between vLLM-HCU
and the HCU vLLM 0.25 release series. It covers the code that is shipped and
maintained after the source-patch migration; migration inventories, experiments,
and historical audit evidence are intentionally outside this document.

> Revision basis: final v0.25.1 behavior. The release Wheel is
> `vllm_hcu-0.25.1+das.9ae3515.dtk2604` with SHA-256
> `779e44f0c402435b5d54f4f30143bd80224f9ba27d8be4335dca7349490f7020`.
> This architecture document records the production contracts validated for
> this release; it is not a claim that every optional model or topology has
> final-Wheel model coverage.

## 1. Design goals

The runtime integration follows these rules:

1. Never rewrite files in the installed `vllm` package.
2. Preserve canonical `vllm.*` import paths for runtime consumers.
3. Register replacements before their target modules are imported.
4. Keep registration explicit and ordered; directory scanning does not define
   runtime behavior.
5. Apply each patch at most once per process and retain the first failure.
6. Bind feature-dependent patches from the Worker process's deserialized
   `vllm_config`.
7. Keep substantial HCU behavior in HCU-owned implementation modules, not in
   callback bodies.
8. Let vLLM 0.25 own shared interfaces and algorithms. Retain only the minimum
   HCU delta or a capability that the target release does not provide.
9. Treat `torch.compile`, Dynamo, vLLM piecewise graphs, and custom-op boundaries
   as runtime ABI. Relational symbolic values must not escape those boundaries.

The accepted dependency contract is installed vLLM `0.25.x`. Every plugin
entry point and both patch phases pass through the same compatibility gate in
`vllm_hcu/compatibility.py` before registering runtime changes.

## 2. Layers and ownership

### 2.1 Plugin entry points

`vllm_hcu/__init__.py` exposes three setuptools entry points:

- `hcu_platform_plugin()` selects `vllm_hcu.platforms.hcu.HCUPlatform` and arms
  the platform phase.
- `hcu_platform_register_model()` prepares lazy patches before registering HCU
  model classes.
- `hcu_platform_register_ops()` prepares lazy patches before importing HCU
  custom operators.

All entry points are safe to call repeatedly in one process. Initialization
failures are latched so a later entry point cannot continue from a partially
registered state.

### 2.2 Patch infrastructure

The modules directly under `vllm_hcu/patch/` are shared infrastructure:

| File | Responsibility |
| --- | --- |
| `__init__.py` | Dependency-light public API for platform and Worker phases |
| `import_coordinator.py` | Exact-name import finder, callbacks, replacements, reload protection, and atomic registration batches |
| `module_exchange.py` | Explicit canonical `vllm.*` to `vllm_hcu.*` whole-module replacement inventory |
| `runtime_state.py` | Process roles, patch records, idempotence, failure latching, feature state, and `patch_report()` |
| `config.py` | Validated `HcuFeatureConfig` sidecar stored under `additional_config["hcu"]` |
| `runtime_callbacks.py` | Small cross-cutting method and symbol adapters |
| `tokenizer_callbacks.py` | Lazy tokenizer compatibility adapters |

The coordinator matches complete module names. It does not replace
`builtins.__import__`, scan packages, or intercept unrelated descendants.

### 2.3 Dispatchers and adapters

`patch/platform/__init__.py` owns process-wide registration. Its `core_fix/`
adapters cover configuration, environment, parser, and registry compatibility;
its `framework_opt/` adapters cover Scheduler, multiprocessing executor, engine
output, distributed state, and KV/PD integration.

`patch/worker/__init__.py` owns Worker-only registration and feature activation.
Its adapter groups are:

- `core_fix/`: model-family compatibility.
- `op_opt/`: attention, MLA/FLA/Mamba, quantization, GEMM, AITER, MoE, and
  DeepEP integration.
- `framework_opt/`: communicators, all-to-all, DBO, speculative decoding, and
  forward-context integration.

Each dispatcher contains an explicit ordered inventory. An adapter declares its
patch ID, exact target module, validated target symbols, and application
function. Importing an adapter must not eagerly import its target module.

### 2.4 HCU-owned implementations

Callbacks should validate and connect behavior, not contain large copied
implementations. Runtime ownership is divided as follows:

- `runtime_compat/` contains focused compatibility implementations for linear
  parameters, scaled matrix multiplication, weight loading, LoRA, and vision
  prompt handling.
- `model_executor/layers/` contains HCU linear, attention, MoE, DeepEP,
  DeepGEMM, and quantization implementations.
- `models/` contains the HCU model classes and model registration table.
- `ops/` contains HCU custom operators and explicit fallbacks.
- `platforms/` contains platform discovery, device policy, and environment
  configuration.
- `v1/` contains the HCU Worker, model runner, attention stack, Scheduler,
  multiprocessing executor, KV-cache integration, and speculative decoding
  runtime.

## 3. Runtime lifecycle

### 3.1 Platform phase

During vLLM plugin discovery, `apply_platform_patches()` performs this ordered
sequence under a process-local lock:

1. Validate the installed vLLM release series.
2. Detect and record the current process role.
3. Install the exact import coordinator.
4. Atomically register cold Worker replacements and the complete module
   exchange inventory.
5. Register platform core, tokenizer, runtime-method, and framework callbacks.
6. Drain callbacks whose target modules were already imported safely.

Whole-module replacements are published before callbacks. Registration uses an
atomic coordinator batch so another thread cannot observe a partially visible
replacement inventory.

### 3.2 Worker phase

`vllm_hcu/v1/worker.py::HcuGPUWorker.__init__` calls
`apply_worker_patches(vllm_config)` before the upstream Worker constructor. This
is the last reliable point before model-runner, communicator, and custom-op
imports begin.

The Worker phase:

1. Re-applies the idempotent platform phase for direct Worker construction.
2. Marks the authoritative process role as `Worker`.
3. Reads and validates `additional_config["hcu"]` after serialization.
4. Rebinds the CompilationConfig sidecar in the spawned process.
5. Arms the ordered Worker callback inventory.
6. Resolves feature flags and records whether each patch is enabled.
7. Raises any previously latched or immediately detected required failure.

After model loading, `validate_worker_patches()` requires enabled terminal patch
chains to have reached the applied state. Feature-off and model-specific targets
that were never imported may remain armed.

## 4. Import actions

The coordinator supports two production integration mechanisms.

### 4.1 Whole-module replacement

`module_exchange.py` maps an exact canonical module to an HCU-owned module. For
example:

```text
vllm.v1.attention.ops.deepseek_v4_ops.fused_compress_quant_cache
    -> vllm_hcu.v1.attention.ops.deepseek_v4_ops.fused_compress_quant_cache
```

Consumers import the canonical `vllm.*` name. The coordinator resolves that
name to the HCU implementation while maintaining one canonical module identity.
The replacement is lazy: registration does not import either module.

Replacements use a fail-closed late policy. If the canonical upstream module
was imported before its replacement was armed, initialization fails instead of
silently mixing upstream and HCU implementations.

Use whole-module replacement when HCU owns the module implementation or when
module-level registration and identity must remain atomic.

### 4.2 Post-import callback

A callback runs against one exact target module after its loader completes. If
the target already exists and is safe to inspect, registration applies the
callback immediately. Target shape and callable signatures are validated before
mutation.

Use a callback for a small method replacement, registry entry, exported symbol,
or feature selection that does not justify owning the complete module.

Callbacks must be idempotent and should install version-neutral markers using
the `_vllm_hcu_*` prefix when a marker is necessary.

## 5. Configuration contract

HCU-only settings are represented by `HcuFeatureConfig` and stored in:

```python
vllm_config.additional_config["hcu"]
```

This sidecar avoids adding fields to upstream vLLM dataclasses and preserves
serialization across Main, EngineCore, and Worker boundaries. Configuration is
normalized at the CLI/config boundary and validated again after Worker
deserialization. Unknown keys and invalid values fail early.

Patch registration and feature activation are separate. A callback may be
armed for import-order safety while its feature state is disabled; disabled
callbacks do not become required terminal patches.

## 6. Safety and backend ownership contracts

### 6.1 MLA prefix-cache policy

HCU MLA plus prefix caching is not enabled for the v0.25.1 release contract. The
public HCU platform configuration hook uses the target-owned
`ModelConfig.use_mla` signal and sets the shared
`VllmConfig.cache_config.enable_prefix_caching` to `False` before EngineCore,
Scheduler, KV-cache, executor, or Worker construction and serialization. It
emits one stable warning before EngineCore observes the final value.

This decision is process-wide configuration policy. Launch wrappers, model-path
matching, request sidecars, scheduler reordering, or Worker-local changes must
not substitute for it. Every consumer must observe the same final
`VllmConfig`.

### 6.2 Channel quantization ownership

The accepted v0.25.1 Channel-FP8 product route keeps the target release's
public backend selection as the pure-TP compute owner:

- compressed-tensors selects
  `ChannelWiseTorchFP8ScaledMMLinearKernel` for Channel-FP8 linear layers;
- the HCU scaled-mm boundary validates HCU layout and metadata, then delegates
  the dense calculation to target vLLM `triton_scaled_mm`;
- pure-TP FP8 MoE explicitly selects the target `triton` or `aiter` backend;
- non-DeepEP `auto` delegates to the target vLLM oracle;
- DP+EP `deepep_auto` selects the HCU contiguous/masked DeepGEMM experts.

`patch_scaled_mm_linear_kernel.py` adds the narrow prequantized-input bridge to
the reviewed target kernel. A supplied `(quantized_activation, scale)` pair
bypasses duplicate activation quantization and is forwarded through the target
scaled-mm owner. The adapter reports this capability only on the reviewed HCU
Channel-FP8 target class; unsupported kernels keep upstream behavior or fail at
the compatibility gate instead of consuming the tuple silently.

Channel-INT8 remains a separate contract. Pure TP supports the target Triton
backend and the HCU explicit-AITER extension; DP+EP uses the HCU
contiguous/masked DeepGEMM experts. Block/PERBLOCK fallback is not equivalent
to the reviewed channel/token W8A8 route.

### 6.3 Dynamic-shape and custom-op boundaries

This release treats graph boundaries as ABI. Python validation must not construct,
return, or thread `torch.SymBool`, SymPy `Equality`, or another relational
Boolean across a vLLM piecewise or HCU custom-op boundary. In particular,
membership tests and scale-shape checks involving dynamic token counts cannot
be allowed to become scalar graph outputs.

The approved pattern is:

1. Preserve friendly type, rank, layout, dtype, device, and scale-shape errors
   for concrete eager calls.
2. During compilation, avoid constructing relational symbolic values in the
   outer Python wrapper.
3. Keep the custom-op fake implementation limited to output tensor
   shape/dtype/device/stride metadata; it must not perform real computation or
   emit a relational Boolean.
4. In the real custom-op implementation, repeat the required contract checks
   with concrete runtime dimensions before calling target vLLM Triton compute.
5. Validate the returned tensor's exact shape, dtype, and device before it
   crosses back through the boundary.

This rule applies both to `vllm_hcu/runtime_compat/scaled_mm.py` and the
prequantized-input adapter in
`vllm_hcu/patch/worker/op_opt/patch_scaled_mm_linear_kernel.py`. A fix is not
complete with eager unit tests alone: focused coverage must use multiple token
counts in one strict-dynamic graph, reject symbolic Boolean graph values, run a
real standalone Inductor compile, and then pass the exact model's compile and
cudagraph gate. The T4 validation run exercised this path through full model startup and
GSM8K-100.

### 6.4 Native resource ownership

Conditionally allocated device, IPC, stream, event, or workspace resources are
initialized to inert values. Ownership is recorded only after successful
acquisition, and teardown releases only resources owned by that instance.
Kernel or collective success without clean construction and teardown is not a
passing lifecycle contract.

## 7. Failure and observability

`PATCH_REGISTRY` stores one record per patch ID, including:

- process ID and role;
- exact target names;
- armed, applied, skipped, or failed status;
- feature-enabled state;
- failure type and detail.

Registration and application are idempotent. The first compatibility or
application failure is retained and re-raised on later lifecycle entry, so the
process cannot retry a partially completed custom-op or registry mutation.

`patch_report()` exposes the registry for diagnostics. `vllm-hcu-doctor`
performs read-only checks for version compatibility, installed source
integrity, plugin entry points, and package metadata.

## 8. Extension guidelines

When adding a runtime integration:

1. Place substantial behavior in the owning HCU runtime package.
2. Choose an exact whole-module replacement or a small post-import callback.
3. Define a stable patch ID and explicit target inventory entry.
4. Validate required classes, functions, attributes, and signatures before
   changing the target.
5. Keep imports of the target module out of dispatcher and adapter module scope.
6. Make application idempotent and allow failures to propagate to the shared
   registry.
7. Bind feature-dependent behavior through `HcuFeatureConfig` rather than new
   upstream config fields.
8. Add focused tests for ordering, late import behavior, idempotence, target
   drift, feature-off behavior, and failure latching.
9. For dynamic shapes, add strict-dynamic and standalone-compile tests that
   prove no relational symbolic value crosses a piecewise/custom-op boundary.
10. For quantized routes, test backend selection and positive route evidence;
    an import-only or output-shape-only test is insufficient.
11. For native ownership, test both successful operation and natural teardown
    on every required topology.

The production boundary check in `tools/check_production_boundary.py` should
remain clean after every change.
