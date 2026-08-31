# Unified AITER MoE Routing Design

## Context

The HCU plugin currently exposes `moe_backend=aiter`, but its implementations do
not use one routing contract:

- W16A16 forces AITER ASM through `spec_sol_type=MoeSolutionType.ASM`, performs
  an ASM-specific weight shuffle during model loading, and calls
  `fused_experts_asm_impl` directly.
- compressed-tensors INT8/FP8 W8A8 uses the public `aiter.moe` configuration and
  execution APIs, but owns a separate configuration and weight cache.
- SlimQuant/Marlin INT8 W8A8 owns another copy of the same selection and cache
  logic.
- W4A16 calls the public AITER API but falls back independently.

The installed HCU AITER package (`0.1.5+das185...`) already exposes
`MoeSolutionType.MOE_C`, `ASM`, `TRITON`, and `CK`. Its
`get_aiter_moe_config()` chooses among supported candidates when
`spec_sol_type` is omitted, and `aiter_moe()` dispatches the selected solution.
The implementation will treat that public API as the source of truth. The
internal AITER GitLab repository at
`http://42.228.13.241:10068/dcutoolkit/deeplearing/aiter/-/tree/main` is a
reference for the same contract; access credentials must never be stored in
this repository, a Git remote, a plan, or test output.

## Goals

1. Make every plugin-owned AITER MoE path use AITER's public configuration and
   execution APIs.
2. Let AITER choose ASM, MOE_C, Triton, or CK from the complete runtime problem
   description.
3. Fall back to vLLM's native Triton fused experts only when AITER explicitly
   reports that no solution exists.
4. Preserve canonical expert weights so a layer may choose different AITER
   solutions for different runtime token counts.
5. Apply one shuffle policy to all supported AITER MoE quantization types.
6. Verify routing and numerical behavior with unit tests and small synthetic
   operator tests, without loading or serving a model.

## Non-goals

- Changing explicit `moe_backend=triton`, `deep_gemm`, `dpsk_deep_gemm`, or
  other non-AITER backend selection.
- Adding a fallback for AITER import, ABI, shuffle, or kernel execution errors.
- Making AITER depend on vLLM or moving vLLM's fallback into AITER.
- Starting a vLLM server, loading a checkpoint, or running model accuracy
  evaluation.
- Expanding AITER's kernel coverage or changing its candidate priority order.

## Selected Architecture

Add a plugin-owned adapter at
`vllm_hcu/model_executor/layers/fused_moe/aiter_moe_dispatch.py`. The adapter is
the only plugin component that knows how to turn a vLLM MoE problem into the
public AITER configuration, weight-layout, expert-map, and execution contracts.
Call sites retain their vLLM-specific fallback invocation because the fallback
arguments differ by quantization method.

The runtime flow is:

```text
vLLM selected moe_backend=aiter
              |
              v
build AiterMoeProblem
(M,E,N1,N2,K,top_k,dtype,quant_type,activation,block_size,use_shuffle)
              |
              v
aiter.moe.get_aiter_moe_config()  [no spec_sol_type]
          |                                  |
    status=True                         status=False
          |                                  |
          v                                  v
prepare weights for solution          call-site vLLM Triton fallback
and normalize expert map
          |
          v
aiter.moe.aiter_moe()
```

### Adapter responsibilities

The adapter will define a typed problem description and focused helpers for:

- calling `get_aiter_moe_config()` without `spec_sol_type`;
- returning `None` only when AITER reports `status=False`; a true status paired
  with an empty or invalid config is a contract error;
- caching config results by the complete runtime problem key;
- treating the load-time `M=1` probe as an ordinary cache entry, never as a
  global capability gate for other runtime `M` values;
- preparing and caching solution-specific shuffled weights;
- preserving canonical weights for solutions whose `need_shuffle` is false;
- converting vLLM's EP expert map only when the selected AITER solution
  requires the ASM mask representation;
- calling `aiter_moe()` with common output, routing, scale, zero-point, block,
  and shuffle metadata;
- logging the selected solution once per problem key and logging framework
  fallback once per problem key.

The adapter must not catch imports, unexpected AITER return values, shuffle
errors, ABI errors, or kernel execution errors. Those are faults, not
capability misses.

### Call-site responsibilities

- `aiter_runtime.py` maps the vLLM custom-op ABI for W16A16 into the adapter.
  It removes direct ASM selection, solution-ID extraction, and direct
  `fused_experts_asm_impl` calls. When the adapter reports no solution, it calls
  vLLM's `fused_experts_impl` with the original canonical weights.
- `unquantized_fused_moe_method.py` keeps W16A16 weights canonical at load time
  and initializes the vLLM modular MoE kernel without permanently converting
  the parameters to an ASM layout. ASM guards and the
  `_hcu_aiter_moe_asm_packed` state are removed.
- `compressed_tensors_moe_runtime.py` delegates compressed-tensors INT8/FP8
  config lookup, weight preparation, EP-map conversion, and execution to the
  adapter. When AITER reports no solution it calls vLLM's Triton fused experts.
- `compressed_tensors_moe_marlin.py` delegates its explicit AITER INT8 W8A8
  path to the same adapter. Its AITER load path continues to retain canonical
  weights. A no-solution result uses vLLM Triton, not the LMSlim Marlin path.
- `patch_fused_moe.py` delegates W4A16 selection and weight preparation to the
  adapter. Its existing `original()` call is the no-solution vLLM Triton
  fallback.

## Configuration and Compatibility

### Unified shuffle switch

Introduce `VLLM_HCU_USE_AITER_MOE_SHUFFLE`, parsed as a boolean with a default
of `True`. Every plugin-owned call to `get_aiter_moe_config()` passes its value
as integer `use_shuffle=1` or `0`, including W16A16, INT8/FP8 W8A8, and W4A16.
The switch is a selection hint only. Weight transformation occurs only when the
returned config has `need_shuffle=True`.

The former W16A16-specific switch is removed rather than retained as an alias.
Only the unified switch can affect routing, and its unset default is `True`.

`VLLM_HCU_USE_AITER_MOE_CONFIG` remains parseable for one release but no longer
disables config lookup. Explicitly setting it to false emits one deprecation
warning because unified dispatch always requires an `AiterMoeConfig`.

### Error and fallback policy

The adapter distinguishes a capability miss from a fault:

- `status=False` or an empty config returned with a false status is a capability
  miss and invokes the call-site's vLLM Triton fallback.
- A true status paired with an empty/invalid config is a contract error and
  raises.
- Import failure, missing public symbols, ABI mismatch, shuffle failure,
  invalid shuffled tensors, and AITER kernel failure raise with problem and
  solution context.
- vLLM Triton errors propagate unchanged. There is no third fallback.
- Existing validation for unsupported non-default vLLM metadata remains
  fail-closed; it must not be silently reclassified as a no-solution result.

## Canonical Weights and Caching

AITER may select a different solution for different `M`, and different
solutions require different layouts. The registered layer parameters therefore
remain canonical. The adapter stores derived tensors outside the registered
parameters.

The config cache key includes:

- `M`, `E`, `N1`, `N2`, `K`, and `top_k`;
- input dtype and device;
- quantization type, activation, block size, and shuffle hint.

The derived-weight cache key includes:

- source tensor identity, `_version`, shape, dtype, and device for both weights;
- quantization type and solution type;
- a stable representation of layout-affecting config data;
- block shape and shuffle hint.

The cache is bounded. Config caches evict the least-recently-used entry beyond
128 entries and per-weight layout caches do the same beyond 8 entries. A source
weight update changes `_version` and invalidates derived layouts. Changing `M`
may select another config without mutating or replacing canonical parameters.

## EP Map Handling

The adapter keeps the solution-specific behavior currently required by HCU
AITER:

- ASM receives the validated integer mask representation expected by its
  sorter.
- MOE_C, Triton, and CK receive the original vLLM global-to-local expert map.
- No-map TP cases pass `None`.
- Shape, dtype, global expert count, and sentinel/mask expectations are
  validated before an ASM launch to prevent invalid global expert IDs from
  indexing local weights.

Framework fallback receives the original vLLM expert map, never an
ASM-converted mask.

## Logging

For each previously unseen problem key, debug logging records the AITER
`solution_type`, quantization type, `M/E/N/K/top_k`, dtype, and shuffle decision.
A no-solution fallback logs one warning per problem key with the same shape
metadata. Normal repeated calls do not emit repeated messages.

No log may include access tokens, complete weight values, or other credentials.

## Tests

### Unit and contract tests

Tests are written first and must be observed failing before implementation.
They cover:

1. W16A16, W4A16, INT8 W8A8, and FP8 W8A8 all omit `spec_sol_type` and pass the
   unified `use_shuffle` value.
2. The new shuffle variable defaults true, overrides the legacy variable, and
   accepts the legacy alias when the new variable is absent.
3. AITER configs selecting ASM, MOE_C, Triton, and CK reach `aiter_moe()` with
   solution-appropriate weights and expert maps.
4. `status=False` invokes each call site's vLLM Triton fallback with canonical
   weights and the original expert map.
5. AITER import, config-contract, shuffle, and execution exceptions never call
   the fallback.
6. Configs for different `M` values do not collide.
7. Derived weights are reused for the same source generation and invalidated
   after an in-place source update.
8. W16A16 post-load processing preserves canonical parameters and initializes
   the modular kernel.
9. Existing tests and static guards that require direct W16A16 ASM are replaced
   with unified-routing assertions.

### Synthetic HCU operator validation

No model is loaded. Small device tensors cover:

- BF16 W16A16;
- channel-wise INT8 W8A8;
- channel-wise FP8 W8A8.

For several `M` values, the validation records the AITER-selected solution and
compares unified-dispatch output with the existing vLLM Triton implementation
using a reference RMS signal floor plus dtype-appropriate NMAE, NRMSE, and
maximum-absolute-error limits. A zero-output negative control must fail the
same comparison. A fixed `gfx938` gate requires at least one supported AITER
route per target quantization so a dependency upgrade cannot turn every case
into a skip. Individual unsupported problem shapes may still skip because
framework fallback is valid for a specific `M`. The validation separately
injects or identifies a no-solution case to prove framework fallback and
injects an AITER execution error to prove fail-closed behavior.

The final verification runs the focused runtime-patch tests, the MoE scoped
suite, and the synthetic operator tests. It does not start a server, load a
checkpoint, or run a large-model test.

## Acceptance Criteria

- No plugin-owned `get_aiter_moe_config()` call forces a solution type.
- All plugin-owned AITER MoE config calls receive the unified shuffle hint.
- W16A16 no longer pre-shuffles registered parameters for ASM.
- AITER-selected ASM, MOE_C, Triton, and CK solutions use `aiter_moe()`.
- Only an explicit AITER no-solution result invokes vLLM Triton.
- AITER faults remain visible and never silently fall back.
- Explicit non-AITER backends retain existing behavior.
- Focused tests and available synthetic HCU operator tests pass without loading
  a model.
