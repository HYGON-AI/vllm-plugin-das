# vLLM v0.25 HCU AITER Quantized W8A8 MoE Design

## Goal

Enable compressed-tensors FP8 W8A8 and INT8 W8A8 MoE models to select the
HCU AITER expert backend with `--moe-backend aiter` on vLLM v0.25.1, while
preserving the v0.25 modular-kernel lifecycle and the existing
`slimquant_marlin` comparison path.

The acceptance models are:

- INT8: `/models/Qwen3.5-35B-A3B-W8A8`
- FP8: `/models/Qwen3.5-35B-A3B-CHANNEL-FP8`

## Current State

vLLM v0.25.1 already maps explicit FP8 `aiter` selection to `AiterExperts`,
but the HCU plugin rejects channel-weight/dynamic-token FP8 unless the user
selects Triton. The target FP8 load-time conversion also calls the legacy
`rocm_aiter_ops.shuffle_weights` interface instead of the installed HCU
`aiter.moe` configuration and shuffle APIs.

The target INT8 oracle has no AITER enum or mapping. Explicit
`--moe-backend aiter` therefore fails in
`CompressedTensorsW8A8Int8MoEMethod.__init__` before weights load with:

```text
moe_backend='aiter' is not supported for Int8 MoE.
Expected one of ['triton', 'humming'].
```

The plugin's `compressed_tensors_moe_marlin.py` contains an older direct
AITER INT8 path gated by `VLLM_ROCM_USE_AITER` and
`VLLM_ROCM_USE_AITER_MOE`. That path belongs to the custom
`slimquant_marlin` quantization method, does not implement the official
`--moe-backend aiter` contract, and bypasses v0.25 backend selection.

## Selected Architecture

Keep compressed-tensors methods target-owned. Extend the v0.25 backend and
expert boundaries through plugin sidecar patches:

1. An INT8 oracle patch adds an `AITER` member, maps the explicit runner
   backend, returns `AiterExperts`, and preserves canonical INT8 weights at
   load time.
2. The existing FP8 oracle patch preserves canonical FP8 weights for AITER
   rather than applying the legacy shuffle.
3. The channel-FP8 adapter accepts explicit `aiter` and `triton`, then verifies
   that the target selected the matching backend.
4. The ROCm AITER expert adapter advertises the INT8 channel/token quant key
   and intercepts only FP8-W8A8 or INT8-W8A8 calls.
5. A plugin-owned quantized runtime translates `FusedMoEQuantConfig` into the
   installed `aiter.moe.get_aiter_moe_config`,
   `aiter_moe_shfl_weight`, and `aiter_moe` ABI.

The target `AiterExperts.apply()` remains responsible for the v0.25 output
workspace contract. Its existing `set_`/`copy_` logic consumes the tensor
returned by the patched `rocm_aiter_fused_experts` wrapper. Prepare/finalize,
routing, expert-parallel maps, shared-expert scheduling, and graph ownership
remain in the target modular kernel.

## Why `aiter_runtime.py` Is Not Modified

`vllm_hcu/model_executor/layers/fused_moe/aiter_runtime.py` is the compatibility
layer for the target `_aiter_ops` ABI and unquantized W16A16 dispatch. The new
FP8/INT8 API is selected using `FusedMoEQuantConfig`, which is available at the
`rocm_aiter_fused_experts` boundary but is no longer available after lowering
to `_aiter_ops.fused_moe`.

Quantized configuration, scales, zero points, and public weight shuffle
therefore stay in
`vllm_hcu/model_executor/layers/quantization/compressed_tensors_moe_runtime.py`.
This also keeps the change independent of the W16A16 AITER branch and MR.

## Backend Selection

### INT8

The HCU INT8 oracle adds `Int8MoeBackend.AITER` without changing the target's
automatic priority order. Only explicit `--moe-backend aiter` selects it.
`triton`, `humming`, `cpu`, and `auto` preserve target behavior.

For AITER, `convert_to_int8_moe_kernel_format` returns the canonical weights
unchanged. Runtime configuration decides whether a selected solution requires
shuffle.

### FP8

The target already has `Fp8MoeBackend.AITER`. For channel-weight,
dynamic-token compressed tensors, the plugin requires explicit
`--moe-backend aiter` or `--moe-backend triton`; `auto` remains rejected so an
environment flag cannot silently change the precision path.

The HCU FP8 oracle returns canonical weights unchanged for AITER. Other FP8
backends, including DPSK DeepGEMM and Triton, retain their existing conversion.

## Quantized Runtime Contract

The runtime accepts the full v0.25 expert call: hidden states, canonical
weights, top-k tensors, `FusedMoEConfig`, activation, expert map,
`FusedMoEQuantConfig`, optional prequantized activation scale, and output dtype.

It enforces:

- FP8 uses `MoeQuantType.FP8_W8A8`.
- INT8 uses `MoeQuantType.W8A8`.
- Both weights and both weight scales must be tensors.
- Top-k weights and IDs are matching rank-2 tensors.
- `apply_router_weight_on_input=True` is rejected for these HCU paths.
- Unsupported quant configs delegate to the audited target function rather
  than being reinterpreted.

The AITER config cache key includes token count, expert dimensions, top-k,
activation dtype, activation name, and quant type. A solution requesting
shuffle uses `aiter_moe_shfl_weight`; the result is cached on the canonical
weight tensor using both tensor identity/generation and solution identity.
Weight replacement or in-place reload invalidates the cache.

The AITER call uses `inplace=False`, float32 top-k weights, int32 top-k IDs,
the v0.25 expert map/`FusedMoEConfig.num_experts`, weight and activation scales, the
selected shuffle flag, and the requested output dtype.

## Error Handling

Explicit AITER selection fails early with an HCU-specific error if:

- the installed `aiter.moe` API or required quant enum is absent;
- no solution exists for the runtime shape;
- scales or tensor layouts are incomplete;
- a shuffle helper returns incompatible tensors;
- the target selects a backend different from the explicit request; or
- a currently unsupported routing contract is requested.

There is no silent fallback from explicit AITER to Triton or
`slimquant_marlin`.

## Testing and Acceptance

### CPU-safe contract tests

- INT8 oracle maps explicit AITER, resolves `AiterExperts`, and leaves weights
  canonical.
- Automatic and non-AITER INT8 selection retain target behavior.
- FP8 explicit AITER is accepted and its old load-time shuffle is bypassed.
- FP8 explicit Triton remains selectable.
- `AiterExperts` advertises the INT8 channel/token quant pair.
- Runtime maps FP8 and INT8 to the correct AITER quant type and forwards exact
  scales, expert maps, output dtype, and activation.
- Runtime caches configs and shuffled weights and invalidates them on weight
  generation changes.
- Unquantized and unrelated quant schemes still call the target implementation.

### Hardware operator comparison

Run deterministic small-shape FP8 and INT8 expert inputs through HCU AITER and
the target Triton expert implementation. Compare output shape, finiteness, and
numerical error using tolerances established from the actual quantized output;
record maximum and mean absolute error rather than accepting only kernel
completion.

### Full-model comparison

For each Qwen3.5 model, run the same deterministic prompt through graph-enabled
services and verify non-empty, non-garbled output. Run EvalScope HumanEval with
32 samples, thinking disabled and temperature zero.

INT8 compares:

- compressed-tensors plus `--moe-backend aiter`
- `--quantization slimquant_marlin`

FP8 compares:

- compressed-tensors plus `--moe-backend aiter`
- compressed-tensors plus `--moe-backend triton`

Acceptance requires successful startup, logs proving the requested expert
backend, no NaN/Inf or device fault, and no unexplained HumanEval pass@1
regression relative to the paired baseline. Raw scores and commands are
reported; a difference is not hidden by changing generation parameters.

## Branch and MR Boundary

Implement on `fix/hcu-v025-aiter-w8a8-quantized`, based directly on remote
`fix/glm51-pp-mtp-mrv2`. Do not include the separate W16A16 branch. The final
MR contains only quantized AITER backend integration, tests, documentation,
and captured validation commands/results.

## Out of Scope

- Changing automatic INT8 backend priority.
- Replacing or deleting `slimquant_marlin`.
- Copying the v0.21 compressed-tensors `apply()` implementations.
- Modifying upstream `/models/zb/vllm_025/vllm` sources.
- Extending AITER to INT8-W8A16, FP8-W8A16, INT4, MXFP4, or unsupported
  activation/routing layouts.
