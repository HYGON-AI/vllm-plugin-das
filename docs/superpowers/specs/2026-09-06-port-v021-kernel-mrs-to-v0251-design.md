# Port v0.21 Kernel MRs to v0.25.1 Design

## Goal

Port the behavior introduced by vLLM-HCU merge requests 270, 271, 272,
and 273 from the v0.21 plugin to the `v0.25.1` branch of
`HYGON-AI/vllm-plugin-das`. The result must preserve the v0.25.1 sidecar
architecture, default every newly introduced optimization switch to enabled,
and keep a working upstream fallback for each optional HCU dependency.

## Source Changes

The source behavior is defined by these v0.21 commits:

| MR | Commit | Behavior to port |
| --- | --- | --- |
| 270 | `c1b95f70bf179a954d7205922fb575cf2cfdacc4` | Route Qwen GDN sigmoid-gating recurrent updates through AITER when available. |
| 271 | `14a680b5b3c4ce83dce3adae7e61aa2b772996ab` | Route supported Qwen GDN decode convolution updates through `causal_conv1d`. |
| 272 | `a4c14b2f67d75003e3db9cd043b989aa9673f86a` | Prefer AITER HIP for `chunk_gated_delta_rule_fwd_h`, with lower-tier fallbacks. |
| 273 | `f82655b0c6c6fd2c8988c26308d9fefa6c0d1fb6` and `85d01aebf9620db36db28cc83dba616204f4be47` | Prefer AITER HIP for `chunk_fwd_o` and remove unrelated environment variables. |

The v0.21 text-replacement patches will not be copied. v0.25.1 uses exact
post-import callbacks, so each behavior will be implemented in the existing
target-native adapters.

## Scope

The implementation will change only the v0.25.1 plugin repository. It will
not modify the vLLM source checkout, AITER, `causal_conv1d`, model files, or
system Python installation.

Expected production files are:

- `vllm_hcu/platforms/envs.py`
- `vllm_hcu/patch/worker/op_opt/patch_fla_chunk_delta_h.py`
- `vllm_hcu/patch/worker/op_opt/patch_fla_chunk_o.py`
- `vllm_hcu/patch/worker/op_opt/patch_gdn_causal_conv1d.py`
- `vllm_hcu/patch/worker/op_opt/patch_gdn_linear_attention.py`

Tests will extend the existing environment-routing, FLA/Mamba adapter, and
real-v0.25.1 ownership suites. No unrelated refactor is in scope.

## Environment Contract

The three existing parent switches remain unchanged:

- `VLLM_HCU_USE_CUSTOM_OPS`, default enabled.
- `VLLM_HCU_USE_CUSTOM_AITER_FLA`, default enabled.
- `VLLM_HCU_USE_CUSTOM_CAUSAL_CONV1D`, default enabled.

The port introduces the following switches, all defaulting to enabled (`1` or
`True` semantics) and accepting the repository's existing `1`/`true` Boolean
forms:

| Variable | Purpose | Parent gate |
| --- | --- | --- |
| `VLLM_HCU_USE_AITER_FUSED_SIGMOID_GATING_DELTA_RULE_UPDATE` | Select the AITER sigmoid-gating update for compatible Qwen GDN calls. | `VLLM_HCU_USE_CUSTOM_OPS` |
| `VLLM_HCU_USE_AITER_CHUNK_GATED_DELTA_RULE_HIP` | Prefer the AITER HIP state-update kernel. | `VLLM_HCU_USE_CUSTOM_OPS` and `VLLM_HCU_USE_CUSTOM_AITER_FLA` |
| `VLLM_HCU_USE_CHUNK_FWD_KERNEL_O` | Prefer the AITER HIP output kernel. | `VLLM_HCU_USE_CUSTOM_OPS` |

`VLLM_HCU_USE_AITER_CHUNK_GATED_DELTA_RULE_HIP` does not control `chunk_o`,
matching the cleanup in MR 273. Explicitly setting any selector to `0` or
`false` disables only that route and preserves lower-tier behavior.

## Runtime Design

### Qwen GDN sigmoid gating

`patch_gdn_linear_attention` will continue to own only symbols local to
`vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn`. It will wrap the
module-local `fused_sigmoid_gating_delta_rule_update` while leaving the
canonical FLA export and all non-Qwen consumers untouched.

When both the global custom-op switch and the new sigmoid switch are enabled,
the wrapper selects AITER's top-level HIP
`vllm_fused_sigmoid_gating_delta_rule_update` when it imports and `A_log` has
the same dtype as `q`, as required by that kernel. A dtype-incompatible HIP
call falls through to AITER's Triton
`fused_sigmoid_gating_delta_rule_update`, which supports v0.25.1's FP32
`A_log` with BF16 activations. If no compatible AITER callable imports, the
wrapper calls the original vLLM function. Import availability is resolved
without mutating the canonical module. Once an AITER callable is selected,
runtime errors from it are propagated rather than silently hiding kernel
failures.

The existing v0.25.1 fused decode path and NN-layout wrapper remain intact.
The new wrapper composes with those behaviors instead of replacing the class
or copying `_forward_core`.

### Qwen GDN causal convolution

`patch_gdn_causal_conv1d` will preserve its current NN-layout normalization
and add a local routing layer around `causal_conv1d_update`. The external HCU
call accepts `x`, state, weight, bias, activation, and state indices. Calls
using only that compatible decode contract route to the external kernel when
both existing causal-convolution gates are enabled. Calls carrying v0.25.1
features unsupported by the external API—such as accepted-token or query
location metadata—remain on the original vLLM implementation.

This keeps speculative or future call semantics correct while accelerating
the two decode forms represented by MR 271. `causal_conv1d_fn` remains scoped
to its current NN-layout adaptation. The canonical causal-convolution module,
Kimi, Olmo, MambaMixer, MambaMixer2, and ShortConv remain unmodified.

### FLA state-update routing

`patch_fla_chunk_delta_h` will use this priority order when custom ops are
enabled:

1. AITER HIP `chunk_gated_delta_rule_fwd_vllm_hip_blockdim64` when its new
   selector and `VLLM_HCU_USE_CUSTOM_AITER_FLA` are enabled and the callable
   imports successfully.
2. Existing AITER Triton launch helper when
   `VLLM_HCU_USE_CUSTOM_AITER_FLA` is enabled and the callable imports.
3. The original vLLM 0.25.1 function.

The adapter will reuse vLLM-prepared sequence indices and offsets and preserve
the caller's chunk size, final-state request, new-value request, and state
layout. Missing optional imports select the next tier. Shape, dtype, or launch
errors raised after selection are not caught.

### FLA output routing

`patch_fla_chunk_o` will use this priority order when custom ops are enabled:

1. AITER HIP `chunk_fwd_o_vllm_hip_blockdim64` when its selector is enabled
   and the callable imports successfully.
2. Existing AITER Triton launch helper when
   `VLLM_HCU_USE_CUSTOM_AITER_FLA` is enabled and the callable imports.
3. The original vLLM 0.25.1 function.

The HIP call will receive the caller's `chunk_size`, scale, sequence metadata,
and state-layout contract. The Triton route will retain support for the
caller's preallocated `core_attn_out`. If the HIP route is selected while a
preallocated output exists, its returned tensor will be copied into that
buffer when needed so the v0.25.1 output-buffer contract remains observable.
Missing imports fall through; selected-kernel runtime errors propagate.

## Error Handling and Compatibility

- Exact v0.25.1 target signatures remain checked before adapters are armed.
- Applying any adapter twice remains idempotent.
- Optional dependency absence causes deterministic fallback, not startup
  failure.
- An incompatible target vLLM signature raises the existing
  `PatchCompatibilityError` during patch application.
- Environment access remains lazy through `vllm_hcu.platforms.envs`.
- The change does not monkey-patch shared canonical modules.

## Test Strategy

Implementation follows red-green-refactor. Before production changes, tests
will demonstrate failures for:

- all three new switches being enabled by default and disabled by `0`;
- HIP-first, AITER-Triton-second, and vLLM fallback routing for both FLA
  functions;
- preservation of keyword arguments, return tuples, and preallocated output;
- Qwen-local sigmoid selection and safe fallback without changing canonical
  FLA exports;
- Qwen-local compatible causal-convolution routing, unsupported-call fallback,
  NN-layout normalization, and non-Qwen ownership boundaries;
- idempotent adapter application and clear target-signature rejection.

Verification will include:

1. Focused environment, FLA/Mamba, and GDN ownership tests against the local
   vLLM 0.25.1 source checkout.
2. The repository contract suite via `tools/run_patch_tests.py`.
3. Static formatting and compilation checks used by the repository.
4. A real HCU smoke inference using `/models/Qwen3.5-35B-A3B`, with new
   switches left unset to prove their default-enabled behavior. The test must
   load the model, generate non-empty deterministic output, and show no kernel
   or patch compatibility exception in its log.

The pre-change focused baseline is 105 passed tests with zero failures when
`VLLM_V0251_SOURCE_ROOT=/models/vllm_0251` is supplied.

## Delivery and Review

Work will be committed on `feat/port-021-kernel-mrs-v0251` with author
`alexanderbin123 <1414695739@qq.com>`. The branch will be pushed to
`HYGON-AI/vllm-plugin-das` and opened as one pull request targeting
`v0.25.1`, with source MR/commit mapping and exact verification evidence in
the description. An independent code review will inspect the complete diff;
all critical and important findings will be resolved before final handoff.
