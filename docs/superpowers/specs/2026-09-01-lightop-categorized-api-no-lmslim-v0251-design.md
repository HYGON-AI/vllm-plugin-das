# LightOp Categorized API and LMSlim Removal for v0.25.1

## Context

The `v0.25.1` branch of `vllm-plugin-das` calls LightOp through a mixture of
top-level exports, `lightop.op`, `lightop.gemmopt`, and four LMSlim-backed
imports. LMSlim is being retired as an external runtime dependency, while the
installed LightOp 0.6 package exposes the migrated implementations through
categorized modules.

This design is based on:

- `/models/Lightop&Lmslim.md`, synchronized on 2026-07-22;
- installed LightOp
  `0.6.0+das.dtk2604.torch2110.2608171227.g8c835c`, including its public
  category `__all__` exports and Python call signatures;
- pull request #27 (`refactor: adapt LightOp 0.6 categorized APIs`) at
  `571bc4c`, used as a tested reference rather than copied without review;
- current `origin/v0.25.1` at `85a4ad5`, which contains call sites added after
  pull request #27 was opened.

The clean baseline contract suite is 1162 passed, 49 deselected, and 14
third-party Torch deprecation warnings.

## Goals

1. Use `lightop.<category>.<name>` at every production LightOp call site.
2. Remove every production import and call of the external LMSlim package.
3. Preserve the existing runtime ownership and lazy-import boundaries.
4. Adapt ABI changes instead of treating the migration as import-only.
5. Expose plugin-owned LightOp tuning through `VLLM_HCU_*` environment names.
6. Cover both pull request #27 and LightOp users added later on `v0.25.1`.
7. Verify the result with portable tests, live HCU kernel tests, and
   `/models/Qwen3.5-35B-A3B-W8A8`.
8. Deliver one new pull request targeting `v0.25.1`.

## Non-goals

- Do not modify or rebuild the installed LightOp package.
- Do not remove `das-install lmslim` from the Dockerfile in this pull request.
- Do not rename LightOp's internal `_lmslim_native` modules or
  `torch.ops.lmslim` registration namespace; these are implementation details
  owned by LightOp, not imports or calls made by this plugin.
- Do not redesign unrelated quantization, attention, MoE, or patch lifecycle
  behavior.
- Do not retain external LMSlim as a runtime fallback.

## Integration Strategy

The work starts from the latest `origin/v0.25.1`. The test-first commits from
pull request #27 are replayed where they still match the latest branch, and
their production commits are then replayed with conflicts resolved against
the current owners. New tests are written before adapting call sites added
after pull request #27.

The resulting branch must not rely on pull request #27 being merged. It will
contain the complete categorized API migration and target `v0.25.1` directly.

## Runtime Architecture

### Strict public API boundary

Production code imports only from a documented category:

- activation and gating: `lightop.activation`;
- attention, sparse MQA, QKV, and RoPE: `lightop.attention`;
- GEMM and grouped GEMM: `lightop.gemm_ops`;
- MoE routing, EP, and Marlin experts: `lightop.moe`;
- normalization: `lightop.norm`;
- quantization: `lightop.quant`;
- sampling: `lightop.sampling`;
- tensor helpers: `lightop.tensor`.

Required optimized paths fail with an owner-specific error if the categorized
symbol is unavailable. They do not retry an old LightOp path or LMSlim after
an import, signature, kernel, dtype, shape, or device failure.

Existing algorithmic fallbacks remain valid when they are part of the
feature's design rather than compatibility with an obsolete API. Examples
include using `torch.cat` when the optional LightOp concatenation helper is
unavailable and retaining existing DeepGEMM or Triton backend selection.

### LMSlim-to-LightOp calls

The plugin currently uses four external LMSlim interfaces. Their call names
and argument expressions remain unchanged; only the owning module changes:

| Current external API | Required public API |
| --- | --- |
| `lmslim.layers.fused_moe...fused_experts_impl_fp8_marlin` | `lightop.moe.fused_experts_impl_fp8_marlin` |
| `lmslim.layers.fused_moe...fused_experts_impl_int8_marlin` | `lightop.moe.fused_experts_impl_int8_marlin` |
| `lmslim.layers.gemm.int8_utils.per_token_quant_int8` | `lightop.quant.per_token_quant_int8` |
| `lmslim.quant_ops.hipblaslt_w8a8_gemm` | `lightop.gemm_ops.hipblaslt_w8a8_channelwise_gemm` |

The installed LightOp implementation returns the same `(status, output)`
contract for channel-wise W8A8 GEMM and exposes compatible Marlin expert and
INT8 quantization signatures. Existing result validation stays in the plugin.

### Categorized LightOp migration

The migration follows the public map in `/models/Lightop&Lmslim.md` and the
installed category exports. Important ABI changes are handled explicitly:

- FP8 per-token quantization uses
  `per_token_quant_fp8(x, dtype=..., out_q=..., out_scale=...)`.
- dynamic RMS quantization consumes returned `(quantized, scale)` tensors.
- Gemma RMSNorm uses input-first ordering and `out=out`.
- MoE alignment uses `moe_align_block_size_out` with preallocated outputs and
  explicit EP/fill mode flags.
- MQA uses categorized signatures, removes legacy logical-size arguments, and
  provides FP32 contiguous weights.
- DeepSeek V4 uses the categorized KVNorm-aware int32 insertion kernel. Each
  caller passes raw versus normalized KV consistently so KV normalization is
  performed exactly once, and slot mappings are contiguous int32 tensors.

### Post-PR-27 owners

The latest branch adds owners that pull request #27 could not cover:

- `vllm_hcu/models/deepseek_v4_dspark.py`;
- `vllm_hcu/patch/worker/core_fix/patch_deepseek_v4_attention.py`;
- the current DeepSeek V4 DeepEP expert implementation and its clamped
  activation helpers;
- their portable and live-HCU accuracy tests.

These owners are inspected against the installed LightOp API independently.
They are not assumed to be correct merely because the older pull request
passes.

## Environment Variable Boundary

User-facing configuration owned by this plugin uses the `VLLM_HCU_*` prefix:

- `VLLM_HCU_FUSED_MOE_CHUNK_SIZE`;
- `VLLM_HCU_USE_GLOBAL_MOE_CACHE`;
- existing `VLLM_HCU_USE_FUSED_RMS_QUANT`;
- `VLLM_HCU_USE_FUSE_SILU_AND_MUL`.

A dependency-light bootstrap helper runs before any plugin-owned LightOp
import. It translates explicitly configured values to the neutral aliases
that installed LightOp 0.6 already accepts:

| Plugin setting | LightOp-supported setting |
| --- | --- |
| `VLLM_HCU_FUSED_MOE_CHUNK_SIZE` | `VLLM_FUSED_MOE_CHUNK_SIZE` |
| `VLLM_HCU_USE_GLOBAL_MOE_CACHE` | `VLLM_USE_GLOBAL_CACHE13` |
| `VLLM_HCU_USE_FUSED_RMS_QUANT` | `USE_FUSED_RMS_QUANT` |
| `VLLM_HCU_USE_FUSE_SILU_AND_MUL` | `VLLM_USE_FUSE_SILU_AND_MUL` |

An explicitly configured dependency-level alias is not overwritten. A
conflicting plugin value is rejected with a clear configuration error rather
than silently choosing one. Legacy `LMSLIM_*` input remains recognized only
long enough to emit a once-per-process deprecation warning and bridge to the
new plugin name; repository-owned image configuration and documentation use
only `VLLM_HCU_*` after this change.

Precedence is deterministic: matching new, neutral, and legacy values are
accepted; conflicting values raise before LightOp import. When only a legacy
value is present, the bootstrap copies it to the new plugin name and neutral
alias after warning. When none is present, the bootstrap leaves the
environment untouched so LightOp owns the default.

`LMSLIM_USE_LIGHTOP` has no supported neutral alias and is unnecessary for a
caller that directly selects LightOp. Repository examples remove this
explicit setting instead of inventing an unsupported `LIGHTOP_*` name.

## Error Handling and Observability

- Missing required categorized exports name the exact expected symbol.
- Runtime kernel failures propagate without trying a legacy implementation.
- Environment conflicts name both variables and their values.
- A legacy environment variable produces one deprecation warning per process.
- Logs never claim that external LMSlim is a supported runtime fallback.

## Test Design

### Test-first contract coverage

1. An AST/import-boundary test rejects production `lmslim` imports and direct
   calls through obsolete LightOp namespaces.
2. A categorized export contract imports every symbol used by production and
   verifies membership in the installed category's public `__all__`.
3. Fake-module tests prove each owner selects the categorized callable and
   passes the required ABI.
4. Missing categorized APIs fail closed even when obsolete symbols are
   present in the fake package.
5. Environment tests cover translation, precedence, conflict errors,
   deprecated input warnings, and bootstrap-before-LightOp ordering.
6. DeepSeek V4 tests verify exactly-once KV normalization, int32 contiguous
   slots, argument order, cache output, and untouched-slot sentinels.

### Verification layers

After focused red-green cycles:

1. run all focused owner and environment tests;
2. run `python tools/run_patch_tests.py --suite contract`;
3. run repository lint, compile, production-boundary, and patch-coverage
   checks applicable to the changed files;
4. run live HCU LightOp numerical tests on the available BW1100 devices;
5. search the final production tree for `lmslim`, `lightop.op`,
   `lightop.gemmopt`, and moved top-level exports and account for every result.

### Model validation

Use an isolated Python environment with the feature worktree installed
without replacing unrelated system packages. Start vLLM with
`/models/Qwen3.5-35B-A3B-W8A8` using the repository's documented W8A8 command,
wait for health, issue a deterministic generation request, and terminate the
server cleanly.

Acceptance requires:

- successful model load and engine startup;
- successful generation with non-empty output;
- no import of the external `lmslim` package by plugin production code;
- logs showing the intended LightOp quantization/MoE path or equivalent
  runtime evidence;
- no kernel, dtype, shape, or device error attributable to the migration.

## Delivery

The branch is `feat/lightop-api-no-lmslim-v0251`, based on `85a4ad5`, and the
new pull request targets `v0.25.1`. The pull request describes the categorized
API map, removed LMSlim dependencies, environment bridge, executed tests,
live-HCU evidence, and model command/result. The existing pull request #27 is
credited as a reference but is not a merge dependency.
