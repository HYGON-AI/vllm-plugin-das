# LightOp Categorized API Adaptation for v0.25.1

## Context

The `v0.25.1` branch of `vllm-plugin-das` still calls LightOp through a mix of
top-level exports, `lightop.op`, `lightop.gemmopt`, and LMSlim. LightOp 0.6
exposes these kernels through categorized modules such as
`lightop.attention`, `lightop.moe`, and `lightop.quant`. Several moves are
simple namespace changes, while others also change argument order, output
ownership, or the operation performed.

This adaptation uses the following references:

- `OpenDAS/vllm-hcu` tag `v0.21.0`, commit
  `059fc4491364afabd9494bd0ee47426b8cc9bdcd` (`V0.21.0 lightop接口适配`).
- The follow-up MQA correction at
  `1d4a0d787156878159e304f9283b12193720c76f`, which requires LightOp MQA
  weights to be FP32 and contiguous.
- The installed LightOp
  `0.6.0+das.dtk2604.torch2110.2608171227.g8c835c`, whose categorized exports
  and callable signatures were inspected directly.
- GitHub `origin/v0.25.1` at
  `8c3d880b1b1f0b73ff7313a37d1a33b4693fc01b`.

## Goals

1. Move every production LightOp call site on `v0.25.1` to the applicable
   categorized API.
2. Preserve the branch's existing runtime-adapter and lazy-import boundaries.
3. Retain deprecated compatibility only when the old and new interfaces are
   ABI-compatible.
4. Fail closed when an ABI-changed API is unavailable.
5. Cover the `v0.25.1`-specific DeepSeek V4 and fused RMS quant paths in
   addition to the interfaces migrated by the reference commit.
6. Deliver the work as one independent pull request targeting `v0.25.1`.

## Non-goals

- Do not introduce a repository-wide LightOp facade or dependency-injection
  subsystem.
- Do not change `lightop.sampling`; it already uses the categorized API.
- Do not redesign unrelated HCU operators, environment flags, or runtime patch
  lifecycle.
- Do not treat CPU mocks as proof of HCU kernel numerical correctness.
- Do not preserve legacy execution for APIs whose data flow or ABI changed.

## Architecture

The migration is performed in place at each existing ownership boundary. A
module that currently imports a kernel lazily continues to do so lazily; a
runtime adapter continues to own validation and exception translation. This
avoids changing plugin import order or HCU initialization behavior.

Each migrated call belongs to one of three policies:

1. **Compatible move:** import the categorized API first, then fall back to the
   deprecated export with a `warning_once`. Only import absence is handled;
   kernel execution failures propagate.
2. **Changed ABI:** require the categorized API and adapt the caller to its new
   contract. Absence produces a targeted error and never invokes the legacy
   API.
3. **Optional optimization:** prefer the categorized API but retain the
   existing portable implementation when no LightOp implementation is
   available. `ds_cat` is the only three-level path:
   `lightop.tensor.ds_cat`, deprecated top-level `lightop.ds_cat`, then
   `torch.cat`.

No broad `except Exception` will be added around LightOp imports or calls.
Compatibility selection catches only `ImportError` and `AttributeError`.

## Interface Migration Map

### Attention and sparse MLA

| Current owner | New API | Policy and adaptation |
| --- | --- | --- |
| `model_executor/layers/attention_runtime.py` | `lightop.attention.split_qkv_rms_rotary_embedding_fuse_with_kv_store_quant` | Compatible top-level fallback. |
| `v1/attention/ops/rocm_aiter_mla_sparse.py` | `lightop.attention.mqa_logits` | Use the categorized signature and remove the explicit logical dimensions used by `lightop.op.mqa_logits`. The compatible fallback is the deprecated top-level `lightop.mqa_logits`, never the incompatible `lightop.op` symbol. |
| same | `lightop.attention.paged_mqa_logits` | Compatible `lightop.gemmopt.paged_mqa_logits` fallback; preserve schedule metadata and clean-logit behavior. |
| same | `lightop.attention.top_k_per_row_prefill` and `top_k_per_row_decode` | Compatible move from `lightop.op`. |

All chunked, non-paged, and paged LightOp MQA paths pass
`weights.float().contiguous()`. Existing AITER and PyTorch fallbacks remain
unchanged.

### Activation and GEMM

| Current owner | New API | Policy |
| --- | --- | --- |
| `ops/fuse_silu_mul_quant.py`, `ops/silu_and_mul.py` | `lightop.activation.*` | Compatible top-level/`lightop.op` fallback. |
| `fused_moe/experts/{deep_gemm_moe,batched_deep_gemm_moe,dpsk_v4_deep_gemm_moe}.py` | `lightop.activation.fuse_silu_mul_fp8_quant*` | Compatible top-level fallback while preserving current import timing. |
| same DeepGEMM expert files | `lightop.gemm_ops.m_grouped_w8a8_gemm_nt_{contig,masked}` | Compatible `lightop.gemmopt` fallback where applicable. |
| `quantization/int8_runtime.py` | `lightop.gemm_ops.hipblaslt_w8a8_channelwise_gemm` | Categorized API first, LMSlim fallback with a deprecation warning. |

### MoE

| Current owner | New API | Policy and adaptation |
| --- | --- | --- |
| `fused_moe/deep_gemm_utils.py` | `lightop.moe.ep_scatter` and `ep_gather` | Compatible `lightop.op` fallback. |
| `fused_moe/router_runtime.py`, `ops/fuse_moe_gate.py` | `lightop.moe.moe_fused_gate` | Compatible `lightop.op` fallback. |
| `quantization/compressed_tensors/compressed_tensors_moe_marlin.py` | `lightop.moe.fused_experts_impl_{fp8,int8}_marlin` | Categorized API first, LMSlim fallback with a deprecation warning. |
| `patch/worker/op_opt/moe/patch_moe_align_block_size.py` | `lightop.moe.moe_align_block_size_out` | Changed ABI. Pass the preallocated output tensors plus explicit `is_ep=False` and `is_fuse_fill=False`; do not call legacy `moe_align_block_size`. |

The MoE align adapter retains its existing sentinel initialization because
LightOp only writes routed tokens when fused fill is disabled.

### Norm and quantization

| Current owner | New API | Policy and adaptation |
| --- | --- | --- |
| `ops/rms_norm.py` | `lightop.norm.fused_add_rms_norm`, `rmsnorm_forward_autograd` | Compatible top-level/`lightop.op` fallback. |
| `ops/gemma_rms_norm.py` | `lightop.norm.gemma_fused_add_rmsnorm` | Compatible fallback for fused-add. |
| same | `lightop.norm.gemma_rmsnorm(x, weight, epsilon, out=out)` | Changed from the legacy output-first ABI; categorized API is required. |
| `model_executor/layers/quantization/lightop_fp8_runtime.py` | `lightop.quant.per_token_quant_fp8(x, dtype=..., out_q=..., out_scale=...)` | Changed from the legacy output-first ABI; categorized API is required. |
| `ops/fuse_rms_norm_quant.py` | `lightop.norm.rms_norm_dynamic_per_token_quant(...)` | Changed from preallocated outputs to returned `(quantized, scale)` tensors; categorized API is required. |
| `quantization/int8_runtime.py` | `lightop.quant.per_token_quant_int8` | Categorized API first, LMSlim fallback with a deprecation warning. |

Existing runtime validation of shapes, dtypes, status, and bias behavior is
retained after the namespace migration.

### Tensor utility

`ops/test_concat.py` resolves `lightop.tensor.ds_cat`, then the deprecated
top-level export. If neither exists it stores no unresolved callable and uses
`torch.cat`. Import failures are logged through the project logger rather than
printed to stdout.

## DeepSeek V4 Data Flow

The DeepSeek V4 path cannot be migrated by renaming the function. The current
flow is:

1. `fused_q_kv_rmsnorm` normalizes QR and KV.
2. `wq_b` projects normalized QR into Q.
3. Legacy
   `fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert` performs head Q norm,
   RoPE, KV quantization, and cache insertion on the already normalized KV.

The categorized function is
`lightop.attention.fused_deepseek_v4_qnorm_rope_kvnorm_rope_quant_insert_int32`.
Its native symbol takes an extra const Tensor before the mutable cache output;
this is the KV RMSNorm weight, consistent with the operation name and the
non-inserting companion API.

The new flow is therefore:

1. Apply `self.q_norm` to QR only.
2. Leave KV unnormalized until cache insertion.
3. Project normalized QR with `wq_b`.
4. Call the categorized `_int32` fused kernel with Q, raw KV,
   `self.kv_norm.weight`, cache, the existing contiguous int64 slot mapping,
   positions, RoPE cache,
   epsilon, and block size.

This preserves one KV normalization and moves it into the new fused kernel.
The legacy DeepSeek V4 kernel is not a fallback because it requires a
different upstream data flow; attempting to share the new flow would skip KV
normalization, while retaining the old flow would normalize KV twice.

## Errors and Observability

- Compatible fallback logs one warning naming both the missing categorized API
  and the deprecated API selected.
- Required changed-ABI APIs raise an owner-specific `RuntimeError` (or the
  adapter's existing error type) with the exact missing categorized symbol.
- A categorized kernel that imports successfully but fails at execution is
  allowed to raise its original error, except where an existing adapter adds
  operation dimensions to the message.
- Compatibility logic does not hide binary loading errors, signature errors,
  invalid shapes, or device failures behind a legacy retry.
- Error messages refer to LightOp, not LMSlim, when the categorized LightOp
  path is required.

## Test Design

### Export contract

Add a categorized API import test covering the reference commit's required
exports:

- `lightop.activation`: SiLU/mul quant variants and `silu_and_mul_opt`.
- `lightop.attention`: MQA/paged MQA metadata and logits, split QKV, and sparse
  top-k kernels.
- `lightop.gemm_ops`: channelwise and grouped W8A8 GEMMs.
- `lightop.moe`: EP gather/scatter, Marlin experts, align-out, and fused gate.
- `lightop.norm`: RMS/Gemma functions.
- `lightop.quant`: FP8 and INT8 per-token quantization.
- `lightop.tensor`: `ds_cat`.

Extend that contract for `v0.25.1` with the categorized DeepSeek V4 fused
kernel and dynamic RMS quant function.

### Portable contract tests

Use fake LightOp module trees to verify:

1. Categorized exports are selected before deprecated exports.
2. Compatible fallbacks execute and warn once.
3. Changed-ABI paths reject a missing categorized export even when a legacy
   symbol is present.
4. FP8 quant passes `out_q` and `out_scale` by keyword.
5. Dynamic RMS quant consumes its returned tensors.
6. Gemma RMSNorm uses `out=out` instead of output-first ordering.
7. MoE align passes all preallocated outputs and explicit mode flags.
8. MQA removes legacy logical-size arguments and normalizes weights to FP32
   contiguous layout for chunked, non-paged, and paged paths.
9. DeepSeek V4 performs QR-only pre-normalization and passes raw KV plus the KV
   RMSNorm weight to the categorized fused kernel in the required order.
10. `ds_cat` reaches `torch.cat` when neither LightOp export exists.

Update existing LMSlim and `lightop.op` fakes in runtime-patch tests so they
exercise the new primary path rather than succeeding accidentally through a
deprecated mock.

### Verification commands

The implementation is accepted only after:

1. Focused tests for every changed owner pass.
2. `python tools/run_patch_tests.py --suite contract` passes.
3. Relevant formatting and static checks configured by the repository pass.
4. A production-code search accounts for every remaining `lightop.op`,
   `lightop.gemmopt`, top-level moved export, and LMSlim kernel import.
5. Available HCU kernel tests are run when the host supports them; unavailable
   device coverage is listed explicitly in the pull request.

The pre-change baseline is 947 passed, 37 deselected, and 14 warnings for the
contract suite on commit `8c3d880`.

## Delivery

Work is isolated on branch `feat/lightop-categorized-api-v0251`, based directly
on `origin/v0.25.1`. The pull request will target `v0.25.1` and will contain
only this LightOp adaptation, its tests, and supporting documentation. Its
description will list:

- the old-to-new interface map;
- compatible fallbacks versus strict requirements;
- the DeepSeek V4 normalization data-flow change;
- executed tests and hardware limitations;
- reference commits `059fc449` and `1d4a0d7`.

Merging is attempted only after the branch is pushed, the pull request checks
are green, and repository permissions and review policy permit it.
