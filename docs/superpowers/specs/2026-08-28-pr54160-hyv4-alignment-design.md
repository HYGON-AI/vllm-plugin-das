# HYV4 PR #54160 Alignment Design

## Goal

Bring the vLLM v0.25.1 HCU HYV4 plugin into behavioral alignment with the
current vLLM PR #54160 head while preserving the already validated HCU TP8,
ModelOpt MXFP8, native MTP, and boltops iHC paths.

## Baseline

- Plugin branch: `feat/hy-v4-mtp-blockwise-v0251-merge` at `dea47e5`.
- Reference PR head: `b22b0ea44cbe19504845b2db7f6461e356f73ffd`.
- Target checkpoint: `/models/Hy4-preview-FP8-Testing`.
- Acceptance topology: TP8, expert parallel disabled, Triton MoE, native MTP
  with three speculative tokens.
- Existing eager TP8/MTP parity must remain unchanged.

## Constraints

- Implement compatibility through `vllm_hcu`; do not edit the checked-out
  vLLM source tree under `/models/zb/vllm_025/vllm`.
- Preserve v0.25.1 APIs and exact-import patch lifecycle validation.
- Add a failing regression test before each production behavior change.
- Keep HCU-specific split Q/KV projections and ModelOpt MXFP8 layouts unless a
  reference change fixes correctness rather than NVIDIA-only performance.
- Do not port Rust parser code into the Python plugin. The packaged vLLM Rust
  extension can only be aligned by rebuilding a vLLM wheel.
- Do not invent an HCU gated-MLA fused operator. Retain the correct eager
  projection/sigmoid/product path until boltops or another HCU library exposes
  an equivalent kernel.

## Architecture

### 1. Core architecture compatibility adapter

Add one idempotent platform-core patch that applies three HYV4 registrations
to the imported v0.25.1 `vllm.config.vllm` and model architecture converter:

1. Add `HYV4ForCausalLM` to `DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES`.
2. Make `hy_v4` and `hy_v4_mtp` return true from the converter's
   `is_deepseek_mla()` without changing other model types.
3. Extend v0.25.1's breakable-CUDAGraph auto-enable decision to cover
   `HYV4ForCausalLM` and `HYV4MTPModel` while preserving an explicit user value
   of `VLLM_USE_BREAKABLE_CUDAGRAPH`.

The patch must validate every target symbol/signature and fail closed with
`PatchCompatibilityError` on an incompatible vLLM build.

### 2. Python HYV4 parser alignment

Backport the current Python parser behavior without depending on main-branch
Rust or serving APIs:

- `tool_choice=auto` must not install a structural tag.
- Required and named tool choices continue to use HYV4 native structural
  tokens.
- Streaming distinguishes the atomic-token free-generation path from the
  string-marker guided path, so ordinary `<` content is never held back.
- A single engine delta may emit every complete tool call already buffered.
- Split structural markers and incremental JSON arguments remain prefix stable.

Keep the v0.25.1 local structural-tag builder because the upstream registry API
is unavailable in this release.

### 3. Generic blockwise FP8 loading

Generalize indexer dequantization to derive group dimensions independently and
accept these scale layouts:

- per-channel `[out]` or `[out, 1]`;
- ModelOpt MXFP8 `[out, in / 32]` with UE8M0 bytes;
- two-dimensional blockwise `[out / block_m, in / block_n]`, including
  128-by-128 blocks.

For legacy `Fp8Config` MTP expert weights, normalize checkpoint `.scale` names
to the fused `*_scale_inv` parameter and reinterpret raw `uint8` UE8M0 bytes as
`torch.float8_e8m0fnu`. ModelOpt quantization keeps its existing raw-byte
contract.

The MTP FP8 config inherits the checkpoint `activation_scheme`, forcing
`dynamic` only for blockwise weights as upstream does.

### 4. Memory-efficient FP32 logits

When `enable_lm_head_fp32` is true, expose `head_dtype="float32"` on the HYV4
config and keep LM-head weights in the model dtype. Use v0.25.1's
`LogitsProcessor` FP32 output path rather than materializing FP32 weights or
casting the complete weight-backed projection input unnecessarily.

Compatibility with ModelOpt-excluded LM heads must be tested. If v0.25.1's
`LogitsProcessor` rejects `UnquantizedLinearMethod`, add a narrowly scoped core
patch matching upstream; otherwise retain the current embedding-method
workaround.

### 5. Preserve already aligned runtime fixes

No structural rewrite is planned for these already covered behaviors:

- native HYV4 MTP registration and `hc_mult=1` draft-buffer contract;
- target/draft top-k buffer sharing;
- `set_skip_topk()` and `compact_topk_indices()`;
- sink-capable sparse MQA behavior;
- zero-initialized padded attention heads;
- boltops `ihc_pre`, `ihc_post`, and `ihc_head` dispatch;
- strict fail-closed parameter completeness;
- HCU Triton/AITER MoE selection.

Regression tests will prove these behaviors remain intact.

### 6. Performance boundary

The following upstream NVIDIA-specific optimizations are documented but are not
correctness requirements for this plugin change:

- HPC fused gated-MLA GEMM;
- NVIDIA TRT-LLM FP8 MoE;
- fused Q/KV latent projection, which conflicts with the validated HCU layout;
- Rust HY unified parser.

They may be implemented later only when matching HCU operators or a rebuilt
vLLM wheel are available.

## Error Handling

- Core patches are idempotent and validate exact targets before mutation.
- Unsupported scale shapes raise a message containing weight shape, scale
  shape, dtype, and inferred block dimensions.
- Incomplete weight/scale pairs remain fatal at the end of loading.
- Parser state never emits inconsistent argument suffixes; a lost prefix is
  withheld and logged rather than sending corrupt JSON.
- Explicit user graph/runner environment choices always override defaults.

## Test Strategy

### Static and CPU tests

- Core registration tests for target and MTP architectures, idempotence, and
  non-HYV4 delegation.
- Parser tests for auto/required/named, ordinary `<`, split markers, multiple
  calls in one delta, and incremental arguments.
- Quantization tests for per-channel, MXFP8, 128-by-128, UE8M0 reinterpretation,
  ModelOpt preservation, and activation scheme inheritance.
- LM-head tests for config `head_dtype`, parameter dtype, excluded heads, and
  target/MTP logits dtype.
- Existing HYV4 model, attention, iHC, MTP, weight-loading, and parser suites.

### HCU integration

1. TP8 eager baseline generation with the current ModelOpt checkpoint.
2. TP8 eager native MTP3 parity; speculative token IDs must equal baseline.
3. TP8 non-eager run with breakable CUDAGraph enabled for target and MTP.
4. Default MRV2 selection without setting `VLLM_USE_V2_MODEL_RUNNER`, plus an
   explicit opt-out check.
5. OpenAI-compatible streaming tool calls covering auto and required modes.

Hardware failures caused by unavailable devices or pinned CI images are
reported separately from code failures; they do not count as passing evidence.

## Acceptance Criteria

- Every new regression test is observed failing before its implementation.
- All focused and full runnable plugin tests pass.
- `git diff --check` and Python compile checks pass.
- Existing TP8 eager/MTP3 parity remains unchanged.
- Non-eager target and MTP start successfully and complete deterministic
  generation without capture, shared-buffer, NaN, or parameter-loading errors.
- The worktree contains no unrelated user changes.
