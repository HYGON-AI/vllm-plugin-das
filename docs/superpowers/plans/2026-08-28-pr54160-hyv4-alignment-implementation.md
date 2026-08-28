# HYV4 PR #54160 Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align the vLLM v0.25.1 HCU HYV4 plugin with the correctness and runtime behavior of vLLM PR #54160 without regressing the validated TP8 ModelOpt MXFP8 and native-MTP path.

**Architecture:** Add fail-closed exact-import adapters for vLLM core model classification and defaults, backport current Python parser semantics, generalize FP8 scale loading at the HYV4 model boundary, and switch FP32 logits to vLLM's `head_dtype` path. Keep NVIDIA-only Rust/HPC kernels and the HCU split-Q/KV layout unchanged.

**Tech Stack:** Python 3.10, PyTorch/HIP, vLLM 0.25.1, vllm-hcu exact-import patches, pytest, boltops, ModelOpt MXFP8.

**Spec:** `docs/superpowers/specs/2026-08-28-pr54160-hyv4-alignment-design.md`

## Global Constraints

- Modify only the plugin worktree; do not edit `/models/zb/vllm_025/vllm`.
- Add and observe a failing test before every production behavior change.
- Preserve TP8, expert-parallel-disabled, Triton MoE, ModelOpt MXFP8, MTP3, and boltops iHC behavior.
- Do not port Rust parser code, NVIDIA HPC gated-MLA, TRT-LLM MoE, or fused Q/KV projection.
- Explicit user environment choices override MRV2 and breakable-CUDAGraph defaults.
- Every exact-import patch is idempotent and fails closed on incompatible vLLM symbols.

---

### Task 1: HYV4 core runtime classification

**Files:**
- Create: `vllm_hcu/patch/platform/core_fix/patch_hy_v4_vllm_config.py`
- Create: `vllm_hcu/patch/platform/core_fix/patch_hy_v4_model_arch_config.py`
- Modify: `vllm_hcu/patch/platform/core_fix/__init__.py`
- Create: `tests/models/hy_v4/test_runtime_config.py`
- Modify: `tests/patch/test_platform_dispatcher.py`

**Interfaces:**
- Consumes: `vllm.config.vllm.DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES`, `VllmConfig.__post_init__`, and `ModelArchConfigConvertorBase.is_deepseek_mla`.
- Produces: two standard adapter modules, each exposing `apply_to_module(module: ModuleType) -> bool`, and registered under independent exact module callbacks and markers.

- [ ] **Step 1: Write failing classification tests**

Use real imported vLLM modules and controlled lightweight owners. Assert these literal outcomes:

```python
assert "HYV4ForCausalLM" in vllm_config.DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES
assert convertor(HYV4Config()).is_deepseek_mla() is True
assert convertor(SimpleNamespace(model_type="hy_v4_mtp", kv_lora_rank=512)).is_deepseek_mla() is True
assert convertor(SimpleNamespace(model_type="unrelated")).is_deepseek_mla() is False
```

Exercise the wrapped post-init against an HYV4 architecture with the graph env absent and assert it sets `VLLM_USE_BREAKABLE_CUDAGRAPH=1`; set the env to `0` first and assert it remains `0`.

- [ ] **Step 2: Verify RED**

Run: `pytest -q tests/models/hy_v4/test_runtime_config.py`

Expected: failures showing HYV4 absent from the V2 set, MLA false, and graph env unset.

- [ ] **Step 3: Implement the minimal exact-import adapters**

Replace the immutable V2 frozenset with its union. Wrap `is_deepseek_mla` so only `hy_v4` and `hy_v4_mtp` return true before delegating. Wrap the v0.25.1 post-init at its existing architecture boundary and set the graph env only when absent. Validate symbols and positional signatures with `_common` helpers.

- [ ] **Step 4: Verify GREEN and inventory**

Run: `pytest -q tests/models/hy_v4/test_runtime_config.py tests/patch/test_platform_dispatcher.py`

Expected: all selected tests pass and callback inventory contains both new exact targets.

- [ ] **Step 5: Commit**

Run: `git add vllm_hcu/patch/platform/core_fix tests/models/hy_v4/test_runtime_config.py tests/patch/test_platform_dispatcher.py && git commit -m 'fix(hy-v4): align runtime model classification'`

### Task 2: Python tool-parser streaming parity

**Files:**
- Modify: `vllm_hcu/tool_parsers/hy_v4_tool_parser.py`
- Modify: `tests/hy_v4/test_parsers.py`

**Interfaces:**
- Consumes: `HYV4ToolExtractor.extract_tool_calls_streaming(..., tools, *, guided=False)`.
- Produces: atomic-token and guided string-marker paths plus `_is_guided(request) -> bool`; `get_structural_tag()` returns `None` for auto/unspecified choice.

- [ ] **Step 1: Write failing parser tests**

Add literal cases that catch the upstream regressions:

```python
assert parser.get_structural_tag(auto_request) is None
assert extractor.extract_tool_calls_streaming(
    "a", "a < b", " < b", [1], [1, 2], [2], None
)["content"] == " < b"
```

Add a guided split-marker sequence using chunks `"<tool_"`, `"calls:hcu>"`, then two complete calls in one final delta; assert names `{0: "weather", 1: "date"}` and exact JSON argument strings. Assert one complete delta returns both calls rather than only the first.

- [ ] **Step 2: Verify RED**

Run: `pytest -q tests/hy_v4/test_parsers.py -k 'auto_structural or ordinary_less or guided_split or complete_delta'`

Expected: auto returns a structural tag and/or extractor lacks the `guided` keyword and atomic behavior.

- [ ] **Step 3: Backport current upstream Python behavior**

Add the `guided` keyword, retain string overlap buffering only for guided decoding, use token IDs for normal marker detection, drain all completed calls in a loop, and pass `guided=self._is_guided(request)` from the wrapper. Preserve the local v0.25.1 structural-tag builder and post-tool content support.

- [ ] **Step 4: Verify GREEN**

Run: `pytest -q tests/hy_v4/test_parsers.py tests/hy_v4/test_registration.py`

Expected: parser and registry suites pass.

- [ ] **Step 5: Commit**

Run: `git add vllm_hcu/tool_parsers/hy_v4_tool_parser.py tests/hy_v4/test_parsers.py && git commit -m 'fix(hy-v4): align tool parser streaming semantics'`

### Task 3: Generic FP8 indexer dequantization

**Files:**
- Modify: `vllm_hcu/models/hy_v4/model.py`
- Modify: `tests/models/hy_v4/test_weight_loading.py`

**Interfaces:**
- Produces: `_dequantize_indexer_fp8(weight, scale) -> torch.Tensor`, accepting per-channel, MXFP8 1x32, and two-dimensional block scale layouts.
- Consumes: `scaled_dequantize` and `GroupShape` from vLLM quantization utilities when available in v0.25.1; retains `dequant_mxfp8_to_bf16` for the validated ModelOpt layout.

- [ ] **Step 1: Write failing 128-by-128 and per-channel tests**

Construct a 256x256 FP8 tensor and a 2x2 float scale tensor with hand-selected constant quadrants. Assert each 128x128 output quadrant equals its literal scale times the FP8 value. Add `[out]` scale coverage equivalent to `[out, 1]`. Keep the existing uint8 `[out, in/32]` test.

- [ ] **Step 2: Verify RED**

Run: `pytest -q tests/models/hy_v4/test_weight_loading.py -k 'indexer and (128 or rank_one)'`

Expected: `ValueError` from the current fixed-shape loader.

- [ ] **Step 3: Implement group-shape-derived dequantization**

Normalize rank-one scale to `[out, 1]`, reinterpret UE8M0 bytes only for the relevant layout, derive `group_m=out/scale_rows` and `group_n=in/scale_cols`, validate exact divisibility, and call the v0.25.1 scaled-dequant primitive. Keep the ModelOpt MXFP8 helper for `[out, in/32]` so current HCU behavior is unchanged.

- [ ] **Step 4: Verify GREEN**

Run: `pytest -q tests/models/hy_v4/test_weight_loading.py`

Expected: all weight-loading tests pass.

- [ ] **Step 5: Commit**

Run: `git add vllm_hcu/models/hy_v4/model.py tests/models/hy_v4/test_weight_loading.py && git commit -m 'fix(hy-v4): support generic blockwise indexer scales'`

### Task 4: Legacy FP8 MTP scale normalization

**Files:**
- Modify: `vllm_hcu/models/hy_v4/mtp.py`
- Modify: `tests/models/hy_v4/test_mtp.py`

**Interfaces:**
- Produces: `_prepare_mtp_fp8_expert_scale(quant_config, name, loaded_weight) -> tuple[str, torch.Tensor]`.
- Updates: `_create_mtp_quant_config` to inherit `activation_scheme` unless blockwise forces `dynamic`.

- [ ] **Step 1: Write failing scale tests**

Use an actual/lightweight `Fp8Config` with `weight_block_size=[128, 128]` and `is_scale_e8m0=True`. Pass `model.layers.78.mtp_block.mlp.experts.gate_up_proj.scale` plus raw uint8 bytes. Assert the returned name ends in `.weight_scale_inv`, dtype is `torch.float8_e8m0fnu`, and byte representation is unchanged. Use a non-`Fp8Config` ModelOpt-like object and assert raw name/dtype remain unchanged. Assert non-block FP8 preserves literal `activation_scheme="static"` while blockwise returns `dynamic`.

- [ ] **Step 2: Verify RED**

Run: `pytest -q tests/models/hy_v4/test_mtp.py -k 'legacy_fp8 or activation_scheme'`

Expected: helper missing and static activation incorrectly becomes dynamic.

- [ ] **Step 3: Implement normalization before expert mapping**

Copy the conditional upstream normalization and invoke it immediately after checkpoint-layer name rewriting in `load_weights`. Do not reinterpret ModelOpt raw-byte scales.

- [ ] **Step 4: Verify GREEN**

Run: `pytest -q tests/models/hy_v4/test_mtp.py tests/models/hy_v4/test_weight_loading.py`

Expected: all MTP and loader tests pass.

- [ ] **Step 5: Commit**

Run: `git add vllm_hcu/models/hy_v4/mtp.py tests/models/hy_v4/test_mtp.py && git commit -m 'fix(hy-v4): normalize legacy blockwise MTP scales'`

### Task 5: Memory-efficient FP32 logits

**Files:**
- Modify: `vllm_hcu/models/hy_v4/config.py`
- Modify: `vllm_hcu/models/hy_v4/model.py`
- Modify: `vllm_hcu/models/hy_v4/mtp.py`
- Modify: `tests/models/hy_v4/test_registration.py`
- Modify: `tests/models/hy_v4/test_weight_loading.py`
- Modify: `tests/models/hy_v4/test_mtp.py`

**Interfaces:**
- Produces: `HYV4Config.head_dtype == "float32"` when enabled and unset.
- Preserves: model-dtype LM-head parameters with FP32 logits through `LogitsProcessor`.

- [ ] **Step 1: Write failing config/head tests**

Assert `HYV4Config(enable_lm_head_fp32=True).head_dtype == "float32"`, an explicit `head_dtype="bfloat16"` is preserved, and disabled FP32 does not invent the field. Construct target/MTP heads with fake `ParallelLMHead` recording `params_dtype`; assert it receives `None`, while `compute_logits` receives the original hidden-state dtype and returns FP32 through the fake logits processor.

- [ ] **Step 2: Verify RED**

Run: `pytest -q tests/models/hy_v4/test_registration.py tests/models/hy_v4/test_weight_loading.py tests/models/hy_v4/test_mtp.py -k 'head_dtype or fp32_head'`

Expected: missing `head_dtype` and explicit FP32 parameter allocation.

- [ ] **Step 3: Implement the head-dtype path**

Set the config field before `super().__post_init__`. Remove target `params_dtype=torch.float32` and hidden-state FP32 casting. Remove MTP projection casting to weight dtype. Keep the excluded-head quantization workaround unless a focused real `LogitsProcessor` test proves v0.25.1 requires the upstream `UnquantizedLinearMethod` compatibility patch.

- [ ] **Step 4: Verify GREEN**

Run: `pytest -q tests/models/hy_v4/test_registration.py tests/models/hy_v4/test_weight_loading.py tests/models/hy_v4/test_mtp.py`

Expected: all selected tests pass.

- [ ] **Step 5: Commit**

Run: `git add vllm_hcu/models/hy_v4 tests/models/hy_v4 && git commit -m 'perf(hy-v4): keep fp32 logits without fp32 head weights'`

### Task 6: Full regression and HCU validation

**Files:**
- Modify if required by test plumbing: `tests/integration/models/test_hy_v4_smoke.py`
- Runtime logs: `/tmp/vllm-hcu-integration/logs/`

**Interfaces:**
- Consumes: Tasks 1-5.
- Produces: fresh static, eager TP8/MTP3, default-MRV2, and non-eager breakable-CUDAGraph evidence.

- [ ] **Step 1: Run static verification**

Run: `python -m compileall -q vllm_hcu/models/hy_v4 vllm_hcu/tool_parsers vllm_hcu/patch/platform/core_fix`

Run: `pytest -q tests/models/hy_v4 tests/hy_v4 tests/patch/test_platform_dispatcher.py tests/runtime_patch/test_worker_framework_opt.py`

Expected: zero failures.

- [ ] **Step 2: Run repository checks**

Run: `git diff --check`

Run the repository's configured formatter/linter on changed Python files if present; otherwise run `ruff check` only when `ruff` is installed.

Expected: zero errors.

- [ ] **Step 3: Run eager TP8/MTP3 parity**

Run the existing `tests/integration/models/test_hy_v4_smoke.py` MTP parity case with `/models/Hy4-preview-FP8-Testing`, TP8, Triton, expert parallel disabled, and three speculative tokens.

Expected: speculative token IDs equal baseline token IDs for every prompt.

- [ ] **Step 4: Run default-MRV2 and graph validation**

Start HYV4 once without `VLLM_USE_V2_MODEL_RUNNER` and assert the resolved config uses V2. Start again with `VLLM_USE_V2_MODEL_RUNNER=0` and assert opt-out. Then run target and MTP without `--enforce-eager`, with breakable graphs enabled, and assert deterministic generation plus absence of capture/shared-buffer/NaN/parameter errors.

- [ ] **Step 5: Run parser API validation**

Start an OpenAI-compatible endpoint with HYV4 parsers. Send streaming auto and required tool requests. Assert auto permits plain content containing `<`, and required returns valid HYV4-native tool-call deltas with stable concatenated JSON arguments.

- [ ] **Step 6: Final verification commit**

If Task 6 required test-plumbing changes, commit them with `test(hy-v4): cover PR 54160 runtime alignment`. Otherwise retain the prior focused commits and record exact commands/log paths in the final report.
