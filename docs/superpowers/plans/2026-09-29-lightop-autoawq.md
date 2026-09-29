# LightOp AutoAWQ Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in public-LightOp AutoAWQ W4A16 backend with exact fallback to the current v0.28.1 method.

**Architecture:** A hybrid quant linear method owns the eligibility decision before layout mutation. Eligible weights are converted once with pure Torch plus the public LightOp repack; all other layers delegate their full lifecycle to the original vLLM quant method.

**Tech Stack:** Python, PyTorch, vLLM 0.28.1 quantization API, public LightOp GEMM API, pytest, HCU.

**Spec:** `docs/superpowers/specs/2026-09-29-lightop-autoawq-design.md`

## Global Constraints

- Base all work on the latest `origin/v0.28.1-dev`.
- LMSlim is deprecated; use only the categorized public LightOp GEMM API.
- `VLLM_HCU_USE_LIGHTOP_AWQ` defaults to disabled and is subordinate to `VLLM_HCU_USE_CUSTOM_OPS`.
- LightOp eligibility is FP16, W4 zero-point, group size 128, and an explicitly tuned K/N pair.
- Decide the provider before layout mutation and retain no duplicate packed weights on eligible layers.
- Keep this change in an AWQ-specific MR separate from Qwen3 work.

## Review Focus

- BF16 model dtype must retain the original v0.28.1 method and untouched standard weights; Task 2 tests it.
- An untuned K/N pair must retain the original method and layout; Task 2 tests it.
- Bias and multi-dimensional input must preserve upstream output shape and value semantics; Task 1 tests them.
- A missing or ABI-incompatible LightOp package must fail over before conversion; Task 2 tests it.
- Checkpoint qzeros must keep raw AutoAWQ zero semantics through the `+64` metadata encoding; Task 1 tests it.

---

### Task 1: Conversion and hybrid linear method

**Files:**
- Create: `vllm_hcu/model_executor/layers/quantization/lightop_autoawq.py`
- Create: `tests/runtime_patch/test_lightop_autoawq.py`
- Modify: `tests/runtime_patch/test_lightop_categorized_api.py`
- Modify: `tests/patch/test_lightop_api_boundary.py`

**Interfaces:**
- Consumes: a vLLM `LinearMethodBase` delegate and standard AutoAWQ `qweight`, `qzeros`, and `scales` parameters.
- Produces: `convert_awq_to_lightop_layout(qweight, qzeros, scales, group_size) -> tuple[Tensor, Tensor]`, `is_lightop_awq_shape_supported(k, n) -> bool`, and `LightOpAutoAWQLinearMethod(delegate, quant_config)`.

- [ ] **Step 1: Write failing tests** for exact nibble order, raw qzero metadata encoding, supported-shape table, LightOp apply with bias/reshaping, delegation, and categorized API ownership.
- [ ] **Step 2: Run focused tests.** Run: `pytest -q tests/runtime_patch/test_lightop_autoawq.py tests/runtime_patch/test_lightop_categorized_api.py tests/patch/test_lightop_api_boundary.py`. Expected: tests fail because the conversion and method do not exist.
- [ ] **Step 3: Implement the pure-Torch conversion and hybrid method.** Import public LightOp functions lazily; delegate without mutation unless eligibility has already been established.
- [ ] **Step 4: Re-run focused tests.** Expected: PASS.
- [ ] **Step 5: Commit.** Commit message: `feat(awq): add lightop w4a16 linear method`.

### Task 2: AutoAWQ selection and runtime registration

**Files:**
- Create: `vllm_hcu/patch/worker/op_opt/patch_auto_awq.py`
- Modify: `vllm_hcu/platforms/envs.py`
- Modify: `vllm_hcu/patch/worker/__init__.py`
- Modify: `tests/runtime_patch/test_lightop_autoawq.py`
- Modify: `tests/patch/test_worker_dispatcher.py`

**Interfaces:**
- Consumes: Task 1 `LightOpAutoAWQLinearMethod` and `is_lightop_awq_shape_supported`.
- Produces: `lightop_awq_enabled() -> bool` and `apply_patch(module: ModuleType | None = None) -> None` wrapping `AutoAWQConfig.get_quant_method` exactly once.

- [ ] **Step 1: Write failing tests** for opt-in/master precedence, FP16 selection, BF16/group/zero-point fallback, absent public export fallback, exact signature, and idempotence.
- [ ] **Step 2: Run focused tests.** Run: `pytest -q tests/runtime_patch/test_lightop_autoawq.py tests/patch/test_worker_dispatcher.py`. Expected: selection and dispatcher tests fail because the patch is absent.
- [ ] **Step 3: Implement policy, selector wrapper, and dispatcher registration.** Wrap only eligible AutoAWQ linear methods and keep the selected original method as the delegate.
- [ ] **Step 4: Run focused and patch suites.** Run: `pytest -q tests/runtime_patch/test_lightop_autoawq.py tests/patch`. Expected: PASS.
- [ ] **Step 5: Commit.** Commit message: `feat(awq): route eligible layers to lightop`.

### Task 3: HCU operator validation and MR documentation

**Files:**
- Create: `tests/accuracy/test_lightop_autoawq_accuracy.py`
- Create: `docs/lightop_autoawq_validation.md`

**Interfaces:**
- Consumes: Task 2 runtime-selected backend.
- Produces: reproducible HCU accuracy/performance evidence and any available service-level evidence.

- [ ] **Step 1: Write a failing HCU test** comparing converted LightOp output to dequantized FP16 reference and the existing v0.28.1 backend for tuned Qwen-like shapes and M in `1,2,8,16,32,64,128`.
- [ ] **Step 2: Run on HCU.** Expected: the test exercises public repack/GEMM and reveals any kernel-call or tolerance mismatch.
- [ ] **Step 3: Fix only demonstrated defects and rerun.** Expected: declared FP16 tolerance passes and timing table is emitted.
- [ ] **Step 4: Search local models and run one service test if an AutoAWQ model exists.** Otherwise record the exact inventory search and the limitation.
- [ ] **Step 5: Run repository verification.** Run: `pytest -q`. Expected: PASS apart from explicitly documented environment skips.
- [ ] **Step 6: Commit validation documentation.** Commit message: `docs(awq): record lightop backend validation`.
