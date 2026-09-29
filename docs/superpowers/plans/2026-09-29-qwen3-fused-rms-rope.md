# Qwen3 Fused RMSNorm and Rotary Embedding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a safe, default-enabled LightOp fused Q/K RMSNorm plus RoPE path for Qwen3 text attention on HCU.

**Architecture:** A categorized LightOp custom-op wrapper owns the mutable kernel call. An exact Qwen3 attention patch selects it only for the supported one-dimensional text path and delegates every unsupported case to the original vLLM implementation.

**Tech Stack:** Python, PyTorch custom ops, vLLM 0.28.1, public LightOp attention API, pytest, HCU.

**Spec:** `docs/superpowers/specs/2026-09-29-qwen3-fused-rms-rope-design.md`

## Global Constraints

- Base all work on the latest `origin/v0.28.1-dev`.
- Use only `lightop.attention.rms_rotary_embedding_fuse`; no LMSlim or private LightOp namespace.
- Preserve the original Qwen3 forward path before mutation for every unsupported input.
- `VLLM_HCU_USE_FUSED_RMS_ROPE` defaults to enabled and is subordinate to `VLLM_HCU_USE_CUSTOM_OPS`.
- Keep this change in a Qwen3-specific MR separate from AWQ work.

## Review Focus

- A two-dimensional M-RoPE `positions` tensor must execute the original method unchanged; Task 2 tests it.
- Dual-chunk attention must execute the original method unchanged; Task 2 tests it.
- Missing or incompatible rotary cache metadata must execute the original method unchanged; Task 2 tests it.
- The vendor kernel may mutate Q/K, so routing must complete before the call; Task 1 and Task 2 test the mutation boundary.
- Reapplying the runtime patch must be idempotent and retain one original method; Task 2 tests it.

---

### Task 1: Public fused operator and feature policy

**Files:**
- Create: `vllm_hcu/ops/rms_rope.py`
- Modify: `vllm_hcu/platforms/envs.py`
- Modify: `tests/runtime_patch/test_lightop_categorized_api.py`
- Create: `tests/runtime_patch/test_qwen3_fused_rms_rope.py`
- Modify: `tests/patch/test_lightop_api_boundary.py`

**Interfaces:**
- Consumes: `vllm_hcu.platforms.envs.optional_custom_op_enabled(bool) -> bool`.
- Produces: `fused_rms_rotary_embedding(positions, query, key, head_size, cos_sin_cache, is_neox, weight_q, weight_k, epsilon) -> tuple[Tensor, Tensor]` and `fused_qwen3_rms_rope_enabled() -> bool`.

- [ ] **Step 1: Write failing tests** for default/disabled/master-off policy, mutable custom-op registration, fake tensor shapes, public categorized export, and production API-boundary compliance.
- [ ] **Step 2: Run the focused tests.** Run: `pytest -q tests/runtime_patch/test_qwen3_fused_rms_rope.py tests/runtime_patch/test_lightop_categorized_api.py tests/patch/test_lightop_api_boundary.py`. Expected: new policy/operator tests fail because their interfaces do not exist.
- [ ] **Step 3: Implement the minimal policy and wrapper.** Import the vendor function lazily from `lightop.attention`, declare Q/K mutation, and register the fake implementation without importing LightOp.
- [ ] **Step 4: Re-run the focused tests.** Expected: PASS.
- [ ] **Step 5: Commit.** Commit message: `feat(qwen3): add fused rms rope operator`.

### Task 2: Exact Qwen3 attention routing

**Files:**
- Create: `vllm_hcu/patch/worker/op_opt/patch_qwen3_attention.py`
- Modify: `vllm_hcu/patch/worker/__init__.py`
- Modify: `tests/runtime_patch/test_qwen3_fused_rms_rope.py`
- Modify: `tests/patch/test_worker_dispatcher.py`

**Interfaces:**
- Consumes: Task 1 `fused_rms_rotary_embedding(...)` and `fused_qwen3_rms_rope_enabled()`.
- Produces: `apply_patch(module: ModuleType | None = None) -> None` for exact target `vllm.model_executor.models.qwen3.Qwen3Attention.forward`.

- [ ] **Step 1: Write failing tests** for exact signature, supported QKV/fused/attention/output routing, opt-out, 2-D positions, dual chunk, missing cache, and duplicate application.
- [ ] **Step 2: Run the focused tests.** Run: `pytest -q tests/runtime_patch/test_qwen3_fused_rms_rope.py tests/patch/test_worker_dispatcher.py`. Expected: patch/routing tests fail because the adapter is absent.
- [ ] **Step 3: Implement the minimal exact patch and dispatcher registration.** Decide all eligibility before QKV projection; preserve the upstream return contract and original callable.
- [ ] **Step 4: Re-run focused and patch suites.** Run: `pytest -q tests/runtime_patch/test_qwen3_fused_rms_rope.py tests/patch`. Expected: PASS.
- [ ] **Step 5: Commit.** Commit message: `feat(qwen3): fuse qk rmsnorm and rotary embedding`.

### Task 3: Hardware evidence and MR documentation

**Files:**
- Create: `docs/qwen3_fused_rms_rope_validation.md`
- Modify: `tests/accuracy/test_lightop_qwen_rmsnorm_gated_accuracy.py` or create a focused HCU accuracy file if separation is clearer.

**Interfaces:**
- Consumes: Task 2 patched Qwen3 service path.
- Produces: reproducible operator and service validation evidence with exact commands.

- [ ] **Step 1: Add a failing HCU accuracy/performance test** comparing the fused operator with separate vLLM RMSNorm and RoPE for Qwen3-8B dimensions.
- [ ] **Step 2: Run the test on HCU.** Expected: it exercises the new operator and initially exposes any call-contract mismatch.
- [ ] **Step 3: Fix only demonstrated contract defects, then run the HCU test.** Expected: BF16 accuracy tolerance passes and timing is recorded.
- [ ] **Step 4: Run Qwen3-8B baseline/candidate service checks** including `curl --noproxy '*' /health`, `/v1/models`, HumanEval 0-7, and throughput A/B; record exact commands and results.
- [ ] **Step 5: Run repository verification.** Run: `pytest -q`. Expected: PASS apart from explicitly documented environment skips.
- [ ] **Step 6: Commit validation documentation.** Commit message: `docs(qwen3): record fused rms rope validation`.
