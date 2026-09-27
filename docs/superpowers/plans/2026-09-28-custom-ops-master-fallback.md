# HCU Custom-Op Master Fallback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `VLLM_HCU_USE_CUSTOM_OPS=0` reliably bypass managed LightOp/AITER optimizations while preserving explicit MoE backend selection and the FP8 QSA exception.

**Architecture:** Centralize effective master/child precedence in `vllm_hcu.platforms.envs`, apply it at provider-selection boundaries before imports or layout conversion, and delegate to audited BoltOPs, Triton, upstream vLLM, or vllm_hcu-native paths. Reject unsupported master-off cache/layout combinations before serving instead of silently executing a custom provider.

**Tech Stack:** Python, pytest, vLLM MRV2 runtime patches, LightOp, AITER, BoltOPs, Triton, HCU/ROCm eight-card hardware.

**Spec:** `docs/superpowers/specs/2026-09-28-custom-ops-master-fallback-design.md`

## Global Constraints

- `VLLM_HCU_USE_CUSTOM_OPS` defaults to enabled.
- The master switch controls managed optional sub-operators, not the selected MoE expert backend.
- Explicit `--moe-backend aiter`, `triton`, or `deep_gemm` must not be rewritten.
- FP8 QSA E4M3/E5M2 reader and cache writer remain independent of the master.
- Provider selection must occur before provider-specific weight/cache layout conversion.
- Missing safe fallback must produce an early, explicit error rather than execute LightOp/AITER under master-off.
- Changes, validation evidence, server commands, and skill updates go to existing MR #163.

## Review Focus

- Master off with a child flag explicitly on must bypass the child provider; covered in Tasks 1-4.
- Explicit AITER MoE plus master off must retain AITER experts but bypass a managed LightOp router; covered in Tasks 1 and 2.
- FP8 QSA plus master off must retain its dedicated reader/writer; covered in Task 4.
- Proprietary layout conversion followed by fallback must be impossible; covered in Tasks 3 and 4.
- CUDA graph capture must not hide a first-call provider change or exception-based retry; covered in Task 5.

---

### Task 1: Central effective policy and backend boundary

**Files:**
- Modify: `vllm_hcu/platforms/envs.py`
- Modify: `tests/accuracy/test_environment_routing.py`
- Modify: `tests/runtime_patch/test_quant_gemm_aiter.py`

**Interfaces:**
- Produces: `optional_custom_op_enabled(feature_enabled: bool = True) -> bool`
- Preserves: `is_aiter_moe_requested(moe_config: object | None = None) -> bool` semantics for explicit MoE selection

- [ ] **Step 1: Write failing policy tests**

Add a four-case master/child matrix for `optional_custom_op_enabled`, plus a test that explicit AITER MoE remains requested while the master is off.

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `pytest -q tests/accuracy/test_environment_routing.py tests/runtime_patch/test_quant_gemm_aiter.py -k 'optional_custom_op or explicit_aiter_moe_ignores_custom_ops_master'`

Expected: FAIL because the effective helper and boundary regression do not yet exist.

- [ ] **Step 3: Implement the effective-policy helper**

Add `optional_custom_op_enabled(feature_enabled: bool = True) -> bool` to `envs.py`; it returns `custom_ops_enabled() and bool(feature_enabled)` without changing raw child getters or MoE backend resolution.

- [ ] **Step 4: Run focused and environment-routing tests**

Run: `pytest -q tests/accuracy/test_environment_routing.py tests/runtime_patch/test_quant_gemm_aiter.py -k 'optional_custom_op or explicit_aiter_moe_ignores_custom_ops_master or master_custom_ops_gate'`

Expected: PASS.

- [ ] **Step 5: Commit**

Commit message: `refactor: centralize optional custom-op policy`

### Task 2: LightOp router and core operator fallbacks

**Files:**
- Modify: `vllm_hcu/ops/fuse_moe_gate.py`
- Modify: `vllm_hcu/model_executor/layers/fused_moe/router_runtime.py`
- Modify: `vllm_hcu/patch/worker/op_opt/moe/patch_fused_topk_bias_router.py`
- Modify as required: `vllm_hcu/ops/rms_norm.py`, `vllm_hcu/ops/rms_norm_gated.py`, `vllm_hcu/ops/gemma_rms_norm.py`, `vllm_hcu/ops/silu_and_mul.py`, `vllm_hcu/ops/topk_topp_sample.py`
- Test: `tests/runtime_patch/test_lightop_sqrtsoftplus_routing.py`
- Test: `tests/runtime_patch/test_lightop_ops_api.py`
- Test: `tests/runtime_patch/test_lightop_rmsnorm_gated.py`
- Test: `tests/runtime_patch/test_hcu_sampler.py`
- Test: `tests/models/hy_v4/test_moe.py`

**Interfaces:**
- Consumes: `optional_custom_op_enabled(feature_enabled: bool = True) -> bool`
- Produces: managed LightOp dispatchers that delegate to their pre-existing upstream/native callable when disabled

- [ ] **Step 1: Write failing no-call and Hy4 parity tests**

Add tests that install raising LightOp sentinels, set master off and child on, execute each dispatcher, and assert the native result.  For Hy4 grouped routing, compare ids and weights to the upstream router for identical logits, correction bias, and scaling.

- [ ] **Step 2: Run the focused LightOp tests and verify RED**

Run: `pytest -q tests/runtime_patch/test_lightop_sqrtsoftplus_routing.py tests/runtime_patch/test_lightop_ops_api.py tests/runtime_patch/test_lightop_rmsnorm_gated.py tests/runtime_patch/test_hcu_sampler.py tests/models/hy_v4/test_moe.py -k 'master_off or custom_ops_master'`

Expected: at least one uncovered dispatcher fails by reaching its LightOp sentinel or missing the centralized policy.

- [ ] **Step 3: Apply the effective policy at LightOp selection boundaries**

Replace duplicated master/child expressions with the Task 1 helper.  Keep existing upstream/native callables as the fallback and do not alter expert-backend selection or provider-specific MoE expert code.

- [ ] **Step 4: Run all focused LightOp/router tests**

Run: `pytest -q tests/runtime_patch/test_lightop_sqrtsoftplus_routing.py tests/runtime_patch/test_lightop_ops_api.py tests/runtime_patch/test_lightop_rmsnorm_gated.py tests/runtime_patch/test_hcu_sampler.py tests/models/hy_v4/test_moe.py`

Expected: PASS.

- [ ] **Step 5: Commit**

Commit message: `fix: gate optional LightOp routes with custom-op master`

### Task 3: Optional AITER sub-operator fallbacks

**Files:**
- Modify as required: `vllm_hcu/patch/worker/op_opt/patch_fla_chunk_o.py`, `vllm_hcu/patch/worker/op_opt/patch_fla_chunk_delta_h.py`, `vllm_hcu/patch/worker/op_opt/patch_gdn_base.py`, `vllm_hcu/model_executor/layers/mhc.py`
- Test: `tests/runtime_patch/test_gdn_v0251_ownership.py`
- Test: `tests/runtime_patch/test_attention_mla_fla_mamba.py`

**Interfaces:**
- Consumes: `optional_custom_op_enabled(feature_enabled: bool = True) -> bool`
- Produces: managed AITER auxiliary dispatchers that choose BoltOPs/Triton/native before kernel execution

- [ ] **Step 1: Write failing AITER no-call tests**

For FLA/GDN and MHC selectors, set master off with the AITER child on, install a raising AITER sentinel, and assert the existing BoltOPs/Triton/native result is selected.

- [ ] **Step 2: Run focused tests and verify RED**

Run: `pytest -q tests/runtime_patch/test_gdn_v0251_ownership.py tests/runtime_patch/test_attention_mla_fla_mamba.py -k 'master_off or custom_ops_master'`

Expected: uncovered selectors fail by reaching AITER or by lacking the common policy.

- [ ] **Step 3: Gate only optional AITER sub-operators**

Use the Task 1 helper in selection code.  Do not change AITER MoE request resolution, AITER MoE shuffle/config, or the explicit MoE expert backend.

- [ ] **Step 4: Run complete focused AITER suites**

Run: `pytest -q tests/runtime_patch/test_gdn_v0251_ownership.py tests/runtime_patch/test_attention_mla_fla_mamba.py`

Expected: PASS.

- [ ] **Step 5: Commit**

Commit message: `fix: bypass optional AITER ops under master-off`

### Task 4: Attention, cache, quantization, and exceptions

**Files:**
- Modify: `vllm_hcu/v1/attention/ops/flashmla.py`
- Modify: `vllm_hcu/v1/attention/backends/mla/flashmla_sparse.py`
- Modify: `vllm_hcu/runtime_compat/scaled_mm.py`
- Modify: `vllm_hcu/patch/worker/op_opt/patch_compressed_tensors_w8a8_int8.py`
- Modify: `vllm_hcu/patch/worker/op_opt/patch_custom_ops.py`
- Modify as required: `vllm_hcu/patch/worker/core_fix/patch_deepseek_v4_attention.py`
- Test: `tests/runtime_patch/test_attention_mla_fla_mamba.py`
- Test: `tests/runtime_patch/test_fp8_channel_triton_product.py`
- Test: `tests/runtime_patch/test_quant_gemm_aiter.py`
- Test: `tests/runtime_patch/test_qwen4_exp_qsa_fp8.py`

**Interfaces:**
- Consumes: `optional_custom_op_enabled(feature_enabled: bool = True) -> bool`
- Preserves: BoltOPs sparse-MLA and target-Triton dense-GEMM fallbacks already added in MR #163
- Preserves: FP8 QSA reader/writer independence

- [ ] **Step 1: Write failing cache-helper and exception tests**

Assert that master-off E5M2/DeepSeek cache helpers delegate to the original implementation or are rejected during compatibility validation without importing LightOp.  Assert separately that QSA E4M3/E5M2 reader/writer still execute with master off.

- [ ] **Step 2: Run focused tests and verify RED**

Run: `pytest -q tests/runtime_patch/test_attention_mla_fla_mamba.py tests/runtime_patch/test_fp8_channel_triton_product.py tests/runtime_patch/test_quant_gemm_aiter.py tests/runtime_patch/test_qwen4_exp_qsa_fp8.py -k 'master_off or custom_ops_master or qsa_fp8'`

Expected: E5M2 cache-helper coverage fails because it currently imports LightOp unconditionally; QSA exception tests pass or identify an accidental broad gate.

- [ ] **Step 3: Complete selection-time fallback/rejection**

Use the effective policy for sparse MLA and quant GEMM selectors.  For E5M2/cache-layout paths, delegate before the LightOp import when the original supports the ABI; otherwise add an early compatibility error naming the unsupported cache dtype and master setting.  Do not gate QSA FP8 reader/writer.

- [ ] **Step 4: Run all attention and quantization regression tests**

Run: `pytest -q tests/runtime_patch/test_attention_mla_fla_mamba.py tests/runtime_patch/test_fp8_channel_triton_product.py tests/runtime_patch/test_quant_gemm_aiter.py tests/runtime_patch/test_qwen4_exp_qsa_backend.py tests/runtime_patch/test_qwen4_exp_qsa_fp8.py`

Expected: PASS.

- [ ] **Step 5: Commit**

Commit message: `fix: complete master-off attention and GEMM fallbacks`

### Task 5: Static audit, full tests, hardware accuracy, and handoff

**Files:**
- Modify: `/models/upgrading-vllm-hcu/SKILL.md` or its routed references
- Modify: `/models/upgrading-vllm-hcu/references/validation.md`
- Modify: `/models/upgrading-vllm-hcu/references/failure-ledger.md` if a new failure signature is confirmed
- Modify: MR #163 description/comment

**Interfaces:**
- Consumes: all managed selectors and exceptions from Tasks 1-4
- Produces: reproducible validation evidence and updated skill guidance

- [ ] **Step 1: Run a source audit for unmanaged provider entry points**

List every production LightOp/AITER import and classify it as managed fallback, explicit MoE/backend-owned, QSA exception, or required HCU primitive.  Treat any unclassified entry as a test/code gap.

- [ ] **Step 2: Run the complete repository test suite**

Run: `pytest -q`

Expected: PASS; if environment-only failures remain, record every failing test by name and separate them from product regressions.

- [ ] **Step 3: Run eight-card master-off hardware validation**

Validate the GLM sparse-MLA fallback and Hy4 DP+EP+MTP3 path, including graph mode where supported.  Capture route evidence proving BoltOPs/Triton/native use and absence of managed LightOp/AITER calls.  Run the established HumanEval eight-prompt accuracy gate.

- [ ] **Step 4: Update the upgrade skill and MR #163**

Document the effective policy, MoE boundary, QSA exception, exact server commands, commit/wheel/topology, readiness evidence, provider evidence, and accuracy results.  Push the branch and post the command/evidence comment to MR #163.

- [ ] **Step 5: Final verification and commit**

Re-run the changed focused suites after documentation edits, verify `git diff --check`, inspect the complete diff against `origin/v0.28.1-dev`, and commit with message `docs: record custom-op fallback validation`.
