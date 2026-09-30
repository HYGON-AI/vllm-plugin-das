# GLM-5.3 MTP Kpool and mHC Correctness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Backport the official speculative-kpool ring correction and make the custom-op master select official native mHC when disabled.

**Architecture:** Patch the frozen target's GLM tail-cache spec and AMD kpool helpers at worker initialization, preserving pool-width semantics while adding an independently sized physical ring. Select native versus BoltOPs mHC once during decoder construction, before any graph capture or kernel execution.

**Tech Stack:** Python, pytest, PyTorch, Triton, vLLM MRV2 runtime patches, HCU/ROCm.

**Spec:** `docs/superpowers/specs/2026-09-28-glm53-mtp-kpool-mhc-correctness-design.md`

## Global Constraints

- Target vLLM remains the v0.28.1-dev frozen API; newer official source is a comparison/backport input only.
- Pool completion remains keyed by `index_kpool`; only physical tail-ring addressing uses ring width.
- `VLLM_HCU_USE_CUSTOM_OPS=0` must reach official mHC `forward_native` and no BoltOPs/local TileLang mHC kernel.
- `VLLM_HCU_USE_CUSTOM_OPS=1` keeps the existing BoltOPs mHC route.
- MoE expert backend selection, sparse attention backend selection, and quantized GEMM routing are out of scope.

## Review Focus

- No-spec decode must retain a one-pool ring and existing behavior; covered in Task 1.
- MTP3 rejected pool-completing drafts must not overwrite redo inputs; covered in Task 1.
- Ring size must divide the attention block so prefix-cache alignment is unchanged; covered in Task 1.
- Reapplying worker patches must be idempotent and reject stale markers; covered in Tasks 1 and 2.
- Materialized child flags must not override a live master-off environment; covered in Task 2.

---

### Task 1: Backport speculative kpool tail-ring correctness

**Files:**
- Create: `vllm_hcu/v1/attention/ops/glm5next_kpool_ring.py`
- Modify: `vllm_hcu/patch/worker/core_fix/patch_glm5next_channel_fp8.py`
- Modify: `tests/runtime_patch/test_glm53_channel_fp8.py`

**Interfaces:**
- Produces: `install_glm5next_kpool_ring(kpool_ops: ModuleType) -> bool`
- Produces: patched `Glm5NextTailCache.get_kv_cache_spec(vllm_config)` using an aligned physical ring
- Preserves: existing `kpool_seed_tail_cache` and `kpool_decode_update_and_maybe_write_cache_batched` call signatures

- [ ] **Step 1: Write failing ring-spec and kernel-dispatch tests**

Add literal cases for pool4 with zero drafts -> ring4 and MTP3 -> ring8. Add a fake-launcher test proving seed/decode pass `RING=8` while retaining `KPOOL`/`POOL_SIZE=4`, plus a CPU mirror showing a rejected completing draft corrupts ring4 but not ring8.

- [ ] **Step 2: Run focused tests and verify RED**

Run: `pytest -q tests/runtime_patch/test_glm53_channel_fp8.py -k 'tail_ring or rejected_completing_draft'`

Expected: FAIL because the tail cache still returns block size 4 and the installed kernels have no independent ring width.

- [ ] **Step 3: Implement the official ring behavior against the frozen API**

Install ABI-checked replacement seed/decode helpers based on official `2617fe9383`; compute `ring = pool * next_power_of_2(ceil((pool + num_speculative_tokens) / pool))`, require cache block divisibility, and patch the target module before model execution.

- [ ] **Step 4: Run focused and complete GLM patch tests**

Run: `pytest -q tests/runtime_patch/test_glm53_channel_fp8.py`

Expected: PASS.

- [ ] **Step 5: Commit**

Commit message: `fix: preserve GLM kpool tail across rejected drafts`

### Task 2: Select official native mHC under master-off

**Files:**
- Modify: `vllm_hcu/patch/worker/core_fix/patch_glm5next_channel_fp8.py`
- Modify: `tests/runtime_patch/test_glm53_native_mhc.py`

**Interfaces:**
- Consumes: `optional_custom_op_enabled(VLLM_HCU_USE_AITER_MHC) -> bool`
- Produces: decoder-construction selection between `_bind_glm5next_native_mhc` and `_bind_glm5next_boltops_mhc`

- [ ] **Step 1: Write failing production-selection tests**

Construct the patched decoder with real fake-op objects under master off/on. Assert master-off calls official native equations and never imports the HCU mHC backend; assert master-on binds BoltOPs.

- [ ] **Step 2: Run focused tests and verify RED**

Run: `pytest -q tests/runtime_patch/test_glm53_native_mhc.py -k 'decoder_selects'`

Expected: master-off FAIL because decoder initialization always binds BoltOPs.

- [ ] **Step 3: Implement construction-time provider selection**

Resolve the live effective policy after official decoder initialization. Bind native mHC when off and lazily import/bind BoltOPs only when on.

- [ ] **Step 4: Run the complete mHC and GLM patch suites**

Run: `pytest -q tests/runtime_patch/test_glm53_native_mhc.py tests/runtime_patch/test_glm53_channel_fp8.py`

Expected: PASS.

- [ ] **Step 5: Commit**

Commit message: `fix: use native GLM mHC when custom ops are disabled`

### Task 3: Regression, hardware accuracy, review, and handoff

**Files:**
- Modify as evidence requires: `/models/upgrading-vllm-hcu/references/validation.md`
- Modify as evidence requires: `/models/upgrading-vllm-hcu/references/failure-ledger.md`
- Modify: MR #163 comment

**Interfaces:**
- Consumes: Tasks 1-2 behavior
- Produces: reproducible HCU evidence and current service commands

- [ ] **Step 1: Run focused and repository tests**

Run focused suites, `git diff --check`, then repository `pytest -q`; record every unrelated/environment failure by name.

- [ ] **Step 2: Run HCU kernel and service validation**

On available cards, run the official rejected-draft kernel scenario, repeated
mHC probes, TP4 MTP3 master-off/on service checks, and prefix-cache repeated
prompts.

- [ ] **Step 3: Run HMMT25 accuracy**

Run the complete HMMT25 gate with the corrected code and compare master-off/on
outputs, max-token terminations, and scores.  The packaged February 2025 split
contains 30 records, so report 30/30 rather than silently claiming 32 inputs.

- [ ] **Step 4: Review and document**

Perform a fresh whole-branch review, fix important findings RED-to-GREEN, update the upgrade skill evidence, and prepare the exact MR comment. Do not push or post until explicitly authorized for the resulting commits.
