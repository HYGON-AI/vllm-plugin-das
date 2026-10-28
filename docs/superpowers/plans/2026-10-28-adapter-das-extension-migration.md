# adapter-das Extension Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate the vLLM 0.32 DAS optimization wheel to the `adapter-das` Extension API v1 without changing operator behavior.

**Architecture:** The plugin keeps one optimization package and one extension factory. Build metadata, imports, verifier expectations, and tests move atomically from `hcu_vllm_adapter` to `hcu_adapter`; vLLM platform/general ownership remains external.

**Tech Stack:** Python 3.11+, setuptools entry points, vLLM 0.32 OOT registries, pytest.

**Spec:** `docs/superpowers/specs/2026-10-28-adapter-das-extension-migration-design.md`

## Global Constraints

- Entry-point group is exactly `hcu_adapter.extensions` with name `das_hcu`.
- Runtime imports API types only from `hcu_adapter.extensions`.
- Distribution dependency is `sgl-kernel>=0.1.0` and no `hcu-vllm-adapter` dependency remains.
- The plugin owns no `vllm.platform_plugins` or `vllm.general_plugins` entry point.
- RMSNorm and SiluAndMul implementation behavior is unchanged.

## Review Focus

- Stale old-adapter strings in runtime/build/verifier code must fail a contract test.
- Installed ownership checks must distinguish the adapter and plugin distributions.
- API version mismatch must fail before any LightOp/vLLM import or registration.
- Live tests must assert the selected platform is `hcu_adapter.bridges.vllm.HcuPlatform`.
- Disabling master or operator switches must still reach upstream vLLM fallbacks.

---

### Task 1: Pin the new package boundary

**Files:**
- Modify: `tests/vllm032/test_package_boundary.py`
- Modify: `tests/vllm032/test_extension.py`
- Modify: `tests/vllm032/test_compat_lock.py`

**Interfaces:**
- Produces: executable contract for the new dependency, namespace, group, and absence of stale adapter ownership.

- [ ] **Step 1: Change tests to require `sgl-kernel`, `hcu_adapter.extensions`, and the new extension group; add a stale-namespace scan.**
- [ ] **Step 2: Run the focused tests and verify they fail against the old metadata/imports.**
- [ ] **Step 3: Commit as `test: pin adapter-das extension boundary`.**

### Task 2: Migrate the plugin runtime contract

**Files:**
- Modify: `adapter_extension/pyproject.toml`
- Modify: `adapter_extension/src/hcu_vllm_das/extension.py`
- Modify: `adapter_extension/README.md`

**Interfaces:**
- Consumes: Adapter `hcu_adapter.extensions` API v1.
- Produces: one `das_hcu` extension entry point and unchanged optimization capabilities.

- [ ] **Step 1: Replace the distribution dependency, entry-point group, and API imports.**
- [ ] **Step 2: Run extension/package/operator portable tests and verify they pass.**
- [ ] **Step 3: Build the wheel and inspect metadata for the single allowed extension entry.**
- [ ] **Step 4: Commit as `vllm032: migrate plugin to adapter-das extension API`.**

### Task 3: Installed and live-HCU contract migration

**Files:**
- Modify: `adapter_extension/tools/verify_installed.py`
- Modify: `tests/vllm032/test_verify_installed.py`
- Modify: `tests/vllm032/test_rms_norm_hcu.py`
- Modify: `tests/vllm032/test_silu_and_mul_hcu.py`
- Modify: `docs/deployment/vllm032-hcu-adapter-migration-guide-cn.md`

**Interfaces:**
- Consumes: Task 2 wheel and adapter-das wheel.
- Produces: installation attestation for exact adapter/plugin ownership and updated live-HCU commands.

- [ ] **Step 1: Write failing installed/live tests for adapter-das entry points, module origins, and selected platform identity.**
- [ ] **Step 2: Update verifier and deployment guide to the new wheel pair and namespaces.**
- [ ] **Step 3: Run all `tests/vllm032`, build/install both wheels in isolation, and run the verifier.**
- [ ] **Step 4: Run available HCU RMSNorm/SiluAndMul live tests; record unrun model/HumanEval gates explicitly.**
- [ ] **Step 5: Commit as `validation: verify adapter-das plugin pairing`.**
