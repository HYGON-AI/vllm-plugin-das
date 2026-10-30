# vLLM 0.32 PR #310 Runtime Contract Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the three reproduced vLLM 0.32 runtime integration defects and publish one internally consistent Plugin candidate wheel, compat lock, and deployment guide.

**Architecture:** Preserve official vLLM 0.32 behavior for every non-AITER backend and add AITER only at the narrow oracle boundary. Carry the canonical expert index-map through the Plugin class, derive the AITER mask only at the vendor call, and preserve the official `rms_norm` custom-op identity independently of the OOT registry key. Commit runtime code first, build that exact clean commit, then record its wheel hash in a documentation-only follow-up commit to avoid circular artifact identity.

**Tech Stack:** Python 3.12, pytest, vLLM 0.32 oracle/OOT APIs, PyTorch wheel tooling, JSON compat lock.

**Spec:** [PR #310 review](https://github.com/HYGON-AI/vllm-plugin-das/pull/310#issuecomment-6518979300) plus the 2026-10-30 re-review supplied by the user.

## Global Constraints

- Base is official vLLM `v0.32.0` commit `5df63a7203cdb9328d824a27b682e77c7b9928be`.
- Plugin master and child optimizations remain default-off.
- Official vLLM behavior remains authoritative for non-AITER backends and `SiluAndMul`.
- AITER kernels remain vendor-owned; the Plugin only adapts selection and argument contracts.
- No force push, no mutation of shared installed environments, and no unrecorded HCU claim.

## Review Focus

- `auto`, `triton`, `humming`, and CPU enum members must still reach official factories after enum extension.
- Linear and round-robin EP maps on both ranks must reach Triton fallback as `-1/local-slot` maps, never AITER masks.
- Native AITER EP calls must receive a 0/1 mask with the sentinel slot preserved.
- `['none', '+rms_norm']` and `['all', '-rms_norm']` must control the OOT replacement by the official name.
- The recorded wheel must contain both `lightop-rms-norm` and `aiter-int8-moe` and pass installed verification.

---

### Task 1: Preserve Official INT8 Oracle Backends

**Files:**
- Modify: `tests/vllm032/test_aiter_int8_oracle.py`
- Modify: `adapter_extension/src/hcu_vllm_das/moe/int8_oracle.py`

**Interfaces:**
- Consumes: official `Int8MoeBackend` values and oracle functions.
- Produces: extended `Int8MoeBackend` with `AITER`, while official members continue through official factories.

- [ ] Add realistic enum-sensitive fake-oracle tests for auto/TRITON/HUMMING/CPU selection and weight conversion.
- [ ] Run the focused tests and observe the current `Unknown Int8 MoE backend` failure.
- [ ] Delegate non-AITER members without converting them back to the detached old enum.
- [ ] Run the focused oracle suite and confirm it passes.
- [ ] Commit the task.

### Task 2: Separate AITER Mask from Triton Expert Map

**Files:**
- Modify: `tests/vllm032/test_aiter_int8_oracle.py`
- Modify: `tests/vllm032/test_aiter_int8_provider.py`
- Modify: `adapter_extension/src/hcu_vllm_das/moe/int8_oracle.py`
- Modify: `adapter_extension/src/hcu_vllm_das/moe/fused_moe_aiter.py`

**Interfaces:**
- Consumes: canonical expert map shaped `[global + shared]` with `-1/local-slot` values.
- Produces: canonical map for Triton fallback; derived int32 0/1 mask plus sentinel for native AITER.

- [ ] Add parameterized rank/placement/fallback tests and a native AITER mask test.
- [ ] Run the focused tests and observe mask/index-map failures.
- [ ] Set the Plugin experts class to consume canonical maps and derive the vendor mask at the AITER call boundary.
- [ ] Run provider/oracle suites and confirm they pass.
- [ ] Commit the task.

### Task 3: Preserve Official RMSNorm Custom-op Identity

**Files:**
- Modify: `tests/vllm032/test_rms_norm.py`
- Modify: `adapter_extension/src/hcu_vllm_das/rms_norm.py`

**Interfaces:**
- Consumes: OOT lookup key `RMSNorm` and official custom-op name `rms_norm`.
- Produces: OOT replacement registered by class key but enabled/disabled by official custom-op directives.

- [ ] Extend the fake runtime to model the real decorator name mutation and compilation `custom_ops` list.
- [ ] Add tests for `none/+rms_norm` and `all/-rms_norm`; observe failures.
- [ ] Restore `HcuRMSNorm.name` to `RMSNorm.name` after OOT registration.
- [ ] Run the RMSNorm suite and confirm it passes.
- [ ] Commit the task.

### Task 4: Build and Lock the Actual Candidate

**Files:**
- Modify: `validation/v0.32.0/compat-lock.json`
- Modify: `tests/vllm032/test_compat_lock.py`
- Modify: `docs/deployment/vllm032-hcu-adapter-migration-guide-cn.md`
- Create: `docs/validation/vllm032-pr310-runtime-contract-candidate.md`

**Interfaces:**
- Consumes: clean runtime commit from Tasks 1-3.
- Produces: exact Plugin wheel SHA256/build commit, installed capability evidence, and explicitly scoped pending HCU gates.

- [ ] Run all portable v0.32 tests on the clean runtime commit.
- [ ] Build the Plugin wheel from a clean archive of that commit and verify package contents.
- [ ] Install the locked vLLM, Adapter, and Plugin wheels in a fresh isolated environment and run `verify_installed.py`.
- [ ] Update compat-lock tests first and observe the old identity failure.
- [ ] Update compat lock, deployment table, and validation record with exact identities and evidence; do not inherit unrerun HCU claims.
- [ ] Run focused gates, the complete portable suite, `git diff --check`, and artifact hash verification.
- [ ] Commit the documentation/lock task and update PR #310 without force pushing.
