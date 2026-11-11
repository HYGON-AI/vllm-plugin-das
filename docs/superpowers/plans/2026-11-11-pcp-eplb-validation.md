# PCP EPLB Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the HCU plugin admit validated PCP expert-parallel EPLB configs before vLLM 0.25.1's TP/DP-only validator rejects them.

**Architecture:** Wrap `ParallelConfig._validate_parallel_config` in the existing platform core-fix lifecycle and rebuild its Pydantic decorator graph. Preserve field validation and initialization; restore EPLB/R16 fields after the original after-validator checks the rest of the config.

**Tech Stack:** Python 3.10, Pydantic dataclasses/core, pytest, vLLM-HCU runtime patch framework.

**Spec:** `docs/superpowers/specs/2026-11-11-pcp-eplb-validation-design.md`

## Global Constraints

- Target `HYGON-AI/vllm-plugin-das:v0.25.1` and pinned vLLM 0.25.1 only.
- Do not mutate installed vLLM source files and do not require `PYTHONPATH` overlays.
- Preserve all upstream validation outside `TP1/DP1/PCP>1 + EPLB + EP`.
- Fail closed on incompatible Pydantic or `ParallelConfig` structure.

## Review Focus

- R16 input object must not be mutated while validation is replayed.
- Unsupported platform and EP-disabled inputs must retain upstream errors.
- PCP1 must retain the upstream TP/DP EPLB error.
- Preserve direct and nested Pydantic construction and communicator selection.
- Repeated plugin patch application must not stack validator proxies.

---

### Task 1: ParallelConfig validator compatibility

**Files:**
- Create: `vllm_hcu/patch/platform/core_fix/patch_parallel_config.py`
- Modify: `vllm_hcu/patch/platform/core_fix/__init__.py`
- Modify: `vllm_hcu/patch/platform/core_fix/patch_hcu_config.py`
- Modify: `tests/runtime_patch/test_hyv4_pcp_static_eplb_config.py`
- Modify: `tests/patch/test_platform_dispatcher.py`

**Interfaces:**
- Consumes: `ParallelConfig._validate_parallel_config`, Pydantic decorator metadata and `rebuild_dataclass`.
- Produces: an idempotent HCU after-validator installed by the dedicated `patch_parallel_config` adapter before `patch_vllm_config`.

- [x] **Step 1: Run the existing PCP16 construction test and record the expected RED failure.**

Run: `pytest -q tests/runtime_patch/test_hyv4_pcp_static_eplb_config.py::test_upstream_parallel_config_counts_pcp_for_eplb`

Expected: FAIL containing `EPLB requires tensor_parallel_size or data_parallel_size`.

- [x] **Step 2: Add focused failing tests.**

Cover R16 preservation and input non-mutation, PCP1 rejection, EP-disabled rejection, nested validation, invalid fields, communicator selection, and idempotent application.

- [x] **Step 3: Implement the minimal after-validator and lifecycle registration.**

Gate on validated target fields, copy the EPLB config before zeroing redundant experts, delegate to the original after-validator, restore fields in `finally`, and expose compatibility failures through `PatchCompatibilityError`.

- [x] **Step 4: Run focused and lifecycle tests.**

Run: `pytest -q tests/runtime_patch/test_hyv4_pcp_static_eplb_config.py tests/patch/test_plugin_lifecycle.py`

Expected: all selected tests pass.

- [x] **Step 5: Run broader platform patch tests and lint.**

Run: `pytest -q tests/runtime_patch/test_glm52_pcp_config.py tests/runtime_patch/test_hyv4_static_r16.py tests/patch/test_platform_dispatcher.py`

Run: repository formatter/linter on changed Python files.

Result: 225 passed, 101 subtests passed across the focused, lifecycle,
GLM52 PCP, HYV4 R16, platform dispatcher and patch coverage suites. Ruff passes
for the new adapter and the updated PCP EPLB regression file. Existing
torch.jit deprecation warnings and a SharedSignalPool exit warning remain.

- [ ] **Step 6: Commit, self-review, push, and open the MR.**

Commit title: `fix(hyv4): admit PCP expert parallel EPLB configs`

MR target: `v0.25.1`; include the real startup failure and post-fix startup evidence.
