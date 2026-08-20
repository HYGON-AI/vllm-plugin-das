# vLLM 0.25 HCU AITER W16A16 Dispatch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make vLLM 0.25 explicit AITER unquantized MoE prefer the HCU direct W16A16 ASM interface used by the vLLM 0.21 plugin.

**Architecture:** Retain the existing `rocm_aiter_fused_experts` request context and HCU-owned `_aiter_ops` replacement. Change only the W16A16 config branch in `aiter_runtime.fused_moe_impl`: resolve a solution id, then call the same direct ASM function used by the non-config branch.

**Tech Stack:** Python 3.12, PyTorch, vLLM 0.25.1, HCU AITER, pytest.

**Spec:** `docs/superpowers/specs/2026-08-20-v025-aiter-w16a16-dispatch.md`

## Global Constraints

- Base branch is `origin/v0.25.1`.
- `/models/zb/vllm_025/vllm` is read-only.
- `UnquantizedMoeBackend.AITER` remains independent of automatic env gates.
- Quantized/scaled inputs preserve the upstream fallback exactly.
- The final result is delivered as a separate branch and MR.

---

### Task 1: Direct configured ASM dispatch

**Files:**
- Modify: `vllm_hcu/model_executor/layers/fused_moe/aiter_runtime.py:742`
- Modify: `tests/runtime_patch/test_quant_gemm_aiter.py:1060`

**Interfaces:**
- Consumes: `get_w16a16_moe_solution_id(...) -> str` and `fused_experts_asm_impl(...) -> torch.Tensor`.
- Produces: `fused_moe_impl(...)` that always uses direct ASM for valid W16A16 AITER requests and optionally supplies `solution_id`.

- [ ] **Step 1: Write the failing configured-dispatch test**

  Add a behavior test that installs fakes for both `aiter.moe.aiter_moe` and
  `aiter.fused_moe_asm_wna16.fused_experts_asm_impl`, enables the config flag,
  and asserts the result comes from direct ASM with literal
  `solution_id="4+9"`; the unified API fake raises if called.

- [ ] **Step 2: Run the test and verify RED**

  Run:
  `pytest -q tests/runtime_patch/test_quant_gemm_aiter.py -k configured_w16a16_prefers_direct_asm`

  Expected: failure because the current implementation calls
  `aiter.moe.aiter_moe`.

- [ ] **Step 3: Implement the minimal dispatch change**

  Replace the `aiter.moe.aiter_moe` branch with:

  ```python
  solution_id = get_w16a16_moe_solution_id(...)
  direct_kwargs["solution_id"] = solution_id
  return fused_experts_asm_impl(..., **direct_kwargs)
  ```

  Build `direct_kwargs` once so configured and non-configured paths share the
  same direct call.

- [ ] **Step 4: Run focused GREEN**

  Run the exact RED command and the existing explicit-AITER/config tests.

- [ ] **Step 5: Commit the behavior change**

  Commit message: `fix(hcu): prefer direct AITER W16A16 dispatch`

### Task 2: Preserve fallback and ABI contracts

**Files:**
- Modify: `tests/runtime_patch/test_quant_gemm_aiter.py`

**Interfaces:**
- Consumes: `fused_moe_impl` from Task 1.
- Produces: regression coverage for non-W16A16 delegation and 0.25-only ABI validation.

- [ ] **Step 1: Add or tighten behavior tests**

  Assert quantized/scaled input delegates to the supplied original callable,
  and configured direct ASM preserves `expert_map`, `global_num_experts`,
  `use_shuffle`, output dtype, and optional `gemm1_limit`.

- [ ] **Step 2: Run focused tests**

  Run:
  `pytest -q tests/runtime_patch/test_quant_gemm_aiter.py -k 'aiter and (w16a16 or explicit)'`

- [ ] **Step 3: Run adjacent regression suites**

  Run:
  `pytest -q tests/runtime_patch/test_quant_gemm_aiter.py tests/runtime_patch/test_moe_deepep.py tests/runtime_patch/test_worker_framework_opt.py`

- [ ] **Step 4: Run static verification**

  Run `python -m compileall -q vllm_hcu tests` and `git diff --check` using the
  repository environment.

- [ ] **Step 5: Commit any test-only refinement**

  Commit message: `test(hcu): cover AITER W16A16 dispatch contracts`

### Task 3: HCU acceptance and separate MR

**Files:**
- Create or modify only an existing integration report location if the repository already tracks hardware reports.

**Interfaces:**
- Consumes: committed branch from Tasks 1-2.
- Produces: hardware evidence, pushed branch, and a standalone GitHub MR.

- [ ] **Step 1: Run resource preflight**

  Verify the plugin and vLLM trees are clean, the serving port is free, and
  the required HCU devices have no unrelated owners. Do not kill unrelated
  processes.

- [ ] **Step 2: Launch the target model**

  Start `/models/Qwen3.5-35B-A3B` with `--moe-backend aiter` using the known
  vLLM 0.25 environment and capture logs proving the selected backend and ASM
  route.

- [ ] **Step 3: Run the bounded request**

  Run one deterministic EvalScope/HumanEval request with a 4096-token output
  limit, check for malformed/repeated output, and confirm server health.

- [ ] **Step 4: Verify cleanup and branch state**

  Stop only the owned server process group, verify the port and devices are
  released, rerun the focused CPU suite, and confirm both repositories are
  clean.

- [ ] **Step 5: Push and open the MR**

  Push `fix/hcu-v025-aiter-w16a16-dispatch` and open a standalone MR targeting
  `v0.25.1`, including RED/GREEN, CPU, hardware, and cleanup evidence.
