# Qwen3.5 TP2 LightOp W16A16 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Safely enable the opt-in LightOp W16A16 backend for Qwen3.5-35B-A3B TP2 through an 8192-token scheduler budget.

**Architecture:** Correct the scratch/output lifetime layout at the modular-kernel boundary, then narrowly admit the Qwen3.5 TP2 shape and gfx936 M=1 config workaround. Preserve fallback for missing configs and every unsupported model/device combination.

**Tech Stack:** Python 3.12, PyTorch 2.11 DTK 26.10, vLLM 0.28.1, LightOp 0.6, pytest

**Spec:** `docs/superpowers/specs/2026-10-20-qwen35-lightop-tp2-design.md`

## Global Constraints

- LightOp remains opt-in behind both existing environment flags and `moe_backend=auto`.
- Preserve BF16, unmodified SiLU, no bias/shared-expert fusion, pure TP, and no EP/DP/PCP/SP restrictions.
- Admit only `(256,2048,512,8)` and `(256,2048,256,8)` problem shapes.
- Apply the M=1-to-M=2 lookup only to gfx936/80 CU Qwen3.5 TP2.
- Maximum supported scheduler token budget is 8192 for Qwen3.5 TP2; the existing N=512 shape remains capped at 16.
- Missing or invalid LightOp configs must select the official backend before weight packing.

## Review Focus

- vLLM output aliasing `workspace13` must never overlap live reduction input; Task 1 tests exact storage ranges.
- Swapping scratch roles must not overlap GEMM1/activation or GEMM2 input/output; Task 1 captures every operator view.
- Other architectures must keep native M=1 configs; Task 2 tests gfx936 and non-gfx936 lookup separately.
- Unsupported MoE shapes and token budgets above 8192 must still fall back before LightOp probing/packing; Task 2 tests both boundaries.
- Config preflight must not make 8192 Python/package lookups; Task 2 asserts only M=1 and M=max are probed.

---

### Task 1: Remove modular output/cache overlap

**Files:**
- Modify: `vllm_hcu/model_executor/layers/fused_moe/experts/lightop_w16a16_moe.py`
- Modify: `vllm_hcu/model_executor/layers/fused_moe/lightop_w16a16_runtime.py`
- Modify: `tests/runtime_patch/test_lightop_w16a16_moe.py`

**Interfaces:**
- Consumes: vLLM's common allocation for `workspace13` and expert output.
- Produces: non-overlapping `cache1/cache2/cache3/output` live ranges without extra Qwen3.5 scratch allocation.

- [x] Add a failing test that allocates output exactly as vLLM does and captures all LightOp operator tensor ranges.
- [x] Verify RED because current reduction input overlaps output.
- [x] Swap workspace shapes and runtime cache carving according to the approved lifetime design.
- [x] Verify the focused test and complete LightOp contract file pass.
- [x] Commit with `fix(moe): isolate lightop reduction output`.

### Task 2: Admit Qwen3.5 TP2 and bounded configs

**Files:**
- Modify: `vllm_hcu/model_executor/layers/fused_moe/experts/lightop_w16a16_moe.py`
- Modify: `vllm_hcu/model_executor/layers/fused_moe/lightop_w16a16_runtime.py`
- Modify: `vllm_hcu/patch/worker/op_opt/moe/patch_unquantized_oracle.py`
- Modify: `tests/runtime_patch/test_lightop_w16a16_moe.py`

**Interfaces:**
- Consumes: LightOp config lookup and existing opt-in oracle.
- Produces: Qwen3.5 TP2 eligibility through 8192 tokens, constant-count config preflight, and gfx936/80-CU one-token config substitution.

- [x] Add failing tests for N=256 eligibility, 8192/8193 boundary, constant-count preflight, and gfx936-only M=1 substitution.
- [x] Verify every test fails for the intended current gate/lookup behavior.
- [x] Implement the minimal shape, budget, preflight, and lookup changes.
- [x] Verify all focused tests and the complete LightOp contract file pass.
- [x] Commit with `fix(moe): enable qwen35 lightop tp2 configs`.

### Task 3: HCU and model validation

**Files:**
- No production file changes unless a new RED test first reproduces a discovered defect.

**Interfaces:**
- Consumes: Tasks 1 and 2 through the worktree on the current DTK 26.10 environment.
- Produces: operator, graph, service readiness, request correctness, and performance evidence.

- [ ] Run an actual LightOp operator with vLLM-style aliased output for M=1,2,8,16 on both visible devices.
- [ ] Run changed-input graph replay with the aliased layout.
- [ ] Start Qwen3.5-35B-A3B TP2 with the requested 8192-token budget and wait for readiness.
- [ ] Send a deterministic request and record output/status.
- [ ] Compare short-request latency against the official auto backend using identical settings.
- [ ] Stop owned processes and verify all HCU memory is released.
