# AutoAWQ BF16 Triton Compatibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove HCU AutoAWQ small-batch GEMM compiles and executes with BF16 activations and keep that production DTK behavior covered in CI.

**Architecture:** Keep the production DTK Triton kernel, whose frontend explicitly supports BF16/FP16 dot output, and exercise it through the real vLLM AutoAWQ method. Register a device accuracy regression on gfx936 and gfx938 instead of maintaining a plugin-owned kernel fork.

**Tech Stack:** Python, PyTorch 2.11, Triton 3.6, vLLM 0.28.1, pytest, HCU/DTK 26.10.

**Spec:** `docs/superpowers/specs/2026-10-30-autoawq-bf16-triton-design.md`

## Global Constraints

- Cover M=1, 255, and 256 with K=N=128 and group size 128.
- Exercise real Triton compilation/execution for the HCU-only regression.
- Preserve FP16, LightOp fallback, and non-AWQ behavior.

## Review Focus

- BF16 M=1 and M=255 must use a supported accumulator type and return BF16.
- M=256 must retain the dequantize-plus-matmul branch.
- FP16 must remain numerically consistent with dequantized reference output.
- LightOp-disabled BF16 must use the fixed standard AutoAWQ route.
- A DTK wheel that loses BF16 dot-output support must fail the real-device CI.

---

### Task 1: Reproduce the production failure

**Files:**
- Inspect: installed `vllm/model_executor/layers/quantization/utils/awq_triton.py`

**Interfaces:**
- Consumes: vLLM `awq_gemm_triton` and `awq_dequantize_triton`.
- Produces: a recorded production-HCU reproduction at M=1/255/256.

- [ ] Run FP16 and BF16 probes against the installed kernels.
- [ ] Confirm the exact compiler exception and dtype/token boundary.

### Task 2: Add a real-device BF16 regression

**Files:**
- Create: `tests/accuracy/test_autoawq_triton_bf16.py`
- Modify: `tests/hcu_ci_registry.py`

**Interfaces:**
- Consumes: the installed production DTK/vLLM AutoAWQ kernels.
- Produces: tests asserting BF16 M=1/255 compile/execute and M=256 stays on dequantize on gfx936/gfx938.

- [ ] Write the focused BF16 test with real Triton kernels where HCU is available.
- [ ] Isolate LightOp's BF16 dtype fallback by forcing shape capability true.
- [ ] Run it against HEAD and record the production-kernel result.

### Task 3: Verify no production patch is required

**Files:**
- Inspect: installed DTK Triton `semantic.py`.
- Verify: existing HCU platform and LightOp AutoAWQ tests.

**Interfaces:**
- Consumes: the real-kernel result from Task 2.
- Produces: evidence that the DTK frontend supports BF16 and surrounding dispatch remains green.

- [ ] Confirm DTK's BF16/FP16 dot-output semantic branches.
- [ ] Do not add a kernel fork when the production path succeeds.
- [ ] Run platform and LightOp AutoAWQ test files.

### Task 4: Verify, commit, and update the PR

**Files:**
- Verify all changed files and generated wheel.

**Interfaces:**
- Consumes: Tasks 1-3.
- Produces: pushed PR head with exact commands, results, and test boundaries.

- [ ] Rerun the real HCU probe against the installed production wheel.
- [ ] Run the repository's relevant contract suite and report any unrelated failure.
- [ ] Review the final diff, commit, push without force, and update PR #317.
