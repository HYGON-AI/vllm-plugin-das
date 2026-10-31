# vLLM 0.32 HCU FlashAttention E5M2 KV Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add model-independent `fp8_e5m2` KV-cache support to the Plugin-owned vLLM 0.32 HCU `FLASH_ATTN` backend and validate it across layouts, Graph, and model families.

**Architecture:** vLLM keeps cache allocation and native writes; the Plugin narrowly adapts the upstream constructor capability check and converts byte-backed K/V views to E5M2 at the public vendor boundary. Adapter and vLLM remain unchanged, and the implementation contains no model-name routing.

**Tech Stack:** Python, PyTorch/HIP, vLLM 0.32 attention seams, HCU `flash_attn_varlen_func`, pytest, isolated wheels.

**Spec:** `docs/superpowers/specs/2026-10-31-vllm032-flash-attn-e5m2-design.md`

## Global Constraints

- Plugin-only runtime change; do not modify vLLM or Adapter.
- Preserve official `reshape_and_cache_flash`, LBHNC/LBNHC, block size 64, and default Graph mode.
- Do not change global platform FP8 dtype or add model/checkpoint allowlists.
- Keep MLA, sparse MLA, multimodal-prefix, hybrid, DCP, PCP, sliding-window, sink, and RSWA behavior unchanged and fail closed.
- Use TDD for every production behavior and compare model E5M2 with an otherwise identical BF16-KV run.

## Review Focus

- Positional versus keyword `kv_cache_dtype` constructor calls must both restore `fp8_e5m2` after the upstream guard.
- K/V may arrive at the vendor boundary as uint8, E4M3-viewed bytes, or already E5M2; only one-byte storage may be reinterpreted.
- BF16/FP16 and any unsupported quantized dtype must keep their existing paths and validation results.
- E5M2 must keep query compute dtype and pass K/V descales without host synchronization or graph-visible data-dependent branching.
- HND/NHD padded blocks and overwrite/replay must preserve storage offsets, strides, and changed-input results.

---

### Task 1: Portable E5M2 backend contract

**Files:**
- Modify: `tests/vllm032/test_flash_attention.py`
- Modify: `adapter_extension/src/hcu_vllm_das/attention/flash_attn.py`

**Interfaces:**
- Consumes: vLLM 0.32 `FlashAttentionBackend`, `FlashAttentionImpl`, and existing `_run_flash_attention` seam.
- Produces: E5M2-aware constructor state, backend validation, and vendor-boundary K/V dtype normalization.

- [ ] **Step 1: Write failing portable tests**

Add tests that require `fp8_e5m2` advertisement and validation, keyword and positional constructor restoration, `supports_quant_query_input=False`, and K/V reinterpretation to `torch.float8_e5m2` while q, scales, layouts, and non-E5M2 calls remain unchanged.

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `python -m pytest -q tests/vllm032/test_flash_attention.py`

Expected: failures state that E5M2 is not advertised/accepted and vendor receives the wrong K/V dtype.

- [ ] **Step 3: Implement the minimal Plugin adaptation**

Extend the HCU backend capability methods, pass `auto` only through the upstream constructor when the requested dtype is E5M2, restore E5M2 state and compute-dtype q behavior, and reinterpret one-byte K/V at `_run_flash_attention` before vendor dispatch. Delegate every non-E5M2 combination to upstream behavior.

- [ ] **Step 4: Run focused and portable v0.32 tests**

Run: `python -m pytest -q tests/vllm032/test_flash_attention.py`

Expected: all focused tests pass.

Run: `VLLM_TARGET_ROOT=<candidate-site-packages> python -m pytest -q tests/vllm032`

Expected: `126 + new tests` pass and only hardware tests skip.

- [ ] **Step 5: Commit**

Commit the test and implementation with a root-cause message describing the upstream NVIDIA E5M2 guard and HCU vendor byte contract.

### Task 2: Live HCU writer-reader and Graph matrix

**Files:**
- Modify: `tests/vllm032/test_flash_attention_hcu.py`

**Interfaces:**
- Consumes: Task 1's E5M2 vendor-boundary normalization.
- Produces: numerical and changed-input Graph evidence for native writer to Plugin reader.

- [ ] **Step 1: Write E5M2 HCU tests before any further production change**

Cover LBHNC/LBNHC, prefill/decode, MHA/GQA, padded two-block storage, block 1 writes, cache overwrite, and Graph replay. Build the reference from `torch.float8_e5m2`-quantized K/V bytes with literal tolerances.

- [ ] **Step 2: Run the E5M2 subset and verify its current result**

Run: `VLLM032_HCU_TEST=1 <candidate-python> -m pytest -q tests/vllm032/test_flash_attention_hcu.py -k e5m2`

Expected: pass after Task 1; any numerical/provider failure triggers systematic debugging, not a fallback backend.

- [ ] **Step 3: Run the full HCU attention matrix**

Run: `VLLM032_HCU_TEST=1 <candidate-python> -m pytest -q tests/vllm032/test_flash_attention_hcu.py`

Expected: existing BF16/FP16 rows and new E5M2 rows all pass.

- [ ] **Step 4: Commit**

Commit the HCU tests separately so the hardware evidence is reviewable.

### Task 3: Final wheel and installed-state verification

**Files:**
- Modify: `docs/validation/vllm032-flash-attn-varlen.md`

**Interfaces:**
- Consumes: Task 1 implementation and Task 2 test matrix.
- Produces: reproducible final artifact identity and installed validation evidence.

- [ ] **Step 1: Build and hash a fresh wheel**

Build from the branch HEAD, record wheel name/SHA256, install with `--no-deps --target` into a new isolated directory, and verify module origins do not resolve from the source tree.

- [ ] **Step 2: Run package, compatibility, installed-state, portable, and live HCU gates from the isolated installation**

Expected: every required capability is present, package ownership is unchanged, and the final wheel repeats the complete HCU matrix.

- [ ] **Step 3: Record commands and results in the validation document**

Distinguish source tests, installed-wheel tests, numerical tolerances, Graph replay, skipped scope, and artifact hashes.

- [ ] **Step 4: Commit**

Commit the artifact and HCU evidence documentation.

### Task 4: Cross-model BF16/E5M2 A/B

**Files:**
- Modify: `docs/validation/vllm032-flash-attn-varlen.md`

**Interfaces:**
- Consumes: Task 3 isolated final wheel.
- Produces: checkpoint-specific model accuracy and runtime evidence.

- [ ] **Step 1: Run Qwen dense controls**

Run otherwise identical BF16-KV and E5M2-KV services with explicit `FLASH_ATTN`, both layouts across the available Qwen rows, default Graph, deterministic changed-input requests, prefix reuse, and HumanEval/0-15.

- [ ] **Step 2: Run Meta-Llama dense controls**

Use `/model/LLM-Research/Meta-Llama-3.1-8B-Instruct` with identical BF16/E5M2 commands and evaluation protocol; retain raw samples, finish reasons, token counts, and scores.

- [ ] **Step 3: Run one non-Qwen MoE control**

Prefer `/model/Hy3-preview-W8A8` at a fitting multi-card topology with `--moe-backend aiter`; if its vLLM/model/provider contract is independently blocked, preserve the reproduced blocker and try `/model/MiniMax-M2.5-bf16`. Dense fallback is not a substitute.

- [ ] **Step 4: Record A/B results and teardown**

Document exact model paths, layouts, topology, Graph mode, dtype logs, provider evidence, samples, scores, and any blocker. Stop all owned processes and verify device memory returns to baseline.

- [ ] **Step 5: Commit**

Commit cross-model evidence without weakening unsupported backend boundaries.

### Task 5: Final verification and PR update

**Files:**
- Modify only if review finds an issue in the files above.

**Interfaces:**
- Consumes: Tasks 1-4.
- Produces: reviewed, pushed PR head with reproducible evidence.

- [ ] **Step 1: Run final gates**

Run full portable v0.32 tests, final-wheel HCU matrix, package/installed-state checks, `git diff --check`, residual-process checks, and device-memory checks.

- [ ] **Step 2: Prepare and perform whole-branch review**

Review against the spec, plan, upstream-first ownership, the five Review Focus items, and all ledger rulings. Fix Critical/Important findings through RED-GREEN once.

- [ ] **Step 3: Push and update PR #321**

Post artifact hashes, commands, raw model A/B results, explicit limitations, and the final head SHA. Confirm the PR is cleanly mergeable.
