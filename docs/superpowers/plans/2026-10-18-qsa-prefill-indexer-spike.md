# Qwen4Exp HCU Prefill Indexer Spike Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove whether a request-major QSA prefill score kernel beats the existing HCU CUTLASS kernel, and integrate it as an opt-in serving path only if it clears the correctness and performance gates.

**Architecture:** Add one HCU Triton operation that consumes existing QSA prefill metadata and reuses paged K tiles across rows of a request. Benchmark it directly against `flash_attn.mqa_paged_score_func`; only after a 10% target-shape win, add a narrow runtime adapter while leaving decode, mixed batches, cache handling, top-k expansion, and FP8 sparse attention unchanged.

**Tech Stack:** Python 3.10, PyTorch 2.11 DTK build, HCU Triton, Hygon FlashAttention/CUTLASS, pytest, synchronized HIP events.

**Spec:** `docs/superpowers/specs/2026-10-18-qsa-prefill-indexer-spike-design.md`

## Global Constraints

- Target hardware is BW1100 with vLLM `0.28.1+dtk2604.torch2110.2609171627.g77acaf`.
- Keep the existing FP8 sparse-attention reader and all QSA cache layouts unchanged.
- Dense prefill must be disabled in every comparison.
- Do not route decode, speculative decode, or mixed batches through the prefill kernel.
- The kernel advancement gate is numerical parity plus at least 10% lower median score time for batch 16, input 4096, over at least three rounds.
- Stop the failing shape after an HCU VMFault and collect evidence before relaunching.

## Review Focus

- A packed prefill chunk that begins in the middle of a request must use the correct absolute Q row and relative logits row; Task 1 tests chunk offsets.
- Prefix/history prefill must read the requesting row's page table and preserve distinct visible lengths; Task 1 tests nonzero history and permuted pages.
- Invalid/padded score columns must never influence top-k; Task 1 compares selected sets using the exact visible counts.
- Oversized logits must be bounded by chunking rather than one batch-sized allocation; Task 1 tests the chunk partition helper and Task 2 reports peak allocation.
- Decode and mixed batches must retain the existing CUTLASS route; Task 3 tests both fallbacks if the kernel gate advances.

---

### Task 1: Request-major prefill score kernel

**Files:**
- Create: `vllm_hcu/v1/attention/ops/qsa_prefill_indexer.py`
- Create: `tests/kernels/test_qsa_prefill_indexer.py`

**Interfaces:**
- Consumes: existing packed Q tensor, compressed K pages, page table, `query_start_loc`, and precomputed `visible_blocks` from vLLM QSA metadata.
- Produces: `qsa_mqa_paged_prefill(q, k_cache, page_table, query_start_loc, visible_blocks, max_query_len, *, query_offset=0, num_queries=None) -> torch.Tensor` and `_prefill_chunks(rows, columns, max_logits_bytes) -> tuple[tuple[int, int], ...]`.

- [ ] **Step 1: Write the failing numerical and validation tests**

  Add GPU tests comparing valid score cells and top-k sets with `flash_attn.mqa_paged_score_func` for context-free and prefixed paged inputs, including permuted physical pages and a nonzero chunk offset. Add CPU tests for chunk boundaries and malformed metadata.

- [ ] **Step 2: Run the focused test and verify RED**

  Run: `pytest -q tests/kernels/test_qsa_prefill_indexer.py`

  Expected: collection fails because `qsa_prefill_indexer` does not exist.

- [ ] **Step 3: Implement the minimal prefill operation**

  Port only the upstream request-major prefill tiling idea. Preserve explicit HCU tensor strides, validate the supported contiguous-last-dimension layout before launch, use existing `visible_blocks`, and chunk temporary logits to a fixed byte budget.

- [ ] **Step 4: Run focused and neighboring tests**

  Run: `pytest -q tests/kernels/test_qsa_prefill_indexer.py tests/runtime_patch/test_qwen4_exp_qsa_backend.py`

  Expected: all tests pass with no HCU fault.

- [ ] **Step 5: Commit**

  Run: `git add vllm_hcu/v1/attention/ops/qsa_prefill_indexer.py tests/kernels/test_qsa_prefill_indexer.py && git commit -m "feat(qsa): prototype request-major prefill scorer"`

### Task 2: CUTLASS comparison and advancement decision

**Files:**
- Create: `benchmarks/kernels/benchmark_qsa_prefill_indexer.py`
- Create: `docs/validation/pr248-qsa-prefill-indexer-spike.md`

**Interfaces:**
- Consumes: Task 1's `qsa_mqa_paged_prefill` and the installed `flash_attn.mqa_paged_score_func` baseline.
- Produces: reproducible correctness, timing, and peak-allocation evidence plus a written pass/fail decision for the 10% gate.

- [ ] **Step 1: Write the benchmark contract test**

  Add an import/argument test that requires the benchmark to expose the three mandated cases, at least three timing rounds, synchronized HIP timing, numerical checks, top-k checks, and peak allocation reporting.

- [ ] **Step 2: Run the contract test and verify RED**

  Run: `pytest -q tests/kernels/test_qsa_prefill_indexer.py -k benchmark_contract`

  Expected: fail because the benchmark module is absent.

- [ ] **Step 3: Implement the benchmark**

  Cover batch 4 × 4096 context-free, batch 16 × 4096 context-free, and batch 4 × (8192 prefix + 4096 chunk). Use identical inputs and chunking for both paths, report environment/version/commit, medians for each round, speedup, top-k agreement, and additional peak memory.

- [ ] **Step 4: Run the benchmark and write the decision record**

  Run: `HIP_VISIBLE_DEVICES=0 python benchmarks/kernels/benchmark_qsa_prefill_indexer.py --rounds 3 --warmup 5 --repetitions 10`

  Expected: all correctness checks pass; the validation note records actual measurements and explicitly says whether the batch-16 target clears 10%.

- [ ] **Step 5: Commit**

  Run: `git add benchmarks/kernels/benchmark_qsa_prefill_indexer.py docs/validation/pr248-qsa-prefill-indexer-spike.md tests/kernels/test_qsa_prefill_indexer.py && git commit -m "bench(qsa): compare prefill scorer with CUTLASS"`

### Task 3: Conditional opt-in serving route

**Files:**
- Create: `vllm_hcu/patch/worker/op_opt/patch_qwen4_exp_qsa_prefill_indexer.py`
- Create: `tests/runtime_patch/test_qwen4_exp_qsa_prefill_indexer.py`
- Modify: `vllm_hcu/platforms/envs.py`
- Modify: `vllm_hcu/patch/platform/core_fix/patch_vllm_config.py`
- Modify: `vllm_hcu/patch/worker/__init__.py`
- Modify: `tests/accuracy/test_environment_routing.py`
- Modify: `tests/patch/test_config.py`
- Modify: `tests/runtime_patch/test_worker_framework_opt.py`

**Interfaces:**
- Consumes: Task 1's prefill scorer and Task 2's passing advancement decision.
- Produces: `VLLM_HCU_QSA_PREFILL_INDEXER=1`, additionally gated by `VLLM_HCU_USE_CUSTOM_OPS`, routing only prefill-only QSA selection batches to the specialized scorer.

- [ ] **Step 1: Stop if Task 2 did not clear the gate**

  Record the failed gate in the execution ledger and omit every remaining Task 3 and Task 4 code change.

- [ ] **Step 2: Write failing environment and routing tests**

  Test default-off/master-gate behavior, pure context-free prefill, prefixed prefill, decode, speculative decode, mixed batches, disabled mode, repeated patch application, and output-buffer identity.

- [ ] **Step 3: Run tests and verify RED**

  Run: `pytest -q tests/runtime_patch/test_qwen4_exp_qsa_prefill_indexer.py tests/accuracy/test_environment_routing.py tests/patch/test_config.py tests/runtime_patch/test_worker_framework_opt.py`

  Expected: new environment/routing assertions fail before implementation.

- [ ] **Step 4: Implement the minimal opt-in adapter**

  Reuse existing vLLM top-k and expansion operations. Route only when metadata has zero decode rows and at least one prefill row; otherwise call the original `_select` unchanged. Perform all shape/capability checks before launching a kernel.

- [ ] **Step 5: Run focused and full neighboring suites**

  Run: `pytest -q tests/kernels/test_qsa_prefill_indexer.py tests/runtime_patch/test_qwen4_exp_qsa_backend.py tests/runtime_patch/test_qwen4_exp_qsa_prefill_indexer.py tests/runtime_patch/test_qwen4_exp_qsa_dense_prefill.py tests/accuracy/test_environment_routing.py tests/patch/test_config.py tests/runtime_patch/test_worker_framework_opt.py`

  Expected: all tests pass.

- [ ] **Step 6: Commit**

  Run: `git add vllm_hcu tests && git commit -m "feat(qsa): add opt-in prefill indexer route"`

### Task 4: Conditional serving A/B

**Files:**
- Modify: `docs/validation/pr248-qsa-prefill-indexer-spike.md`

**Interfaces:**
- Consumes: Task 3's opt-in route and the existing BS16 input-4096/output-1024 harness.
- Produces: three-warmup/five-measurement A/B results, trace attribution, accuracy smoke result, and VMFault audit.

- [ ] **Step 1: Install the worktree plugin without dependency changes**

  Run: `pip install --force-reinstall --no-deps <worktree>`

  Expected: installed plugin source hashes match the worktree commit.

- [ ] **Step 2: Run OFF and ON serving measurements**

  Keep model, DP4/EP4, MTP, scheduler, sampling, max batched tokens, and dense-prefill setting identical. Run three warmups and five measured rounds per side at batch 16, input 4096, output 1024.

- [ ] **Step 3: Verify trace and system health**

  Confirm the ON trace contains the prefill scorer, decode retains the CUTLASS path, all requests succeed, and `dmesg` contains no new HCU VMFault or invalid-address event.

- [ ] **Step 4: Record and commit the outcome**

  Add TTFT, TPOT, throughput, request success, accuracy smoke, trace totals, and the go/no-go decision to the validation note.

  Run: `git add docs/validation/pr248-qsa-prefill-indexer-spike.md && git commit -m "docs(qsa): record prefill indexer validation"`

  Expected: the note distinguishes kernel gain from end-to-end gain and does not recommend PR 248 unless both correctness and reproducible serving performance pass.

