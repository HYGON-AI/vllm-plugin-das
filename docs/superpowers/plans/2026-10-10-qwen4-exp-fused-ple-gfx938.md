# Qwen4Exp Fused PLE gfx938 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Backport upstream vLLM's AMD fused Qwen4Exp PLE path into the HCU plugin and retain it only if unit and gfx938 validation demonstrate correctness and useful improvement.

**Architecture:** Ship the upstream plain-Triton PLE kernels in a plugin-owned common module, integrate them through the existing cold Qwen4Exp PLE module exchange, and preserve the plugin-owned INT8/SlimQuant ETP/UVA storage layer. Use a narrow signature-checked AMD model patch for packed `kv_proj` loading and residual ownership, while `VLLM_HCU_USE_CUSTOM_OPS=0` retains the installed vLLM fallback.

**Tech Stack:** Python 3.10, PyTorch 2.11, Triton/DTK 26.04, vLLM 0.28.1, pytest, gfx938 HCU runtime, EvalScope HumanEval.

**Spec:** `docs/superpowers/specs/2026-10-10-qwen4-exp-fused-ple-gfx938-design.md`

## Global Constraints

- Base the branch on PR #199 commit `5d5fe2f4ea42340a9b42f001bdcc33e59da66482`; do not amend PR #199.
- Compare behavior with upstream vLLM main `90ba2f34b5a29d85b8301672e5735676f78fa00c` and PR #60021 merge commit `e2414ec1dbb177b54ac1bbce60e94769ebacbb28`.
- Do not change the installed vLLM package or require a vLLM version newer than 0.28.1.
- Preserve compressed-tensors INT8, SlimQuant INT8/unquantized, ETP-across-DP, UVA offload, prefetch, MTP3, and FULL_AND_PIECEWISE behavior.
- Every optimized operator must obey `VLLM_HCU_USE_CUSTOM_OPS`; the disabled state must retain a supported fallback.
- CPU offload is supported only with a live UVA alias and sufficient pinned-memory capacity; do not claim it from unit tests alone.
- Keep PLE metadata-builder PR #58114, DeepSeek V4.1 shared host tables, THP, and new quantization formats out of this change.
- Follow test-first development: observe the focused test fail for the missing behavior before editing production code.
- Use `apply_patch` for source and documentation edits.

## Review Focus

- Mixed MTP/prefill batches must preserve token routing, update the correct PLE state rows, and add the decoder residual exactly once; Task 3 adds mixed-batch and residual assertions.
- ETP-across-DP must handle uneven token counts and a rank with zero local tokens without selecting another rank's rows; Task 2 adds slot/gather/select coverage.
- Side-stream UVA lookup must never consume n-gram IDs whose graph-pool storage has been reused; Task 2 extends the persistent-buffer lifetime test.
- Packed `kv_proj` must load separate key/value checkpoint tensors and preserve quantization exclude semantics; Task 3 adds mapper and packed-module tests.
- `VLLM_HCU_USE_CUSTOM_OPS=0` must avoid importing or registering fused PLE operators while keeping the official fallback usable; Task 4 adds cold-import and dispatcher tests.

---

### Task 1: Plugin-owned fused PLE Triton operators

**Files:**

- Create: `vllm_hcu/models/qwen4_exp/common/__init__.py`
- Create: `vllm_hcu/models/qwen4_exp/common/ops/__init__.py`
- Create: `vllm_hcu/models/qwen4_exp/common/ops/ple.py`
- Create: `tests/runtime_patch/test_qwen4_exp_fused_ple_ops.py`

**Interfaces:**

- Consumes: PyTorch tensors, vLLM `direct_register_custom_op`, current platform PDL capability, and `PleShortConvAttentionMetadata` tensor layouts.
- Produces: `ple_ngram_ids(..., output: torch.Tensor | None = None) -> torch.Tensor`, `ple_gate(...) -> tuple[torch.Tensor, torch.Tensor]`, and `ple_conv(..., mode: Literal["decode", "spec", "prefill"], ...) -> None` with the argument and mutation contracts from upstream #60021.

- [ ] **Step 1: Write failing operator contract tests**

  Add tests named `test_fused_ple_ops_export_upstream_interfaces`, `test_ple_ngram_ids_matches_reference`, `test_ple_gate_matches_reference`, and parametrized `test_ple_conv_matches_reference`. Assert importable signatures first; on an accelerator, compare exact IDs and numerical outputs for decode, prefill, speculative decode, mixed token maps, EOS boundaries, empty requests, and both cache layouts.

- [ ] **Step 2: Run the focused test and verify RED**

  Run: `pytest -q tests/runtime_patch/test_qwen4_exp_fused_ple_ops.py`

  Expected: FAIL because `vllm_hcu.models.qwen4_exp.common.ops.ple` does not exist.

- [ ] **Step 3: Port the minimal upstream common operators**

  Copy the three plain-Triton kernels and thin custom-op wrappers from upstream #60021 into `vllm_hcu.models.qwen4_exp.common.ops.ple`. Preserve PDL capability gating, mutation declarations, request-count validation, state-layout handling, and token-map behavior; change only imports and plugin namespace details required by vLLM 0.28.1.

- [ ] **Step 4: Run operator tests and formatting checks**

  Run: `pytest -q tests/runtime_patch/test_qwen4_exp_fused_ple_ops.py`

  Run: `pre-commit run --files vllm_hcu/models/qwen4_exp/common/__init__.py vllm_hcu/models/qwen4_exp/common/ops/__init__.py vllm_hcu/models/qwen4_exp/common/ops/ple.py tests/runtime_patch/test_qwen4_exp_fused_ple_ops.py`

  Expected: PASS; accelerator-only numerical tests may skip only when no accelerator is visible.

- [ ] **Step 5: Commit Task 1**

  ```bash
  git add vllm_hcu/models/qwen4_exp/common tests/runtime_patch/test_qwen4_exp_fused_ple_ops.py
  git commit -m "feat(qwen4-exp): add fused PLE Triton operators"
  ```

### Task 2: Fused n-gram IDs with ETP and UVA prefetch

**Files:**

- Modify: `vllm_hcu/models/qwen4_exp/amd/ple_layer.py`
- Modify: `vllm_hcu/patch/worker/core_fix/patch_qwen4_exp_ple_conv.py`
- Modify: `tests/runtime_patch/test_qwen4_exp_ple_conv.py`
- Modify: `tests/runtime_patch/test_qwen4_exp_ple_prefetch.py`

**Interfaces:**

- Consumes: Task 1 `ple_ngram_ids`, existing `HcuPLEVocabParallelEmbedding._get_dp_gather_slot`, `_gather_dp_ids`, `_select_embeddings`, and persistent `_hcu_prefetch_ids_buffer`/`_hcu_prefetch_rows_buffer`.
- Produces: `Qwen4ExpNGramEmbedding.compute_ngram_ids(..., output: torch.Tensor | None = None) -> torch.Tensor` without padded request workspaces; prefetch continues to write IDs into persistent storage before UVA lookup.

- [ ] **Step 1: Write failing integration tests**

  Add `test_fused_ngram_removes_padded_request_workspace`, `test_fused_ngram_writes_requested_output_buffer`, `test_fused_ngram_handles_empty_dp_slot`, and `test_prefetch_ids_survive_graph_break_with_etp_padding`. Assert no `padded_buffer`/`positions_buffer`, output-buffer identity, uneven counts `[0, 3, 1, 2]`, correct local-row selection, and persistent ID storage across a simulated eager graph break.

- [ ] **Step 2: Run the focused tests and verify RED**

  Run: `pytest -q tests/runtime_patch/test_qwen4_exp_ple_conv.py tests/runtime_patch/test_qwen4_exp_ple_prefetch.py`

  Expected: FAIL because the existing implementation constructs padded buffers and `compute_ngram_ids` has no `output` contract.

- [ ] **Step 3: Integrate fused ID generation**

  Replace the Python shift/scatter path in the HCU replacement module with Task 1 `ple_ngram_ids`. During prefetch, slice `_hcu_prefetch_ids_buffer` to the local token count and pass it as `output`; retain ETP gather, local shard masking, UVA lookup, row reduction, and local slot selection unchanged.

- [ ] **Step 4: Simplify the graph-break adapter**

  Update `patch_qwen4_exp_ple_conv.py` so its opaque n-gram wrapper delegates to the fused ID path without reintroducing temporary ID ownership. Keep strict module/signature checks and the existing sentinel contract until all prefetch tests prove it can be safely reduced.

- [ ] **Step 5: Run focused and storage regressions**

  Run: `pytest -q tests/runtime_patch/test_qwen4_exp_ple_conv.py tests/runtime_patch/test_qwen4_exp_ple_prefetch.py tests/runtime_patch/test_qwen4_exp_ple_int8_offload.py`

  Expected: PASS with uneven/empty DP-slot and persistent-buffer cases covered.

- [ ] **Step 6: Commit Task 2**

  ```bash
  git add vllm_hcu/models/qwen4_exp/amd/ple_layer.py vllm_hcu/patch/worker/core_fix/patch_qwen4_exp_ple_conv.py tests/runtime_patch/test_qwen4_exp_ple_conv.py tests/runtime_patch/test_qwen4_exp_ple_prefetch.py
  git commit -m "feat(qwen4-exp): fuse PLE ngram id generation"
  ```

### Task 3: Packed KV projection, fused gate, and fused convolution

**Files:**

- Modify: `vllm_hcu/models/qwen4_exp/amd/ple_layer.py`
- Create: `vllm_hcu/patch/worker/core_fix/patch_qwen4_exp_fused_ple.py`
- Modify: `vllm_hcu/patch/worker/core_fix/__init__.py`
- Modify: `vllm_hcu/patch/worker/__init__.py`
- Modify: `tests/runtime_patch/test_qwen4_exp_fused_ple_ops.py`
- Create: `tests/runtime_patch/test_qwen4_exp_fused_ple_model.py`

**Interfaces:**

- Consumes: Task 1 `ple_gate`/`ple_conv`, vLLM `MergedColumnParallelLinear`, `WeightsMapper`, AMD `Qwen4ExpDecoderLayer`, `Qwen4ExpModel`, `Qwen4ExpForCausalLM`, and `Qwen4ExpForConditionalGeneration`.
- Produces: `Qwen4ExpPLELayer.kv_proj`; patch adapter `apply_to_module(module: ModuleType) -> bool` that installs exact packed-weight metadata and fused residual ownership only when the audited vLLM 0.28.1 targets match. The adapter leaves the MTP draft forward path unchanged.

- [ ] **Step 1: Write failing model-integration tests**

  Add `test_fused_ple_uses_one_replicated_kv_projection`, `test_key_and_value_checkpoint_weights_map_to_kv_shards`, `test_quantization_excludes_match_original_projection_names`, `test_decoder_adds_residual_exactly_once`, `test_fused_ple_does_not_wrap_mtp_forward`, and parametrized `test_fused_conv_mixed_batch_state_and_residual`. Use synthetic modules for mapper/dispatcher contracts and accelerator reference comparison for state/output semantics.

- [ ] **Step 2: Run the focused tests and verify RED**

  Run: `pytest -q tests/runtime_patch/test_qwen4_exp_fused_ple_ops.py tests/runtime_patch/test_qwen4_exp_fused_ple_model.py`

  Expected: FAIL because the current PLE layer owns separate `key_proj`/`value_proj`, composed gating, and the old convolution path.

- [ ] **Step 3: Integrate `kv_proj` and fused gate**

  Replace the two replicated projections with one replicated `MergedColumnParallelLinear` using output sizes `[hc_hidden_size, hidden_size]`. Split its output and call Task 1 `ple_gate`, preserving model dtype, norm weights, epsilon, dequantization, and quantization configuration.

- [ ] **Step 4: Integrate fused convolution**

  Replace the existing decode/prefill/spec branches with Task 1 `ple_conv`, retain both vLLM cache layouts, and pass the decoder residual exactly as specified by upstream #60021. Keep `qwen4_exp_ple_short_conv` a graph splitting op because it consumes runtime request metadata.

- [ ] **Step 5: Add the narrow model compatibility patch**

  Implement `patch_qwen4_exp_fused_ple.apply_to_module` with exact signature and class-shape checks. Extend the weight mapper with `ple.key_proj -> ple.kv_proj[0]` and `ple.value_proj -> ple.kv_proj[1]`, add `kv_proj` packed metadata to causal and conditional model classes, and wrap decoder forward so the fused PLE result already containing the residual is not added a second time. Make repeated application idempotent and roll back partial mutation on failure.

- [ ] **Step 6: Register the adapter and run focused regressions**

  Run: `pytest -q tests/runtime_patch/test_qwen4_exp_fused_ple_ops.py tests/runtime_patch/test_qwen4_exp_fused_ple_model.py tests/runtime_patch/test_qwen4_exp_ple_conv.py tests/runtime_patch/test_qwen4_exp_ple_prefetch.py tests/runtime_patch/test_qwen4_exp_ple_int8_offload.py tests/runtime_patch/test_qwen4_exp_pp.py`

  Expected: PASS, including mixed MTP/prefill state updates and one residual addition.

- [ ] **Step 7: Commit Task 3**

  ```bash
  git add vllm_hcu/models/qwen4_exp/amd/ple_layer.py vllm_hcu/patch/worker/core_fix vllm_hcu/patch/worker/__init__.py tests/runtime_patch/test_qwen4_exp_fused_ple_ops.py tests/runtime_patch/test_qwen4_exp_fused_ple_model.py
  git commit -m "feat(qwen4-exp): enable fused PLE execution"
  ```

### Task 4: Master-gate fallback and complete unit verification

**Files:**

- Modify: `vllm_hcu/platforms/envs.py`
- Modify: `vllm_hcu/patch/module_exchange.py`
- Modify: `tests/patch/test_module_exchange.py`
- Modify: `tests/patch/test_worker_dispatcher.py`
- Modify: `tests/runtime_patch/test_qwen4_exp_fused_ple_model.py`

**Interfaces:**

- Consumes: `custom_ops_enabled()`, exact import coordinator, Task 3 patch adapter.
- Produces: a cold-import selection in which custom ops enabled loads the HCU fused PLE replacement and custom ops disabled leaves the installed vLLM fallback untouched.

- [ ] **Step 1: Write failing fallback tests**

  Add `test_fused_ple_exchange_requires_custom_ops_master`, `test_custom_ops_zero_does_not_import_fused_ple_ops`, `test_official_ple_fallback_remains_importable`, `test_fused_ple_patch_fails_closed_on_signature_drift`, and `test_fused_ple_logs_selected_mode_once`. Assert cold coordinator registrations, `sys.modules` contents, and the one-time fused/fallback audit log, not only helper return values.

- [ ] **Step 2: Run the fallback tests and verify RED**

  Run: `pytest -q tests/patch/test_module_exchange.py tests/patch/test_worker_dispatcher.py tests/runtime_patch/test_qwen4_exp_fused_ple_model.py`

  Expected: FAIL because the module exchange currently depends only on the prefetch request and has no fused-PLE fallback contract.

- [ ] **Step 3: Implement master-gated cold selection**

  Separate fused-PLE module eligibility from UVA prefetch eligibility. Arm the HCU PLE replacement under `custom_ops_enabled()` and apply the prefetch-specific adapters only under `ple_prefetch_enabled()`. Preserve exact import ordering, emit one fused/fallback audit message, and reject a canonical PLE module imported before exchange registration.

- [ ] **Step 4: Run all Qwen4Exp tests**

  Run: `pytest -q tests/patch/test_module_exchange.py tests/patch/test_worker_dispatcher.py tests/runtime_patch/test_qwen4_exp_*.py`

  Expected: PASS.

- [ ] **Step 5: Run the complete plugin suite and static checks**

  Run: `pytest -q`

  Run: `pre-commit run --all-files`

  Run: `git diff --check origin/codex/engram-across-dp...HEAD`

  Expected: all tests and hooks pass; report any unrelated baseline failure by exact test name instead of suppressing it.

- [ ] **Step 6: Commit Task 4**

  ```bash
  git add vllm_hcu/platforms/envs.py vllm_hcu/patch/module_exchange.py tests/patch/test_module_exchange.py tests/patch/test_worker_dispatcher.py tests/runtime_patch/test_qwen4_exp_fused_ple_model.py
  git commit -m "test(qwen4-exp): verify fused PLE fallback policy"
  ```

### Task 5: gfx938 acceptance, documentation, skill, and stacked PR

**Files:**

- Modify: `/models/gfx938-vllm-0281-validated-model-commands.md`
- Modify: `/models/upgrading-vllm-hcu/SKILL.md`
- Modify: `/models/upgrading-vllm-hcu/references/validation.md`
- Modify only if needed for measured issues: files owned by Tasks 1-4 and their tests

**Interfaces:**

- Consumes: completed Tasks 1-4, Qwen3.8 SlimQuant n-gram INT8 checkpoint, four free gfx938 cards, EvalScope, current PR #199 serve/eval commands.
- Produces: runtime evidence, an integration/no-integration decision, updated reusable commands and rules, pushed branch, and one PR stacked on #199.

- [ ] **Step 1: Establish the hardware baseline**

  Verify four cards are free and record plugin/vLLM commits. Run the existing accelerator-resident TP1/DP4/EP4/ETP4 MTP3 FULL_AND_PIECEWISE command from the command document before installing the candidate. Record startup, PLE mode, memory, HumanEval16, MTP acceptance, and a fixed random-serving benchmark.

- [ ] **Step 2: Install the candidate and run functional acceptance**

  Build/install the plugin branch without changing the vLLM wheel. Run the same topology with `--engram-config '{"cpu_offload":false,"embedding_across_dp":true}'`, `allgather_reducescatter`, and AITER MoE. Require successful startup, fused-path log evidence, and HumanEval16 equal to the current 16/16 result.

- [ ] **Step 3: Measure the optimization**

  Repeat identical workloads for control and candidate starts. Record PLE kernel launches, output tokens/s, TPOT, TTFT, MTP acceptance, and peak accelerator memory. Integrate only if correctness holds and the candidate shows repeatable throughput/latency improvement or materially removes the padded workspace; otherwise retain the research commits locally and do not publish a performance claim.

- [ ] **Step 4: Attempt CPU offload only when capacity permits**

  Run `cpu_offload=true`, `embedding_across_dp=true`, and `VLLM_HCU_PLE_PREFETCH_STREAM=1` only after confirming enough host RAM and locked/pinned capacity. Record per-rank and aggregate pinned allocation; if allocation fails, document the capacity limit and keep CPU offload unvalidated rather than changing semantics to pageable memory.

- [ ] **Step 5: Update commands and skill**

  Add the exact official EvalScope client command, server environment, topology, fused/fallback behavior, accuracy, performance, and CPU-offload status to all three documentation targets. Run the skill validator specified by the skill package and `git diff --check` for local Markdown changes.

- [ ] **Step 6: Review, push, and open the stacked PR**

  Use the verification-before-completion and requesting-code-review workflows, fix confirmed findings, then push `codex/qwen4-exp-fused-ple-gfx938`. Open one PR against `codex/engram-across-dp`, include upstream attribution, serve/eval commands, A/B evidence, limitations, and link PR #199 as the base dependency.

- [ ] **Step 7: Post-publication verification**

  Confirm the PR base/head, mergeability, commits, changed files, and comments through the GitHub API. Ensure no credential appears in git history, remote URLs, process arguments, logs, or PR text.
