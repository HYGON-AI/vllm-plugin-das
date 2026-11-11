# Qwen4Exp DeepEP Auto Phase-Aware CUDA Graph Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make opt-in Qwen4Exp SlimQuant W4A8 serving stable on DP4/EP4 with DeepEP auto, using HT/contiguous experts for prefill and LL/masked experts for decode while retaining the compatible CUDA Graph path.

**Architecture:** Treat the concrete vLLM CUDA Graph runtime mode as the authoritative DeepEP delegate contract: FULL selects LL, PIECEWISE selects HT, and NONE keeps the existing DP-synchronized dynamic phase selection. Propagate V2 dummy-run phase explicitly through an invocation-local scope so idle ranks remain collective participants without falsely forcing HT, and preserve `FULL_AND_PIECEWISE` unless hardware proves HT piecewise unsupported, in which case use `FULL_DECODE_ONLY`.

**Tech Stack:** Python, PyTorch distributed, vLLM v0.28.1 runner V2, DeepEP, DeepGEMM, pytest, ROCm/HCU

**Spec:** `docs/superpowers/specs/2026-10-18-qwen4exp-deepep-auto-graph-design.md`

## Global Constraints

- The feature remains opt-in only through `--enable-expert-parallel --all2all-backend deepep_auto --moe-backend deep_gemm`.
- The default AITER plus all-gather/reduce-scatter route and MR #247 behavior must remain unchanged.
- `FULL` means pure decode and must select DeepEP LL plus masked DeepGEMM.
- `PIECEWISE` means prefill/mixed and must select DeepEP HT plus contiguous DeepGEMM.
- `NONE` keeps DP-synchronized selection: empty ranks are neutral; LL is allowed only when every active rank is pure decode; missing or mixed evidence selects HT.
- Target and native MTP draft must obey the same graph-mode and invocation-phase contract.
- Do not infer request phase from token count.
- Do not enable DBO/ubatching or change KV-cache dtype, MTP length, attention selection, workspace allocation policy, fixed DeepEP modes, or model defaults.
- Use red-green TDD, preserve DeepSeek-V4 and GLM behavior, and reserve enough device memory for the 6.08 GiB HT workspace during hardware validation.
- Stack the final MR on MR #247; do not claim a performance improvement unless measured.

## Review Focus

- A DP rank that becomes idle during another rank's decode must contribute explicit decode-dummy phase and must not switch the group back to HT; Task 3 tests this.
- A dummy-run exception must restore invocation-local phase so the next real request cannot inherit stale state; Task 3 tests this.
- Graph capture with no real `InputBatch` phase must select FULL=LL and PIECEWISE=HT without an extra DP phase collective; Task 2 tests this.
- An uncaptured/missing-phase execution must fail closed to HT, while a zero-token rank remains neutral during DP synchronization; Task 2 tests this.
- If HT PIECEWISE capture is unsupported by the installed stack, decode FULL graphs must remain enabled through `FULL_DECODE_ONLY` rather than globally disabling graphs; Task 4 tests and hardware-gates this.

---

### Task 1: Lock Qwen4Exp Factory Admission

**Status:** Completed in `c351110e` before this plan was written; retained here
as a checked prerequisite so execution starts at Task 2 without rewriting
history.

**Files:**
- Modify: `vllm_hcu/model_executor/layers/fused_moe/deepep_runtime.py:126-149`
- Test: `tests/runtime_patch/test_quant_gemm_aiter.py:4380-4410`

**Interfaces:**
- Consumes: `model_architectures(vllm_config) -> tuple[str, ...]` and the existing DeepEP-auto/DeepGEMM capability checks.
- Produces: `require_supported_deepep_deepgemm_model(vllm_config) -> None` admitting `Qwen4ExpForCausalLM` and `Qwen4ExpForConditionalGeneration` without weakening any other gate.

- [x] **Step 1: Extend the factory-selection test to pin both Qwen wrapper names**

  `test_slimquant_w4a8_deepep_auto_uses_w4a8_deepgemm_factory_not_aiter`
  now covers the two approved names and preserves the existing unsupported
  model tests. `Qwen4ExpMTP` is not added: the draft is constructed from the
  admitted target config and the installed checkpoint advertises
  `Qwen4ExpForConditionalGeneration` at the top level.

- [x] **Step 2: Run the focused test and verify the unsupported wrappers fail before implementation**

  Run: `pytest -q tests/runtime_patch/test_quant_gemm_aiter.py -k 'slimquant_w4a8_deepep_auto'`

  Expected before the allow-list change: the two approved Qwen cases FAIL with the DeepEP/DeepGEMM unsupported-model error; the unrelated rejection case PASS.

- [x] **Step 3: Add only the two approved Qwen4Exp architecture strings to the existing allow-list**

  Preserve the HIPC operator, quantization, topology, and backend checks around the architecture gate.

- [x] **Step 4: Run the focused factory tests**

  Run: `pytest -q tests/runtime_patch/test_quant_gemm_aiter.py -k 'slimquant_w4a8_deepep_auto'`

  Expected: PASS.

- [x] **Step 5: Commit the factory support**

  ```bash
  git add vllm_hcu/model_executor/layers/fused_moe/deepep_runtime.py tests/runtime_patch/test_quant_gemm_aiter.py
  git commit -m "feat(moe): enable Qwen4Exp DeepEP DeepGEMM"
  ```

### Task 2: Make Runtime Graph Mode the Delegate Contract

**Files:**
- Modify: `vllm_hcu/forward_context_runtime.py:61-114`
- Modify: `vllm_hcu/patch/worker/framework_opt/patch_forward_context.py:152-220`
- Test: `tests/runtime_patch/test_worker_framework_opt.py:1249-1515`

**Interfaces:**
- Consumes: `CUDAGraphMode.NONE`, `CUDAGraphMode.PIECEWISE`, `CUDAGraphMode.FULL`, request phase from `get_deepep_auto_request_phase()`, and `_synchronize_deepep_auto_phase(...)`.
- Produces: `choose_deepep_auto_low_latency(vllm_config, num_tokens, num_tokens_across_dp, batch_descriptor, attn_metadata=None, is_prefilling=None, cudagraph_runtime_mode=None) -> bool` and a patched `set_forward_context` wrapper that passes its concrete runtime mode.

- [ ] **Step 1: Write graph-mode selector tests**

  Add tests asserting:

  - FULL returns `True` with no phase evidence and does not call `_synchronize_deepep_auto_phase`;
  - PIECEWISE returns `False` even if stale phase says decode and does not call the synchronizer;
  - NONE calls the synchronizer and returns its result;
  - NONE with missing phase fails closed to `False`;
  - a zero-token rank passes `local_active=False` to the synchronizer and can follow the active ranks' global decode result;
  - non-`deepep_auto` behavior remains `False` for every runtime mode.

- [ ] **Step 2: Run the selector tests and verify the new runtime-mode cases fail**

  Run: `pytest -q tests/runtime_patch/test_worker_framework_opt.py -k 'deepep_auto_forward_mode or deepep_auto_graph_mode or dp_global_phase_on_empty_ranks'`

  Expected: FAIL because `cudagraph_runtime_mode` is not accepted or FULL/PIECEWISE still use phase inference.

- [ ] **Step 3: Add `cudagraph_runtime_mode` to the selector and branch before dynamic phase synchronization**

  Compare against `CUDAGraphMode.FULL` and `CUDAGraphMode.PIECEWISE`; only `NONE` or an absent mode may inspect request phase and perform the DP phase collective. Reject any non-concrete composite runtime mode with `ValueError` so capture cannot silently bind the wrong delegate.

- [ ] **Step 4: Pass the concrete graph mode from the patched forward-context entry point**

  In `hcu_set(...)`, forward the existing `cudagraph_runtime_mode` argument to `choose_deepep_auto_low_latency(...)`. Do not add a second collective or mutable global delegate.

- [ ] **Step 5: Run focused and surrounding framework tests**

  Run: `pytest -q tests/runtime_patch/test_worker_framework_opt.py -k 'deepep_auto or forward_context'`

  Expected: PASS.

- [ ] **Step 6: Commit the graph-mode contract**

  ```bash
  git add vllm_hcu/forward_context_runtime.py vllm_hcu/patch/worker/framework_opt/patch_forward_context.py tests/runtime_patch/test_worker_framework_opt.py
  git commit -m "fix(moe): bind DeepEP auto to graph runtime mode"
  ```

### Task 3: Propagate V2 Dummy-Run Phase Safely

**Files:**
- Modify: `vllm_hcu/v1/hcu_model_runner_v2.py:1-30,317-425`
- Create: `tests/runtime_patch/test_hcu_model_runner_v2_deepep_auto.py`
- Test: `tests/runtime_patch/test_deepseek_v4_pcp.py:117-151`

**Interfaces:**
- Consumes: `deepep_auto_request_phase_scope()`, `set_deepep_auto_request_phase(is_prefilling: object) -> None`, `_dummy_run(num_tokens, *, uniform_decode=False, ...)`, and the existing `execute_model(...)` override.
- Produces: `_deepep_auto_dummy_phase_scope(is_prefilling: bool | None)` backed by a `ContextVar[bool | None]`; `_dummy_run` publishes `not uniform_decode` only around its nested execution, and `execute_model` copies that value into the request phase scope before delegating.

- [ ] **Step 1: Write isolated dummy-phase tests**

  Test that:

  - `_dummy_run(..., uniform_decode=True)` exposes `is_prefilling=False` inside nested `execute_model`;
  - `_dummy_run(..., uniform_decode=False)` exposes `is_prefilling=True`;
  - success and raised exceptions both restore the prior value;
  - nested scopes restore in LIFO order;
  - when `deepep_auto` is disabled, `_dummy_run` does not publish a phase and existing runner behavior is unchanged;
  - a real batch's `prepare_inputs()` value remains authoritative when no dummy scope exists.

- [ ] **Step 2: Run the new tests and verify they fail**

  Run: `pytest -q tests/runtime_patch/test_hcu_model_runner_v2_deepep_auto.py tests/runtime_patch/test_deepseek_v4_pcp.py -k 'phase'`

  Expected: FAIL because the invocation-local dummy holder and propagation do not exist.

- [ ] **Step 3: Add the invocation-local dummy phase scope**

  Define a module-private `ContextVar` and context manager with token/reset semantics. In `_dummy_run`, enter it only for `deepep_auto`, publish `not bool(uniform_decode)`, and delegate in `try/finally` through the context manager.

- [ ] **Step 4: Seed the request phase when `execute_model` enters its existing scope**

  After entering `deepep_auto_request_phase_scope()` and before delegating to the base runner, copy a non-`None` dummy phase into `set_deepep_auto_request_phase(...)`. Real batches continue to set their phase in `prepare_inputs()`.

- [ ] **Step 5: Run dummy, runner, and phase regression tests**

  Run: `pytest -q tests/runtime_patch/test_hcu_model_runner_v2_deepep_auto.py tests/runtime_patch/test_deepseek_v4_pcp.py tests/runtime_patch/test_worker_framework_opt.py -k 'phase or dummy or deepep_auto'`

  Expected: PASS, including exception restoration and feature-off cases.

- [ ] **Step 6: Commit the dummy-rank fix**

  ```bash
  git add vllm_hcu/v1/hcu_model_runner_v2.py tests/runtime_patch/test_hcu_model_runner_v2_deepep_auto.py tests/runtime_patch/test_deepseek_v4_pcp.py
  git commit -m "fix(worker): propagate DeepEP phase to DP dummy runs"
  ```

### Task 4: Preserve Phase-Compatible CUDA Graph Policy

**Files:**
- Modify: `vllm_hcu/patch/platform/core_fix/patch_compilation_config.py:165-190`
- Test: `tests/runtime_patch/test_platform_hcu_config.py:870-975`
- Test: `tests/runtime_patch/test_worker_framework_opt.py`

**Interfaces:**
- Consumes: `CompilationConfig.cudagraph_mode`, `CUDAGraphMode.FULL_AND_PIECEWISE`, `CUDAGraphMode.FULL_DECODE_ONLY`, and the Task 2 graph-mode selector.
- Produces: configuration that preserves `FULL_AND_PIECEWISE` for `deepep_auto`; if the hardware gate rejects HT PIECEWISE, a narrowly documented fallback maps only `deepep_auto` to `FULL_DECODE_ONLY`, retaining FULL decode graphs and eager prefill/mixed execution.

- [ ] **Step 1: Replace the global-disable test with phase-compatible policy tests**

  Assert that DP `deepep_auto` no longer changes `FULL_AND_PIECEWISE` to `NONE`, unrelated backends are unchanged, and an already explicit `NONE` remains `NONE`. Add target/draft capture assertions through the common forward-context wrapper: FULL selects LL and PIECEWISE selects HT for both invocations without phase evidence.

- [ ] **Step 2: Run the new policy tests and verify the preservation case fails**

  Run:

  ```bash
  pytest -q tests/runtime_patch/test_platform_hcu_config.py -k 'compilation_adapter and deepep_auto'
  pytest -q tests/runtime_patch/test_worker_framework_opt.py -k 'target_mtp_graph_mode'
  ```

  Expected: FAIL because the current patch rewrites `FULL_AND_PIECEWISE` to `NONE`.

- [ ] **Step 3: Remove only the `deepep_auto` global CUDA Graph downgrade**

  Keep upstream fixed-HT safety behavior and every unrelated compilation setting intact. Add a one-shot effective-policy log only if existing startup logs do not already expose the resulting graph mode.

- [ ] **Step 4: Run the configuration and graph-contract regressions**

  Run:

  ```bash
  pytest -q tests/runtime_patch/test_platform_hcu_config.py -k 'compilation_adapter or deepep_auto'
  pytest -q tests/runtime_patch/test_worker_framework_opt.py -k 'deepep_auto or target_mtp_graph_mode'
  ```

  Expected: PASS with `FULL_AND_PIECEWISE` preserved.

- [ ] **Step 5: Perform the short DP4 hardware capture gate**

  Start the service with `--gpu-memory-utilization 0.82 --enable-expert-parallel --all2all-backend deepep_auto --moe-backend deep_gemm`, `FULL_AND_PIECEWISE`, DP4/EP4, MTP3, and `--max-num-batched-tokens 16384`. Send a warmed BS16 input-4096/output-128 request twice, then inspect all four engine logs for successful FULL and PIECEWISE capture plus prefill HT/decode LL evidence.

  Expected: service startup completes, the second request completes, and no rank reports capture, cache-key, collective-timeout, or layout mismatch errors.

- [ ] **Step 6: If and only if Step 5 proves HT PIECEWISE unsupported, add the fail-closed fallback test and implementation**

  Change the policy test to assert `deepep_auto + FULL_AND_PIECEWISE -> FULL_DECODE_ONLY`, implement that single mapping in `patch_compilation_config.py`, log the fallback once, rerun Steps 4-5, and record the exact hardware error in the MR. Do not change fixed DeepEP modes or explicit user `NONE`.

- [ ] **Step 7: Commit the effective graph policy**

  ```bash
  git add vllm_hcu/patch/platform/core_fix/patch_compilation_config.py tests/runtime_patch/test_platform_hcu_config.py tests/runtime_patch/test_worker_framework_opt.py
  git commit -m "fix(graph): preserve decode graphs for DeepEP auto"
  ```

### Task 5: Run Software Regression and Review Gates

**Files:**
- Modify as required by failures: only files already named in Tasks 1-4
- Test: `tests/runtime_patch/test_quant_gemm_aiter.py`
- Test: `tests/runtime_patch/test_worker_framework_opt.py`
- Test: `tests/runtime_patch/test_platform_hcu_config.py`
- Test: `tests/runtime_patch/test_deepseek_v4_pcp.py`
- Test: `tests/runtime_patch/test_moe_deepep.py`
- Test: `tests/runtime_patch/test_hcu_model_runner_v2_deepep_auto.py`

**Interfaces:**
- Consumes: all Task 1-4 interfaces.
- Produces: one reviewed software change set with no regression in DeepSeek-V4, GLM, feature-off, AITER, or MR #247 paths.

- [ ] **Step 1: Run syntax and focused unit suites**

  Run:

  ```bash
  python -m compileall -q vllm_hcu
  pytest -q tests/runtime_patch/test_quant_gemm_aiter.py tests/runtime_patch/test_worker_framework_opt.py tests/runtime_patch/test_platform_hcu_config.py tests/runtime_patch/test_deepseek_v4_pcp.py tests/runtime_patch/test_moe_deepep.py tests/runtime_patch/test_hcu_model_runner_v2_deepep_auto.py
  ```

  Expected: all selected tests PASS.

- [ ] **Step 2: Inspect the branch diff against MR #247**

  Run: `git diff --check feature/int8-allgatherv...HEAD && git diff --stat feature/int8-allgatherv...HEAD && git status --short`

  Expected: no whitespace errors, only scoped files and docs changed, and a clean worktree.

- [ ] **Step 3: Review the five failure modes from Review Focus against tests and implementation**

  Confirm each named test actually exercises its failure mode; correct gaps with a failing test before changing code.

- [ ] **Step 4: Commit any review-only correction**

  ```bash
  git add <only-the-corrected-files>
  git commit -m "test(moe): cover DeepEP auto graph edge cases"
  ```

  Skip this commit when no correction is needed.

### Task 6: Validate DP4/EP4 Stability, Performance, Accuracy, and Trace

**Files:**
- Create outside Git: `/models/qwen4exp_deepep_auto_graph_<timestamp>/`
- Create outside Git: `/models/<new-rank0-trace>.pt.trace.json.gz`

**Interfaces:**
- Consumes: the exact opt-in server configuration and the effective graph policy from Task 4.
- Produces: reproducible logs and JSON results for stability, BS16 performance, HumanEval32 accuracy, and FULL/PIECEWISE trace evidence.

- [ ] **Step 1: Record environment and launch the DP4/EP4 service**

  Save commit SHA, package versions, free memory, environment variables, and the exact server command. Include `ROCBLAS_VERBOSE_TENSILE_ERROR=0`, `VLLM_ROCM_USE_AITER_CUSTOM_AR=1`, `VLLM_HCU_PLE_PREFETCH_STREAM=1`, `--api-server-count 1`, `--max-num-batched-tokens 16384`, and `--gpu-memory-utilization 0.82`. Keep CPU offload/embedding flags aligned with the accepted MR #247 baseline command.

- [ ] **Step 2: Prove idle-rank liveness**

  Run repeated uneven requests until three DP ranks are idle while one continues decoding, then require all responses to finish and scan every engine log for collective timeouts.

  Expected: no hang, no shared-memory timeout, and decode LL remains selected for active plus uniform-decode dummy ranks.

- [ ] **Step 3: Measure BS16 input-4096/output-1024 after warmup**

  Run three warmups and five measured rounds with the same prompts/seeds as the MR #247 baseline. Report median TTFT, median TPOT, output throughput, end-to-end latency, and spread; compare against MR #247 without claiming causality from a single round.

- [ ] **Step 4: Run HumanEval first 32 at concurrency 32**

  Use the established repository evaluation command and deterministic sampling parameters from the prior baseline. Report pass count/pass@1 and any request failures.

- [ ] **Step 5: Capture and inspect the warmed second BS16 trace**

  Capture input-4096/output-128 with graphs enabled after warmup. Copy the full rank-0 trace to `/models`, report its absolute path, and verify prefill uses HT/contiguous execution while decode replays FULL LL/masked execution; disclose if the Task 4 fallback makes prefill eager NONE.

- [ ] **Step 6: Stop all test processes and verify device memory is released**

  Expected: no serving or benchmark processes remain and the four devices return to their pre-test memory state.

### Task 7: Deliver the Stacked MR

**Files:**
- Modify: `docs/superpowers/specs/2026-10-18-qwen4exp-deepep-auto-graph-design.md` only if measured evidence changes an explicit assumption
- Modify: `docs/superpowers/plans/2026-10-18-qwen4exp-deepep-auto-graph.md` checkbox status during execution
- GitHub: new branch and MR stacked on `feature/int8-allgatherv` / MR #247

**Interfaces:**
- Consumes: clean verified commits and Task 6 artifacts.
- Produces: one reviewable stacked MR for functional Qwen4Exp DeepEP-auto support.

- [ ] **Step 1: Run final verification immediately before publishing**

  Re-run Task 5 Step 1, `git diff --check feature/int8-allgatherv...HEAD`, and `git status --short`.

  Expected: tests PASS, no diff errors, and no unintended/untracked files.

- [ ] **Step 2: Review the complete stacked diff**

  Review architecture admission, selector ordering, ContextVar restoration, target/MTP parity, graph policy, and feature-off paths. Resolve every concrete finding with a reproducing test before publishing.

- [ ] **Step 3: Push a dedicated feature branch**

  Push the current branch to `https://github.com/HYGON-AI/vllm-plugin-das` without modifying MR #247's branch.

- [ ] **Step 4: Create the stacked MR targeting `feature/int8-allgatherv`**

  The description must identify this as functional support, summarize the dummy-rank root cause, state the effective FULL/PIECEWISE or FULL_DECODE_ONLY policy, list unit and hardware results, link/quote relevant artifact paths, and explicitly state whether performance improved.

- [ ] **Step 5: Add a reproducibility comment**

  Post the exact service command, benchmark command, HumanEval command, warmup count, measured TTFT/TPOT/throughput, trace path, and the memory-headroom requirement. Do not post credentials or unrelated environment values.

- [ ] **Step 6: Report the MR URL and unresolved limitations**

  Call out any PIECEWISE fallback, the 6.08 GiB HT workspace headroom, and whether results justify performance claims.
