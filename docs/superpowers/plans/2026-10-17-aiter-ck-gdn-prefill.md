# AITER CK GDN Prefill Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a shape-gated AITER CK fused GDN prefill path that preserves existing provider controls, supports fixed-sequence `FULL_AND_PIECEWISE`, falls back safely for unsupported inputs, and passes operator plus Qwen3.5-35B end-to-end validation.

**Architecture:** Patch vLLM's `ChunkGatedDeltaRule.forward_native` at the complete prefill boundary. A new audited AITER CK adapter owns symbol/signature validation, input capability checks, graph-safe metadata normalization, and workspace acquisition through vLLM's existing per-ubatch/lane `WorkspaceManager`; the original native path remains the complete fallback.

**Tech Stack:** Python, PyTorch/HCU Graph, vLLM runtime patch callbacks, AITER CK, pytest, isolated wheel builds, OpenAI-compatible vLLM API benchmarking.

**Spec:** `docs/superpowers/specs/2026-10-17-aiter-ck-gdn-prefill-design.md`

## Global Constraints

- Preserve the installed vLLM `0.28.1` GDN call contract and one-interface/one-owner rule.
- Use existing `VLLM_HCU_USE_CUSTOM_OPS` and `hcu-aiter` selection; add no provider environment switch.
- Keep `FULL_AND_PIECEWISE` enabled in required model validation.
- CK eligibility requires `Dk=Dv=128` and local `Hv/Hqk` in `{1,2,4}`.
- Fixed single-sequence CK calls omit `cu_seqlens`; packed-varlen graph capture uses the original path.
- Use vLLM `WorkspaceManager` per ubatch/lane; do not allocate one workspace per layer.
- Propagate CK runtime failures after selection; only capability/input checks may fall back.
- Qwen3.5-27B ratio 3 remains on the existing path.

## Review Focus

- `cu_seqlens` shape `[2]` whose values do not describe `[0,tokens]`: reject CK without a host read during graph capture.
- Concurrent/DBO calls: obtain independent workspace views from the active workspace-manager slot.
- Locked workspace smaller than the CK requirement: fail before launch with the manager's allocation error, never allocate during capture.
- `core_attn_out` capacity/layout: preserve official backing-buffer behavior without returning detached storage.
- Explicit `triton`, explicit `hcu-boltops`, and master-switch-off configurations: never reach CK.

---

### Task 1: Audited AITER CK adapter and capability contract

**Files:**
- Create: `vllm_hcu/patch/worker/op_opt/_aiter_ck_gdn.py`
- Modify: `tests/runtime_patch/test_attention_mla_fla_mamba.py`

**Interfaces:**
- Produces: `resolve_aiter_ck_gdn() -> AiterCkGdnOps | None` with audited `prefill` and `workspace_bytes` callables.
- Produces: `aiter_ck_unsupported(...) -> str | None` for dimensions, ratios, dtypes, layouts, metadata, and graph mode.
- Produces: `aiter_ck_workspace(ops, q, v, cu_seqlens, flags) -> torch.Tensor` backed by `current_workspace_manager().get_simultaneous`.
- Consumes: existing provider selection from `_gdn_backend.requested_backend("prefill")`.

- [ ] **Step 1: Write failing contract tests**

Add tests for missing AITER, non-callable symbols, signature drift, supported ratios 1/2/4, ratio 3, non-128 dimensions, incompatible dtypes/layouts, fixed metadata, packed-varlen eager, packed-varlen capture, and master/provider gating. Assert exact fallback reasons.

- [ ] **Step 2: Run tests and verify RED**

Run: `pytest -q tests/runtime_patch/test_attention_mla_fla_mamba.py -k 'ck_gdn'`

Expected: failures because `_aiter_ck_gdn` and its interfaces do not exist.

- [ ] **Step 3: Implement the minimal audited adapter**

Use lazy imports and exact signature checks matching installed AITER 0.1.7. Treat absence as optional capability and signature drift as `PatchCompatibilityError`. Normalize only framework-proven single-sequence metadata to `None`; never read device metadata on the host during capture.

- [ ] **Step 4: Add workspace-manager tests**

Test required byte propagation, returned `uint8` view size, reuse, ubatch/lane isolation through the manager, locked-manager growth failure, and absence of direct `torch.empty` allocation in the adapter.

- [ ] **Step 5: Run tests and verify GREEN**

Run: `pytest -q tests/runtime_patch/test_attention_mla_fla_mamba.py -k 'ck_gdn'`

Expected: all selected tests pass.

- [ ] **Step 6: Commit**

```bash
git add vllm_hcu/patch/worker/op_opt/_aiter_ck_gdn.py tests/runtime_patch/test_attention_mla_fla_mamba.py
git commit -m "feat(gdn): add audited AITER CK prefill adapter"
```

### Task 2: Complete-prefill runtime routing

**Files:**
- Create: `vllm_hcu/patch/worker/op_opt/patch_gdn_ck_prefill.py`
- Modify: `vllm_hcu/patch/worker/__init__.py`
- Modify: `vllm_hcu/patch/worker/op_opt/patch_gdn_linear_attention.py`
- Modify: `tests/runtime_patch/test_gdn_v0251_ownership.py`
- Modify: `tests/runtime_patch/test_attention_mla_fla_mamba.py`

**Interfaces:**
- Consumes: Task 1's resolver, capability check, and workspace provider.
- Produces: an idempotent patch for `ChunkGatedDeltaRule.forward_native` preserving its exact signature.
- Produces: one-time execution/fallback logs that distinguish fused CK from the existing per-operator frontend.

- [ ] **Step 1: Write failing routing tests**

Cover master on/off, requested `hcu-aiter`/`hcu-boltops`/`triton`, supported and unsupported shapes, fixed and varlen graph states, missing AITER, output-final-state false, nonzero state, and `core_attn_out`. Assert the original method receives untouched arguments on fallback and CK runtime exceptions propagate.

- [ ] **Step 2: Run tests and verify RED**

Run: `pytest -q tests/runtime_patch/test_gdn_v0251_ownership.py tests/runtime_patch/test_attention_mla_fla_mamba.py -k 'gdn and (ck or prefill)'`

Expected: failures because the complete-prefill patch is absent.

- [ ] **Step 3: Implement the minimal runtime patch**

Bind the official signature, query the existing provider selector, apply Task 1 capability checks, acquire workspace, call CK with `use_exp2=False`, `transpose_state=True`, and `qk_l2norm=False`, restore vLLM's leading batch dimension, honor final-state semantics, and copy into a validated `core_attn_out` view when supplied.

- [ ] **Step 4: Register callback ownership and update logs**

Register the new callback after the Qwen GDN module is importable and before execution. Change the earlier frontend log so it reports fused CK when eligible without claiming per-operator resolution for calls that CK owns. Retain exact explicit-Triton behavior.

- [ ] **Step 5: Run related runtime tests**

Run: `pytest -q tests/runtime_patch/test_gdn_v0251_ownership.py tests/runtime_patch/test_attention_mla_fla_mamba.py`

Expected: all tests pass; only established Torch deprecation warnings remain.

- [ ] **Step 6: Commit**

```bash
git add vllm_hcu/patch/worker/op_opt/patch_gdn_ck_prefill.py vllm_hcu/patch/worker/__init__.py vllm_hcu/patch/worker/op_opt/patch_gdn_linear_attention.py tests/runtime_patch/test_gdn_v0251_ownership.py tests/runtime_patch/test_attention_mla_fla_mamba.py
git commit -m "feat(gdn): route supported prefill through AITER CK"
```

### Task 3: Native HCU correctness, graph, and performance gates

**Files:**
- Create: `tests/kernels/test_aiter_ck_gdn_prefill.py`
- Create: `tests/kernels/benchmark_aiter_ck_gdn_prefill.py`
- Modify: `docs/validation/gdn-aiter-ck-prefill.md`

**Interfaces:**
- Consumes: the public patched complete-prefill call from Task 2.
- Produces: reproducible native correctness/Graph tests and a non-CI benchmark emitting JSON.

- [ ] **Step 1: Write native correctness tests**

Parameterize fixed lengths `1,16,63,64,65,127,128,256,512,1024,2048,4096`, ratios 1/2/4, fixed and packed-varlen cases, zero/nonzero state, and final-state behavior. Generate legal normalized q/k and compare CK with official FLA and the pre-integration mixed pipeline using finite checks, max/mean absolute error, and NRMSE thresholds justified by the probe evidence.

- [ ] **Step 2: Run native correctness tests**

Run: `HIP_VISIBLE_DEVICES=0 pytest -q tests/kernels/test_aiter_ck_gdn_prefill.py -m native`

Expected: supported cases pass; ratio 3 and incompatible inputs select the original path with the asserted reason.

- [ ] **Step 3: Add Graph replay tests**

Capture a fixed 65-token call, replay after in-place input changes, and compare replay with eager CK. Assert packed-varlen capture selects the original path and does not call CK with `cu_seqlens`.

- [ ] **Step 4: Run Graph tests**

Run: `HIP_VISIBLE_DEVICES=0 pytest -q tests/kernels/test_aiter_ck_gdn_prefill.py -m graph`

Expected: fixed replay is numerically exact to eager CK; packed-varlen fallback passes.

- [ ] **Step 5: Add and run the benchmark**

The benchmark compares official FLA, the pre-integration mixed pipeline, and fused CK with prebuilt metadata and reused workspace. It records cold time separately and reports P50/P90/mean, tokens/s, workspace bytes, errors, and source/provider versions for at least three seeds.

Run: `HIP_VISIBLE_DEVICES=0 python tests/kernels/benchmark_aiter_ck_gdn_prefill.py --lengths 1,64,65,128,512,1024,2048,4096 --repetitions 3 --json-out /data/models/aiter_ck_gdn_operator_results.json`

Expected: JSON contains every requested shape and all three providers; no non-finite result.

- [ ] **Step 6: Document measured support and commit**

Record the supported ratios, workspace scaling, fixed Graph behavior, packed-varlen Graph limitation, commands, versions, and raw result location.

```bash
git add tests/kernels/test_aiter_ck_gdn_prefill.py tests/kernels/benchmark_aiter_ck_gdn_prefill.py docs/validation/gdn-aiter-ck-prefill.md
git commit -m "test(gdn): validate AITER CK fused prefill"
```

### Task 4: Isolated wheels and Qwen3.5-35B end-to-end comparison

**Files:**
- Modify: `docs/validation/gdn-aiter-ck-prefill.md`
- Create outside Git: `/data/models/validation/gdn-aiter-ck-prefill/<baseline-sha>/`
- Create outside Git: `/data/models/validation/gdn-aiter-ck-prefill/<candidate-sha>/`

**Interfaces:**
- Consumes: baseline commit `68f20511` (current mixed behavior), the final candidate commit, `/model/Qwen3.5-35B-A3B`, and the same installed vLLM/AITER/BoltOps/Torch/DTK stack.
- Produces: isolated wheels, checksums, server logs, request outputs, provider evidence, performance JSON, and a validation verdict.

- [ ] **Step 1: Build and isolate both wheels**

Build baseline and candidate from immutable SHAs with `ROCM_PATH=/opt/dtk`, record wheel names and SHA256, install each with `--no-deps --target`, and verify parent plus worker import paths. Do not overwrite the system plugin.

- [ ] **Step 2: Start baseline Qwen3.5-35B TP4**

Use BF16, the same memory/utilization and request limits chosen for both runs, and `FULL_AND_PIECEWISE`. Capture startup, actual mixed-provider logs, graph mode, attention/MoE providers, memory, and readiness. Signal only the API PID during cleanup.

- [ ] **Step 3: Run baseline correctness workloads**

Run a fixed greedy corpus spanning short, 63/64/65-boundary, 1K, 4K, and longest-practical prompts; dynamic concurrent requests; and three repeated shared-prefix requests. Save raw inputs/outputs, HTTP status, token counts, prefix-cache metrics, and errors.

- [ ] **Step 4: Run baseline performance workloads three times**

After warmup, record TTFT, TPOT/ITL, end-to-end latency distributions, request/token throughput, error rate, and peak HCU memory for single-request and concurrent workloads. Keep compile/cold-start time separate.

- [ ] **Step 5: Repeat Steps 2–4 with the candidate**

Use byte-identical workload files and settings. Require logs proving fused CK execution on eligible prefill and no unintended fallback. Preserve `FULL_AND_PIECEWISE` and prefix caching.

- [ ] **Step 6: Compare and issue the verdict**

Require deterministic greedy output parity or investigate every difference; successful fixed Graph replay; HTTP success; explainable prefix hits; and clean teardown. Report three-run distributions and whether prefill-critical end-to-end workloads improve by roughly 5% without a repeatable unexplained regression elsewhere.

- [ ] **Step 7: Update validation evidence and commit**

Link the raw artifact directories, exact commands, versions, SHAs, checksums, results, failures, and remaining limitations from `docs/validation/gdn-aiter-ck-prefill.md`.

```bash
git add docs/validation/gdn-aiter-ck-prefill.md
git commit -m "docs(gdn): record CK prefill model validation"
```

### Task 5: Final regression, review, and PR update

**Files:**
- Modify only files required by discovered regressions or review findings.

**Interfaces:**
- Consumes: Tasks 1–4 and the existing PR branch.
- Produces: verified branch and updated PR evidence.

- [ ] **Step 1: Run focused regression suites**

Run: `pytest -q tests/runtime_patch/test_attention_mla_fla_mamba.py tests/runtime_patch/test_gdn_v0251_ownership.py tests/kernels/test_aiter_ck_gdn_prefill.py`

Expected: all selected tests pass.

- [ ] **Step 2: Run repository-wide tests**

Run: `pytest -q`

Expected: report every failure by name; distinguish known environment failures from regressions. Do not call the full suite green unless exit status is zero.

- [ ] **Step 3: Run static/build checks**

Run `git diff --check`, Python compilation for changed modules, the available project linter, and a wheel build. Verify the wheel imports after `torch` from an isolated target.

- [ ] **Step 4: Request code review and resolve findings**

Review provider ownership, graph safety, workspace concurrency, exact ABI checks, fallback boundaries, logs, and test evidence. Re-run affected tests after every change.

- [ ] **Step 5: Push and update PR #245**

Push the verified commits to `fork/fix/gdn-provider-log` and update the PR body with operator results, model validation, performance distributions, artifacts, and remaining AITER limitations.
