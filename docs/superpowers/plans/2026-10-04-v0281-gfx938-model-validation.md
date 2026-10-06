# vLLM 0.28.1 gfx938 Model Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebase the gfx938 FP8 sparse-indexer work onto PR #180, validate the pinned vLLM 0.28.1 stack on 15 local models, fix plugin-owned failures, and publish exactly one new stacked plugin PR with updated HCU upgrade guidance.

**Architecture:** Preserve the existing PRs as a three-layer stack: PR #180, rebased PR #181, then one new validation/fix branch. Build and test clean wheels, drive every model through the existing EvalScope server harness, isolate failures at their owning plugin boundary, and record final gfx938 evidence separately from historical results on other chips.

**Tech Stack:** Python 3.10, PyTorch 2.11 DTK 26.04, vLLM 0.28.1, pytest, EvalScope, HCU/HIP, AITER, BoltOPs, FlashAttention/FlashMLA, LightOp, DeepEP, DeepGEMM, setuptools/wheel, GitHub REST API.

**Spec:** `docs/superpowers/specs/2026-10-04-v0281-gfx938-model-validation-design.md`

## Global Constraints

- Pin `vllm==0.28.1+dtk2604.torch2110.2609171627.g77acaf` from `https://pypi.sourcefind.cn/nightly/dtk/` and record its wheel SHA256.
- Keep Model Runner V2, prefix caching, requested attention/MoE providers, native FP8 KV cache, and the official default CUDA Graph policy enabled in acceptance runs.
- Run HumanEval samples 0-15 on all 15 supported full checkpoints; exclude DeepSeek-V4.1 and treat `dspark_qwen3_8b_block7` as a component artifact.
- TP is mandatory for every model. DP8 is additional evidence and cannot replace a TP run.
- Use HND/BHSD only for FlashAttention; preserve native FlashMLA layouts.
- Final evidence must import vLLM and the plugin from isolated wheel targets, not a source checkout.
- Do not modify vLLM core or create a second new PR. Stop and report a missing upstream extension point.
- Never print, persist, or commit the supplied GitHub credential. Use `git credential fill` and clear shell variables immediately.
- Update `/models/upgrading-vllm-hcu/SKILL.md` and evidence references only with verified findings.

## Review Focus

- A BF16/FP16 cache dtype must still normalize to the unquantized HIPC ABI while `fp8_e4m3` remains unchanged; Task 2 pins this in the rebased cache-writer tests.
- HND must reach vendor FlashAttention as `bhsd` without changing FlashMLA cache views; Task 5 covers both layout families.
- Model Runner V2 evidence must come from the concrete worker/runner import path rather than the generic V1-engine log; Task 3 asserts the command and Task 4 records runtime markers.
- DP/EP/MTP must keep graph-padded and inactive ranks collective-safe; Task 9 runs DP8 and a combined TP+DP+EP+MTP topology with repeated accuracy.
- HumanEval must bypass all local proxies and never reuse stale prediction caches; Task 3 adds an environment regression and unique owned work directories.

---

### Task 1: Create isolated execution worktrees and freeze the PR inputs

**Files:**
- Read: `docs/superpowers/specs/2026-10-04-v0281-gfx938-model-validation-design.md`
- Read: `/models/upgrading-vllm-hcu/SKILL.md`
- Create later on final branch: `docs/validation/v0281-gfx938-provenance.md`

**Interfaces:**
- Consumes: PR #180 head `f24e99be620c3a0d3611a7d96079bb7be7e63de8`, PR #181 head `3f73a9cb63935e78d1e8db1bd19958d64d0389f9`, shared base `9788dc633fb3d0a0a0a124b5886f62c07b3de765`, design commit `e1b7b9a5abb10786617c671101b4b66601df6c2e`.
- Produces: isolated PR #181 rebase worktree and a final-branch worktree rooted at the verified rebased #181 head.

- [ ] **Step 1: Invoke `superpowers:using-git-worktrees` and verify repository state**

Run read-only worktree detection and require `GIT_DIR == GIT_COMMON`, a non-detached branch, and a clean design worktree before creating execution worktrees.

- [ ] **Step 2: Fetch the exact base and PR heads without changing the design branch**

Run:

```bash
git fetch origin \
  refs/heads/v0.28.1-dev:refs/remotes/origin/v0.28.1-dev \
  refs/heads/fix/flash-attn-unquantized-cache-dtype:refs/remotes/origin/pr180 \
  refs/heads/codex/fix-gfx938-fp8-indexer-reader-v0281:refs/remotes/origin/pr181
```

Expected: the three refs resolve to the frozen SHAs above. If any head moved, record the new SHA, re-read its complete diff and PR comments, and amend the plan before rebasing.

- [ ] **Step 3: Create the PR #181 rebase worktree**

Create `/models/.worktrees/vllm-plugin-das-pr181-on-pr180` on a local branch `work/pr181-on-pr180` at `origin/pr181`. Do not reuse `/models/vllm-plugin-das`, which holds the committed design.

- [ ] **Step 4: Record the pre-rebase ranges**

Save `git log --reverse --format='%H %s' 9788dc6..origin/pr180` and `9788dc6..origin/pr181` in the task notes. Verify PR #180 has 10 commits and PR #181 has 8 commits before proceeding.

### Task 2: Rebase and revalidate existing PR #181 on PR #180

**Files:**
- Resolve if conflicted: `vllm_hcu/patch/worker/core_fix/patch_glm5next_channel_fp8.py`
- Resolve if conflicted: `vllm_hcu/v1/attention/ops/rocm_aiter_mla_sparse.py`
- Test: `tests/kernels/test_gfx938_fp8_indexer_reader.py`
- Test: `tests/runtime_patch/test_glm53_channel_fp8.py`
- Test: `tests/runtime_patch/test_sparse_indexer_loading.py`
- Test: every test file changed by `origin/pr180..work/pr181-on-pr180`

**Interfaces:**
- Consumes: PR #180 dtype normalization and PR #181 direct gfx938 paged-indexer reader.
- Produces: rebased #181 behavior in which unquantized cache names normalize correctly, explicit AITER dispatch is preserved, and gfx938 FP8 pages use the direct reader without forced PIECEWISE mode.

- [ ] **Step 1: Rebase the eight PR #181 commits**

Run `git rebase --onto origin/pr180 9788dc633fb3d0a0a0a124b5886f62c07b3de765 work/pr181-on-pr180`.

- [ ] **Step 2: Resolve overlaps by contract**

In `patch_glm5next_channel_fp8.py`, retain PR #180's current GLM projection/cache dtype behavior and PR #181's cache-wrapper normalization before physical page inference. In `rocm_aiter_mla_sparse.py`, retain PR #180's BF16 physical page view plus PR #181's gfx938 direct FP8 reader and explicit-AITER/page-size-one fallbacks. Do not restore the full-cache linearization as the default gfx938 path.

- [ ] **Step 3: Review the semantic rebase with `range-diff`**

Run:

```bash
git range-diff 9788dc633fb3..3f73a9cb6393 origin/pr180..HEAD
git diff --check origin/pr180..HEAD
```

Expected: all eight logical PR #181 commits remain attributable; differences are limited to conflict integration with PR #180.

- [ ] **Step 4: Run the focused PR #181 and overlap tests**

Run:

```bash
pytest -q \
  tests/kernels/test_gfx938_fp8_indexer_reader.py \
  tests/runtime_patch/test_glm53_channel_fp8.py \
  tests/runtime_patch/test_sparse_indexer_loading.py \
  tests/runtime_patch/test_flash_attention_and_pp.py \
  tests/runtime_patch/test_attention_mla_fla_mamba.py
```

Expected: PASS with no collection or setup errors.

- [ ] **Step 5: Run every test file changed by the complete stacked diff**

Build the file list from `git diff --name-only origin/pr180..HEAD -- 'tests/**/*.py'`, run every listed file, then run `pytest -q tests/runtime_patch`. Expected: all pass; a filtered subset is not sufficient.

- [ ] **Step 6: Review and update the existing PR #181 branch**

Review `origin/pr180..HEAD` before and after the rebase result, then push with `--force-with-lease` to `codex/fix-gfx938-fp8-indexer-reader-v0281`. Update PR #181's base to `fix/flash-attn-unquantized-cache-dtype` and replace its forced-PIECEWISE/source-checkout validation caveat with the actual rebased test evidence. Do not claim full model acceptance yet.

### Task 3: Add the gfx938 HumanEval16 matrix contract on the final branch

**Files:**
- Create: `tests/models/v0281_gfx938_humaneval16.yaml`
- Create: `tests/integration/server/test_evalscope_v0281_gfx938_humaneval16.py`
- Modify: `tests/integration/server/evalscope_server.py`
- Modify: `tests/integration/server/test_evalscope_report_threshold.py`
- Modify: `tests/models/README.md`
- Cherry-pick: `docs/superpowers/specs/2026-10-04-v0281-gfx938-model-validation-design.md`
- Cherry-pick: `docs/superpowers/plans/2026-10-04-v0281-gfx938-model-validation.md`

**Interfaces:**
- Consumes: `load_profiled_config()`, `server_command()`, `evalscope_command()`, and `run_evalscope_server_test()` from `tests.integration.server.evalscope_server`.
- Produces: 15 named profiles with exact model paths, TP values, backends, MTP/parser controls, HumanEval limit 16, score 1.0, and unique work directories; `_server_environment(config) -> dict[str, str]` that removes local HTTP/HTTPS/ALL proxy variables; optional prefix-probe and final metrics snapshots owned by the existing server harness.

- [ ] **Step 1: Create the final branch and preserve the approved design**

After Task 2 is pushed, create `/models/.worktrees/vllm-plugin-das-v0281-gfx938-validation` at the rebased #181 head on branch `codex/validate-v0281-gfx938-models`. Cherry-pick both documentation commits from `design/v0281-gfx938-model-validation` so the approved design and this implementation plan are preserved in the final stacked PR.

- [ ] **Step 2: Write failing proxy and matrix contract tests**

Add `test_server_environment_clears_all_proxy_spellings_and_sets_local_no_proxy()` asserting removal of `HTTP_PROXY`, `HTTPS_PROXY`, `ALL_PROXY`, `http_proxy`, `https_proxy`, and `all_proxy`, plus both `NO_PROXY` and `no_proxy` containing `127.0.0.1,localhost`.

Add parameterized `test_gfx938_profile_contract(profile, model, tp, attention, kv_layout)` covering all 15 model profiles. Assert HumanEval limit 16, deterministic sampling, unique `work_dir`, no `--enforce-eager`, no forced `--compilation-config`, prefix caching, and the expected TP. Assert HND is present only in FlashAttention profile environments.

Add focused tests for an optional `server.prefix_probe` contract: two identical OpenAI chat requests are sent after readiness, `/metrics` is captured before and after, `vllm:prefix_cache_hits_total` increases, and the final metrics body is written below the owned work directory. Mock HTTP in this portable test; do not start a model.

- [ ] **Step 3: Run the contract tests and verify they fail**

Run:

```bash
pytest -q \
  tests/integration/server/test_evalscope_report_threshold.py \
  tests/integration/server/test_evalscope_v0281_gfx938_humaneval16.py
```

Expected: FAIL because the matrix file/test does not exist yet and proxy variables are preserved.

- [ ] **Step 4: Implement the proxy boundary and matrix YAML**

Update `_server_environment()` to remove all six proxy variables before installing identical local-only `NO_PROXY` and `no_proxy` values. The YAML must contain exactly the 15 spec model paths and TP gates. Use `FLASH_ATTN` plus `VLLM_KV_CACHE_LAYOUT=HND` for FlashAttention profiles, `FLASHMLA_SPARSE` with no forced layout for sparse-MLA profiles, `--moe-backend aiter` where the model is MoE-capable, and model-specific reasoning controls from the spec.

For MTP-capable profiles use `{"method":"mtp","num_speculative_tokens":3}` unless the current model implementation supports only another official method/count. MiniMax must use `--reasoning-parser minimax_m2` and must not infer MTP support from `num_mtp_modules`.

Implement the optional probe inside `run_evalscope_server_test()` using the existing proxy-free opener. Fill the configured served model name into two identical request bodies, require HTTP 200 and coherent non-empty content, compare `vllm:prefix_cache_hits_total`, and save the final `/metrics` response as `logs/metrics.prom`. Profiles without `server.prefix_probe` must preserve current behavior.

- [ ] **Step 5: Make the parameterized external test selectable one profile at a time**

Expose `VLLM_HCU_GFX938_PROFILE` as the selected profile name. The marked external test must call `run_evalscope_server_test()` with the profile's required HCU count and model-specific label. Do not parametrize all 15 live services into an ordinary portable test run.

- [ ] **Step 6: Run contracts to green and commit**

Run the two files from Step 3 and `pytest -q tests/patch/test_plugin_lifecycle.py`. Expected: PASS. Review the diff, commit as `test: add gfx938 v0.28.1 HumanEval matrix`, and review the committed diff again.

### Task 4: Freeze wheel provenance and validate installed artifacts

**Files:**
- Create: `docs/validation/v0281-gfx938-provenance.md`
- Test: `tests/patch/test_compatibility_gate.py`
- Test: `tests/patch/test_setup_packaging.py`
- Test: `tests/patch/test_plugin_lifecycle.py`

**Interfaces:**
- Consumes: exact vLLM wheel version and clean final-branch plugin commit.
- Produces: immutable wheel checksums, install roots, package versions, source SHAs, and parent/child import-root evidence.

- [ ] **Step 1: Download and checksum the pinned vLLM wheel**

Use `pip download --no-deps` into `/models/artifacts/v0281-gfx938/`, then run `sha256sum`. Do not reinstall the global environment.

- [ ] **Step 2: Create a clean vLLM target**

Install the downloaded wheel with `--no-deps --target /models/.installs/vllm-v0281-gfx938-g77acaf` and verify `vllm`, `_C`, `_rocm_C`, `_C_stable_libtorch`, and `_moe_C_stable_libtorch` resolve below that root.

- [ ] **Step 3: Build and install a clean plugin wheel**

From a clean committed final branch run `ROCM_PATH=/opt/dtk MAX_JOBS=16 python setup.py bdist_wheel`, checksum the wheel, and install it with `--no-deps --target /models/.installs/vllm-plugin-v0281-gfx938-<sha>`.

- [ ] **Step 4: Validate parent and spawned-child imports**

With `PYTHONNOUSERSITE=1` and the two install roots ahead of other paths, print and assert vLLM, `vllm_hcu`, and `vllm_hcu.hcu_ops` locations in the parent and a spawned child. Run the three listed test files against those roots.

- [ ] **Step 5: Record provenance and commit**

Write exact filenames, SHA256 values, versions, source SHAs, module paths, DTK/Torch/Python/provider versions, and the missing/present `librt.so` state to the provenance document. Commit as `docs: record gfx938 validation provenance` after pre/post diff review.

### Task 5: Validate FlashAttention TP models and HND/BHSD E4M3

**Files:**
- Config: `tests/models/v0281_gfx938_humaneval16.yaml`
- Evidence: `docs/validation/v0281-gfx938-provenance.md`
- Failure tests if needed: `tests/runtime_patch/test_flash_attention_and_pp.py`
- Failure source if needed: `vllm_hcu/v1/attention/backends/fa_utils.py`, `vllm_hcu/v1/attention/backends/flash_attn.py`, or `vllm_hcu/v1/kv_cache.py`

**Interfaces:**
- Consumes: installed wheel roots and the matrix runner.
- Produces: TP HumanEval16, prefix reuse, MRV2, default Graph, HND-to-BHSD, and E4M3 evidence for Qwen3-8B, Qwen2-57B, and Qwen3-30B.

- [ ] **Step 1: Run Qwen3-8B TP2 BF16 control and E4M3 candidate**

Run the selected profile twice with identical requests, first `--kv-cache-dtype auto`, then `fp8_e4m3`. Require HumanEval16 16/16, nonzero prefix hits, HND/bhsd markers, `HcuGPUModelRunnerV2`, and default Graph capture.

- [ ] **Step 2: Run Qwen2-57B TP2 and Qwen3-30B TP2**

Require 16/16 on each canonical profile, AITER config/fallback evidence for the MoE paths, and clean teardown.

- [ ] **Step 3: Apply the failure loop if any run fails**

Invoke `superpowers:systematic-debugging`, read `/models/upgrading-vllm-hcu/references/failure-ledger.md`, reduce the failure to `test_flash_attention_and_pp.py`, then implement the minimal fix in the owning file above. Run the focused test, full changed-test file, and affected model again before committing.

- [ ] **Step 4: Append exact commands and results to provenance**

Record output score, graph mode, layout, dtype, prefix counters, provider selection, server/eval log paths, and teardown status for all three models.

### Task 6: Validate Qwen hybrid/GDN TP models

**Files:**
- Config: `tests/models/v0281_gfx938_humaneval16.yaml`
- Evidence: `docs/validation/v0281-gfx938-provenance.md`
- Failure tests if needed: `tests/runtime_patch/test_attention_mla_fla_mamba.py`, `tests/runtime_patch/test_qwen35_mtp_shared_gate_loading.py`, `tests/runtime_patch/test_lightop_rmsnorm_gated.py`, or `tests/runtime_patch/test_quant_gemm_aiter.py`
- Failure source if needed: the corresponding owner under `vllm_hcu/patch/worker/op_opt/`, `vllm_hcu/patch/worker/core_fix/`, or `vllm_hcu/model_executor/layers/quantization/`

**Interfaces:**
- Consumes: matrix runner and provider-selection evidence format.
- Produces: TP2/TP4 HumanEval16 results for six Qwen hybrid checkpoints, with MTP3 where supported and E4M3 on Qwen3.5 plus Qwen3.8 Flash-Next.

- [ ] **Step 1: Run both Qwen3.5-35B profiles at TP2 with MTP3**

Require 16/16, target/draft default Graph capture, per-step metadata rebuild where fused update is unsupported, nonzero prefix hits, and MTP drafted/accepted metrics. Run E4M3 plus BF16/auto control on at least one profile.

- [ ] **Step 2: Run Qwen3.6-27B and Qwen3.8-27B at TP2**

Require 16/16, default Graph, prefix reuse, and their configured quantized provider paths.

- [ ] **Step 3: Run both Qwen3.8 Flash-Next profiles at TP4 with MTP3**

Require 16/16, official hybrid manager block evidence, FlashAttention HND/bhsd, AITER or documented Triton fallback, and separate channel-FP8 versus SlimQuant provider markers. Include E4M3 on the channel-FP8 profile.

- [ ] **Step 4: Apply the same TDD failure loop and record results**

Use the owning test/source pair listed above; never add scheduler or block-size overrides. Commit each independently reviewable fix, then append final model evidence.

### Task 7: Validate sparse-MLA TP models and the direct gfx938 FP8 reader

**Files:**
- Config: `tests/models/v0281_gfx938_humaneval16.yaml`
- Evidence: `docs/validation/v0281-gfx938-provenance.md`
- Failure tests if needed: `tests/runtime_patch/test_glm53_channel_fp8.py`, `tests/runtime_patch/test_sparse_indexer_loading.py`, `tests/kernels/test_gfx938_fp8_indexer_reader.py`
- Component test: `tests/runtime_patch/test_deepseek_v4_dspark_patches.py`
- Failure source if needed: `vllm_hcu/patch/worker/core_fix/patch_glm5next_channel_fp8.py`, `vllm_hcu/v1/attention/ops/rocm_aiter_mla_sparse.py`, `vllm_hcu/v1/attention/ops/fp8_paged_mqa_gfx938.py`

**Interfaces:**
- Consumes: rebased #181 direct reader and sparse-MLA profiles.
- Produces: TP4/TP8 HumanEval16 evidence for DeepSeek-V4, three GLM-5 checkpoints, and Hy4, plus E4M3 direct-reader coverage where supported.

- [ ] **Step 1: Run DeepSeek-V4 TP4 and record its actual draft capability**

Require 16/16, sparse-MLA backend/provider markers, TP4, prefix reuse, default Graph, and only the speculative method supported by the registered implementation. Attempt ordinary E4M3 only if the model does not require `fp8_ds_mla`.

- [ ] **Step 2: Run the three GLM-5 checkpoints at TP8**

Use `glm45` only when required by the template. Require 16/16, `FLASHMLA_SPARSE`, AITER lookup/fallback, default Graph, prefix hits, and MTP3 where supported. Run the GLM-5.3 E4M3/indexer profile and verify the direct gfx938 reader marker rather than full-cache linearization.

- [ ] **Step 3: Run Hy4 at TP8**

Require 16/16, its supported sparse attention/MoE path, default Graph, MTP evidence where supported, and clean teardown.

- [ ] **Step 4: Apply the sparse-reader TDD failure loop and record results**

Distinguish writer, cache view, page inference, reader, and evaluation-parser failures before editing. Preserve official allocation and use zero-copy page views only at owning HIPC callsites.

- [ ] **Step 5: Validate the unattached DSpark component artifact**

Run `pytest -q tests/runtime_patch/test_deepseek_v4_dspark_patches.py` against `/models/dspark_qwen3_8b_block7` metadata where the tests support an override. Record it as a component-contract result only; do not report HumanEval or standalone service support.

### Task 8: Validate MiniMax TP4 without false MTP assumptions

**Files:**
- Config: `tests/models/v0281_gfx938_humaneval16.yaml`
- Evidence: `docs/validation/v0281-gfx938-provenance.md`
- Failure tests if needed: the smallest current MiniMax model-loading or attention test selected after traceback ownership is known

**Interfaces:**
- Consumes: MiniMax matrix profile with `--reasoning-parser minimax_m2`.
- Produces: TP4 HumanEval16 evidence without claiming embedded MTP solely from `num_mtp_modules`.

- [ ] **Step 1: Run the TP4 canonical profile**

Require 16/16, FlashAttention, MRV2, default Graph, prefix reuse, INT8 dense/MoE provider markers, and the official reasoning parser.

- [ ] **Step 2: Probe registered speculative capability read-only**

If vLLM registers Eagle3, test only that official path as a separate profile. If it rejects `method=mtp`, record that as expected capability behavior rather than a failure.

- [ ] **Step 3: Preserve long-reasoning failures before retry**

For any completion ending at `max_tokens`, retain the failed artifact and repeat the same fixed sample with the predeclared larger reasoning budget before diagnosing kernels.

### Task 9: Validate DP8/EP8/MTP3 and combined TP+DP+EP+MTP

**Files:**
- Modify: `tests/models/v0281_gfx938_humaneval16.yaml`
- Modify: `tests/integration/server/test_evalscope_v0281_gfx938_humaneval16.py`
- Evidence: `docs/validation/v0281-gfx938-provenance.md`
- Failure tests if needed: `tests/runtime_patch/test_moe_deepep.py`, `tests/runtime_patch/test_hcu_model_runner_cudagraph_contract.py`, and the owning MTP/DP metadata test
- Failure source if needed: `vllm_hcu/model_executor/layers/fused_moe/deepep_runtime.py` or the owning runner/DP adapter

**Interfaces:**
- Consumes: a one-card-capable MTP MoE checkpoint and validated TP profiles.
- Produces: preferred `DP8+TP1+EP8+MTP3+deepep_low_latency` evidence plus one combined TP/DP/EP/MTP topology.

- [ ] **Step 1: Add failing distributed profile contract tests**

Assert the DP8 profile has DP=8, TP=1, EP enabled, MTP3, and `deepep_low_latency`. Assert the combined profile has either DP4+TP2+EP8 or DP2+TP4+EP8, MTP3, and low-latency DeepEP. Assert neither forces eager or PIECEWISE.

- [ ] **Step 2: Add the profiles and run portable contracts**

Start with `Qwen3.5-35B-A3B-W8A8`. Run the matrix contract test and `tests/runtime_patch/test_moe_deepep.py` before the live service.

- [ ] **Step 3: Run DP8 with concurrent real requests**

Use EvalScope batch 8 or a concurrent prefix probe so all DP ranks receive work. Require per-rank HTTP activity, backend selection, Graph capture, prefix counters, and MTP drafted/accepted metrics. Run HumanEval16 twice in the same healthy service.

- [ ] **Step 4: Run a combined TP topology**

Try DP4+TP2+EP8 first; use DP2+TP4+EP8 only when memory or model constraints require it. Compare MTP3 with the same topology without MTP.

- [ ] **Step 5: Diagnose collective failures before changing code**

Invoke systematic debugging. Audit inactive ranks, buffer ownership, graph padding, microbatching, all-to-all manager selection, and per-step draft metadata. Do not bypass `sync_cudagraph_and_dp_padding` or buffer cleanup based on backend name alone.

- [ ] **Step 6: Record unsupported DP8 honestly**

If DP8 is impossible after the audited capability checks, keep the logs and reason, retain the successful combined topology, and do not force a model or kernel semantic change merely to produce DP8.

### Task 10: Perform conditional SGLang-DAS performance work

**Files:**
- Read when triggered: `/models/upgrading-vllm-hcu/references/sglang-qwen35-gap.md`
- Read when triggered: `/models/zb/sglang-das`
- Create when triggered: one focused benchmark under `benchmarks/`
- Test/modify when triggered: only the current owning operator files identified by dispatch evidence

**Interfaces:**
- Consumes: steady-state baseline/candidate measurements and actual model tensor contracts.
- Produces: either a documented “not triggered” decision or one independently tested operator optimization with at least approximately 5% repeatable end-to-end-path improvement.

- [ ] **Step 1: Compare warmed steady-state results against rebased #181**

Use identical model, command, request set, seed, layout, and cache dtype. Synchronize HCU timing and exclude cold compilation.

- [ ] **Step 2: Decide whether the performance gate triggers**

If there is no material regression and no contract-compatible measured opportunity, record “not triggered” and make no performance code change.

- [ ] **Step 3: If triggered, write a failing parity/dispatch benchmark first**

Pin real shape, dtype, layout, varlen metadata, graph replay, and provider fallback. Include conversion/copy cost. Reject an eager-only win or a result below the repeatability threshold.

- [ ] **Step 4: Implement and commit only the minimal owned operator route**

Run numerical parity, fallback, graph capture/replay, focused tests, affected HumanEval16, and steady-state measurement before committing.

### Task 11: Run the final candidate closure on all 15 models

**Files:**
- Modify: `docs/validation/v0281-gfx938-provenance.md`
- Test: every Python test file changed by `origin/pr181..HEAD`
- Test: `tests/runtime_patch/`

**Interfaces:**
- Consumes: all accepted fixes and matrix profiles.
- Produces: one exact final commit SHA with a complete 15-model gfx938 result matrix.

- [ ] **Step 1: Run source closure before the final wheel**

Run `git diff --check`, every changed test file, full `tests/runtime_patch`, packaging/lifecycle tests, and the matrix contract tests. Expected: all pass.

- [ ] **Step 2: Commit all reviewed production/test/documentation changes**

Review the complete intended diff relative to rebased #181, commit in independently reviewable units, and review each committed diff.

- [ ] **Step 3: Build and install the final clean wheel**

Build only from a clean commit, record SHA256, install in a fresh target and fresh `VLLM_CACHE_ROOT`, and repeat parent/child import-root checks.

- [ ] **Step 4: Run all 15 HumanEval16 profiles on the final SHA**

Require 16/16 per model, the specified TP, MRV2, default Graph, prefix reuse, intended provider, and complete teardown. After every profile, run `/opt/dtk/bin/rocm-smi --showmeminfo vram --showpids` and require no owned worker PID plus memory returned to the recorded idle baseline. Previously passing results from an earlier candidate SHA do not close this gate.

- [ ] **Step 5: Finalize the evidence matrix**

For each model record command, versions, layout/dtype, TP/DP/EP/MTP, graph mode, provider selection, prefix hits, draft metrics, HumanEval score, log/result paths, and teardown status. List DeepSeek-V4.1 as excluded, not failed.

### Task 12: Update and validate the upgrading-vllm-hcu skill

**Files:**
- Modify: `/models/upgrading-vllm-hcu/SKILL.md`
- Modify: `/models/upgrading-vllm-hcu/references/validation.md`
- Modify when a new failure was diagnosed: `/models/upgrading-vllm-hcu/references/failure-ledger.md`
- Modify when source routes changed: `/models/upgrading-vllm-hcu/references/sglang-qwen35-gap.md`

**Interfaces:**
- Consumes: final commands, SHAs, scores, failures, fallbacks, and limitations from Task 11.
- Produces: concise reusable rules for the v0.28.1 gfx938 stack without unverified workarounds.

- [ ] **Step 1: Invoke both `skill-creator` and `superpowers:writing-skills`**

Read both skills completely before editing the local skill. Follow their validation requirements in addition to this plan.

- [ ] **Step 2: Write only evidence-backed findings**

Add the pinned wheel/provenance rule, #180/#181 ownership, HND-to-BHSD behavior, gfx938 E4M3 writer/reader contract, MRV2 evidence, per-model parser rules, TP sizing lessons, distributed topology result, and any confirmed failure signature/fix.

- [ ] **Step 3: Keep failures and exclusions explicit**

Record unsuccessful DP8 attempts, AITER config fallbacks, truncation-sensitive HumanEval results, and the DeepSeek-V4.1 exclusion as such. Do not promote a diagnostic eager/PIECEWISE run to guidance.

- [ ] **Step 4: Validate the skill**

Run the checks required by `skill-creator` and `writing-skills`, scan for contradictions/placeholders/secrets, and re-read every changed reference from start to finish.

### Task 13: Final review, publish the one new PR, and report

**Files:**
- Review: complete `origin/pr181..HEAD` range
- Update: new GitHub PR title/body and existing PR #181 base/body
- Do not commit: credentials, raw tokens, large logs, wheel artifacts, model outputs

**Interfaces:**
- Consumes: final clean branch, passing closure, complete evidence matrix, and validated local skill.
- Produces: one new stacked PR and a findings-first code review report.

- [ ] **Step 1: Invoke `superpowers:requesting-code-review`**

Review correctness, unintended scope, vLLM interface ownership, model-path interactions, performance regressions, missing tests, and the entire stacked range. Report findings by severity with file references; fix all blocking findings and rerun affected gates.

- [ ] **Step 2: Invoke `superpowers:verification-before-completion`**

Re-run the commands required by that skill against the final commit. Do not infer success from earlier output.

- [ ] **Step 3: Push the final branch without exposing credentials**

Use the configured credential helper and `git push origin HEAD:codex/validate-v0281-gfx938-models`. Confirm the remote SHA matches local HEAD.

- [ ] **Step 4: Create exactly one new stacked PR**

Create the PR with base `codex/fix-gfx938-fp8-indexer-reader-v0281`, title `fix: validate v0.28.1 gfx938 model compatibility`, and a body containing the full validation matrix, exact wheel/plugin SHAs, tests, known limitations, and AI-assistance disclosure. Do not open another PR.

- [ ] **Step 5: Review the remote range and PR rendering**

Fetch the remote head, compare it to the rebased #181 base, rerun `git diff --check`, inspect the PR files/commits via the REST API, and ensure no earlier-base commits or secrets leaked into the diff.

- [ ] **Step 6: Deliver the final report**

Lead with code-review findings. Then provide PR links, exact SHAs, test totals, the 15 HumanEval16 results, E4M3/BHSD/MRV2/Graph evidence, distributed results, performance decision, skill files updated, and any remaining external limitations.
