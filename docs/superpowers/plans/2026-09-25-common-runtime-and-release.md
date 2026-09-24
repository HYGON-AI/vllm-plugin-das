# Common HCU Runtime and Release Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Carry only evidenced, still-missing v0.25.1 common runtime improvements into v0.28.1-dev, update the upgrade skill from measured results, and publish one reviewed Hy4-plus-common MR.

**Architecture:** Audit each source-only commit by functional owner before changing target code. Migrate one obsolete test-root contract and bound the target sparse-indexer Torch fallback; treat scheduler, communicator, GDN/FLA, LightOp, and other candidates as conditional evidence decisions, not automatic cherry-picks. Release only after the paired Hy4 MRV2 plan's device and accuracy gates.

**Tech Stack:** git, Python 3.10, installed OpenDAS vLLM 0.28.1, pytest, PyTorch/HIP, HCU plugin, GitHub PR, /models/upgrading-vllm-hcu skill.

**Spec:** docs/superpowers/specs/2026-09-25-hy4-v0281-runtime-alignment-design.md

## Global Constraints

- Keep plugin base origin/v0.28.1-dev@93021650a5c768121507d5b8241d0e848756e44e and source origin/v0.25.1@6ea7b12f3d77d4564d613aac31c5e6c5e7a8641d pinned until refreshed.
- Same MR as the Hy4 MRV2 plan, but separate commits and tests; no whole-commit cherry-picks.
- VLLM_USE_V2_MODEL_RUNNER=1 is mandatory; reject any V1-only scheduler/communicator optimization.
- The legacy DeviceCommunicatorBase.__init__ use_all2all patch is out; target ParallelConfig owns that selection.
- Do not change scheduler defaults without same-workload correctness and performance evidence.
- Do not globally install or replace vLLM/Torch/HIP packages; never print or commit access tokens.
- Update /models/upgrading-vllm-hcu only after actual target validation, explicitly recording failed gates.

## Review Focus

- Long decode with short neighboring requests and invalid unused block-table entries: bounded fallback must mask unused pages and avoid host .item() during graph replay (Task 3 tests).
- Prefill with empty causal intervals or scales shaped [N] vs [N,1]: preserve negative-infinity mask and numerical agreement while bounding temporary workspace (Task 3 tests).
- Missing installed vLLM target or unexpected import root: dispatcher test must fail with a clear target-root error, not point to v0.25.1 (Task 2 test).
- PCP+EP enabled under MRV2 but missing a target hook: reject unsupported configuration rather than carry a V1 communicator monkeypatch (Task 4 test).
- Target ref advances before publication: re-audit changed dispositions and rerun affected tests, or stop publication with the refreshed SHA documented (Task 5 check).

---

### Task 1: Audit all source-only commits and target owners

**Files:**
- Create: docs/hy4_v0281_commit_dispositions.md

**Interfaces:**
- Consumes: git merge-base and git log of frozen refs, target source tree, the approved design spec.
- Produces: commit-by-commit owner, target equivalent/gap, selected path, test, and disposition; Task 4 acts on this matrix.

- [ ] **Step 1: Refresh refs and reproduce the source-only list.**

~~~bash
git fetch origin v0.25.1 v0.28.1-dev
git merge-base origin/v0.25.1 origin/v0.28.1-dev
git log --no-merges --format='%h %s' origin/v0.28.1-dev..origin/v0.25.1
~~~

If the result differs from the spec's 28 commits, update the spec before proceeding. Include every commit exactly once in docs/hy4_v0281_commit_dispositions.md.

- [ ] **Step 2: For each runtime candidate, inspect final source and target owners.**

~~~bash
git show --stat --oneline 021625d b701be3 4f1f266 325dee8 8565e54 9218102
git diff origin/v0.28.1-dev..origin/v0.25.1 -- vllm_hcu/v1/attention/ops/rocm_aiter_mla_sparse.py
rg -n 'use_all2all|steady_decode|FusedMoEFactory|LightOp|fp8_paged_mqa_logits_torch' vllm_hcu
~~~

Use the spec's five audit lanes. The matrix records status as adapted, target-equivalent, intentionally excluded, or unverified. For Hy4 entries, link the Hy4 plan's owning task. For generic sparse indexer, link Task 3 here. For the V1 steady-decode route, record exclusion and the exact V1/TP1/no-prefix/no-MTP guards. Do not call an untested route equivalent.

- [ ] **Step 3: Review the matrix against the 28-line git log and commit.**

~~~bash
git add docs/hy4_v0281_commit_dispositions.md
git commit -m "docs: classify v0.25.1 source-only runtime changes"
~~~

### Task 2: Remove legacy v0.25.1 test-root coupling

**Files:**
- Modify: tests/patch/test_platform_dispatcher.py

**Interfaces:**
- Consumes: importlib.util.find_spec("vllm") or VLLM_TARGET_ROOT.
- Produces: installed-target-root subprocess checks matching test_plugin_lifecycle.py.

- [ ] **Step 1: Reproduce the collection failure.**

~~~bash
PYTHONNOUSERSITE=1 PYTHONPATH=. pytest -q tests/patch/test_platform_dispatcher.py
~~~

Expected baseline: RuntimeError naming absent VLLM_V0251_SOURCE_ROOT, before any plugin test runs.

- [ ] **Step 2: Add a focused root-selection test using the installed vLLM package.**

~~~python
def test_target_root_is_installed_vllm():
    import importlib.util
    from pathlib import Path
    import vllm
    spec = importlib.util.find_spec("vllm")
    assert spec is not None and spec.origin is not None
    assert Path(vllm.__file__).resolve().is_relative_to(TARGET_VLLM_ROOT)
~~~

The test file already imports TARGET_VLLM_ROOT at module level. Keep an explicit VLLM_TARGET_ROOT override for isolated installs.

- [ ] **Step 3: Mirror the target-root setup in tests/patch/test_plugin_lifecycle.py.**

~~~python
import importlib.util

_target_root_override = os.environ.get("VLLM_TARGET_ROOT")
if _target_root_override:
    TARGET_VLLM_ROOT = Path(_target_root_override).resolve()
else:
    _spec = importlib.util.find_spec("vllm")
    if _spec is None or _spec.origin is None:
        raise RuntimeError("vllm package is not discoverable")
    TARGET_VLLM_ROOT = Path(_spec.origin).resolve().parents[1]
~~~

Replace VLLM_V0251_SOURCE_ROOT in the child-process environment and assertion with VLLM_TARGET_ROOT. Keep the target-root path check; do not merely remove provenance validation.

- [ ] **Step 4: Run dispatcher and lifecycle tests, then commit.**

~~~bash
PYTHONNOUSERSITE=1 PYTHONPATH=. pytest -q tests/patch/test_platform_dispatcher.py tests/patch/test_plugin_lifecycle.py
git add tests/patch/test_platform_dispatcher.py
git commit -m "test: resolve platform dispatcher against installed vLLM"
~~~

### Task 3: Bound generic sparse-indexer Torch fallback

**Files:**
- Modify: vllm_hcu/v1/attention/ops/rocm_aiter_mla_sparse.py:607
- Create: tests/runtime_patch/test_sparse_indexer_torch_fallback.py

**Interfaces:**
- Consumes: fp8_paged_mqa_logits_torch(q, kv_cache, weights, context_lens, block_tables, max_model_len) and fp8_mqa_logits_torch(q, kv, weights, cu_seqlen_ks, cu_seqlen_ke).
- Produces: same logits shape and mask, graph-capturable shape-only loops, <=8 MiB temporary decode/prefill score workspace.

- [ ] **Step 1: Port source tests for reference parity, variable decode lengths, long-context workspace, empty-interval prefill, and graph replay using apply_patch.**

~~~python
assert max(workspaces) <= 8 * 1024 * 1024
assert torch.isneginf(out[row, row:]).all()
torch.testing.assert_close(actual, expected, atol=0.001, rtol=0.0002)
~~~

The exact source test names are test_long_context_bounds_decode_workspace, test_prefill_bounds_workspace_and_preserves_masks, test_decode_graph_replay, and test_prefill_logits_graph_replay in origin/v0.25.1:tests/runtime_patch/test_sparse_indexer_torch_fallback.py. Adapt device markers to the current HCU pytest conventions without reducing assertions.

- [ ] **Step 2: Run CPU cases; expect long-context/memory and graph-safety failures on the target's .item()/full-einsum implementation.**

~~~bash
PYTHONNOUSERSITE=1 PYTHONPATH=. pytest -q tests/runtime_patch/test_sparse_indexer_torch_fallback.py -k 'not graph'
~~~

- [ ] **Step 3: Adapt the source's bounded shape-only page/query loops to target signatures using apply_patch.**

~~~python
num_pages = min(
    block_tables.shape[1],
    (max_model_len + block_size - 1) // block_size,
)
logits = torch.full(
    (batch_size, next_n, max_model_len),
    float("-inf"), dtype=torch.float32, device=q.device,
)
~~~

Copy the complete final gather/mask computation from
origin/v0.25.1:vllm_hcu/v1/attention/ops/rocm_aiter_mla_sparse.py
functions fp8_paged_mqa_logits_torch and fp8_mqa_logits_torch through
apply_patch. The excerpt above pins the output shape and page bound. Preserve
source chunk sizing, use tensor context lengths rather than .item(), and keep
the public output ABI. For prefill, chunk both query and key axes before
einsum and mask empty intervals as -inf.

- [ ] **Step 4: Run CPU, HCU graph-replay, and adjacent sparse-indexer tests; inspect allocations.**

~~~bash
PYTHONNOUSERSITE=1 PYTHONPATH=. pytest -q tests/runtime_patch/test_sparse_indexer_torch_fallback.py tests/runtime_patch/test_sparse_indexer_loading.py
~~~

Record actual peak workspace and graph replay result. If capture fails due target core ABI, diagnose before changing the graph policy.

- [ ] **Step 5: Commit.**

~~~bash
git add vllm_hcu/v1/attention/ops/rocm_aiter_mla_sparse.py tests/runtime_patch/test_sparse_indexer_torch_fallback.py
git commit -m "fix: bound generic sparse-indexer fallback workspace"
~~~

### Task 4: Resolve remaining common runtime candidates

**Files:**
- Modify: docs/hy4_v0281_commit_dispositions.md
- Modify if a tested target gap exists: vllm_hcu/model_executor/layers/fused_moe/moe_runner.py
- Test if that gap exists: tests/runtime_patch/test_moe_deepep.py

**Interfaces:**
- Consumes: Task 1 matrix and Task 3 sparse-indexer result.
- Produces: documented target-equivalent/excluded status or a small MRV2-safe common patch with tests.

- [ ] **Step 1: Run existing focused target tests for SlimQuant, native FP8 KV, MoE, PCP+EP, LightOp, GDN/FLA, and scheduler.**

~~~bash
PYTHONNOUSERSITE=1 PYTHONPATH=. pytest -q tests/runtime_patch/test_hcu_cache_kernel_source.py tests/runtime_patch/test_moe_deepep.py tests/runtime_patch/test_worker_framework_opt.py tests/runtime_patch/test_attention_mla_fla_mamba.py tests/runtime_patch/test_glm52_pcp_config.py
~~~

The source's test_scheduler_adapters.py is absent on the target; classify
4f1f266 by code/contract review and run target scheduling integration only if
a current-owner regression is identified.

- [ ] **Step 2: Compare source behavior of 021625d, b701be3, 325dee8, and 8565e54 to target code and test observations.** Update each matrix row with the exact owner and one of: equivalent, missing-and-selected, unsupported MRV2, or out of this MR. Do not port b701be3's DeviceCommunicatorBase.__init__ use_all2all override or 4f1f266's V1 steady-decode scheduler. M-RoPE from 4f1f266 is outside this Hy4 run unless a specific target regression test fails.

- [ ] **Step 3: For a selected MoE gap only, write a failing target test and a narrow target-owner fix.**

~~~python
def test_moe_runner_preserves_valid_token_counts_for_mrv2():
    from inspect import signature
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
    assert "valid_token_counts" in str(signature(MoERunner.forward))
~~~

This interface test alone does not authorize a monkeypatch. If target already has the parameter, mark it equivalent and make no product-code edit. If an actual plugin call drops the value, add a behavioral assertion to tests/runtime_patch/test_moe_deepep.py that captures the forwarded tensor, then change only that call site and rerun the test.

- [ ] **Step 4: Run changed test files plus Hy4 MRV2 regression suite; commit only a real targeted gap or the completed matrix.**

~~~bash
PYTHONNOUSERSITE=1 PYTHONPATH=. pytest -q tests/runtime_patch/test_moe_deepep.py tests/runtime_patch/test_sparse_indexer_torch_fallback.py tests/models/hy_v4 tests/patch/test_plugin_lifecycle.py
git add docs/hy4_v0281_commit_dispositions.md
git commit -m "docs: finalize common runtime dispositions"
~~~

If a narrow code fix was required, stage that exact product file and owning test in a separate commit after its red/green test cycle.

### Task 5: Re-review, update skill, and publish one MR

**Files:**
- Modify: /models/upgrading-vllm-hcu/SKILL.md
- Modify: /models/upgrading-vllm-hcu/references/plugin-migration.md
- Modify: /models/upgrading-vllm-hcu/references/validation.md
- Modify: docs/hy4_v0281_validation.md
- Modify: docs/hy4_v0281_commit_dispositions.md

**Interfaces:**
- Consumes: completed Hy4 plan, Tasks 1-4, fresh remote refs, complete final diff, device/accuracy artifacts.
- Produces: reviewed plugin MR and updated skill path, with no unsupported success claims.

- [ ] **Step 1: Refresh both remote refs; compare the base and source SHAs to those in the spec and matrix.**

~~~bash
git fetch origin v0.25.1 v0.28.1-dev
git rev-parse origin/v0.25.1 origin/v0.28.1-dev
git merge-base HEAD origin/v0.28.1-dev
git diff --check origin/v0.28.1-dev...HEAD
~~~

If target advanced, rebase the feature branch, re-evaluate affected API owners, and rerun all changed tests plus the live Hy4 gates before publication.

- [ ] **Step 2: Review every changed file against the spec and matrix, fix blocking findings, then rerun all owning tests and HumanEval/0-7 comparisons.** Include installed-artifact bootstrap, actual HcuGPUModelRunnerV2 construction, AITER/fallback route, graph capture, sparse route, native MTP3 and FP8 KV, and process teardown. Preserve failures and retry logs.

- [ ] **Step 3: Load superpowers:writing-skills and skill-creator, then update the named skill with only verified target migration lessons using apply_patch.**

~~~text
Hy4 v0.28.1-dev: VLLM_USE_V2_MODEL_RUNNER=1 is explicit. Verify the
HcuGPUModelRunnerV2 worker in a live TP8 run; target vLLM already owns Hy4
config/parsers/MTP conversion. Keep only plugin-owned model, quantization,
sparse attention, and MoE adapters. Record separate target/MTP3/FP8-KV
HumanEval/0-7 outcomes and any failed Graph gate.
~~~

Integrate this content into the skill's existing sections rather than appending a duplicate rule. Update plugin-migration.md with concrete target API changes and validation.md with exact successful and failed commands/log locations. Keep the skill outside the plugin Git commit.

- [ ] **Step 4: Run final tests and syntax/skill checks, and record their output.**

~~~bash
PYTHONNOUSERSITE=1 PYTHONPATH=. pytest -q tests/models/hy_v4 tests/patch/test_platform_dispatcher.py tests/patch/test_plugin_lifecycle.py tests/patch/test_clean_process_bootstrap.py tests/runtime_patch/test_sparse_indexer_torch_fallback.py tests/runtime_patch/test_sparse_indexer_loading.py
git diff --check origin/v0.28.1-dev...HEAD
~~~

Read the updated SKILL.md and both updated references in full. Confirm no token, stale SHA, impossible test command, or unproven success statement.

- [ ] **Step 5: Publish a single reviewed MR against v0.28.1-dev only after required gates pass.**

~~~bash
git status --short
git push -u origin feat/hy4-v0281-alignment
gh pr create --base v0.28.1-dev --head feat/hy4-v0281-alignment --title "Adapt Hy4 and common HCU runtime to v0.28.1-dev" --body-file docs/hy4_v0281_validation.md
~~~

Add a PR comment with exact final server commands, model path, MRV2 env, resolved Graph mode, result/log paths, HumanEval per-item outcomes, and the commit-disposition matrix. If any required gate is still unverified, do not state that it passed and do not publish a success-labelled MR. Final handoff includes /models/upgrading-vllm-hcu/SKILL.md.
