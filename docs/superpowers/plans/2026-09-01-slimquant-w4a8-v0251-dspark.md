# SlimQuant W4A8 Latest v0.25.1 and DSpark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild MR #38 on the latest `v0.25.1`, retain the base branch's unified AITER SlimQuant W4A8 design, add only missing DP8+EP8 `deepep_auto` W4A8 DeepGEMM support, and validate DSpark on TP8 and DP8+EP8.

**Architecture:** Reconstruct the feature branch from `origin/v0.25.1@85a4ad5`, then port a minimal W4A8 DeepEP expert implementation into the existing `deepep_auto` modular-kernel lifecycle. Pure TP remains owned by the base branch's unified AITER dispatcher. Existing HCU DSpark target and draft models are reused without new speculative-decoding architecture.

**Tech Stack:** Python 3.10, PyTorch/HCU, vLLM 0.25.1, AITER, DeepEP, DeepGEMM HIPC W4A8, pytest, EvalScope, Git/GitHub API.

**Spec:** `docs/superpowers/specs/2026-09-01-slimquant-w4a8-v0251-dspark-design.md`

## Global Constraints

- Base all production changes on current `origin/v0.25.1@85a4ad5` or a newer fetched tip if it advances before reconstruction.
- Pure TP SlimQuant W4A8 must use the base branch's unified AITER dispatcher; do not restore the former LightOp/AITER split.
- DP8+EP8 must expose only `deepep_auto`; HT/LL selection belongs to the existing forward snapshot.
- Do not add or modify `.github`, workflows, or scripts.
- Preserve authorship as `zhangzbb <1414695739@qq.com>`.
- Never store or print access tokens in files, Git configuration, remotes, logs, or MR content.
- Do not claim DSpark support without successful live startup and request evidence.

---

### Task 1: Reconstruct MR #38 on the latest base

**Files:**
- Preserve: `docs/superpowers/specs/2026-09-01-slimquant-w4a8-v0251-dspark-design.md`
- Preserve: `docs/superpowers/plans/2026-09-01-slimquant-w4a8-v0251-dspark.md`
- Remove from final diff: obsolete MR-owned DeepSeek-V4, sampler, LightOp, and duplicate AITER changes already provided by the latest base

**Interfaces:**
- Consumes: `origin/v0.25.1` and MR #38 head history.
- Produces: a clean reconstructed branch whose initial diff contains only the approved design/plan before runtime implementation.

- [ ] **Step 1: Record recoverable branch state**

Run:

```bash
git status --short
git branch backup/mr38-before-v0251-rebuild HEAD
git fetch origin v0.25.1
git rev-parse origin/v0.25.1
```

Expected: clean worktree, backup branch at the old MR head, and a concrete current base SHA.

- [ ] **Step 2: Inventory old changes against the new base**

Run:

```bash
git diff --name-status origin/v0.25.1...backup/mr38-before-v0251-rebuild
git log --format='%h %an <%ae> %s' origin/v0.25.1..backup/mr38-before-v0251-rebuild
```

Expected: overlapping DeepSeek-V4/AITER files are explicitly identified; no uncommitted user files are lost.

- [ ] **Step 3: Reconstruct the branch**

Create a temporary reconstruction branch from the fetched base and restore only the approved spec and plan from the backup:

```bash
git switch -c rebuild/deepseek-v4-pro-slimquant-w4a8-v0251 origin/v0.25.1
git checkout backup/mr38-before-v0251-rebuild -- \
  docs/superpowers/specs/2026-09-01-slimquant-w4a8-v0251-dspark-design.md \
  docs/superpowers/plans/2026-09-01-slimquant-w4a8-v0251-dspark.md
git status --short
```

Expected: only the two approved documentation files differ from the latest base.

- [ ] **Step 4: Commit the reconstructed documentation**

```bash
git add docs/superpowers/specs/2026-09-01-slimquant-w4a8-v0251-dspark-design.md \
  docs/superpowers/plans/2026-09-01-slimquant-w4a8-v0251-dspark.md
git commit -m "docs: redesign W4A8 integration for latest v0.25.1"
```

Expected: commit author is `zhangzbb <1414695739@qq.com>`.

### Task 2: Characterize the missing W4A8 `deepep_auto` contract

**Files:**
- Test: `tests/runtime_patch/test_moe_deepep.py`
- Test: `tests/runtime_patch/test_quant_gemm_aiter.py`
- Read: `vllm_hcu/model_executor/layers/fused_moe/deepep_runtime.py`
- Read: `vllm_hcu/model_executor/layers/fused_moe/prepare_finalize/deepep_auto.py`
- Read: `vllm_hcu/model_executor/layers/quantization/slimquant_w4a8.py`

**Interfaces:**
- Consumes: latest `deepep_auto` selection snapshot and `SlimQuantW4A8Int8AiterMoEMethod`.
- Produces: failing tests that define DP+EP expert selection without changing TP AITER behavior.

- [ ] **Step 1: Write a failing DP+EP selection test**

Add a test that constructs a W4A8 `FusedMoEQuantConfig` with
`weight_quant_dtype == "int4"`, DP size greater than one, expert parallelism,
and `deepep_auto`. Assert that the SlimQuant method delegates expert creation
to a W4A8 DeepGEMM auto expert factory rather than `apply_aiter_w4a8_moe`.

- [ ] **Step 2: Verify the selection test fails for the missing capability**

Run:

```bash
VLLM_V0251_SOURCE_ROOT=/usr/local/lib/python3.10/dist-packages \
pytest -q tests/runtime_patch/test_quant_gemm_aiter.py -k 'slimquant_w4a8 and deepep_auto'
```

Expected: FAIL because the latest base has no SlimQuant W4A8 DeepEP expert implementation.

- [ ] **Step 3: Write failing snapshot/layout tests**

Add tests proving one `deepep_auto` forward snapshot selects:

```python
"high_throughput" -> contiguous W4A8 expert
"low_latency" -> masked W4A8 expert
```

Include an empty-rank case where all ranks retain the same snapshot and no
local token count is treated as a backend change.

- [ ] **Step 4: Verify snapshot/layout tests fail**

Run:

```bash
VLLM_V0251_SOURCE_ROOT=/usr/local/lib/python3.10/dist-packages \
pytest -q tests/runtime_patch/test_moe_deepep.py -k 'w4a8 and deepep_auto'
```

Expected: FAIL because no W4A8 auto experts are registered.

- [ ] **Step 5: Commit the failing contract tests**

```bash
git add tests/runtime_patch/test_moe_deepep.py \
  tests/runtime_patch/test_quant_gemm_aiter.py
git commit -m "test: define W4A8 DeepEP auto contract"
```

### Task 3: Implement W4A8 DeepGEMM expert layouts

**Files:**
- Create: `vllm_hcu/model_executor/layers/quantization/slimquant_w4a8_deepgemm_runtime.py`
- Modify: `vllm_hcu/model_executor/layers/fused_moe/experts/batched_deep_gemm_moe.py`
- Test: `tests/runtime_patch/test_deep_gemm_utils.py`
- Test: `tests/runtime_patch/test_moe_deepep.py`

**Interfaces:**
- Consumes: canonical packed `[E,N,K/2]` W4A8 parameters and base DeepGEMM expert interfaces.
- Produces: `DeepEPDeepGemmW4A8ContiguousExperts` and `DeepEPDeepGemmW4A8BatchedExperts`, both compatible with the existing modular kernel ABI.

- [ ] **Step 1: Write failing contiguous-layout tests**

Cover pack-once behavior, canonical-parameter preservation, W4A8 channel scale
shape, both GEMM stages, expert-map propagation, and zero-token output.

- [ ] **Step 2: Run the contiguous tests and observe failure**

```bash
VLLM_V0251_SOURCE_ROOT=/usr/local/lib/python3.10/dist-packages \
pytest -q tests/runtime_patch/test_deep_gemm_utils.py -k 'w4a8 and contiguous'
```

Expected: FAIL because the class/API does not exist.

- [ ] **Step 3: Implement the contiguous expert**

Use DeepGEMM's HIPC W4A8 contiguous packing and
`m_grouped_w4a8_gemm_nt_contiguous_hipc`. Store derived packed tensors outside
registered canonical parameters, validate channel scales before packing, and
reuse the base branch's DeepEP permutation and reduction helpers.

- [ ] **Step 4: Run contiguous tests**

Run the command from Step 2.

Expected: PASS.

- [ ] **Step 5: Write failing masked-layout tests**

Cover N32 six-dimensional view shape, pack-once behavior, gate/up and down
scale propagation, `m_grouped_w4a8_gemm_nt_masked_hipc`, masked token counts,
and an empty dispatch.

- [ ] **Step 6: Run the masked tests and observe failure**

```bash
VLLM_V0251_SOURCE_ROOT=/usr/local/lib/python3.10/dist-packages \
pytest -q tests/runtime_patch/test_deep_gemm_utils.py -k 'w4a8 and masked'
```

Expected: FAIL because the masked expert is absent.

- [ ] **Step 7: Implement the masked expert**

Pack canonical weights with `pack_w4a8_moe_hipc_weight`, expose the N32 view
with `view_w4a8_moe_hipc_weight_n32_layout`, and extend the batched DeepGEMM
expert only when `weight_quant_dtype == "int4"`. Preserve all existing W8A8,
FP8, BF16, and MXFP4 branches.

- [ ] **Step 8: Run focused DeepGEMM tests**

```bash
VLLM_V0251_SOURCE_ROOT=/usr/local/lib/python3.10/dist-packages \
pytest -q tests/runtime_patch/test_deep_gemm_utils.py \
  tests/runtime_patch/test_moe_deepep.py -k 'w4a8 or deepep_auto'
```

Expected: PASS.

- [ ] **Step 9: Commit the expert implementation**

```bash
git add vllm_hcu/model_executor/layers/quantization/slimquant_w4a8_deepgemm_runtime.py \
  vllm_hcu/model_executor/layers/fused_moe/experts/batched_deep_gemm_moe.py \
  tests/runtime_patch/test_deep_gemm_utils.py tests/runtime_patch/test_moe_deepep.py
git commit -m "feat: add W4A8 DeepGEMM auto experts"
```

### Task 4: Integrate SlimQuant W4A8 with `deepep_auto`

**Files:**
- Modify: `vllm_hcu/model_executor/layers/quantization/slimquant_w4a8.py`
- Modify: `vllm_hcu/model_executor/layers/fused_moe/deepep_runtime.py`
- Test: `tests/runtime_patch/test_quant_gemm_aiter.py`
- Test: `tests/runtime_patch/test_moe_deepep.py`

**Interfaces:**
- Consumes: the two W4A8 DeepGEMM expert classes from Task 3.
- Produces: DP+EP modular selection that bypasses TP AITER only for `deepep_auto`.

- [ ] **Step 1: Implement strict topology detection**

Select W4A8 DeepEP experts only when data parallel size is greater than one,
expert parallelism is enabled, and the configured all-to-all backend is
`deepep_auto`. Raise a descriptive error for fixed or incompatible DeepEP
settings rather than silently using the TP path.

- [ ] **Step 2: Bind HT and LL experts to the auto snapshot**

Register contiguous and masked experts through the latest base's auto expert
selection interface. Do not duplicate prepare/finalize selection logic.

- [ ] **Step 3: Keep pure TP behavior unchanged**

Assert in tests that TP explicit AITER still calls the shared AITER adapter,
explicit Triton still uses vLLM fallback, and no LightOp runtime is imported.

- [ ] **Step 4: Run focused routing tests**

```bash
VLLM_V0251_SOURCE_ROOT=/usr/local/lib/python3.10/dist-packages \
pytest -q tests/runtime_patch/test_quant_gemm_aiter.py \
  tests/runtime_patch/test_moe_deepep.py -k 'slimquant_w4a8 or deepep_auto'
```

Expected: PASS, including tests written in Task 2.

- [ ] **Step 5: Commit integration**

```bash
git add vllm_hcu/model_executor/layers/quantization/slimquant_w4a8.py \
  vllm_hcu/model_executor/layers/fused_moe/deepep_runtime.py \
  tests/runtime_patch/test_quant_gemm_aiter.py tests/runtime_patch/test_moe_deepep.py
git commit -m "feat: route W4A8 DP EP through deepep auto"
```

### Task 5: Run static, unit, and operator verification

**Files:**
- Verify: all files changed since `origin/v0.25.1`

**Interfaces:**
- Consumes: reconstructed implementation.
- Produces: clean static checks and reproducible test evidence before model startup.

- [ ] **Step 1: Check production boundaries and forbidden changes**

```bash
git diff --check origin/v0.25.1...HEAD
git diff --name-only origin/v0.25.1...HEAD | rg '(^|/)(\.github|workflows|scripts)(/|$)' && exit 1 || true
python tools/check_production_boundary.py
```

Expected: no whitespace errors, forbidden paths, or boundary violations.

- [ ] **Step 2: Run focused runtime tests**

```bash
VLLM_V0251_SOURCE_ROOT=/usr/local/lib/python3.10/dist-packages \
pytest -q tests/runtime_patch/test_quant_gemm_aiter.py \
  tests/runtime_patch/test_moe_deepep.py \
  tests/runtime_patch/test_deep_gemm_utils.py \
  tests/runtime_patch/test_deepseek_v4_dspark_model.py \
  tests/runtime_patch/test_deepseek_v4_dspark_patches.py
```

Expected: all tests pass.

- [ ] **Step 3: Run available HCU operator tests**

```bash
HIP_VISIBLE_DEVICES=0 VLLM_PLUGINS=__disabled__ \
pytest -q tests/accuracy/test_deepseek_v4_dspark_ops.py -s
```

Expected: existing DSpark operators pass; add a focused W4A8 device test if the
base suite does not exercise both HIPC W4A8 layouts.

- [ ] **Step 4: Run the broad relevant suite**

```bash
VLLM_V0251_SOURCE_ROOT=/usr/local/lib/python3.10/dist-packages \
pytest -q tests/runtime_patch
```

Expected: all collected runtime-patch tests pass.

### Task 6: Validate TP8 AITER with DSpark

**Files:**
- Artifact: `/tmp/vllm-w4a8-tp8-aiter-dspark.log`
- Artifact: `/tmp/vllm-w4a8-tp8-aiter-dspark-response.json`

**Interfaces:**
- Consumes: target checkpoint and explicit AITER backend.
- Produces: live startup, output, AITER selection, and speculative acceptance evidence.

- [ ] **Step 1: Confirm resources and terminate no unrelated services**

Run read-only process/device checks and select a free port. Stop only service
processes launched by this task if cleanup is needed.

- [ ] **Step 2: Start TP8+AITER+DSpark**

```bash
vllm serve /models/DeepSeek-V4-Pro-0813-INT4-Channel \
  --host 0.0.0.0 --port 10140 --trust-remote-code \
  --tokenizer-mode deepseek_v4 --distributed-executor-backend mp \
  --quantization slimquant_w4a8 --tensor-parallel-size 8 \
  --moe-backend aiter --kv-cache-dtype fp8 --block-size 256 \
  --max-model-len 4096 --max-num-batched-tokens 512 --max-num-seqs 8 \
  --gpu-memory-utilization 0.9 \
  --served-model-name deepseek-v4-pro-int4-channel \
  --speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}' \
  > /tmp/vllm-w4a8-tp8-aiter-dspark.log 2>&1
```

Expected: API ready and all target/draft workers loaded.

- [ ] **Step 3: Send a deterministic client request**

```bash
curl --noproxy '*' -sS http://127.0.0.1:10140/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4-pro-int4-channel","messages":[{"role":"user","content":"Write a Python hello-world program."}],"temperature":0,"max_tokens":128,"chat_template_kwargs":{"thinking":false}}' \
  > /tmp/vllm-w4a8-tp8-aiter-dspark-response.json
```

Expected: non-empty assistant output.

- [ ] **Step 4: Record route and acceptance evidence**

Inspect the log for unified AITER W4A8 config selection, DSpark proposed and
accepted token counters, and absence of load/runtime errors.

- [ ] **Step 5: Stop the task-owned TP service cleanly**

Expected: only the recorded TP service process tree is terminated.

### Task 7: Validate DP8+EP8 `deepep_auto` with DSpark

**Files:**
- Artifact: `/tmp/vllm-w4a8-dp8-ep8-deepep-auto-dspark.log`
- Artifact: `/tmp/vllm-w4a8-dp8-ep8-deepep-auto-dspark-response.json`

**Interfaces:**
- Consumes: W4A8 auto experts and base DSpark runtime.
- Produces: live DP+EP startup, output, DeepEP/DeepGEMM route, and speculative evidence.

- [ ] **Step 1: Start DP8+EP8+DSpark**

```bash
vllm serve /models/DeepSeek-V4-Pro-0813-INT4-Channel \
  --host 0.0.0.0 --port 10137 --trust-remote-code \
  --tokenizer-mode deepseek_v4 --distributed-executor-backend mp \
  --quantization slimquant_w4a8 --tensor-parallel-size 1 \
  --data-parallel-size 8 --enable-expert-parallel \
  --all2all-backend deepep_auto --kv-cache-dtype fp8 --block-size 256 \
  --max-model-len 4096 --max-num-batched-tokens 512 --max-num-seqs 8 \
  --gpu-memory-utilization 0.9 \
  --served-model-name deepseek-v4-pro-int4-channel \
  --speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}' \
  > /tmp/vllm-w4a8-dp8-ep8-deepep-auto-dspark.log 2>&1
```

Expected: one API service and eight DP workers become ready.

- [ ] **Step 2: Send the same deterministic request**

Use the Task 6 request with port `10137` and write the DP artifact.

Expected: non-empty assistant output.

- [ ] **Step 3: Verify DeepEP auto and DSpark evidence**

Inspect synchronized rank logs for DeepEP dispatch/combine, W4A8 DeepGEMM,
the selected contiguous or masked layout, and proposed/accepted speculative
tokens. Treat rank disagreement, empty-rank collective failure, or fallback to
the TP AITER path as failure.

- [ ] **Step 4: Stop the task-owned DP service cleanly**

Expected: all eight task-owned workers terminate without affecting unrelated processes.

### Task 8: Run HumanEval-32 and consolidate evidence

**Files:**
- Artifact: `/tmp/vllm-hcu-evalscope/deepseek-v4-pro-int4-dspark-tp8/`
- Artifact: `/tmp/vllm-hcu-evalscope/deepseek-v4-pro-int4-dspark-dp8-ep8/`

**Interfaces:**
- Consumes: successful Task 6 and Task 7 services.
- Produces: 32 predictions, 32 reviews, pass@1, latency, and artifact paths per supported topology.

- [ ] **Step 1: Run the repository-owned EvalScope gate for TP8**

Adapt the latest base's `test_evalscope_deepseek_v4_dspark_humaneval.py` model
environment and server arguments to the INT4 checkpoint without creating a
new script.

Expected: exactly 32 prediction and 32 review records.

- [ ] **Step 2: Run the DP8+EP8 gate**

Use the same dataset and scoring contract against the `deepep_auto` service.

Expected: exactly 32 prediction and 32 review records, or a precisely recorded
resource/runtime blocker.

- [ ] **Step 3: Summarize raw results**

Record pass@1, successful sample count, mean latency, TTFT, output throughput,
and report paths. Do not replace raw metrics with normalized-only claims.

### Task 9: Final review, MR update, and safe branch replacement

**Files:**
- Update remotely: MR #38 Summary and its consolidated Summary comment
- Verify locally: complete `origin/v0.25.1...HEAD` diff

**Interfaces:**
- Consumes: verified implementation and artifacts.
- Produces: an updated MR #38 based on the latest design with no unresolved Critical/Important findings.

- [ ] **Step 1: Run final verification**

Repeat Task 5 static and broad tests after the last code change. Confirm the
worktree is clean and every new commit has the required author.

- [ ] **Step 2: Request independent code review**

Review the complete new-base diff, focusing on TP ownership, DP auto snapshot,
W4A8 layout/scale correctness, DSpark draft layers, non-SlimQuant impact, and
empty-rank collectives.

Expected: findings sorted by Critical, Important, and Minor.

- [ ] **Step 3: Fix Critical and Important findings with TDD**

For each accepted finding, first add a failing regression test, make the
minimal fix, rerun focused tests, and then repeat final verification. Request a
targeted re-review of the fix.

- [ ] **Step 4: Replace the MR branch safely**

After verifying the backup branch and exact remote target, move the local MR
branch name to the reconstructed commit and push with lease protection:

```bash
git push --force-with-lease origin \
  HEAD:fix/deepseek-v4-pro-slimquant-w4a8-v0251
```

Expected: MR #38 updates without overwriting an unexpected remote change.

- [ ] **Step 5: Rewrite MR Summary**

Include latest-base ownership, retained W4A8 DP+EP delta, exact TP/DP server
commands, client command, unit/operator results, DSpark route and acceptance
evidence, HumanEval32 metrics, review status, and explicit absence of
`.github`/workflow/script changes.

- [ ] **Step 6: Verify the published MR**

Confirm MR #38 targets `v0.25.1`, its head SHA equals the local verified SHA,
its diff excludes forbidden paths and obsolete patches, and its Summary
matches observed evidence.
