# Step5 and HY4 Custom-Op Master Gating Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `VLLM_HCU_USE_CUSTOM_OPS` override every optional Step5 acceleration introduced by `75aa13701` and the HY4 PCP linear-gate sharding path, without disabling correctness or compatibility behavior.

**Architecture:** Resolve every optional provider as `custom_ops_enabled() and child_enabled` before layout mutation, registration, allocation, or execution. Keep Step5 model support and DSA semantics always available; reject HY4 TP1/PCP8 early when the effective gate shard is disabled because its unsharded layout is not a usable fallback.

**Tech Stack:** Python 3, PyTorch, vLLM 0.28.1 HCU runtime adapters, pytest, eight-card DCU/HCU service validation.

**Spec:** `docs/superpowers/specs/2026-10-15-step5-custom-ops-master-gating-design.md`

## Global Constraints

- `VLLM_HCU_USE_CUSTOM_OPS` defaults to enabled and has higher priority than every managed child switch.
- Existing child defaults and accepted values remain unchanged except that absent `VLLM_HCU_ENABLE_LINEAR_GATE_PCP_SHARD` defaults on only for HY4 TP1/PCP8.
- Master-off must be selected before weight conversion, fused parameter creation, custom-op/splitting-op registration, IPC allocation, or graph capture.
- Step5 config/model registration, checkpoint loading, MTP, distributed correctness, EPLB, and DSA semantics remain active with the master off.
- Explicit MoE backend selection remains outside this master; optional MoE helpers remain inside it.
- Missing safe fallbacks fail during configuration with an actionable error instead of falling through after allocation.
- Environment changes after model construction are unsupported; cold-import tests verify import-time choices.
- Production edits use TDD and each task ends in a focused commit.

## Review Focus

- Master `0` with child `1` must disable the child at config, construction, registration, and execution boundaries; Tasks 1–3 test each boundary.
- HY4 TP1/PCP8 with either master `0` or explicit shard child `0` must fail before weights allocate; Task 1 tests both messages and the no-env positive case.
- Master-off must not translate into `VLLM_STEP5_DISABLE_DSA=1` or remove DSA compile registration; Task 4 adds the regression.
- Import order must not freeze a stale master value across config/runtime selection; Tasks 1–3 use cold subprocess imports and shared resolvers.
- Layout-changing Step5 FP8/fusion children must leave native weights and unfused modules intact under master-off; Task 2 tests constructed methods/layout choices, not constants alone.

---

### Task 1: HY4 PCP linear-gate effective policy

**Files:**
- Modify: `vllm_hcu/platforms/envs.py`
- Modify: `vllm_hcu/models/hy_v4/attention.py`
- Modify: `vllm_hcu/patch/platform/core_fix/patch_vllm_config.py`
- Modify: `tests/models/hy_v4/test_attention.py`
- Modify: `tests/runtime_patch/test_glm52_pcp_config.py`

**Interfaces:**
- Consumes: `custom_ops_enabled() -> bool` and `optional_custom_op_enabled(feature_enabled: bool = True) -> bool` from `vllm_hcu.platforms.envs`.
- Produces: `linear_gate_pcp_shard_enabled(*, tp_size: int, pcp_size: int, pp_size: int = 1) -> bool` in `vllm_hcu.platforms.envs`, imported by both HY4 construction and config validation.

- [ ] **Step 1: Complete failing resolver tests**

Extend the existing tests so they assert:

```python
# no child setting: only TP1/PCP8 defaults on when master is enabled
assert linear_gate_pcp_shard_enabled(tp_size=1, pcp_size=8) is True
assert linear_gate_pcp_shard_enabled(tp_size=1, pcp_size=4) is False
assert linear_gate_pcp_shard_enabled(tp_size=1, pcp_size=8, pp_size=2) is False

# the master always wins
monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "0")
monkeypatch.setenv("VLLM_HCU_ENABLE_LINEAR_GATE_PCP_SHARD", "1")
assert linear_gate_pcp_shard_enabled(tp_size=1, pcp_size=8) is False
```

Add config tests for no-env acceptance, explicit child `0` rejection, and
master `0` plus child `1` rejection. Match error text naming the effective
shard requirement and the blocking switch.

- [ ] **Step 2: Run the focused tests and observe RED**

Run:

```bash
pytest -q tests/models/hy_v4/test_attention.py \
  tests/runtime_patch/test_glm52_pcp_config.py
```

Expected: failures show the current resolver has no topology parameters, the
unset child defaults off, and the master is ignored.

- [ ] **Step 3: Implement the dependency-light effective resolver**

Add `linear_gate_pcp_shard_enabled(*, tp_size: int, pcp_size: int, pp_size: int =
1) -> bool` to `vllm_hcu/platforms/envs.py`. Preserve the current child parser;
when the child is absent, request sharding only for
`(tp_size, pcp_size, pp_size) == (1, 8, 1)`, then pass the requested value
through `optional_custom_op_enabled`.

- [ ] **Step 4: Bind config and model construction to the same resolver**

In `attention.py`, import the resolver and call it with
`get_tensor_model_parallel_world_size()`, `get_pcp_group().world_size`, and the
current `VllmConfig` pipeline-parallel size before choosing
`PCPShardedGateLinear`. In `patch_vllm_config.py`, call it with the validated
`ParallelConfig` sizes and emit separate actionable errors for master-off and
explicit child-off cases.

- [ ] **Step 5: Run the focused tests and observe GREEN**

Run the Step 2 command. Expected: all selected tests pass.

- [ ] **Step 6: Commit Task 1**

```bash
git add vllm_hcu/platforms/envs.py \
  vllm_hcu/models/hy_v4/attention.py \
  vllm_hcu/patch/platform/core_fix/patch_vllm_config.py \
  tests/models/hy_v4/test_attention.py \
  tests/runtime_patch/test_glm52_pcp_config.py
git commit -m "fix(hy4): gate PCP linear shard under custom ops master"
```

### Task 2: Step5 model-local and load-time accelerations

**Files:**
- Modify: `vllm_hcu/models/step5_text.py`
- Modify: `vllm_hcu/models/step5/kernels.py`
- Modify: `vllm_hcu/models/step5_swa_fused.py`
- Modify: `vllm_hcu/model_executor/layers/skinny_lm_head.py`
- Modify: `vllm_hcu/ops/token_fp8_quant.py`
- Modify: `vllm_hcu/patch/platform/core_fix/patch_step5_splitting_ops.py`
- Modify: `vllm_hcu/patch/worker/framework_opt/patch_step5_skinny_lm_head.py`
- Modify: `vllm_hcu/patch/worker/framework_opt/patch_step5_token_fp8.py`
- Create: `tests/runtime_patch/test_step5_custom_ops_master.py`
- Modify: `tests/kernels/test_step5_router_topk.py`
- Modify: `tests/model_executor/test_step5_projection_fusion.py`
- Modify: `tests/runtime_patch/test_step5_optimization_scope.py`

**Interfaces:**
- Consumes: `optional_custom_op_enabled(feature_enabled: bool = True) -> bool`.
- Produces: master-gated import-time values for `FUSE_QKVG`, `BF16_ROUTER`, `FUSED_ROUTER_SHARED`, `FP8_DENSE`, `FP8_BF16_MOE`, `SWA_FUSED_QK`, `FUSED_GATE`, skinny-head enablement, and token-FP8 enablement; `router_bias_func(...)` selects the eager Torch implementation under master-off.

- [ ] **Step 1: Add cold-import master/child matrix tests**

In `test_step5_custom_ops_master.py`, launch a clean Python subprocess for
master/child combinations and assert every listed module value is false when
the master is `0`, including when each child is `1`. Assert existing defaults
and explicit child `0` behavior remain unchanged when the master is `1`.

- [ ] **Step 2: Add behavioral fallback tests**

Add tests that prove master-off:

- builds independent QKV/gate projections and native router/shared-expert paths;
- does not mutate BF16 weights into the optional FP8 dense or expert layouts;
- does not append Step5 fused splitting ops;
- runs Step5 biased top-k through the existing eager Torch implementation even
  for CUDA/HCU-shaped test inputs, without importing `vllm_hcu.ops.step5.router_bias`;
- calls the original logits-head and `_fp8_quantize` implementations without
  importing/executing skinny-head or Triton token-FP8 providers.

- [ ] **Step 3: Run Task 2 tests and observe RED**

Run:

```bash
pytest -q tests/runtime_patch/test_step5_custom_ops_master.py \
  tests/kernels/test_step5_router_topk.py \
  tests/model_executor/test_step5_projection_fusion.py \
  tests/runtime_patch/test_step5_optimization_scope.py
```

Expected: child `1` still enables current optimized providers under master `0`.

- [ ] **Step 4: Gate model-local provider selection**

Wrap the existing child boolean expression at each import-time decision with
`optional_custom_op_enabled(...)`. Do not change child defaults, model-support
imports, checkpoint mappings, or DSA selection. Make `router_bias_func` consult
the master before importing the Triton router and reuse its existing eager
implementation as the fallback.

- [ ] **Step 5: Gate splitting-op and wrapper execution boundaries**

Make splitting-op registration inspect the same effective values. Keep
correctness wrappers installed only where their master-off branch calls the
original implementation without importing the optimized module.

- [ ] **Step 6: Run Task 2 tests and observe GREEN**

Run the Step 3 command. Expected: all selected tests pass, with spies showing
no optimized entry point reached under master-off.

- [ ] **Step 7: Commit Task 2**

```bash
git add vllm_hcu/models/step5_text.py \
  vllm_hcu/models/step5/kernels.py \
  vllm_hcu/models/step5_swa_fused.py \
  vllm_hcu/model_executor/layers/skinny_lm_head.py \
  vllm_hcu/ops/token_fp8_quant.py \
  vllm_hcu/patch/platform/core_fix/patch_step5_splitting_ops.py \
  vllm_hcu/patch/worker/framework_opt/patch_step5_skinny_lm_head.py \
  vllm_hcu/patch/worker/framework_opt/patch_step5_token_fp8.py \
  tests/runtime_patch/test_step5_custom_ops_master.py \
  tests/kernels/test_step5_router_topk.py \
  tests/model_executor/test_step5_projection_fusion.py \
  tests/runtime_patch/test_step5_optimization_scope.py
git commit -m "fix(step5): gate model accelerations under custom ops master"
```

### Task 3: Step5 distributed, MoE, runner, and attention accelerations

**Files:**
- Modify: `vllm_hcu/model_executor/layers/fused_moe/experts/deep_gemm_moe.py`
- Modify: `vllm_hcu/model_executor/layers/fused_moe/moe_runner.py`
- Modify: `vllm_hcu/patch/worker/framework_opt/patch_cuda_communicator.py`
- Modify: `vllm_hcu/v1/hcu_model_runner_v2.py`
- Modify: `vllm_hcu/v1/attention/backends/flash_attn.py`
- Modify: `tests/runtime_patch/test_ep_fused_moe_contract.py`
- Modify: `tests/runtime_patch/test_ipc_all_gather.py`
- Modify: `tests/runtime_patch/test_step5_runtime.py`
- Modify: `tests/runtime_patch/test_step5_custom_ops_master.py`

**Interfaces:**
- Consumes: `optional_custom_op_enabled(feature_enabled: bool = True) -> bool`.
- Produces: master-gated effective values for EP fused MoE/combine, DP IPC
  all-gather, Step5 host sequence lengths, and HCU SWA decode.

- [ ] **Step 1: Add distributed allocation and execution tests**

Assert master `0` plus child `1`:

- never enters `_apply_ep_fused` and never uses `_ep_fused_all_reduce`;
- does not construct EP custom-allreduce or `IpcAllGather` objects;
- leaves communicator teardown safe with `_hcu_ipc_ag is None`;
- uses the process-group all-gather/finalize fallback.

- [ ] **Step 2: Add runner and attention fallback tests**

Assert master `0` plus child `1` leaves `_hsl_enabled` false without allocating
host mirrors, and routes SWA decode through the existing baseline branch.
Include a master `1` control for each feature.

- [ ] **Step 3: Run Task 3 tests and observe RED**

Run:

```bash
pytest -q tests/runtime_patch/test_ep_fused_moe_contract.py \
  tests/runtime_patch/test_ipc_all_gather.py \
  tests/runtime_patch/test_step5_runtime.py \
  tests/runtime_patch/test_step5_custom_ops_master.py
```

Expected: current child-only checks allocate or select optimized providers
under master `0`.

- [ ] **Step 4: Gate distributed and MoE provider boundaries**

Resolve EP fused MoE/combine and DP IPC allocation through
`optional_custom_op_enabled`. Preserve normal MoE backend selection and the
existing collective implementations as the fallback.

- [ ] **Step 5: Gate runner and SWA decode choices**

Apply the same policy to `_USE_HOST_SEQ_LENS` and `_SWA_DECODE_ENABLED` before
host-mirror allocation or attention dispatch. Do not alter metadata ownership,
PCP/DCP restrictions, or baseline attention registration.

- [ ] **Step 6: Run Task 3 tests and observe GREEN**

Run the Step 3 command. Expected: all selected tests pass and allocation spies
remain untouched under master-off.

- [ ] **Step 7: Commit Task 3**

```bash
git add vllm_hcu/model_executor/layers/fused_moe/experts/deep_gemm_moe.py \
  vllm_hcu/model_executor/layers/fused_moe/moe_runner.py \
  vllm_hcu/patch/worker/framework_opt/patch_cuda_communicator.py \
  vllm_hcu/v1/hcu_model_runner_v2.py \
  vllm_hcu/v1/attention/backends/flash_attn.py \
  tests/runtime_patch/test_ep_fused_moe_contract.py \
  tests/runtime_patch/test_ipc_all_gather.py \
  tests/runtime_patch/test_step5_runtime.py \
  tests/runtime_patch/test_step5_custom_ops_master.py
git commit -m "fix(step5): gate runtime accelerations under custom ops master"
```

### Task 4: Full `75aa13701` policy audit and regression closure

**Files:**
- Modify: `vllm_hcu/model_executor/layers/step4_dsa.py`
- Modify: `tests/runtime_patch/test_step5_custom_ops_master.py`
- Modify: `tests/runtime_patch/test_step5_env_contract.py`
- Modify: `tests/model_executor/test_step4_dsa_schedule.py`
- Modify: `tests/model_executor/test_step4_dsa_geometry.py`
- Modify: `docs/step5_runtime_integration.md`
- Modify: `docs/hy4_pcp_ep_profile_memory_alignment.md`

**Interfaces:**
- Consumes: the effective policy patterns established in Tasks 1–3.
- Produces: a reviewed classification of every `75aa13701` runtime path as
  correctness/compatibility, managed optional acceleration, or documented
  required primitive.

- [ ] **Step 1: Audit the complete introducing commit**

List all Python runtime files from:

```bash
git diff-tree --no-commit-id --name-only -r 75aa13701 -- 'vllm_hcu/**/*.py'
```

For each provider selection, layout conversion, custom registration,
communicator allocation, and direct optimized call, record its classification
in the Step5 integration guide. Add any missing optional path to the managed
tests before changing production code. The expected remaining managed DSA
subpaths are per-request/speculative indexer kernels, indexer FP8 dot, BoltOPs
and FlashAttention stage selection, optional FP8 attention math, fused summary
construction, and merged indexer projection. DSA itself, its metadata, and its
Triton baseline remain required model semantics.

- [ ] **Step 2: Add DSA semantic-preservation tests**

With master `0`, assert `is_dsa_disabled()` remains false unless
`VLLM_STEP5_DISABLE_DSA` itself is set, and Step5 config still registers the
DSA attention op. Cold-import `step4_dsa.py` with every optional DSA child
enabled and assert it selects the full-batch indexer, BF16 math, unfused
summaries, separate projections, and Triton baseline stages without importing
BoltOPs/FlashAttention providers. Leave diagnostic/tuning settings such as
slot checks, plan dumps, query chunk size, and graph-region sizing unchanged.

- [ ] **Step 3: Run new audit regressions and observe RED where gaps exist**

Run:

```bash
pytest -q tests/runtime_patch/test_step5_custom_ops_master.py \
  tests/runtime_patch/test_step5_env_contract.py
```

Expected: any newly found unmanaged optional boundary fails its behavioral
test; if none remain after Tasks 2–3, the new DSA and classification tests pass.

- [ ] **Step 4: Close audited gaps without broad refactoring**

Gate the listed DSA subpaths with `optional_custom_op_enabled` before provider
import or execution and preserve their existing baseline branches. The static
audit classifies the other Step5-introduced paths as correctness/compatibility,
debug/tuning settings, or helpers reachable only through one of the already
gated providers. Keep required primitives active and document their exact
semantic reason; do not silently pretend the master disabled them.

- [ ] **Step 5: Update user documentation**

Document the managed feature table, master precedence, DSA exception, HY4
TP1/PCP8 default-on behavior, explicit/master-off errors, and the final audit
classification. Remove the now-unnecessary explicit HY4 child flag from the
normal master-on launch example.

- [ ] **Step 6: Run the complete CPU verification set**

Run:

```bash
pytest -q \
  tests/models/hy_v4/test_attention.py \
  tests/runtime_patch/test_glm52_pcp_config.py \
  tests/runtime_patch/test_step5_custom_ops_master.py \
  tests/runtime_patch/test_step5_env_contract.py \
  tests/runtime_patch/test_step5_optimization_scope.py \
  tests/runtime_patch/test_step5_runtime.py \
  tests/kernels/test_step5_router_topk.py \
  tests/runtime_patch/test_ep_fused_moe_contract.py \
  tests/runtime_patch/test_ipc_all_gather.py \
  tests/model_executor/test_step4_dsa_schedule.py \
  tests/model_executor/test_step4_dsa_geometry.py \
  tests/model_executor/test_step5_projection_fusion.py \
  tests/models/test_deepseek_v4_mtp_compat.py
python tools/check_production_boundary.py
python tools/check_patch_test_coverage.py
git diff --check
```

Expected: all pytest cases and both boundary checks pass; `git diff --check`
prints nothing.

- [ ] **Step 7: Commit Task 4**

Stage only the audited source, tests, and two documentation files, then run:

```bash
git commit -m "test(step5): close custom ops master audit"
```

### Task 5: Eight-card validation and upstream pull request

**Files:**
- Create: `docs/validation/hy4-pcp8-step5-custom-ops-master.md`

**Interfaces:**
- Consumes: verified commits from Tasks 1–4 and the existing supported vLLM
  0.28.1 HCU wheel/environment.
- Produces: reproducible master-on/master-off hardware evidence and an upstream
  pull request from the personal fork.

- [ ] **Step 1: Confirm resources and record the exact environment**

Verify all eight cards are free without killing unrelated processes. Record
branch SHA, vLLM wheel, plugin source path, model paths, and complete environment
in the validation document.

- [ ] **Step 2: Validate HY4 master-on default sharding**

Launch HY4 TP1/PCP8 + EP8 + DeepEP high-throughput + MTP3 without
`VLLM_HCU_ENABLE_LINEAR_GATE_PCP_SHARD`. Require readiness, sharded-gate log or
layout evidence, one smoke request, peak/load memory, and concurrent HumanEval
0–15 results.

- [ ] **Step 3: Validate HY4 fail-fast master-off behavior**

Run the same config with `VLLM_HCU_USE_CUSTOM_OPS=0` and child `1`. Require the
documented config error before model-weight allocation and confirm the server
process group exits cleanly.

- [ ] **Step 4: Validate Step5 master-on and master-off providers**

Run the representative supported Step5 command twice. Master-on must show its
optimized defaults. Master-off must show audited baselines, retain DSA, serve a
prompt longer than 4096 tokens, and pass the existing short correctness gate.

- [ ] **Step 5: Commit validation evidence**

```bash
git add docs/validation/hy4-pcp8-step5-custom-ops-master.md
git commit -m "docs: record custom ops master validation"
```

- [ ] **Step 6: Rebase, re-run post-rebase verification, and request review**

Fetch `origin/v0.28.1-dev`, rebase the topic branch, re-run the Task 4 CPU suite
and any hardware smoke affected by conflicts, then use
`superpowers:requesting-code-review` for a whole-branch review. Fix and reverify
all actionable findings.

- [ ] **Step 7: Push and open the upstream pull request**

Push `codex/fix-hy4-pcp-mtp-draft-config` to
`alexanderbin123/vllm-plugin-das`, then open a pull request against
`HYGON-AI/vllm-plugin-das:v0.28.1-dev`. The PR body must distinguish CPU,
source-level, and hardware evidence and report the exact HumanEval numerator
and denominator without rounding it to a pass.
