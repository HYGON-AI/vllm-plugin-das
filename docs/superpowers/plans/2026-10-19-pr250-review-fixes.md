# PR #250 Review Fixes Implementation Plan

> **For implementer:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task with red-green TDD.

**Goal:** Close the three P1 findings against PR #250 HEAD `7fca90ee009a83a47050d6a348e5dd4b4715511a`: keep fixed-HT Mooncake producers out of CUDA Graph capture, normalize every DeepEP-auto graph mode after all platform rewrites, and reserve HT DeepGEMM workspace for the maximum dispatcher fan-in without double-counting top-k.

**Architecture:** Put the DeepSeek-V4/DSpark/Mooncake role predicate in the existing lightweight runtime-predicate module so compilation policy and MoE dispatch share one source of truth. Bind that fixed-role result beside the immutable HCU feature sidecar, then run one graph-mode normalization policy both from the patched upstream compilation callback and again after HCU's PCP/DCP graph rewrite. Size HT workspaces from received token rows (`local scheduler tokens * dispatchers`); let the child expert's existing `compute_aligned_M(..., num_topk=topk)` account for routes and alignment exactly once.

**Tech Stack:** Python 3, pytest, vLLM 0.28.1 compilation configuration, HCU platform patching, DeepEP, DeepGEMM, ROCm/HIP.

**Spec:** `docs/superpowers/specs/2026-10-18-qwen4exp-deepep-auto-graph-design.md`, amended by the three review findings recorded for PR #250 on 2026-10-19.

---

## Global constraints

- Preserve feature-off behavior and DP=1 behavior byte-for-byte.
- For multi-rank `deepep_auto` (DP>1 or the validated PCP-only topology), the only effective configured graph modes are `FULL_DECODE_ONLY` and `NONE`.
- Fixed Mooncake `kv_producer` (HT) always resolves to `NONE`.
- Fixed Mooncake `kv_consumer` (LL) may use `FULL_DECODE_ONLY`; it must not retain a mixed/prefill `FULL` or `PIECEWISE` mode.
- Dynamic DeepEP auto maps `FULL` and `FULL_AND_PIECEWISE` to `FULL_DECODE_ONLY`, maps `PIECEWISE` to `NONE`, and preserves explicit `NONE`/`FULL_DECODE_ONLY`.
- Reapply that policy after HCU converts full graph modes to `PIECEWISE` for PCP or DCP.
- Do not change default AITER/AGRS behavior, DeepEP fixed-backend behavior, scheduling limits, or the LL capacity calculation in this patch.
- For HT workspace sizing, the `M` passed to the contiguous expert is a received-token-row bound. Do not multiply it by top-k before calling `workspace_shapes`; `compute_aligned_M` already performs that expansion.

### Task 1: Add failing graph-policy and role regression tests

**Files:**
- Modify: `tests/runtime_patch/test_platform_hcu_config.py:880-1060`
- Modify: `tests/runtime_patch/test_platform_hcu_config.py:1980-2120`

**Step 1: Extend the fake graph enum used by compilation-patch tests**

Add distinct fake `FULL` and `PIECEWISE` values, with accurate `has_full_cudagraphs()` and `has_piecewise_cudagraphs()` behavior. Keep identity-based assertions so the tests exercise exact enum normalization rather than string matching.

**Step 2: Write focused failing tests for the callback policy**

Add cases proving:

- dynamic DP auto: `FULL -> FULL_DECODE_ONLY`;
- dynamic DP auto: `FULL_AND_PIECEWISE -> FULL_DECODE_ONLY`;
- dynamic DP auto: `PIECEWISE -> NONE`;
- fixed Mooncake DeepSeek-V4 DSpark producer: every full-capable configured mode resolves to `NONE`;
- fixed Mooncake DeepSeek-V4 DSpark consumer: a full-capable configured mode resolves to `FULL_DECODE_ONLY`;
- explicit `NONE`, DP=1, feature-off, and unrelated all-to-all backends remain unchanged.

Construct producer/consumer configs with the actual predicate inputs (`model_config.architectures`, `speculative_config.method/use_dspark`, and `kv_transfer_config.kv_connector/kv_role`) instead of injecting a private test-only role flag.

**Step 3: Write a failing post-platform-normalization test**

Exercise `HCUPlatform.check_and_update_config` with the validated PCP-only auto topology and with DP>1 auto plus DCP>1. Assert that HCU's intermediate `PIECEWISE` downgrade is normalized again to `NONE`. Add a non-PCP/DCP control that remains `FULL_DECODE_ONLY`.

**Step 4: Run the focused tests and confirm the expected failures**

Run:

```bash
pytest -q tests/runtime_patch/test_platform_hcu_config.py -k 'deepep_auto and (compilation or graph or pcp or dcp)'
```

Expected before implementation: producer, explicit `FULL`, explicit `PIECEWISE`, and post-PCP/DCP assertions fail for the documented reasons.

### Task 2: Centralize fixed Mooncake role selection and graph normalization

**Files:**
- Modify: `vllm_hcu/deepseek_v4_runtime.py:1-40`
- Modify: `vllm_hcu/model_executor/layers/fused_moe/prepare_finalize/deepep_auto.py:15-45`
- Modify: `vllm_hcu/patch/platform/core_fix/patch_compilation_config.py:20-230`
- Modify: `vllm_hcu/platforms/hcu.py:610-675`
- Test: `tests/runtime_patch/test_platform_hcu_config.py`
- Test: `tests/runtime_patch/test_moe_deepep.py`

**Step 1: Move the role predicate to the lightweight shared module**

Move `dspark_mooncake_pd_use_low_latency(vllm_config) -> bool | None` into `deepseek_v4_runtime.py` and export it. Import that function from both the MoE prepare/finalize adapter and compilation patch. Keep `False=HT producer`, `True=LL consumer`, and `None=dynamic/not-applicable` unchanged.

**Step 2: Bind role policy with the HCU feature sidecar**

Extend the weak-reference binding entry in `patch_compilation_config.py` to retain the fixed role result computed from the full `VllmConfig`. Do not add fields to upstream `CompilationConfig` or serialize a second copy. Preserve stale-id cleanup and spawned-worker rebind behavior.

**Step 3: Implement one table-driven normalization function**

Create a small pure normalization helper whose inputs are current graph mode, DP size, `deepep_auto`, fixed-role result, and the five required enum members (`NONE`, `FULL`, `PIECEWISE`, `FULL_AND_PIECEWISE`, `FULL_DECODE_ONLY`). Return the effective mode plus an optional reason for one-shot logging.

Policy table for multi-rank auto:

| Fixed role | Input mode | Effective mode |
|---|---|---|
| HT producer (`False`) | any graph-enabled mode | `NONE` |
| LL consumer (`True`) | `FULL`, `FULL_AND_PIECEWISE` | `FULL_DECODE_ONLY` |
| LL consumer (`True`) | `PIECEWISE` | `NONE` |
| dynamic (`None`) | `FULL`, `FULL_AND_PIECEWISE` | `FULL_DECODE_ONLY` |
| dynamic (`None`) | `PIECEWISE` | `NONE` |
| any | `NONE`, safe `FULL_DECODE_ONLY` | unchanged, except producer forces `FULL_DECODE_ONLY -> NONE` |

Reject enum compatibility drift during patch installation if `FULL` or `PIECEWISE` is missing, just as the patch already rejects missing existing members.

**Step 4: Apply normalization at both configuration boundaries**

Call the helper before upstream `set_splitting_ops_for_v1`, then expose a narrow function that reapplies the same bound policy to the final `VllmConfig`. Invoke it in `HCUPlatform.check_and_update_config` immediately after the PCP/DCP full-to-piecewise block. This second call is mandatory because platform normalization currently runs after the compilation callback.

Do not duplicate role detection or mode tables in `hcu.py`.

**Step 5: Make logs describe the effective safety decision**

Use one-shot logs that distinguish fixed HT forced eager, mixed/prefill graph forced eager, and decode-only FULL retention. Do not claim that every FULL graph uses LL unless the normalized mode is `FULL_DECODE_ONLY` and the role is not fixed HT.

**Step 6: Run the graph and role tests**

Run:

```bash
pytest -q tests/runtime_patch/test_platform_hcu_config.py -k 'compilation_adapter or deepep_auto'
pytest -q tests/runtime_patch/test_moe_deepep.py -k 'mooncake and role'
```

Expected: all new red tests turn green; existing producer/consumer MoE selection remains unchanged.

### Task 3: Add a failing real-shape HT workspace capacity test

**Files:**
- Modify: `tests/runtime_patch/test_moe_deepep.py:3990-4055`

**Step 1: Replace the implementation-mirroring test**

Replace `test_deepep_auto_ht_workspace_reserves_all_local_routed_tokens`, which only records the delegated `M`, with a behavior-oriented test using the real contiguous expert `workspace_shapes` math on the ROCm path.

Use the review's reduced counterexample:

- scheduler tokens `T=64`;
- `num_dispatchers=16`;
- `topk=10`;
- `global_num_experts=512`, `local_num_experts=32`;
- `N=1536`, `K=2048`;
- HCU alignment 256.

Assert the HT temporary workspace has 18,432 aligned rows, not 14,592. Assert the output keeps the real invocation `M`, and assert switching the same auto wrapper to LL does not inherit the HT reserve.

**Step 2: Add a DP4/top-k regression control**

Add a case proving top-k is applied once: the reserve input is bounded by dispatchers, while final aligned rows still include `topk=10` through the real `compute_aligned_M` contract. This protects against “fixing” DP16 by multiplying both dispatcher count and top-k at the wrapper level.

**Step 3: Run the test and confirm the old sizing fails**

Run:

```bash
pytest -q tests/runtime_patch/test_moe_deepep.py -k 'auto_ht_workspace'
```

Expected before implementation: the DP16 case reports the old 14,592-row reserve instead of 18,432.

### Task 4: Fix HT workspace reservation at the correct semantic layer

**Files:**
- Modify: `vllm_hcu/model_executor/layers/fused_moe/experts/dpsk_v4_deep_gemm_moe.py:950-995`
- Test: `tests/runtime_patch/test_moe_deepep.py`

**Step 1: Compute the receive bound from dispatcher fan-in**

For the selected HT delegate, calculate:

```python
max_received_tokens = (
    int(self.moe_config.max_num_tokens) * int(self.num_dispatchers)
)
reserve_m = max(M, max_received_tokens)
```

Call the child contiguous expert's `workspace_shapes(reserve_m, ..., topk, ...)`. Keep the returned output shape from the real current invocation so workspace13/output reuse remains valid.

**Step 2: Remove the pre-expanded top-k bound and correct comments**

Delete `max_num_tokens * experts_per_token` from the wrapper. Document that DeepEP HT may receive token rows from every dispatcher and that the child workspace calculation expands those rows by top-k and per-expert alignment.

**Step 3: Run focused workspace tests**

Run:

```bash
pytest -q tests/runtime_patch/test_moe_deepep.py -k 'workspace or auto_experts'
```

Expected: DP16 receives 18,432 aligned rows; LL sizing and output shape tests remain green.

### Task 5: Update design documentation and run software verification

**Files:**
- Modify: `docs/superpowers/specs/2026-10-18-qwen4exp-deepep-auto-graph-design.md`
- Modify: `docs/superpowers/plans/2026-10-19-pr250-review-fixes.md` only if implementation discoveries change commands or file locations

**Step 1: Correct the effective graph-policy section**

Document that fixed Mooncake P/HT is always eager `NONE`, fixed D/LL can retain decode-only FULL, explicit `FULL` is narrowed, explicit `PIECEWISE` is disabled, and PCP/DCP normalization is followed by the same final guard. Remove language that implies `FULL_DECODE_ONLY` alone guarantees LL for every role.

**Step 2: Correct the workspace section**

Document dispatcher fan-in as the HT receive bound and top-k expansion as the contiguous-expert responsibility. Record that the reduced DP16 counterexample is a capacity-contract test, not a multi-HCU performance result.

**Step 3: Run focused suites**

Run:

```bash
pytest -q tests/runtime_patch/test_platform_hcu_config.py
pytest -q tests/runtime_patch/test_moe_deepep.py
```

If the complete MoE file is too slow for a local iteration, run its focused subsets first, but the full two files are required before claiming completion.

**Step 4: Run repository hygiene checks**

Run:

```bash
git diff --check
python -m compileall -q \
  vllm_hcu/deepseek_v4_runtime.py \
  vllm_hcu/model_executor/layers/fused_moe/prepare_finalize/deepep_auto.py \
  vllm_hcu/patch/platform/core_fix/patch_compilation_config.py \
  vllm_hcu/platforms/hcu.py \
  vllm_hcu/model_executor/layers/fused_moe/experts/dpsk_v4_deep_gemm_moe.py
```

**Step 5: Review the complete diff against all three P1 findings**

Confirm explicitly:

- no fixed producer capture descriptor can execute HT under FULL;
- no configured or platform-normalized mixed/prefill mode can select LL with more rows than the LL buffer;
- HT workspace capacity scales with dispatchers and applies top-k exactly once;
- feature-off, DP=1, AITER/AGRS, and fixed backend paths are unchanged.

### Task 6: Hardware gate before updating PR #250

**Files:**
- No production-code changes expected
- Add evidence to PR #250 only after results are collected

**Step 1: Run role-specific startup/capture validation**

On the target HCU environment, start DeepSeek-V4 DSpark Mooncake producer and consumer separately with DP>1:

- producer log/config must show effective `NONE` and must not enter FULL HT capture;
- consumer must capture only decode FULL with LL;
- repeat with PCP or DCP enabled where supported and confirm the final mode is `NONE`, not `PIECEWISE`.

**Step 2: Run the DP16 capacity gate if hardware is available**

Exercise multiple-sender concentration after workspace lock. If DP16 hardware is unavailable, disclose that the fix is covered by exact capacity math/unit tests only; do not describe it as a distributed hardware reproduction.

**Step 3: Re-run the established DP4/EP4 functional gate**

Use the PR's recorded server/client commands and sufficient free memory. Verify prefill HT, decode LL, repeated uneven-rank completion, capture/replay, and nonzero-down-weight output accuracy. Performance is a regression gate, not a claim that this PR beats AITER/AGRS.

**Step 4: Request code review, then commit and update the existing branch**

Use `superpowers:requesting-code-review`, address findings with `superpowers:receiving-code-review`, and use `superpowers:verification-before-completion` before stating success. Keep the changes on PR #250's existing branch unless the reviewer requests a split.

Suggested commit message:

```text
fix(moe): close deepep auto graph and workspace bounds
```

Post the exact verification commands, software results, hardware scope/limitations, and effective role-specific graph policy to PR #250.
