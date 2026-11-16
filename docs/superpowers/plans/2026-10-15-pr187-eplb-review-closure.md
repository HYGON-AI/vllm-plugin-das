# PR187 EPLB Review Closure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Fix all six confirmed P2 findings and revalidate HY4 offline EPLB from calibration through static serving.

**Architecture:** Capture record-only policy results before the online ROCm migration gate, make static primary selection explicit at BaseRouter, split lifecycle feature activation from static-loader activation, and remove the DeepEP-LL DP-sync bypass. Keep upstream vLLM responsible for policy calculation, DP synchronization, and ordinary online routing.

**Tech Stack:** Python 3.10, pytest, PyTorch, vLLM 0.28.1 HCU runtime, DP8/TP1/EP8 hardware validation.

**Spec:** `docs/superpowers/specs/2026-10-15-pr187-eplb-review-closure-design.md`

## Global Constraints

- Preserve ordinary online EPLB behavior when no offline path or standalone disable flag is configured.
- Keep the supported static topology at DP8/TP1/EP8 with exactly eight redundant experts.
- Record mode must never transfer weights or commit a candidate into the live model.
- Static mode must retain the real replica metadata while routing through the selected rank-local primary.
- Fixed DeepEP-LL must use upstream DP synchronization and DBO selection.
- Use `/models/Hy4-preview-Channel-FP8-w8a8-v2` for final hardware validation and do not enable offline EPLB for the fake-prefill-only gate.

## Review Focus

- Multiple model states in one rearrange call must associate captured candidates with the correct target or MTP owner; Task 1 tests both independently.
- A globally all-zero window must never become a proposal even if policy returns a valid map; Task 1 pins this case.
- Profile rearrangement must retain upstream profiling behavior and never publish; Task 1 keeps the existing profile regression.
- Static primary routing must work with the HCU Torch fallback both enabled and disabled; Task 2 tests both installed mapping paths.
- Standalone `disable_rearrange` must not arm model-loader/static-communicator callbacks; Task 3 asserts the exact feature inventory.

---

### Task 1: Decouple record-only proposals from online migration

**Files:**
- Modify: `vllm_hcu/patch/worker/framework_opt/patch_offline_eplb.py`
- Modify: `tests/runtime_patch/test_offline_eplb.py`

**Interfaces:**
- Consumes: upstream `EplbState.rearrange`, `policy.rebalance_experts`, and the all-reduced logical load passed to the policy.
- Produces: record-only publication driven by captured `(candidate, has_global_load)` results without live transfer or commit.

- [x] **Step 1: Add failing calibration and reset regressions**

Add tests proving: ROCm improvement below five percent still publishes; EP0 `is_dummy=True` publishes when the policy sees globally nonzero load; all-zero global load stays `initial`; target and MTP candidates remain associated with their owners; malformed existing JSON/schema raises and remains byte-for-byte unchanged at session start.

- [x] **Step 2: Run Task 1 tests and witness RED**

Run: `pytest -q tests/runtime_patch/test_offline_eplb.py -k 'rocm or dummy or zero_global or malformed_session or multiple_models'`

Expected: failures show publication still depends on `_commit_eplb_maps`, local dummy still vetoes EP0, and session reset overwrites malformed content.

- [x] **Step 3: Implement policy-result capture and fail-closed reset**

Introduce a private per-call proxy around `self.policy.rebalance_experts` in record-only rearrange. Capture one result per model plus `bool(global_load.any())`, restore `self.policy` in `finally`, let the existing transfer wrapper suppress migration, make the commit wrapper a no-op in record-only mode, then publish eligible captured results on EP0. Parse and `_validate_record_payload` before `_start_offline_record_session` replaces an existing file.

- [x] **Step 4: Run Task 1 tests and full offline lifecycle suite**

Run: `pytest -q tests/runtime_patch/test_offline_eplb.py`

Expected: all tests pass.

- [x] **Step 5: Commit Task 1**

Commit: `git commit -m "fix(eplb): publish record-only policy results"`

### Task 2: Make static locality selection effective at runtime

**Files:**
- Modify: `vllm_hcu/patch/worker/framework_opt/patch_offline_eplb.py`
- Modify: `vllm_hcu/patch/worker/op_opt/moe/patch_base_router.py`
- Modify: `vllm_hcu/model_executor/layers/fused_moe/router_runtime.py` only if a shared helper is needed
- Modify: `tests/runtime_patch/test_offline_eplb.py`
- Modify: `tests/runtime_patch/test_moe_deepep.py`

**Interfaces:**
- Consumes: the rank-local ordered map installed by static `EplbState.add_model`.
- Produces: `_vllm_hcu_static_primary_replica_count` on each `EplbLayerState` and a BaseRouter mapping wrapper that uses it only for static states.

- [x] **Step 1: Add failing physical-ID routing tests**

For a logical expert replicated on physical IDs 0/GPU0 and 132/GPU4, assert 64 rank-0 static tokens all resolve to physical 0 under `nearest` and `locality_fair`. Cover the HCU Torch mapping path and a delegated upstream-style mapping function. Assert an unmarked online state retains hashed distribution across both replicas and the real replica count remains two.

- [x] **Step 2: Run Task 2 tests and witness RED**

Run: `pytest -q tests/runtime_patch/test_offline_eplb.py tests/runtime_patch/test_moe_deepep.py -k 'static_primary or locality_physical_ids or primary_replica'`

Expected: static repeated tokens still split across replica columns.

- [x] **Step 3: Bind and consume explicit static primary state**

After static map ordering, attach a device-local tensor of ones to each installed layer state without changing `logical_replica_count`. Extend the guarded BaseRouter adapter to wrap `_apply_eplb_mapping(self, topk_ids)` and pass that primary count only when the static marker is present; otherwise delegate byte-for-byte to upstream.

- [x] **Step 4: Run routing and integration suites**

Run: `pytest -q tests/runtime_patch/test_eplb_locality_fair_dispatch.py tests/runtime_patch/test_offline_eplb.py tests/runtime_patch/test_moe_deepep.py`

Expected: all tests pass.

- [x] **Step 5: Commit Task 2**

Commit: `git commit -m "fix(eplb): route static tokens through local primaries"`

### Task 3: Restore lifecycle activation and upstream DBO synchronization

**Files:**
- Modify: `vllm_hcu/patch/worker/__init__.py`
- Delete: `vllm_hcu/patch/worker/framework_opt/patch_gpu_dp_utils.py`
- Modify: `tests/patch/test_worker_dispatcher.py`
- Modify: `tests/runtime_patch/test_worker_framework_opt.py`

**Interfaces:**
- Produces: an `eplb_control` feature state for the state-lifecycle adapter and no plugin wrapper around `vllm.v1.worker.gpu.dp_utils.sync_cudagraph_and_dp_padding`.

- [x] **Step 1: Add failing feature and DBO regressions**

Assert `eplb_control` is enabled by either offline path or by standalone `eplb_disable_rearrange=True`, while `offline_eplb` remains false for standalone disable. Assert only the state-lifecycle callback is armed in that case. Replace the fixed-DeepEP-LL skip test with a real upstream-helper regression whose DBO thresholds select two microbatches and whose synchronized state is non-null.

- [x] **Step 2: Run Task 3 tests and witness RED**

Run: `pytest -q tests/patch/test_worker_dispatcher.py tests/runtime_patch/test_worker_framework_opt.py -k 'disable_rearrange or gpu_dp or ubatch or eplb_control'`

Expected: standalone disable does not arm the lifecycle wrapper and fixed DeepEP-LL returns the unsynchronized single-batch descriptor.

- [x] **Step 3: Split the feature gate and remove the bypass adapter**

Add `eplb_control` to `FeatureKey`, gate only `patch_offline_eplb` on it, derive it from either path or the disable flag, and keep loader/mapping/communicator adapters on `offline_eplb`. Remove `patch_gpu_dp_utils` from callback inventory, terminal IDs, import smoke tests, and the filesystem.

- [x] **Step 4: Run dispatcher/framework suites**

Run: `pytest -q tests/patch/test_worker_dispatcher.py tests/runtime_patch/test_worker_framework_opt.py tests/patch/test_plugin_lifecycle.py`

Expected: all tests pass and the installed upstream DP helper remains unwrapped.

- [x] **Step 5: Commit Task 3**

Commit: `git commit -m "fix(runtime): preserve EPLB controls and DBO sync"`

### Task 4: Software and eight-device acceptance

**Files:**
- Modify: `docs/validation/hy4-v0281-offline-eplb.md` if present, otherwise create it
- Modify: PR #187 discussion after evidence is retained

**Interfaces:**
- Consumes: Tasks 1–3 and the existing HY4 calibration/static profiles.
- Produces: test logs, proposal-complete map, static runtime evidence, fake-prefill result, HumanEval result, and reviewed commits on the existing PR branch.

- [x] **Step 1: Run changed and complete software gates**

Run:

```bash
pytest -q \
  tests/patch/test_worker_dispatcher.py \
  tests/patch/test_plugin_lifecycle.py \
  tests/models/hy_v4 \
  tests/runtime_patch/test_eplb_locality_fair_dispatch.py \
  tests/runtime_patch/test_model_loader_static_eplb.py \
  tests/runtime_patch/test_offline_eplb.py \
  tests/runtime_patch/test_moe_deepep.py \
  tests/runtime_patch/test_worker_framework_opt.py
git diff --check
```

Expected: all tests pass and diff check is clean.

- [x] **Step 2: Run eight-device record-only calibration**

Use profile `hy4_preview_channel_fp8_dp8_ep8_mtp3_kvfp8_eplb_calibration`, changing its model path to `/models/Hy4-preview-Channel-FP8-w8a8-v2`. Drive varied concurrent requests through at least one complete EPLB window. Require target and MTP `record_kind=proposal`, exact 256+8 dimensions, zero transfer/live-commit counters, and retain the map SHA-256 and service log.

- [x] **Step 3: Run fresh static service and routing probes**

Use the matching static profile and the exact calibration artifact. Require eight matching fingerprints, actual Model Runner V2, default FULL_AND_PIECEWISE target/draft captures, DeepEP-LL plus DeepGEMM, E4M3 KV, 33 local experts per rank, primary physical IDs consistent with the selected policy, and zero transfer/rearrangement counters.

- [x] **Step 4: Run low-latency fake prefill without offline EPLB**

Start the same model in DP8/TP1/EP8 fixed DeepEP-LL mode without record/load paths. Exercise one long chunked prefill, concurrent uneven prompts that force idle/fake ranks, and a staggered long decode plus inserted prefills. Require all HTTP requests to finish without hang or engine error and confirm the service tears down cleanly.

- [x] **Step 5: Run HumanEval-32 and inspect runtime evidence**

Run deterministic HumanEval/0-31 against the static service with proxy variables cleared, temperature zero, and `reasoning_effort=no_think`. Require 32/32 normalized pass@1, 32 unique predictions and reviews, HTTP success, nonzero drafted/accepted tokens, prefix-hit evidence, no runtime ERROR/Traceback, and zero static transfer/rearrangement events.

- [x] **Step 6: Final review, commit documentation, and push**

Review `git diff 38b179181080e00b184a8bd0e3d2508fbce350bf...HEAD`, re-run affected focused tests after any correction, update validation evidence, commit it, push the existing PR branch, reply to each confirmed review thread with test/runtime evidence, resolve addressed threads, and verify PR draft/mergeability/check state without changing merge state.
