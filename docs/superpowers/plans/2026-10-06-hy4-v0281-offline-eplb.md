# HY4 v0.28.1 Offline EPLB Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add calibrated and static offline EPLB for HY4 target plus MTP on DP8/TP1/EP8 with eight redundant experts, DeepEP low latency, DeepGEMM, E4M3 sparse KV, prefix caching, and default FULL_AND_PIECEWISE Graphs.

**Architecture:** Preserve current vLLM 0.28.1 EPLB, Model Runner V2, RoutedExperts, loader, DeepEP, and DeepGEMM owners. Add narrow HCU sidecar adapters for immutable placement files, record-only proposal publication, pre-load expert replication, static map commit, and locality-aware dispatch; HY4 opts into those adapters only when one offline path is configured.

**Tech Stack:** Python 3.10, PyTorch/HCU, OpenDAS vLLM `77acaf6`, pytest, JSON plus `fcntl`/atomic filesystem operations, DeepEP low latency, DeepGEMM, EvalScope HumanEval.

**Spec:** `docs/superpowers/specs/2026-10-06-hy4-v0281-offline-eplb-design.md`

## Global Constraints

- Work only on `codex/validate-v0281-gfx938-models` and update existing PR #183; do not open a second PR.
- Compare behavior with v0.25.1 commits `c089932`, `3f5cdf5`, and `5e68a65`; do not cherry-pick them wholesale.
- Preserve Model Runner V2, MTP3, prefix caching, public `fp8_e4m3` resolving to `fp8_ds_mla`, explicit `deepep_low_latency`, DeepGEMM, and the default `FULL_AND_PIECEWISE` policy.
- Accepted topology is DP8/TP1/EP8, 256 logical plus 8 redundant experts, 264 physical experts, and 33 physical experts per EP rank.
- Record and static load are synchronous (`use_async=false`); calibration uses `window_size=16` and `step_interval=16`.
- HY4 ordinary dynamic/async EPLB, elastic EP, PP, PCP, DCP, multi-node, W4A8/SlimQuant static loading, `deepep_auto`, and high-throughput Graph claims remain out of scope.
- Static startup must fail closed before serving on malformed, incomplete, changed, or cross-rank-inconsistent maps; it must never fall back to identity placement.
- HumanEval runs only through the repository's secure isolated evaluator; proxy variables are cleared for localhost; tokens and credentials never appear in argv, logs, commits, or PR text.
- Every code task follows RED → GREEN, reviews its complete diff before commit, reviews the exact committed diff afterward, and leaves the worktree clean.

## Review Focus

- A placement file replaced with different bytes but identical size/mtime must produce a different fingerprint and fail any post-bind recheck; Task 2 adds this test.
- A calibration file where only target or only MTP reached a proposal must be rejected as incomplete; Task 5 adds this test.
- Boolean JSON values must not pass as integer expert IDs or counts; Task 2 adds this test.
- A pre-existing malformed record file must remain untouched when publication fails; Task 5 adds this test.
- `initialize_model` aliases imported before patch application must either be patched atomically or fail with an alias-identity error; Task 4 adds this test.

---

### Task 1: Offline EPLB Configuration Contract

**Files:**
- Modify: `vllm_hcu/patch/config.py:18-140`
- Modify: `vllm_hcu/patch/platform/core_fix/patch_engine_args.py:35-160`
- Test: `tests/patch/test_config.py`
- Test: `tests/runtime_patch/test_platform_hcu_config.py`

**Interfaces:**
- Consumes: official `EngineArgs.eplb_config` mapping and existing `additional_config['hcu']` serialization.
- Produces: `HcuFeatureConfig.expert_map_path`, `expert_map_record_path`, `eplb_disable_rearrange`, `eplb_static_dispatch_policy`; `bind_hcu_eplb_config(vllm_config: object) -> None`; `_HCU_EPLB_FIELDS` mapping public nested keys to sidecar fields.

- [ ] **Step 1: Write failing sidecar validation tests**

Add tests asserting defaults, round-trip serialization, non-empty path strings, mutual exclusion, strict `bool` for `eplb_disable_rearrange`, and the exact policies `nearest|locality_fair`; assert unknown values fail before worker use.

- [ ] **Step 2: Run the focused configuration tests and confirm RED**

Run: `pytest -q tests/patch/test_config.py tests/runtime_patch/test_platform_hcu_config.py -k 'eplb or expert_map or static_dispatch'`

Expected: FAIL because the four fields and binder do not exist.

- [ ] **Step 3: Extend the immutable HCU sidecar**

Implement the four fields in `HcuFeatureConfig`, include them in `_FEATURE_FIELDS`, validate them in `__post_init__`, and implement `bind_hcu_eplb_config(vllm_config: object) -> None` to restore `_vllm_hcu_*` attributes on object or mapping parallel configs after deserialization.

- [ ] **Step 4: Write failing EngineArgs extraction tests**

Assert nested `eplb_config` strips `expert_map_path`, `expert_map_record_path`, `disable_rearrange`, and `static_dispatch_policy` before the official constructor sees them, preserves official EPLB fields, rejects conflicts with an explicit HCU sidecar, and survives `create_engine_config()` serialization.

- [ ] **Step 5: Run EngineArgs tests and confirm RED**

Run: `pytest -q tests/runtime_patch/test_platform_hcu_config.py -k 'engine_args and eplb'`

Expected: FAIL because `_normalise_constructor_kwargs` forwards HCU-only keys to official `EPLBConfig`.

- [ ] **Step 6: Implement nested EPLB extraction**

Add `_HCU_EPLB_FIELDS` and merge extracted values through `HcuFeatureConfig.with_updates`; preserve the official mapping and current positional/additional-config conflict behavior.

- [ ] **Step 7: Run all affected configuration tests**

Run: `pytest -q tests/patch/test_config.py tests/runtime_patch/test_platform_hcu_config.py`

Expected: PASS.

- [ ] **Step 8: Review and commit Task 1**

Review: `git diff --check && git diff -- vllm_hcu/patch/config.py vllm_hcu/patch/platform/core_fix/patch_engine_args.py tests/patch/test_config.py tests/runtime_patch/test_platform_hcu_config.py`

Commit: `git commit -m "feat(eplb): add offline map configuration"`

### Task 2: Immutable Static Placement Plans

**Files:**
- Create: `vllm_hcu/model_executor/layers/fused_moe/static_eplb.py`
- Create: `tests/model_executor/layers/fused_moe/test_static_eplb.py`

**Interfaces:**
- Consumes: version-2 JSON `model_maps`, runtime model class, `parallel_config`, and current EP CPU group.
- Produces: frozen `StaticEplbPlan`; `load_static_eplb_plan(path, *, model_key, expected_shape, num_logical_experts, num_redundant_experts, require_proposal=True) -> StaticEplbPlan`; `resolve_offline_eplb_model_key(model, parallel_config) -> str`; `verify_static_plan_across_ep_ranks(plan) -> None`.

- [ ] **Step 1: Write parser, immutability, and fingerprint tests**

Cover canonical absolute paths, SHA-256 over exact file bytes, immutable tuple rows, returned tensor-copy isolation, target and MTP key selection, same-size/same-mtime replacement, and cross-rank all-gather agreement.

- [ ] **Step 2: Write malformed-input tests**

Reject absent `version: 2`, missing key, initial-only entries when `require_proposal=True`, empty rows, wrong dimensions, bool/float/negative/out-of-range IDs, bool counts, metadata mismatches, missing logical experts, incorrect redundancy, PP, and unequal EP fingerprints.

- [ ] **Step 3: Run the new tests and confirm RED**

Run: `pytest -q tests/model_executor/layers/fused_moe/test_static_eplb.py`

Expected: FAIL at import because `static_eplb.py` does not exist.

- [ ] **Step 4: Implement the plan object and loader**

Port the v0.25.1 behavior semantically to current types. Read bytes on every call; cache parsing only by the bytes included in the fingerprint. Require PP size one for this scope, exact layer/physical dimensions, all 256 logical IDs per row, and `logical + redundant == physical`.

- [ ] **Step 5: Implement model-key and rank verification helpers**

Resolve keys to current class names (`HYV4ForCausalLM`, `HYV4MTP`) and compare `(model_key, sha256, shape, counts)` with `all_gather_object` on the EP CPU group.

- [ ] **Step 6: Run Task 2 tests**

Run: `pytest -q tests/model_executor/layers/fused_moe/test_static_eplb.py`

Expected: PASS.

- [ ] **Step 7: Review and commit Task 2**

Review: `git diff --check && git diff -- vllm_hcu/model_executor/layers/fused_moe/static_eplb.py tests/model_executor/layers/fused_moe/test_static_eplb.py`

Commit: `git commit -m "feat(eplb): validate immutable static plans"`

### Task 3: Locality-Aware Replica Dispatch

**Files:**
- Create: `vllm_hcu/model_executor/layers/fused_moe/eplb_dispatch.py`
- Create: `tests/runtime_patch/test_eplb_locality_fair_dispatch.py`

**Interfaces:**
- Consumes: official `logical_to_physical_map`, source `ep_rank`, `ep_size`, node count, and physical expert count.
- Produces: `build_nearest_replica_order(...) -> torch.Tensor`; `build_locality_fair_replica_order(..., seed=42, layer_offset=0) -> torch.Tensor`.

- [ ] **Step 1: Write failing topology and validation tests**

Assert same-GPU then same-node then remote ordering for `nearest`; deterministic balanced primary selection for `locality_fair`; unchanged replica sets; rank-local differences; and rejection of wrong rank, dtype, dimensions, duplicate replicas, missing replicas, invalid IDs, non-divisible EP, and invalid node counts.

- [ ] **Step 2: Run the dispatch tests and confirm RED**

Run: `pytest -q tests/runtime_patch/test_eplb_locality_fair_dispatch.py`

Expected: FAIL at import.

- [ ] **Step 3: Implement the two pure CPU transforms**

Use the v0.25.1 final algorithm, deterministic `random.Random(42 + layer/expert offsets)`, and return the result on the source tensor's device without mutating the input.

- [ ] **Step 4: Run Task 3 tests**

Run: `pytest -q tests/runtime_patch/test_eplb_locality_fair_dispatch.py`

Expected: PASS.

- [ ] **Step 5: Review and commit Task 3**

Review: `git diff --check && git diff -- vllm_hcu/model_executor/layers/fused_moe/eplb_dispatch.py tests/runtime_patch/test_eplb_locality_fair_dispatch.py`

Commit: `git commit -m "feat(eplb): add locality-aware replica routing"`

### Task 4: Pre-Checkpoint Static Direct Loading

**Files:**
- Modify: `vllm_hcu/model_executor/layers/fused_moe/static_eplb.py`
- Create: `vllm_hcu/patch/worker/framework_opt/patch_model_loader_static_eplb.py`
- Create: `vllm_hcu/patch/worker/framework_opt/patch_static_expert_mapping.py`
- Create: `tests/models/static_eplb_test_utils.py`
- Create: `tests/runtime_patch/test_model_loader_static_eplb.py`
- Extend: `tests/model_executor/layers/fused_moe/test_static_eplb.py`

**Interfaces:**
- Consumes: `StaticEplbPlan`, current `RoutedExperts`, instance-owned parameter `weight_loader`, model `get_expert_mapping`, and official `initialize_model(vllm_config, *, prefix='', model_class=None, model_config=None)`.
- Produces: `bind_static_eplb_plan(vllm_config, model) -> StaticEplbPlan | None`; `load_static_logical_expert(...)`; `validate_static_loader(vllm_config, load_config) -> None`; guarded `initialize_model` wrapper.

- [ ] **Step 1: Build generic CPU MoE fixtures and failing direct-load tests**

Assert one logical checkpoint tensor populates every mapped rank-local physical replica, skipped nonlocal slots are harmless, shared experts retain their loader, router bias/global scales are untouched, fused expert tensors are rejected, and `enable_ep_weight_filter=True` fails before mutation.

- [ ] **Step 2: Add binding safety tests**

Cover current `RoutedExperts` ownership, `supports_eplb`, counts, layer ordering, model/inner mapping filters, identical rebind, changed-file rebind failure, unsupported loader owner, unsupported quantization owner, and missing local MoE layers.

- [ ] **Step 3: Run static binding tests and confirm RED**

Run: `pytest -q tests/model_executor/layers/fused_moe/test_static_eplb.py`

Expected: FAIL because binder and direct loader are absent.

- [ ] **Step 4: Implement instance-scoped direct loading**

Bind only parameters whose storage belongs to `RoutedExperts.get_expert_weights()`. Wrap their bound loader functions, preserve shared/global state, install logical checkpoint mappings, and publish model/layer plan attributes only after every target validates.

- [ ] **Step 5: Write loader-seam tests**

Assert binding occurs after model construction but before `load_weights`, nested initialization binds once, allowed formats are `auto|pt|safetensors|npcache|mistral`, unsupported formats fail, and pre-imported aliases are patched atomically or rejected by identity guard.

- [ ] **Step 6: Run loader-seam tests and confirm RED**

Run: `pytest -q tests/runtime_patch/test_model_loader_static_eplb.py`

Expected: FAIL because the two framework adapters are absent.

- [ ] **Step 7: Implement guarded loader and mapping adapters**

Use exact current signatures, `_DEPTH` to avoid nested double binding, and explicit alias inventory for current base/tensorizer loaders. Patch current `RoutedExperts.make_expert_params_mapping` only where legacy inner loaders bypass instance `get_expert_mapping`.

- [ ] **Step 8: Run Task 4 tests**

Run: `pytest -q tests/model_executor/layers/fused_moe/test_static_eplb.py tests/runtime_patch/test_model_loader_static_eplb.py`

Expected: PASS.

- [ ] **Step 9: Review and commit Task 4**

Review all six Task 4 files and confirm no global quantization class or tensor layout is replaced.

Commit: `git commit -m "feat(eplb): load static replicas before checkpoints"`

### Task 5: Record-Only and Static EPLB Lifecycle

**Files:**
- Create: `vllm_hcu/patch/worker/framework_opt/patch_offline_eplb.py`
- Create: `vllm_hcu/patch/worker/framework_opt/patch_eplb_communicator.py`
- Modify: `vllm_hcu/patch/worker/__init__.py:47-420`
- Test: `tests/patch/test_worker_dispatcher.py`
- Create: `tests/runtime_patch/test_offline_eplb.py`
- Extend: `tests/runtime_patch/test_worker_framework_opt.py`

**Interfaces:**
- Consumes: Tasks 1–4, current `EplbState.add_model/step/rearrange`, `_commit_eplb_maps`, `rearrange_expert_weights_inplace`, current communicator classes, and patch dispatcher inventory.
- Produces: `record_offline_expert_map(..., record_kind: Literal['initial','proposal']) -> None`; guarded lifecycle wrappers; static Gloo profile policy; runtime counters/markers for proposal, transfer, and rearrangement events.

- [ ] **Step 1: Write failing atomic recorder tests**

Assert rank-zero-only version-2 publication, target/MTP entry merge, `record_kind` transition from `initial` to `proposal`, lock serialization, temporary-file cleanup, file and parent-directory `fsync`, exact metadata, and preservation of a malformed pre-existing file on failure.

- [ ] **Step 2: Write failing completeness tests**

Assert static HY4 acceptance rejects target-only, MTP-only, initial-only, wrong-count, and mixed-SHA artifacts; both `HYV4ForCausalLM` and `HYV4MTP` must be `proposal` entries before the recorded file is considered calibration-complete.

- [ ] **Step 3: Run recorder tests and confirm RED**

Run: `pytest -q tests/runtime_patch/test_offline_eplb.py -k 'record or complete or atomic'`

Expected: FAIL at import.

- [ ] **Step 4: Implement durable record publication**

Use an adjacent `fcntl` lock, validate the merged payload before mutation, write and `fsync` a same-directory temporary file, `os.replace`, then `fsync` the parent directory. Do not log map contents or credentials.

- [ ] **Step 5: Write failing record-mode lifecycle tests**

Assert initial maps are tagged initial; a real proposal records target and MTP separately; proposal computation calls the official policy but suppresses weight transfer and live commit; profile/dummy work cannot mark completion; elastic rank mappings fail; async is disabled; ordinary online EPLB delegates unchanged.

- [ ] **Step 6: Write failing static-mode lifecycle tests**

Assert add-model rechecks the bound plan and EP fingerprint, commits maps without transfer, installs `locality_fair`, forces Gloo state communicator while preserving DeepEP data-plane selection, disables recording/async, and makes `step`/`rearrange` no-ops with zero transfer/rearrangement counters.

- [ ] **Step 7: Run lifecycle tests and confirm RED**

Run: `pytest -q tests/runtime_patch/test_offline_eplb.py`

Expected: FAIL because current official lifecycle still transfers/commits and has no offline path.

- [ ] **Step 8: Implement exact lifecycle wrappers**

Guard the current signatures and installed wrapper identities. Use a `ContextVar` only around record-only rearrangement, preserve official profile behavior where required for nonstatic configurations, and restore the requested communicator in `finally` after static `add_model` construction.

- [ ] **Step 9: Implement the Gloo reservation adapter**

Override `TorchDistGlooStagedEplbCommunicator.needs_profile_buffer_reservation` to false only if current base/Gloo/NCCL/PyNccl class contracts match and Gloo has no upstream override; leave all other communicators unchanged.

- [ ] **Step 10: Register callbacks and test inventory/order**

Add an `offline_eplb` worker feature derived from either offline path, register loader/mapping/state/communicator callbacks before model and EPLB modules are consumed, and extend dispatcher subprocess tests for enabled, disabled, serialization, and stale-marker paths.

- [ ] **Step 11: Run Task 5 tests**

Run: `pytest -q tests/runtime_patch/test_offline_eplb.py tests/runtime_patch/test_worker_framework_opt.py tests/patch/test_worker_dispatcher.py`

Expected: PASS.

- [ ] **Step 12: Review and commit Task 5**

Review exact target signatures against the installed vLLM source and confirm offline-disabled behavior is byte-for-byte delegation.

Commit: `git commit -m "feat(eplb): add offline record and static lifecycle"`

### Task 6: HY4 Target and MTP Offline EPLB

**Files:**
- Modify: `vllm_hcu/models/hy_v4/moe.py:66-155`
- Modify: `vllm_hcu/models/hy_v4/model.py:239-240,451-461,533-610`
- Modify: `vllm_hcu/models/hy_v4/mtp.py:260-418`
- Create: `tests/models/hy_v4/test_eplb.py`
- Extend: `tests/models/hy_v4/test_moe.py`
- Extend: `tests/models/hy_v4/test_mtp.py`
- Extend: `tests/models/hy_v4/test_weight_loading.py`

**Interfaces:**
- Consumes: Tasks 1–5 and current HY4 checkpoint accounting, `FusedMoEFactory`, model/MTP MoE metadata, and common `MixtureOfExperts` state protocol.
- Produces: HY4 target and MTP owners that allow exactly one offline path, allocate 264/33 expert topology at EP8, require their own bound plan, and reject dynamic/elastic/PP paths.

- [ ] **Step 1: Write failing constructor and count tests**

Assert target and MTP with `num_redundant_experts=8`, EP8, and one offline path produce 256 logical, 264 physical, and 33 local experts; their `FusedMoEFactory` receives `enable_eplb=True` and eight redundant experts. Assert no path, both paths, dynamic-only, async, PP, elastic, non-divisible counts, and missing plan fail with targeted errors.

- [ ] **Step 2: Run constructor tests and confirm RED**

Run: `pytest -q tests/models/hy_v4/test_eplb.py tests/models/hy_v4/test_moe.py -k 'eplb or redundant or static'`

Expected: FAIL at the current unconditional HY4 EPLB guards.

- [ ] **Step 3: Replace unconditional guards with offline-mode validation**

Read the HCU sidecar through Task 1, allow record or static mode only, preserve existing no-EPLB behavior, and compute physical/local counts before expert tensors are allocated.

- [ ] **Step 4: Write failing target/MTP state and loader tests**

Cover distinct `HYV4ForCausalLM` and `HYV4MTP` plan keys, plan-required metadata update/state binding, exact state shapes, MTP one-layer map, replicated Channel-FP8 weight/scale accounting, router bias ownership, complete checkpoint ledger, and rejection of changed maps.

- [ ] **Step 5: Run state/loader tests and confirm RED**

Run: `pytest -q tests/models/hy_v4/test_eplb.py tests/models/hy_v4/test_mtp.py tests/models/hy_v4/test_weight_loading.py -k 'eplb or static or replicated'`

Expected: FAIL because `_require_hyv4_static_plan` is still a rejection stub and record mode has no state contract.

- [ ] **Step 6: Implement HY4 plan lookup and state delegation**

Resolve the outer or inner bound plan without global state, validate exact target/MTP dimensions, delegate tensor installation to the common protocol, and prohibit runtime physical-count changes.

- [ ] **Step 7: Run the complete HY4 unit suite**

Run: `pytest -q tests/models/hy_v4`

Expected: PASS.

- [ ] **Step 8: Review and commit Task 6**

Confirm the diff does not alter sparse attention, HIPC cache writing, checkpoint quantization math, MTP metadata rebuild, or non-EPLB HY4 behavior.

Commit: `git commit -m "feat(hy4): enable offline EPLB for target and MTP"`

### Task 7: Integrated Routing and Regression Closure

**Files:**
- Extend: `tests/runtime_patch/test_moe_deepep.py`
- Extend: `tests/runtime_patch/test_platform_hcu_config.py`
- Extend: `tests/patch/test_plugin_lifecycle.py`
- Extend: `tests/integration/server/test_evalscope_v0281_gfx938_humaneval16.py`
- Modify: `tests/models/v0281_gfx938_humaneval16.yaml`

**Interfaces:**
- Consumes: complete Tasks 1–6 implementation and existing secure model-validation harness.
- Produces: installed-vLLM import/registration coverage, DeepEP/DeepGEMM expert-map coverage, and two checked-in HY4 calibration/static profiles with exact commands.

- [ ] **Step 1: Add failing DeepEP/DeepGEMM map-consumption tests**

Assert low-latency prepare/finalize receives the native global expert map and DeepGEMM masked experts receive the rank-local map/mask without dropping redundant slots. Assert offline-disabled and non-EPLB calls keep current behavior.

- [ ] **Step 2: Run route tests and confirm RED or expose existing equivalence**

Run: `pytest -q tests/runtime_patch/test_moe_deepep.py -k 'eplb and (low_latency or deep_gemm or expert_map)'`

Expected: at least one new assertion FAIL; if all pass without product changes, retain the tests as evidence and do not edit runtime routing.

- [ ] **Step 3: Make only the minimal routing correction, if RED requires it**

Patch the current owner that loses or reshapes the map; do not fork DeepEP/DeepGEMM implementations or change their non-EPLB ABI.

- [ ] **Step 4: Add real import and callback-chain smoke tests**

Assert installed vLLM roots, all offline callbacks applied exactly once, no failed patch records, Model Runner V2 remains required, and HCU config survives worker serialization.

- [ ] **Step 5: Add calibration and static launcher profiles**

Create separate HY4 DP8/TP1/EP8 profiles. Both use MRV2, `FLASHMLA_SPARSE`, DeepGEMM, explicit DeepEP low latency, MTP3, E4M3 KV, prefix caching, model length 4096, batch tokens 1024, eight sequences, and `reasoning_effort=no_think`. Calibration adds EPLB `{window_size:16, step_interval:16, num_redundant_experts:8, use_async:false, expert_map_record_path:<artifact>}`; static replaces the record path with `expert_map_path` and `static_dispatch_policy:locality_fair`.

- [ ] **Step 6: Run changed-file and full software gates**

Run:

```bash
pytest -q \
  tests/patch/test_config.py \
  tests/patch/test_worker_dispatcher.py \
  tests/patch/test_plugin_lifecycle.py \
  tests/model_executor/layers/fused_moe/test_static_eplb.py \
  tests/models/hy_v4 \
  tests/runtime_patch/test_eplb_locality_fair_dispatch.py \
  tests/runtime_patch/test_model_loader_static_eplb.py \
  tests/runtime_patch/test_offline_eplb.py \
  tests/runtime_patch/test_moe_deepep.py \
  tests/runtime_patch/test_platform_hcu_config.py \
  tests/runtime_patch/test_worker_framework_opt.py
pytest -q tests/runtime_patch
```

Expected: all tests pass; existing documented skips remain skips.

- [ ] **Step 7: Review the complete branch diff and commit Task 7**

Run: `git diff --check origin/v0.28.1-dev...HEAD && git diff --stat origin/v0.28.1-dev...HEAD`

Review for current-owner alignment, offline-disabled regressions, global monkeypatch leakage, memory duplication, missing tests, and unrelated changes.

Commit: `git commit -m "test(eplb): cover HY4 offline low-latency route"`

### Task 8: Eight-Device Calibration, Static Accuracy, and Delivery

**Files:**
- Create: `docs/validation/hy4-v0281-offline-eplb.md`
- Modify: `docs/validation/v0281-gfx938-provenance.md`
- Modify after evidence: `/models/upgrading-vllm-hcu/SKILL.md`
- Modify after evidence: `/models/upgrading-vllm-hcu/references/validation.md`
- Modify on failure discoveries: `/models/upgrading-vllm-hcu/references/failure-ledger.md`

**Interfaces:**
- Consumes: Tasks 1–7, eight idle HCU devices, secure EvalScope harness, existing PR #183, and protected GitHub authentication.
- Produces: retained map/log/report artifacts, exact commands, HumanEval16 result, self-review, skill updates, and one pushed update to the existing PR.

- [ ] **Step 1: Freeze runtime provenance and device baseline**

Record branch/plugin/vLLM SHAs, installed package/import roots, wheel checksum if available, Python/Torch/DTK/provider versions, model config checksum, `hy-smi` idle state, listeners, and owned process baseline. Stop if unrelated HCU work is active.

- [ ] **Step 2: Run calibration in a foreground PTY**

Launch the checked-in calibration profile with proxy variables cleared. Retain the PTY session, send a documented varied/concurrent workload until both target and MTP log non-profile proposals, then stop only the owned process group and verify all eight devices return to idle.

- [ ] **Step 3: Validate the recorded artifact offline**

Require version 2, `HYV4ForCausalLM` and `HYV4MTP` proposal entries, exact layer counts, 256/264/8 counts, every logical expert present in every row, SHA-256, no temporary file, and stable reread. Reject initial-only or partial artifacts.

- [ ] **Step 4: Run the fresh static service in a foreground PTY**

Use the static profile and recorded SHA without `--enforce-eager` or manual compilation config. Require eight matching fingerprints, 33 local experts per rank, Gloo static-state communicator, DeepEPLL data plane, DeepGEMM masked experts, MRV2 on all ranks, `fp8_ds_mla`, target/speculator PIECEWISE and FULL capture, and zero transfer/rearrangement counters.

- [ ] **Step 5: Run functional, prefix, and HumanEval gates**

Require health and generation HTTP 200, nine identical prefix requests with aggregate cache-hit growth, nonzero MTP drafted/accepted tokens, then run secure EvalScope HumanEval16 at temperature zero, `max_tokens=2048`, batch eight, and `reasoning_effort=no_think`. Require raw and independently normalized 16/16, 16 unique predictions/reviews, and no runtime ERROR or Traceback.

- [ ] **Step 6: Teardown and retain evidence**

Send Ctrl-C through the owned PTY, wait for API/EngineCore/workers, verify port removal and every owned PID/PGID exit, and require all eight cards to return to the recorded idle memory state.

- [ ] **Step 7: Write evidence-backed docs and skill updates**

Record exact server/client commands, map checksum and summary, test totals, providers, Graph evidence, MTP/prefix metrics, zero-transfer evidence, HumanEval score, teardown, failures/retries, and non-goals. Update the skill only with reproduced conclusions; record failed approaches in the ledger.

- [ ] **Step 8: Run final verification and review gates**

Rerun every test affected by documentation/launcher changes, `git diff --check`, secret scan, complete diff review against refreshed `origin/v0.28.1-dev`, and exact committed-diff review. Fix every blocking finding and rerun its gate.

- [ ] **Step 9: Commit, push, and update existing PR #183**

Commit only plugin-repository evidence and launcher changes in the existing branch. Update `/models/upgrading-vllm-hcu` as a separate local skill artifact because it is not part of the plugin Git worktree. Push through configured GitHub authentication without embedding the token in URLs or output. Add the exact calibration/static server commands, client command, results, review statement, and limitations to PR #183; verify the rendered PR contains no credential.

- [ ] **Step 10: Final remote review**

Fetch the remote branch, compare the exact remote range with `origin/v0.28.1-dev`, confirm local/remote SHAs match, and perform a final findings-first code review. Report no findings only if all required gates and artifacts remain present.
