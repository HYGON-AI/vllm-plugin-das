# vLLM 0.32 Upstream-First Plugin Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver an independently installable DAS Plugin for the vLLM 0.32 release line whose optional LightOp operators are Graph-safe, explicitly selectable, and retained only with evidence that official vLLM does not fully take over the HCU use case.

**Architecture:** Adapter remains the sole owner of vLLM platform/runtime compatibility and activates one Plugin extension entry point. The audit retained only optional ordinary/fused-add `RMSNorm`; warmed evidence delegated `SiluAndMul` completely to official vLLM. Runtime admission uses the 0.32 release line plus API/capability checks; immutable SHAs identify evidence, not runtime locks.

**Tech Stack:** Python 3.12, PyTorch 2.11 custom operators and `torch.compile`, vLLM 0.32 OOT `CustomOp`, LightOp, pytest, HCU `gfx936`.

**Spec:** `docs/superpowers/specs/2026-10-29-vllm032-plugin-onoff-validation-design.md`

## Global Constraints

- Base Plugin work on `HYGON-AI/vllm-plugin-das:v0.32.0@e76220d9c0006cc3c9d886587a4771a31e7d83e8` and target that upstream branch.
- Pair evidence with Adapter `c99886d41f435e594050d47663e03ea94081fdc6` and official vLLM source `5df63a7203cdb9328d824a27b682e77c7b9928be`.
- Keep Plugin out of `vllm.platform_plugins`, `vllm.general_plugins`, Adapter runtime code, and `sitecustomize.py`.
- Prefer official vLLM behavior; no retained optimization may be described as active or faster without dispatch and performance evidence.
- Child operator switches default OFF until an accepted HCU performance criterion exists; explicit ON remains subordinate to `VLLM_HCU_USE_CUSTOM_OPS`.
- Every v0.32 test command replaces inherited `PYTHONPATH` and records `vllm.__version__` and `vllm.__file__`.
- Runtime compatibility is `>=0.32.0,<0.33.0` plus required API/symbol checks; artifact claims remain scoped to the exact tested wheel.

## Review Focus

- A `0.32.1`, prerelease, postrelease, or local build with all required APIs must activate; `0.31.x`, `0.33.x`, invalid versions, and epoch changes must fail before runtime imports (Task 1).
- Fused-add RMSNorm must mutate its owned buffers without returning aliases from the Torch custom operator, while the Python layer still returns `(x, residual)` (Task 2).
- Explicit ON must reach `forward_hip` even when vLLM configures `custom_ops=['none']`; OFF must preserve official eager and compiled fallbacks (Task 3).
- Invalid environment values must fail closed even if official vLLM would otherwise enable its custom op (Task 3).
- Installed-wheel HCU evidence must use fresh ON/OFF processes, changed-input Graph replay, actual selected paths, and warmed repeated measurements (Task 4).

---

### Task 1: Release-Line and Capability Admission

**Files:**
- Modify: `tests/vllm032/test_extension.py`
- Modify: `adapter_extension/src/hcu_vllm_das/extension.py`
- Modify: `adapter_extension/README.md`

**Interfaces:**
- Consumes: `ExtensionContext.vllm_version: str | None` and existing required callable checks.
- Produces: `_require_vllm_032_release_line(raw_version: str | None) -> None`, accepting the 0.32 release line and rejecting every other release before imports.

- [ ] **Step 1: Write the failing release-line boundary tests**

Add literal accepted cases `0.32.1`, `0.32.7+das.1`, and rejected cases `0.31.9`, `0.33.0`, `1!0.32.0`; assert unsupported metadata performs no runtime imports and the error states `>=0.32.0,<0.33.0`.

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `env -u PYTHONPATH HCU_ADAPTER_HOST=none PYTHONPATH=/dev/shm/adapter-das-main-20261029 /dev/shm/adapter-das-pr11-vllm032-qwen2-05b/venv/bin/python -m pytest -q tests/vllm032/test_extension.py -k 'release_line or unsupported_metadata'`

Expected: `0.32.1` cases fail because the current exact-release comparison rejects them.

- [ ] **Step 3: Implement the release-line guard**

Rename the guard and accept only `Version(...).epoch == 0` with `release[:2] == (0, 32)`. Preserve all existing callable checks and fail-before-import ordering.

- [ ] **Step 4: Run focused and full portable tests**

Run focused command from Step 2, then:

`env -u PYTHONPATH HCU_ADAPTER_HOST=none VLLM_TARGET_ROOT=/dev/shm/hcu-vllm032-work/src/vllm PYTHONPATH=/dev/shm/adapter-das-main-20261029 /dev/shm/adapter-das-pr11-vllm032-qwen2-05b/venv/bin/python -m pytest -q tests/vllm032`

Expected: focused PASS; portable suite PASS with only live-HCU skips.

- [ ] **Step 5: Commit**

Commit: `fix(vllm032): admit the capability-checked release line`

### Task 2: Graph-Safe Fused RMSNorm Mutation Contract

**Files:**
- Modify: `tests/vllm032/test_rms_norm.py`
- Modify: `tests/vllm032/test_rms_norm_hcu.py`
- Modify: `adapter_extension/src/hcu_vllm_das/rms_norm.py`

**Interfaces:**
- Consumes: `lightop.norm.fused_add_rms_norm(x, residual, weight, epsilon)` mutating `x` and `residual`.
- Produces: `_fused_add_rms_norm_impl(...) -> None`, `_fused_add_rms_norm_fake(...) -> None`; `HcuRMSNorm.forward_hip(...)` continues to return `(x, residual)`.

- [ ] **Step 1: Write the failing side-effect-only schema tests**

Change the portable fake-runtime assertions so the registered fused implementation and fake return `None`, while the layer-visible result remains the same two mutated objects. Add a live compile test that treats the PyTorch mutable-output alias warning as failure and replays with changed inputs.

- [ ] **Step 2: Run the portable test and verify RED**

Run: `env -u PYTHONPATH HCU_ADAPTER_HOST=none PYTHONPATH=/dev/shm/adapter-das-main-20261029 /dev/shm/adapter-das-pr11-vllm032-qwen2-05b/venv/bin/python -m pytest -q tests/vllm032/test_rms_norm.py`

Expected: FAIL because both registered functions currently return `(x, residual)`.

- [ ] **Step 3: Implement the side-effect-only custom operator**

Return `None` from implementation and fake functions. In `forward_hip`, call the custom op for its side effects and then return the owned `x, residual` tuple.

- [ ] **Step 4: Run focused portable tests**

Run: `env -u PYTHONPATH HCU_ADAPTER_HOST=none PYTHONPATH=/dev/shm/adapter-das-main-20261029 /dev/shm/adapter-das-pr11-vllm032-qwen2-05b/venv/bin/python -m pytest -q tests/vllm032/test_rms_norm.py tests/vllm032/test_silu_and_mul.py`

Expected: PASS.

- [ ] **Step 5: Commit**

Commit: `fix(vllm032): make fused rmsnorm custom op side effect only`

### Task 3: Explicit Graph Reachability and Conservative Defaults

**Files:**
- Modify: `tests/vllm032/test_silu_and_mul.py`
- Modify: `tests/vllm032/test_rms_norm.py`
- Modify: `adapter_extension/src/hcu_vllm_das/environment.py`
- Modify: `adapter_extension/src/hcu_vllm_das/silu_and_mul.py`
- Modify: `adapter_extension/src/hcu_vllm_das/rms_norm.py`
- Modify: `adapter_extension/README.md`

**Interfaces:**
- Consumes: vLLM `CustomOp.enabled() -> bool` and Plugin environment policy.
- Produces: `HcuSiluAndMul.enabled() -> bool` and `HcuRMSNorm.enabled() -> bool`, each true when explicit Plugin policy enables the operator or official vLLM enables the base custom op.

- [ ] **Step 1: Write failing policy and dispatch tests**

Assert child switches default OFF; explicit master+child ON makes each OOT class enabled when its base class reports disabled; Plugin OFF still returns true when the official base reports enabled; invalid Plugin values raise before consulting the base.

- [ ] **Step 2: Run focused tests and verify RED**

Run: `env -u PYTHONPATH HCU_ADAPTER_HOST=none PYTHONPATH=/dev/shm/adapter-das-main-20261029 /dev/shm/adapter-das-pr11-vllm032-qwen2-05b/venv/bin/python -m pytest -q tests/vllm032/test_silu_and_mul.py tests/vllm032/test_rms_norm.py`

Expected: FAIL because child switches default ON and OOT classes inherit vLLM's `custom_ops=['none']` disablement.

- [ ] **Step 3: Implement conservative policy and OOT enablement**

Set both child defaults to false. Override each OOT class's `enabled()` so it evaluates Plugin policy first, then delegates to the official base method; retain complete `forward_cuda`/`forward_native` fallback behavior.

- [ ] **Step 4: Run focused and full portable tests**

Run focused command from Step 2, then the full portable command from Task 1 Step 4.

Expected: all portable tests pass with only live-HCU skips.

- [ ] **Step 5: Commit**

Commit: `fix(vllm032): make lightop paths explicit graph-safe opt-ins`

### Task 4: Installed HCU Takeover Gate and Delivery Evidence

**Audit outcome:** Warmed A/B removed the candidate Silu override and its tests,
switch, capability, module, and OOT registration. After the shared-node
occupancy cleared without interference, the final-wheel live Graph and fresh
non-eager ON/OFF small-model service rows passed. HumanEval and broader-model
precision remain explicitly pending.

**Files:**
- Create: `tests/vllm032/test_operator_graph_hcu.py`
- Modify: `tests/vllm032/test_silu_and_mul_hcu.py`
- Modify: `tests/vllm032/test_rms_norm_hcu.py`
- Modify: `validation/v0.32.0/compat-lock.json`
- Create: `docs/validation/vllm032-upstream-first-operator-ab.md`
- Modify: `docs/superpowers/specs/2026-10-29-vllm032-plugin-onoff-validation-design.md`

**Interfaces:**
- Consumes: the Plugin wheel from Tasks 1-3, Adapter `c99886d`, and vLLM `0.32.0+rocm723`.
- Produces: fresh-process ON/OFF numerical and dispatch results, changed-input compile/Graph replay, warmed repeated A/B measurements, wheel hashes, and final P-032-02/P-032-03 dispositions.

- [x] **Step 1: Write the opt-in Graph integration test and verify it initially fails against the pre-Task-3 wheel**

The test must assert selected OOT class and `_forward_method`, execute ordinary/fused RMSNorm and SiluAndMul with explicit ON, replay changed inputs, reject alias warnings, and compare with fresh-process official OFF output.

- [x] **Step 2: Build and install the clean candidate wheel**

Build from `adapter_extension/`, install with `--no-deps` into the dedicated v0.32 environment, and record SHA256 plus import roots. Inspect metadata for exactly one `hcu_adapter.extensions` entry point and no vLLM platform/general entry points.

- [x] **Step 3: Run live HCU operator and Graph gates**

Use `HIP_VISIBLE_DEVICES=0`, `VLLM_PLUGINS=hcu_adapter`, current Adapter source first on `PYTHONPATH`, and explicit operator switches. Run all three live test files in fresh processes/parameter rows.

Expected: numerical parity, `forward_hip` for explicit ON, official fallback for OFF, changed-input replay, and no mutable-alias warning.

- [x] **Step 4: Run warmed upstream takeover A/B**

Compare official default compiled native against explicit LightOp for representative Qwen dense shapes and ordinary/fused RMSNorm, with identical inputs, at least three independent measurements, warmup excluded, and raw latency distributions recorded. Do not invent a pass threshold; absent a project criterion, keep child defaults OFF and report the measurements without a speedup claim.

- [x] **Step 5: Run small-model non-eager ON/OFF service smoke**

Use a model under `/model`, default `FULL_AND_PIECEWISE` Graph policy, proxy-bypassed `/health`, `/v1/models`, and deterministic chat requests. Start and stop each owned service through its PTY session and require HCU memory to return to baseline.

- [x] **Step 6: Update evidence and run final verification**

Update the compatibility lock to the current three-way SHAs, wheel hashes, commands, raw statuses, and explicit not-run/blocked rows. Run the full portable suite, all live operator/Graph gates, wheel verifier, and `git diff --check`.

Expected: all executed required rows pass; unexecuted model/accuracy/performance acceptance rows remain explicitly pending rather than inheriting old results.

- [ ] **Step 7: Commit**

Commit: `test(vllm032): record upstream takeover and graph evidence`
