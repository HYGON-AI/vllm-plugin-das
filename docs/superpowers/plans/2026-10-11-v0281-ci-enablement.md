# v0.28.1-dev CI Enablement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable release-isolated v0.28.1-dev CI and restore the portable patch-test baseline.

**Architecture:** The checked-out v0.28.1 branch owns its environment lock, while a small resolver converts the target ref plus `HCU_CI_IMAGE_V0281` into explicit workflow outputs. Existing reusable hardware jobs continue to pin the selected image to a digest. Recent v0.28.1 dispatcher changes are captured as explicit compatibility baselines rather than weakening lazy-loading assertions.

**Tech Stack:** Python 3.10, pytest, GitHub Actions YAML, JSON HCU registry/configuration

**Spec:** `docs/superpowers/specs/2026-10-11-v0281-ci-baseline-design.md`

## Global Constraints

- Target branch is `v0.28.1-dev`; do not change v0.25.1 release behavior.
- v0.28.1 CI must use `HCU_CI_IMAGE_V0281` and must never fall back to `vllm:0.25.1-latest`.
- The checked-out environment lock must require the vLLM and vLLM-HCU `0.28.1` prefixes, Torch 2.11.0, and the DTK/ROCm `26.04` release line.
- Keep HCU business adapters lazy even where upstream vLLM 0.28.1 imports framework modules eagerly.
- Keep the HCU-only bootstrap test strict; do not hide a missing locally built `hcu_ops.so`.

## Review Focus

- A fork PR with no `HCU_CI_IMAGE_V0281` must fail with a clear configuration error, not select another image; Task 1 tests this.
- A target ref other than `v0.28.1-dev` must be rejected by the release resolver; Task 1 tests this.
- Hardware jobs must receive the exact image selected by the static gate and then pin it to a digest; Task 2 tests workflow data flow.
- The Kimi marked test must route to an eight-card job carrying both evalscope and checkpoint requirements; Task 4 tests this.
- Upstream eager imports may be present, but importing plugin dispatchers must not eagerly load HCU MoE/runtime implementations; Task 5 tests this.

---

### Task 1: Release environment resolver

**Files:**
- Create: `.github/scripts/hcu_ci/resolve_hcu_ci_environment.py`
- Create: `tests/patch/test_hcu_ci_release_environment.py`

**Interfaces:**
- Consumes: target ref string and an environment mapping containing `HCU_CI_IMAGE_V0281`.
- Produces: `resolve_release_environment(ref: str, environment: Mapping[str, str]) -> ReleaseEnvironment`, where `ReleaseEnvironment.image` is non-empty and `ReleaseEnvironment.environment_lock` is `.github/workflows/configs/hcu-runner-environment.json`; CLI writes `image` and `environment_lock` GitHub outputs.

- [ ] **Step 1: Write failing resolver tests**

Add tests that hand-derive these outcomes: `v0.28.1-dev` plus a digest/tag returns it unchanged and the checked-out lock path; missing `HCU_CI_IMAGE_V0281` raises `ReleaseEnvironmentError`; an unsupported ref raises the same error; CLI failure exits 2 and emits no output file values.

- [ ] **Step 2: Run tests to verify RED**

Run: `pytest -q tests/patch/test_hcu_ci_release_environment.py`

Expected: collection/import failure because the resolver does not exist.

- [ ] **Step 3: Implement the minimal resolver**

Implement the typed result, error, pure resolver, argparse CLI, and atomic append of the two GitHub output keys. Do not add release lines other than `v0.28.1-dev`.

- [ ] **Step 4: Run tests to verify GREEN**

Run: `pytest -q tests/patch/test_hcu_ci_release_environment.py`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add .github/scripts/hcu_ci/resolve_hcu_ci_environment.py tests/patch/test_hcu_ci_release_environment.py
git commit -m "ci: resolve v0.28.1 HCU environment explicitly"
```

### Task 2: Route workflows through the v0.28.1 environment

**Files:**
- Modify: `.github/workflows/hcu-pr-ci.yml:6-300`
- Modify: `.github/workflows/_selected-hcu-tests.yml:8-105`
- Modify: `.github/workflows/quality-gate.yml:3-8`
- Modify: `.github/workflows/patch-coverage.yml:5-24`
- Modify: `.github/workflows/nightly-hcu.yml:5-85`
- Modify: `.github/workflows/validate-docker-image.yml:1-150`
- Modify: `tests/patch/test_hcu_ci_release_environment.py`

**Interfaces:**
- Consumes: Task 1 CLI outputs `image` and `environment_lock`.
- Produces: static-gate job outputs consumed by authorization and selected hardware jobs; all v0.28.1 jobs use the versioned image variable and checked-out lock.

- [ ] **Step 1: Write failing workflow data-flow tests**

Load workflows as YAML data and assert: PR/quality gates target `v0.28.1-dev`; the PR static job invokes the resolver with the PR base ref and exports both values; selected hardware gets the static job's image; reusable and validation workflows have no `0.25.1-latest` fallback; nightly defaults to `v0.28.1-dev` and uses `vars.HCU_CI_IMAGE_V0281`; patch coverage targets v0.28.1 and uses the same image variable.

- [ ] **Step 2: Run tests to verify RED**

Run: `pytest -q tests/patch/test_hcu_ci_release_environment.py`

Expected: assertions fail on v0.25.1 triggers/fallbacks and absent resolver data flow.

- [ ] **Step 3: Apply minimal workflow changes**

Add the resolver step and job outputs to PR CI, pass its image explicitly to reusable jobs, replace legacy fallbacks with fail-closed versioned inputs, and update target/default refs. Preserve immutable digest resolution and existing authorization behavior.

- [ ] **Step 4: Run tests and YAML validation**

Run: `pytest -q tests/patch/test_hcu_ci_release_environment.py tests/patch/test_hcu_ci_selector.py`

Run: `python3 -c 'import pathlib,yaml; [yaml.safe_load(p.read_text()) for p in pathlib.Path(".github/workflows").glob("*.yml")]'`

Expected: tests pass and all workflows parse.

- [ ] **Step 5: Commit**

```bash
git add .github/workflows tests/patch/test_hcu_ci_release_environment.py
git commit -m "ci: enable release-isolated v0.28.1 gates"
```

### Task 3: Align the v0.28.1 runner lock

**Files:**
- Modify: `.github/workflows/configs/hcu-runner-environment.json:1-36`
- Modify: `tests/patch/test_hcu_ci_selector.py:284-385`

**Interfaces:**
- Consumes: existing `_load_environment_lock` and `_check_environment_lock` APIs.
- Produces: a schema-version-1 lock requiring Python 3.10.12, Torch `2.11.0+`, HIP `6.3.`, the DTK/ROCm `26.04` release line, vLLM `0.28.1`, vLLM-HCU `0.28.1`, AITER `0.1.`, and pytest 9.1.1. The vLLM values use prefix matching so release-candidate, nightly, patch, and local build suffixes remain valid.

- [ ] **Step 1: Write a failing branch-lock contract test**

Assert the literal compatibility prefixes above and verify `_check_environment_lock` accepts representative installed v0.28.1 and DTK 26.04 builds but rejects a v0.25.1 vLLM distribution.

- [ ] **Step 2: Run test to verify RED**

Run: `pytest -q tests/patch/test_hcu_ci_selector.py -k 'environment_lock or runner_lock'`

Expected: failure because the lock still requires v0.25.1 and DTK 26.04.

- [ ] **Step 3: Update only the release-specific values**

Keep schema unchanged; use prefix matching for both vLLM distributions and release-line matching for DTK/ROCm.

- [ ] **Step 4: Run test to verify GREEN**

Run: `pytest -q tests/patch/test_hcu_ci_selector.py -k 'environment_lock or runner_lock'`

Run: `python3 .github/scripts/hcu_ci/hcu_ci_preflight.py --check-lock-only`

Expected: tests pass and the lock prints successfully.

- [ ] **Step 5: Commit**

```bash
git add .github/workflows/configs/hcu-runner-environment.json tests/patch/test_hcu_ci_selector.py
git commit -m "ci: lock the v0.28.1 HCU runner environment"
```

### Task 4: Register Kimi K2.5 HumanEval ownership

**Files:**
- Modify: `.github/workflows/configs/hcu-test-map.yaml:394-550,714-855`
- Modify: `tests/hcu_ci_registry.py:301-376`
- Modify: `tests/patch/test_hcu_ci_selector.py:187-205,517-537`

**Interfaces:**
- Consumes: existing literal registry and matrix parser.
- Produces: job `kimi-k25-humaneval` for the marked server test, requiring eight gfx938 cards, evalscope 1.9.1, and model env `VLLM_HCU_KIMI_K25_MODEL`.

- [ ] **Step 1: Extend the existing failing registry test with routing assertions**

Assert the exact test node is registered, its job exists, it requests eight cards, and its requirements contain evalscope plus the Kimi model path contract. Add its selector group and nightly entry assertions.

- [ ] **Step 2: Run tests to verify RED**

Run: `pytest -q tests/patch/test_hcu_ci_selector.py -k 'registered or evalscope or kimi'`

Expected: failure because the marked Kimi file is not registered and no job exists.

- [ ] **Step 3: Add the literal registration, job, nightly entry, and selector groups**

Route changes to the Kimi test/config and shared evalscope runner to `kimi-k25-humaneval`; do not alter its serving command or accuracy threshold.

- [ ] **Step 4: Run tests to verify GREEN**

Run: `pytest -q tests/patch/test_hcu_ci_selector.py`

Run: `python3 .github/scripts/hcu_ci/verify_hcu_registration.py`

Expected: tests pass and every marked HCU test is registered.

- [ ] **Step 5: Commit**

```bash
git add .github/workflows/configs/hcu-test-map.yaml tests/hcu_ci_registry.py tests/patch/test_hcu_ci_selector.py
git commit -m "ci: register Kimi K2.5 HumanEval coverage"
```

### Task 5: Restore dispatcher and patch-coverage baselines

**Files:**
- Modify: `tests/patch/test_platform_dispatcher.py:114-460`
- Modify: `tests/patch/test_worker_dispatcher.py:240-310`
- Modify: `tests/patch/test_patch_coverage.py:14-28`
- Modify or create direct contract test: `tests/runtime_patch/test_glm53_pcp.py`

**Interfaces:**
- Consumes: current `platform_framework_callback_names`, worker dispatcher inventory, and GLM-5.3 PCP adapter.
- Produces: explicit 49-registration/37-callback baseline, exact upstream-v0.28 eager import sets, and a direct behavioral reference for `patch_glm53_pcp_kv_cache`.

- [ ] **Step 1: Add the GLM-5.3 callback and import baselines to tests**

Add the callback at its dependency-ordered position, adjust counts, and replace impossible “none loaded” assertions with exact audited upstream target sets while retaining empty HCU business-adapter intersections. Add a direct adapter contract test that exercises the KV-cache coordinator behavior patched by `patch_glm53_pcp_kv_cache`.

- [ ] **Step 2: Run targeted tests to verify the intended RED evidence**

Run the five failing node IDs from `test_platform_dispatcher.py`, `test_worker_dispatcher.py`, and `test_patch_coverage.py`, plus the new GLM-5.3 contract test.

Expected: existing tests fail with the recorded inventory/count/import differences; the new direct test fails until its adapter reference/fixture is complete.

- [ ] **Step 3: Complete the minimal expectation and fixture updates**

Use the exact observed upstream sets; do not replace them with unconstrained subset assertions. Keep all HCU laziness and idempotence assertions.

- [ ] **Step 4: Run tests to verify GREEN**

Run: `pytest -q tests/patch/test_platform_dispatcher.py tests/patch/test_worker_dispatcher.py tests/patch/test_patch_coverage.py tests/runtime_patch/test_glm53_pcp.py`

Run: `python3 tools/check_patch_test_coverage.py`

Expected: all pass and no adapter is reported untested.

- [ ] **Step 5: Commit**

```bash
git add tests/patch/test_platform_dispatcher.py tests/patch/test_worker_dispatcher.py tests/patch/test_patch_coverage.py tests/runtime_patch/test_glm53_pcp.py
git commit -m "test: align v0.28.1 patch dispatcher contracts"
```

### Task 6: Repair the model-runner lifecycle test double

**Files:**
- Modify: `tests/patch/test_plugin_lifecycle.py:716-755`

**Interfaces:**
- Consumes: upstream `GPUModelRunner.__init__(vllm_config, device)` contract.
- Produces: a complete test double that retains `self.vllm_config`, allowing the HCU wrapper to exercise its real request-scope behavior.

- [ ] **Step 1: Strengthen the failing test fixture contract**

Set `self.vllm_config = vllm_config` in the upstream test double and retain all request-phase assertions. No production fallback is added.

- [ ] **Step 2: Run the previously failing node**

Run: `pytest -q tests/patch/test_plugin_lifecycle.py::test_hcu_model_runner_v2_scopes_request_phase_around_upstream_execute`

Expected: pass, demonstrating the failure was an incomplete double rather than production behavior.

- [ ] **Step 3: Run the lifecycle file**

Run: `pytest -q tests/patch/test_plugin_lifecycle.py`

Expected: all portable tests pass; any HCU-only environment failure is reported separately.

- [ ] **Step 4: Commit**

```bash
git add tests/patch/test_plugin_lifecycle.py
git commit -m "test: mirror v0.28.1 model runner initialization"
```

### Task 7: Verify the CI baseline

**Files:**
- No production changes expected.

**Interfaces:**
- Consumes: Tasks 1-6.
- Produces: static and portable verification evidence for the MR description.

- [ ] **Step 1: Run static control-plane checks**

Run environment-lock parsing, `verify_hcu_registration.py`, nightly/quarantine matrix builds, `check_patch_test_coverage.py`, workflow YAML parsing, and `compileall` over changed Python.

- [ ] **Step 2: Run portable patch tests**

Run: `pytest -q tests/patch`

Expected: all portable tests pass; if the local missing `hcu_ops.so` bootstrap test remains HCU-only, record its exact node and run the CI-equivalent `tools/run_patch_tests.py --suite contract` selection.

- [ ] **Step 3: Review diff and commit verification-only corrections if needed**

Run: `git diff --check && git status --short && git log --oneline origin/v0.28.1-dev..HEAD`

Expected: no whitespace errors or uncommitted files.
