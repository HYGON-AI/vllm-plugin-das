# vLLM Main Plugin Bootstrap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce an installable vLLM-HCU plugin bootstrap explicitly paired with the frozen OpenDAS vLLM main wheel, with lifecycle tests running against that artifact.

**Architecture:** Keep official main as owner of interfaces it already provides and retain plugin callbacks only for HCU behavior. Replace 0.25.1 source-test assumptions with an installed-target contract, bind compatibility to the frozen wheel, then build and install the plugin separately before kernel/model migration.

**Tech Stack:** Python 3.10, pytest, setuptools, PyTorch 2.11.0, DTK 26.04, HIP/CUDAExtension, vLLM plugin entry points.

**Spec:** `docs/superpowers/specs/2026-09-07-vllm-main-58ad1f3-hcu-plugin-migration-design.md`

## Global Constraints

- Official baseline: `vllm-project/vllm@58ad1f3b8973b23943107b51230d594050b42ec3`.
- OpenDAS source: `4574da606553cad5c22448d498f144630a23641e`.
- Required vLLM version: `0.28.1rc1.dev489+g4574da606.das.4574da6.dtk2604`.
- Required vLLM wheel SHA256: `6638d04d0213032ded93b9533bf19f4e267c1b9514fe50aac696dd1b210dcf83`.
- Plugin baseline: `origin/chore/qwen-max-model-len-40960-v0251-clean@9aa55078`.
- Final tests import both products from isolated install targets.
- Do not port an RC4 shim unless a main-specific failing test proves it is needed.

---

### Task 1: Make Lifecycle Tests Target-Agnostic

**Files:**
- Modify: `tests/patch/test_plugin_lifecycle.py`

**Interfaces:**
- Consumes: `VLLM_TARGET_ROOT`, plus the root containing the loaded `vllm_hcu` package.
- Produces: `_fresh_python()` subprocesses with both selected install targets on `PYTHONPATH`.

- [ ] **Step 1: Preserve the observed RED evidence**

The old-source run already reproduced three independent debts: fixed callback adjacency, an engine import against stale 0.25.1 APIs, and the old MRV2 `prepare_inputs` fake signature.

- [ ] **Step 2: Replace the old source-root contract**

Rename the `VLLM_V0251_SOURCE_ROOT` contract to `VLLM_TARGET_ROOT`. Resolve the default from `importlib.util.find_spec("vllm").origin`, validate `vllm/__init__.py`, and put the target before the plugin repository in child `PYTHONPATH`.

Resolve the plugin target independently from `vllm_hcu.__file__`, export it as
`VLLM_HCU_TARGET_ROOT`, and put it after the vLLM target in child `PYTHONPATH`.
Do not substitute the repository root: an installed parent pytest process can
otherwise launch source-plugin children and report a false artifact pass.

- [ ] **Step 3: Assert callback dependency order**

Assert each PCP callback index is less than `platform.framework_opt.mtp_indexer_kv_cache_coordinator`; do not require adjacency because independent Qwen MTP classification may be interposed.

- [ ] **Step 4: Update the MRV2 fake signature**

Use `prepare_inputs(self, scheduler_output, batch_desc, batch_req_state)` and call it with three arguments.

- [ ] **Step 5: Add actual-main import coverage**

In a fresh process import `kv_cache_utils`, `kv_cache_coordinator`, `engine.core`, `gpu.model_runner`, `attention.selector`, and `fused_moe.layer`. Assert each module comes from the target and the patch report has no failures.

- [ ] **Step 6: Run the lifecycle test**

```bash
PYTHONNOUSERSITE=1 VLLM_TARGET_ROOT=/models/.installs/vllm-main-58ad1f3-hcu-4574da6 \
PYTHONPATH=$PWD:/models/.installs/vllm-main-58ad1f3-hcu-4574da6 \
pytest -q tests/patch/test_plugin_lifecycle.py
```

Expected: the baseline debts are removed; any remaining failure is a main-specific adapter defect.

### Task 2: Bind Compatibility to the Frozen Wheel

**Files:**
- Modify: `tests/patch/test_compatibility_gate.py`
- Modify: `tests/patch/test_setup_packaging.py`
- Modify: `vllm_hcu/compatibility.py`
- Modify: `vllm_hcu/version.py`
- Modify: `setup.py`

**Interfaces:**
- Consumes: installed `vllm` distribution metadata.
- Produces: exact version and upstream/OpenDAS provenance in compatibility diagnostics.

- [ ] **Step 1: Write exact-version RED tests**

Accept only `0.28.1rc1.dev489+g4574da606.das.4574da6.dtk2604`. Reject `0.28.0`, `0.28.9`, the official source-only version, and other DAS builds. Require expected/actual versions and both SHAs in diagnostics.

- [ ] **Step 2: Run RED**

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=$PWD:/models/.installs/vllm-main-58ad1f3-hcu-4574da6 \
pytest -q tests/patch/test_compatibility_gate.py tests/patch/test_setup_packaging.py
```

Expected: broad `0.28.x` acceptance and stale plugin version assertions fail.

- [ ] **Step 3: Implement the exact artifact contract**

Set plugin public version to `0.28.1rc1.dev489`; expose expected vLLM version, full official SHA, and full OpenDAS SHA in `vllm_hcu.version`; compare normalized PEP 440 versions exactly; include provenance in `detail()`.

- [ ] **Step 4: Preserve plugin build provenance**

Generate `0.28.1rc1.dev489+das.<plugin-sha>.dtk2604` without rewriting source or changing global Git configuration.

- [ ] **Step 5: Run GREEN**

Repeat Step 2 and require all compatibility/packaging tests to pass.

### Task 3: Build and Install the Plugin Wheel

**Files:**
- Create: `/models/artifacts/vllm-plugin-das-main-58ad1f3/BUILD_PROVENANCE.txt`
- Create: `/models/artifacts/vllm-plugin-das-main-58ad1f3/SHA256SUMS`

**Interfaces:**
- Consumes: `/opt/dtk`, torch 2.11.0, and the isolated OpenDAS vLLM installation.
- Produces: plugin wheel, isolated installation, checksum, and provenance.

- [ ] **Step 1: Commit the artifact source boundary**

```bash
git add setup.py vllm_hcu/version.py vllm_hcu/compatibility.py \
  tests/patch/test_compatibility_gate.py tests/patch/test_plugin_lifecycle.py \
  tests/patch/test_setup_packaging.py docs/superpowers/plans/2026-09-07-vllm-main-plugin-bootstrap.md
git commit -m "build: bootstrap plugin for frozen vllm main"
```

The release build must start after this commit. A wheel built from uncommitted
sources but named with the previous HEAD is a trial artifact and is discarded.

- [ ] **Step 2: Build with bounded parallelism**

```bash
rm -rf build dist
PYTHONNOUSERSITE=1 PYTHONPATH=/models/.installs/vllm-main-58ad1f3-hcu-4574da6 \
ROCM_PATH=/opt/dtk ADD_GIT_VERSION=1 MAX_JOBS=8 python setup.py bdist_wheel
```

- [ ] **Step 3: Archive wheel evidence**

Record official SHA, OpenDAS SHA, plugin SHA, wheel name, DTK, torch, Python, and SHA256 under `/models/artifacts/vllm-plugin-das-main-58ad1f3`.

- [ ] **Step 4: Install without dependencies**

```bash
python -m pip install --no-deps --target \
  /models/.installs/vllm-plugin-das-main-58ad1f3-<plugin-sha> \
  /models/artifacts/vllm-plugin-das-main-58ad1f3/vllm_hcu-*.whl
```

### Task 4: Verify and Record the Bootstrap Gate

**Files:**
- Modify: `docs/superpowers/plans/2026-09-07-vllm-main-plugin-bootstrap.md`
- Modify: `/root/.codex/skills/upgrading-vllm-hcu/references/plugin-migration.md`
- Modify: `/root/.codex/skills/upgrading-vllm-hcu/references/failure-ledger.md`

**Interfaces:**
- Consumes: both isolated install targets.
- Produces: module-origin, platform, patch-report, native-extension, and test evidence.

- [ ] **Step 1: Verify delivered module origins**

Print `vllm.__file__`, `vllm_hcu.__file__`, both distribution versions, selected platform, patch failures, and `vllm_hcu.hcu_ops.__file__` from a fresh process.

- [ ] **Step 2: Run bootstrap tests from installed artifacts**

```bash
PYTHONNOUSERSITE=1 VLLM_TARGET_ROOT=/models/.installs/vllm-main-58ad1f3-hcu-4574da6 \
PYTHONPATH=/models/.installs/vllm-plugin-das-main-58ad1f3-<plugin-sha>:/models/.installs/vllm-main-58ad1f3-hcu-4574da6 \
pytest -q tests/patch/test_compatibility_gate.py tests/patch/test_plugin_lifecycle.py \
  tests/patch/test_setup_packaging.py
```

- [ ] **Step 3: Record evidence only**

Update the plan and upgrade skill with observed results. Bootstrap success advances API Gate 2 only; it does not claim attention, MoE, quantization, graph, MTP, or model support.

- [ ] **Step 4: Commit the evidence update separately**

```bash
git add docs/superpowers/plans/2026-09-07-vllm-main-plugin-bootstrap.md
git commit -m "docs: record frozen main plugin bootstrap evidence"
```
