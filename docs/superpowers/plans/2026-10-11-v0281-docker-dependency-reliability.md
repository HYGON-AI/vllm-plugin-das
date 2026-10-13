# v0.28.1 Docker Dependency Reliability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the v0.28.1 release image install its Ray runtime dependencies from the configured internal package hierarchy with audited versions instead of an unbounded Aliyun-only resolution.

**Architecture:** A focused requirements file owns the Ray/pandas compatibility pair validated in the current v0.28.1 environment. A small install wrapper delegates to the active pip configuration without replacing its index URL. The Dockerfile copies these artifacts and invokes the wrapper in the framework-dependency layer.

**Tech Stack:** Dockerfile, Bash, pip requirements, pytest

**Spec:** `docs/superpowers/specs/2026-10-11-v0281-ci-baseline-design.md`

## Global Constraints

- Preserve the private devpi configuration established earlier in `docker/Dockerfile`.
- Do not use an Aliyun-only `--index-url` for Ray.
- Pin `ray[data,train,tune,serve]` to 2.58.0 and pandas to 2.3.3, matching the audited local v0.28.1 environment.
- Do not change DTK, compiler, vLLM, Torch, or operator installation order.
- The authoritative full image result remains the private self-hosted workflow.

## Review Focus

- The wrapper must honor the configured index and must not inject any public mirror argument; Task 1 tests argv.
- Requirements must keep every requested Ray extra while using one exact Ray version; Task 1 parses and tests the requirement.
- pandas must remain constrained even if Ray metadata changes upstream; Task 1 tests the exact pin.
- Docker must copy requirements before invoking the install wrapper so cache/failure boundaries stay clear; Task 2 validates the build graph.
- A dependency-resolution failure must expose pip's original failing package and return code; Task 1 tests exit propagation.

---

### Task 1: Add a constrained runtime dependency installer

**Files:**
- Create: `docker/requirements-vllm-runtime.txt`
- Create: `scripts/ci/hcu/install_vllm_runtime_dependencies.sh`
- Create: `tests/patch/test_docker_runtime_dependencies.py`

**Interfaces:**
- Consumes: executable `${PYTHON:-python3}` and an optional requirements path argument.
- Produces: the exit status of `python -m pip install --no-cache-dir --requirement <path>` while relying on the existing pip configuration.

- [ ] **Step 1: Write failing installer behavior tests**

Run the wrapper with a recording Python executable and assert exact pip argv, absence of `--index-url`/`--extra-index-url`, and propagation of a nonzero pip exit. Parse requirements with `packaging.Requirement` and assert Ray extras `{data, train, tune, serve}`, Ray `==2.58.0`, and pandas `==2.3.3`.

- [ ] **Step 2: Run tests to verify RED**

Run: `pytest -q tests/patch/test_docker_runtime_dependencies.py`

Expected: failure because the wrapper and requirements file do not exist.

- [ ] **Step 3: Add the minimal requirements and wrapper**

Use `set -euo pipefail`, resolve the supplied/default requirements path, and invoke pip once without index overrides.

- [ ] **Step 4: Run tests to verify GREEN**

Run: `pytest -q tests/patch/test_docker_runtime_dependencies.py`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add docker/requirements-vllm-runtime.txt scripts/ci/hcu/install_vllm_runtime_dependencies.sh tests/patch/test_docker_runtime_dependencies.py
git commit -m "build: constrain v0.28.1 runtime dependencies"
```

### Task 2: Integrate the constrained installer into the image

**Files:**
- Modify: `docker/Dockerfile:55-100`
- Modify: `tests/patch/test_docker_runtime_dependencies.py`

**Interfaces:**
- Consumes: Task 1 wrapper and requirements file copied to `/tmp`.
- Produces: a framework dependency layer that installs Ray/pandas through internal devpi and leaves all subsequent package layers unchanged.

- [ ] **Step 1: Add a failing Docker integration contract**

Parse Dockerfile instructions and assert the requirements and wrapper are copied before the framework dependency `RUN`, that the wrapper is invoked, and that no Ray install command contains the Aliyun index override.

- [ ] **Step 2: Run test to verify RED**

Run: `pytest -q tests/patch/test_docker_runtime_dependencies.py`

Expected: failure on the existing direct unbounded Aliyun Ray install.

- [ ] **Step 3: Replace only the Ray install command**

Copy the two Task 1 files into the image and invoke the wrapper at the existing Ray position. Preserve the rest of the chained framework installs and cache purge.

- [ ] **Step 4: Run tests and a dependency dry run**

Run: `pytest -q tests/patch/test_docker_runtime_dependencies.py`

Run: `python3 -m pip install --dry-run --ignore-installed --requirement docker/requirements-vllm-runtime.txt`

Expected: tests pass and pip resolves both pins from the configured package hierarchy. If local network policy prevents resolution, record that result and rely on the private image workflow for integration.

- [ ] **Step 5: Commit**

```bash
git add docker/Dockerfile tests/patch/test_docker_runtime_dependencies.py
git commit -m "build: use internal index for Ray runtime"
```

### Task 3: Final Docker verification

**Files:**
- No additional changes expected.

**Interfaces:**
- Consumes: Tasks 1-2.
- Produces: reproducibility evidence and the exact private-runner follow-up recorded in the MR.

- [ ] **Step 1: Run local static verification**

Run: `bash -n scripts/ci/hcu/install_vllm_runtime_dependencies.sh`

Run: `pytest -q tests/patch/test_docker_runtime_dependencies.py`

Run: `git diff --check`

Expected: all pass.

- [ ] **Step 2: Trigger or observe the release image workflow after push**

Confirm the Ray layer resolves from the configured internal index, the image is pushed, and `validate-image` receives the exact dated image output. Record the workflow URL and any runner-only failure without masking it.
