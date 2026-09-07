# OpenDAS vLLM Main HCU Wheel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce, install, smoke-test, attest, and push an OpenDAS HCU vLLM wheel based exactly on official main commit `58ad1f3b8973b23943107b51230d594050b42ec3`.

**Architecture:** Keep the OpenDAS fork limited to DTK build and packaging compatibility. Port the requirements represented by RC4 commit `a20aa339f` hunk-by-hunk against current main, using failing contract tests before production changes; do not not cherry-pick that commit because main has materially changed its kernel and packaging sources.

**Tech Stack:** Python 3.10, setuptools/setuptools-scm, CMake, DTK 26.04, DTK torch 2.11, HIP/C++, pytest, Git.

**Spec:** `/models/.worktrees/vllm-plugin-das-main-58ad1f3/docs/superpowers/specs/2026-09-07-vllm-main-58ad1f3-hcu-plugin-migration-design.md`

## Global Constraints

- Official baseline is exactly `58ad1f3b8973b23943107b51230d594050b42ec3`.
- Work only in `/models/.worktrees/vllm-main-58ad1f3-hcu` on branch `main-58ad1f3-hcu`.
- Do not modify the RC4 worktrees.
- Do not add model, scheduler, attention, MoE, quantization, or plugin behavior to the vLLM fork.
- Reuse upstream behavior whenever main already satisfies a DTK requirement.
- The wheel provenance must identify upstream SHA, source SHA, DTK, torch, Python, and the wheel SHA256.
- The build helper must create `/usr/lib/x86_64-linux-gnu/librt.so` only when absent and only when `librt.so.1` exists.
- Each test command runs with local HTTP proxies irrelevant or explicitly disabled.

---

### Task 1: Establish Main Build Contracts

**Files:**
- Create: `/models/.worktrees/vllm-main-58ad1f3-hcu/tests/das/test_dtk_build_contract.py`
- Create: `/models/.worktrees/vllm-main-58ad1f3-hcu/tests/das/test_build_das_hcu_wheel.py`

**Interfaces:**
- Consumes: official main source at the fixed SHA and RC4 test intent from commit `a20aa339f`.
- Produces: static source contracts and an isolated fake-build integration contract for later tasks.

- [ ] **Step 1: Add the failing static build contract**

Create tests that require `gfx928`, `gfx936`, and `gfx938`; DTK detection through `ATen/dtk_macros.h`; the `DCU_ASM` definition; DTK-safe stable ABI handling; a 1024-thread HIP compiler flag; guarded qknorm and GPTQ compatibility shims; and CUDA-only bundled `triton_kernels` packaging. Assert that main wheel versioning contains no hardcoded RC4 version.

- [ ] **Step 2: Add the failing build-helper contract**

Use a temporary fake Python executable and temporary `librt.so.1` target. Require the helper to create the symlink, identify the produced wheel without assuming a release version, emit a valid SHA256 sidecar, and emit these provenance keys:

```text
wheel=
source_sha=
upstream_ref=main
upstream_sha=58ad1f3b8973b23943107b51230d594050b42ec3
dtk_version=
torch_version=
python_version=
```

- [ ] **Step 3: Run tests to verify RED**

Run:

```bash
cd /models/.worktrees/vllm-main-58ad1f3-hcu
python -m pytest -q tests/das/test_dtk_build_contract.py tests/das/test_build_das_hcu_wheel.py
```

Expected: failures for missing DTK source contracts and missing `tools/build_das_hcu_wheel.sh`, not import or collection errors.

- [ ] **Step 4: Commit the failing contracts**

```bash
git add tests/das/test_dtk_build_contract.py tests/das/test_build_das_hcu_wheel.py
git commit -m "test: define main DTK wheel contracts"
```

---

### Task 2: Adapt CMake and HIP Compilation

**Files:**
- Modify: `/models/.worktrees/vllm-main-58ad1f3-hcu/CMakeLists.txt`
- Modify: `/models/.worktrees/vllm-main-58ad1f3-hcu/cmake/utils.cmake`

**Interfaces:**
- Consumes: the static contracts from Task 1 and DTK marker `${TORCH_INSTALL_PREFIX}/include/ATen/dtk_macros.h`.
- Produces: CMake configuration that selects DTK-safe compilation without changing upstream ROCm/CUDA behavior.

- [ ] **Step 1: Compare each RC4 CMake hunk to main**

Classify each hunk from `a20aa339f` as upstream-equivalent, still required, or obsolete. Retain no hunk whose behavior is already present in main.

- [ ] **Step 2: Add DTK architecture and compiler detection**

Extend the supported HIP architectures with `gfx928;gfx936;gfx938`. After `find_package(Torch REQUIRED)`, detect `ATen/dtk_macros.h`, define `DCU_ASM`, and disable the torch stable ABI target version only for DTK torch.

- [ ] **Step 3: Preserve main's stable ABI behavior outside DTK**

Guard stable-libtorch target definitions with the DTK-specific boolean. CUDA and upstream ROCm must retain their main behavior.

- [ ] **Step 4: Use the DTK HIP compiler limit**

Add `--gpu-max-threads-per-block=1024` to the HIP compiler flags in `cmake/utils.cmake` without changing CUDA flags.

- [ ] **Step 5: Keep bundled triton kernels CUDA-only for DTK**

Do not globally remove main's ROCm support. Gate the external `triton_kernels` project so DTK uses its toolchain package while non-DTK main behavior remains explicit and reviewable.

- [ ] **Step 6: Run the static contract**

```bash
python -m pytest -q tests/das/test_dtk_build_contract.py::test_cmake_configures_dtk_architectures_and_compiler_mode
```

Expected: PASS.

- [ ] **Step 7: Commit CMake compatibility**

```bash
git add CMakeLists.txt cmake/utils.cmake
git commit -m "build: configure official main for DTK"
```

---

### Task 3: Adapt Changed Stable-Libtorch Kernels

**Files:**
- Modify: `/models/.worktrees/vllm-main-58ad1f3-hcu/csrc/libtorch_stable/activation_kernels.cu`
- Modify: `/models/.worktrees/vllm-main-58ad1f3-hcu/csrc/libtorch_stable/fused_qknorm_rope_kernel.cu`
- Modify: `/models/.worktrees/vllm-main-58ad1f3-hcu/csrc/libtorch_stable/quantization/gptq/q_gemm.cu`

**Interfaces:**
- Consumes: `DCU_ASM` from Task 2.
- Produces: DTK-safe source selection while preserving main's newer activation implementation.

- [ ] **Step 1: Map main activation call sites by semantic condition**

Locate every main call site that selects `VecTraits<true>::ARCH_MAX_VEC_SIZE` using CUDA toolkit and compute capability conditions. Do not apply RC4 line offsets because this file changed by more than 300 lines after RC4.

- [ ] **Step 2: Add one DTK-aware activation predicate**

Define one source-level predicate or constexpr owner for DTK versus CUDA toolkit support, and use it at every semantically equivalent max-vector selection site. Avoid duplicating independent `DCU_ASM` branches.

- [ ] **Step 3: Guard qknorm and GPTQ compatibility definitions**

Prevent the pre-ROCm-7 `__syncwarp` shim and GPTQ `compat.cuh` include from colliding with DTK definitions. Keep both paths unchanged for non-DTK builds.

- [ ] **Step 4: Run the kernel source contract**

```bash
python -m pytest -q tests/das/test_dtk_build_contract.py::test_dtk_kernel_compatibility_guards_are_present
```

Expected: PASS.

- [ ] **Step 5: Commit kernel compatibility**

```bash
git add csrc/libtorch_stable/activation_kernels.cu csrc/libtorch_stable/fused_qknorm_rope_kernel.cu csrc/libtorch_stable/quantization/gptq/q_gemm.cu
git commit -m "build: select DTK-safe stable kernels"
```

---

### Task 4: Add Reproducible Main Wheel Versioning and Build Helper

**Files:**
- Modify: `/models/.worktrees/vllm-main-58ad1f3-hcu/setup.py`
- Create: `/models/.worktrees/vllm-main-58ad1f3-hcu/tools/build_das_hcu_wheel.sh`
- Create: `/models/.worktrees/vllm-main-58ad1f3-hcu/docs/sourcefind/hcu_delta_manifest.md`

**Interfaces:**
- Consumes: setuptools-scm's official main version, Git HEAD, `ROCM_HOME/.info/rocm_version`, and installed torch metadata.
- Produces: a PEP 440-compatible DAS wheel version, wheel checksum, provenance file, and auditable core delta manifest.

- [ ] **Step 1: Add a failing version contract for main**

Require the DAS version function to accept the SCM-derived base version and append `das.<source-sha>.dtk<digits>` using `+` when no local segment exists and `.` when one already exists. Reject missing Git worktree or DTK version metadata. Do not embed `0.29.0rc4` or infer an unreleased `v0.30.0` tag.

- [ ] **Step 2: Implement DTK wheel version derivation**

Detect DTK torch with `include/ATen/dtk_macros.h`. Derive the base from the same `get_version()` result main uses, append source and DTK identity, set `SETUPTOOLS_SCM_PRETEND_VERSION`, and write `vllm/_version.py` once.

- [ ] **Step 3: Preserve main package selection outside DTK**

Apply CUDA-only exclusions for bundled Triton artifacts only under the DTK/HIP condition required by the container. Do not change ordinary main CUDA packaging.

- [ ] **Step 4: Implement the idempotent build helper**

The script must use `set -euo pipefail`, validate `librt.so.1`, create the missing unversioned link, run `setup.py bdist_wheel`, choose the newest `vllm-*.whl`, produce SHA256, and write full provenance. It must support temporary test overrides through `VLLM_DAS_LIBRT_LINK_PATH`, `VLLM_DAS_LIBRT_TARGET`, and `VLLM_DAS_DTK_VERSION`.

- [ ] **Step 5: Write the main delta manifest**

Record official SHA, every changed official file, reason retained, owner, and validation. Explicitly state that runtime attention, model, scheduler, MoE, and quantization behavior belongs to official vLLM or the plugin.

- [ ] **Step 6: Run all DAS build contracts**

```bash
python -m pytest -q tests/das/test_dtk_build_contract.py tests/das/test_build_das_hcu_wheel.py
```

Expected: all tests PASS with no hardcoded RC4 assertion.

- [ ] **Step 7: Commit packaging and provenance**

```bash
git add setup.py tools/build_das_hcu_wheel.sh docs/sourcefind/hcu_delta_manifest.md tests/das
git commit -m "build: package traceable main DTK wheel"
```

---

### Task 5: Build, Install, and Smoke-Test the Wheel

**Files:**
- Produce: `/models/artifacts/vllm-main-58ad1f3-hcu/*.whl`
- Produce: `/models/artifacts/vllm-main-58ad1f3-hcu/*.whl.sha256`
- Produce: `/models/artifacts/vllm-main-58ad1f3-hcu/*.whl.provenance`
- Produce: `/models/.installs/vllm-main-58ad1f3-hcu-<source-sha>/`

**Interfaces:**
- Consumes: source and build helper from Tasks 2-4.
- Produces: the installed vLLM prerequisite for the plugin migration plan.

- [ ] **Step 1: Record toolchain identity**

```bash
python -c 'import platform, torch; print(platform.python_version()); print(torch.__version__); print(torch.version.hip)'
cat "${ROCM_HOME:-/opt/rocm}/.info/rocm_version"
```

- [ ] **Step 2: Build the wheel with bounded parallelism**

```bash
cd /models/.worktrees/vllm-main-58ad1f3-hcu
MAX_JOBS=16 CMAKE_BUILD_PARALLEL_LEVEL=16 \
  bash tools/build_das_hcu_wheel.sh /models/artifacts/vllm-main-58ad1f3-hcu
```

Expected: one new wheel plus matching checksum and provenance files.

- [ ] **Step 3: Verify the checksum before installation**

```bash
cd /models/artifacts/vllm-main-58ad1f3-hcu
sha256sum -c *.whl.sha256
```

Expected: `OK` for the newly built wheel.

- [ ] **Step 4: Install without dependencies into an isolated target**

```bash
python -m pip install --no-deps --target /models/.installs/vllm-main-58ad1f3-hcu-$(git -C /models/.worktrees/vllm-main-58ad1f3-hcu rev-parse --short=7 HEAD) /models/artifacts/vllm-main-58ad1f3-hcu/vllm-*.whl
```

- [ ] **Step 5: Smoke-test only the installed artifact**

Set `PYTHONNOUSERSITE=1` and put only the isolated target first on
`PYTHONPATH`. Import `vllm`, `vllm._C`, and the main attention/KV-cache
interfaces needed by the plugin. Print `vllm.__version__` and module paths;
all paths must resolve under the isolated target.

- [ ] **Step 6: Run the build contracts against final source**

```bash
python -m pytest -q tests/das
```

Expected: all PASS.

---

### Task 6: Publish OpenDAS Main Baseline

**Files:**
- Modify: no source files.
- Publish: SourceFind branch `main-58ad1f3-hcu`.

**Interfaces:**
- Consumes: committed source, passing tests, installed wheel, and provenance.
- Produces: remote prerequisite consumed by the plugin migration.

- [ ] **Step 1: Verify branch ancestry and bounded delta**

```bash
git merge-base --is-ancestor 58ad1f3b8973b23943107b51230d594050b42ec3 HEAD
git diff --name-status 58ad1f3b8973b23943107b51230d594050b42ec3..HEAD
```

Expected: only the files declared by this plan and its tests/documentation.

- [ ] **Step 2: Push the branch**

```bash
git push --set-upstream origin main-58ad1f3-hcu
```

- [ ] **Step 3: Record the published contract for the next plan**

Capture remote branch SHA, wheel filename, SHA256, provenance path, installed
target, Python version, torch version, and DTK version. The plugin plan may
start only after these values are available.
