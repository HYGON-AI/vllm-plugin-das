# vLLM 0.32 DAS Adapter Extension Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and validate an independently installable vLLM 0.32 DAS optimization wheel that loads through `hcu-vllm-adapter` and ports the LightOp `SiluAndMul` path without taking platform ownership.

**Architecture:** Add a self-contained `adapter_extension/` Python project with the `hcu_vllm_das` namespace; do not package or import the legacy root `vllm_hcu` runtime. A dependency-light entry-point factory gates vLLM 0.32 before idempotently registering one vLLM OOT operator, then installed-wheel and live-HCU checks prove the real boundary.

**Tech Stack:** Python 3.11+, setuptools/PEP 517, `packaging.version`, hcu-vllm-adapter extension API v1, PyTorch 2.11, vLLM 0.32 OOT custom ops, LightOp 0.6, pytest.

**Spec:** `docs/superpowers/specs/2026-10-26-vllm032-adapter-extension-design.md`

## Global Constraints

- DAS reference commit is exactly `cd412a0d53b9b517fbc3dcdf96121dc3daa1096c`.
- Official vLLM target is exactly `v0.32.0` at `5df63a7203cdb9328d824a27b682e77c7b9928be`.
- `hcu-vllm-adapter` remains the sole `vllm.platform_plugins` and `vllm.general_plugins` owner.
- The new wheel declares only `hcu_vllm_adapter.extensions` and packages no `vllm_hcu` files.
- Compatible vLLM metadata has epoch zero and release tuple `(0, 32, 0)`.
- Discovery imports no Torch, vLLM, LightOp, device runtime, or legacy DAS module.
- Both `VLLM_HCU_USE_CUSTOM_OPS=0` and `VLLM_HCU_USE_CUSTOM_SILU_AND_MUL=0` must retain the native vLLM path.
- No model, topology, performance, or unported-capability claim may be inferred from this slice.

## Review Focus

- A wheel containing any `vllm_hcu/` path must fail the package-boundary test in Task 1.
- A malformed, epoch-shifted, or adjacent vLLM version must fail before registration in Task 2.
- Missing LightOp or a changed vLLM OOT API must leave activation unlatched and report failure in Task 2.
- Repeated or concurrent activation must register the capability once in Task 2.
- Either rollback switch must bypass LightOp and call the vLLM native path in Task 3.

---

### Task 1: Optimization-only wheel boundary

**Files:**
- Create: `adapter_extension/pyproject.toml`
- Create: `adapter_extension/README.md`
- Create: `adapter_extension/src/hcu_vllm_das/__init__.py`
- Create: `adapter_extension/src/hcu_vllm_das/version.py`
- Create: `tests/vllm032/test_package_boundary.py`

**Interfaces:**
- Produces: distribution `hcu-vllm-das-plugin`, import package `hcu_vllm_das`, `__version__`, `TARGET_VLLM_RELEASE`, `VLLM_UPSTREAM_COMMIT`, and one entry point named `das_hcu` in group `hcu_vllm_adapter.extensions`.
- Consumes: no root `vllm_hcu` package and no runtime framework imports.

- [ ] **Step 1: Write failing package-boundary tests.**

Assert the project metadata has only the adapter extension group, the target
release/commit constants are exact, importing `hcu_vllm_das` is framework-free,
and a built wheel contains only `hcu_vllm_das/` code with no `vllm_hcu/` path.

- [ ] **Step 2: Run `python3 -m pytest tests/vllm032/test_package_boundary.py -q` and verify RED because `adapter_extension/` does not exist.**

- [ ] **Step 3: Implement the minimal src-layout package and metadata.**

Use static project version `0.1.0.dev0`, Python `>=3.11,<3.15`, dependencies
`hcu-vllm-adapter>=0.1.0.dev0` and `packaging>=23`, and entry-point value
`hcu_vllm_das.extension:extension_spec`. Do not declare vLLM or LightOp as pip
dependencies because production installs audited local wheels with `--no-deps`.

- [ ] **Step 4: Run the focused test and verify GREEN.**

- [ ] **Step 5: Commit with message `feat: add vLLM 0.32 DAS wheel boundary`.**

### Task 2: Adapter extension lifecycle

**Files:**
- Create: `adapter_extension/src/hcu_vllm_das/extension.py`
- Create: `tests/vllm032/test_extension.py`

**Interfaces:**
- Consumes: `hcu_vllm_adapter.extensions.ExtensionContext` and `ExtensionSpec`, plus Task 1 version constants.
- Produces: `extension_spec() -> ExtensionSpec`, `activate_extension(context: ExtensionContext) -> None`, and capability `lightop-silu-and-mul`.

- [ ] **Step 1: Write failing lifecycle tests.**

Test exact factory fields; no framework import during discovery; acceptance of
`0.32.0`, prerelease/dev/post/local builds; rejection of `None`, malformed,
epoch-shifted, 0.31, and 0.33 versions; LightOp resolution before vLLM module
registration; retry after a failed activation; and once-only activation under
repeated and concurrent calls.

- [ ] **Step 2: Run `PYTHONPATH=/models/hcu-vllm-adapter/.worktrees/vllm032-hcu-candidate/src python3 -m pytest tests/vllm032/test_extension.py -q` and verify RED on the missing module.**

- [ ] **Step 3: Implement dependency-light discovery and ordered activation.**

Use a module lock plus a success-only latch. Validate with
`packaging.version.Version`; require callable
`lightop.activation.silu_and_mul_opt`; validate the target
`vllm.model_executor.layers.activation.SiluAndMul.register_oot` and
`vllm.utils.torch_utils.direct_register_custom_op`; import
`hcu_vllm_das.silu_and_mul` last.

- [ ] **Step 4: Run the focused test and verify GREEN.**

- [ ] **Step 5: Commit with message `feat: add DAS adapter extension lifecycle`.**

### Task 3: vLLM 0.32 LightOp SiluAndMul capability

**Files:**
- Create: `adapter_extension/src/hcu_vllm_das/environment.py`
- Create: `adapter_extension/src/hcu_vllm_das/silu_and_mul.py`
- Create: `tests/vllm032/test_silu_and_mul.py`
- Create: `tests/vllm032/test_vllm_032_contract.py`

**Interfaces:**
- Consumes: vLLM 0.32 `SiluAndMul.register_oot`, `forward_cuda`, and `direct_register_custom_op`; LightOp `silu_and_mul_opt(out, input)`.
- Produces: registered op `torch.ops.vllm.hcu_silu_and_mul_lightop`, OOT class `HcuSiluAndMul`, `custom_ops_enabled()`, and `silu_and_mul_enabled()`.

- [ ] **Step 1: Write failing operator and frozen-source contract tests.**

Use lightweight fake Torch/vLLM modules to assert output allocation, LightOp
call wiring, OOT registration, native fallback for each rollback switch, and
idempotent import. Inspect the exact official source root supplied by
`VLLM_TARGET_ROOT` and assert the target class/decorator/custom-op interfaces.

- [ ] **Step 2: Run the tests with `VLLM_TARGET_ROOT=/models/.cache/vllm-0.32.0-source/extracted/vllm-0.32.0 python3 -m pytest tests/vllm032/test_silu_and_mul.py tests/vllm032/test_vllm_032_contract.py -q` and verify RED.**

- [ ] **Step 3: Implement the migrated operator and rollback policy.**

Register the custom op as `hcu_silu_and_mul_lightop`, return a newly allocated
tensor with the last dimension halved, and let `HcuSiluAndMul.forward_hip`
delegate to `forward_cuda` whenever either switch is false.

- [ ] **Step 4: Run Task 3 tests plus Tasks 1-2 tests and verify GREEN.**

- [ ] **Step 5: Commit with message `feat: port LightOp SiluAndMul to vLLM 0.32`.**

### Task 4: Installed-wheel and live-HCU evidence

**Files:**
- Create: `adapter_extension/tools/verify_installed.py`
- Create: `tests/vllm032/test_verify_installed.py`
- Create: `tests/vllm032/test_silu_and_mul_hcu.py`
- Modify: `adapter_extension/README.md`
- Modify: `README.md`
- Create: `docs/validation/v032-adapter-extension-silu.md`

**Interfaces:**
- Consumes: built official vLLM 0.32, adapter, and Task 1 plugin wheels; Task 2 entry point; Task 3 registered OOT implementation.
- Produces: one JSON installed-state attestation and a validation record containing exact wheel hashes, versions, architecture, commands, and numeric result.

- [ ] **Step 1: Write failing verifier and HCU numeric tests.**

The verifier rejects editable/unrecorded installs, any plugin-owned vLLM
platform/general entry point, wrong module origins, wrong release, missing
adapter entry point, inactive extension, or absent `SiluAndMul` OOT registry.
The HCU test compares BF16 LightOp output to a float32 PyTorch reference and
checks both rollback paths.

- [ ] **Step 2: Run portable verifier tests and verify RED.**

- [ ] **Step 3: Implement the installed verifier and documentation.**

Emit one stable JSON object without mutating imports or package files. Document
that only `adapter_extension/` is the vLLM 0.32 install target and the root
package remains 0.28 reference material.

- [ ] **Step 4: Build wheels and create an isolated `--system-site-packages` environment.**

Build the plugin with `python3 -m pip wheel --no-deps adapter_extension`, build
the adapter with `python3 -m pip wheel --no-deps`, install the fixed official
vLLM 0.32 wheel plus both new wheels using `--no-deps`, and record SHA256 for
all three wheels.

- [ ] **Step 5: Run installed verification and one-device HCU numeric A/B.**

Require architecture `gfx938`; run plugin discovery/activation from installed
files only, then run `tests/vllm032/test_silu_and_mul_hcu.py` with one explicitly
owned idle device. Record exact pass/fail output; do not convert skips into a
production pass.

- [ ] **Step 6: Run `python3 -m pytest -q` and report every legacy/reference failure, then run the vLLM 0.32 suite as the release gate.**

The release gate is all `tests/vllm032` tests, installed verifier, wheel-content
inspection, and live HCU numeric tests. Existing 0.28 tests remain reference
evidence and must not be silently deleted or globally skipped.

- [ ] **Step 7: Commit with message `validation: verify vLLM 0.32 DAS extension`.**
