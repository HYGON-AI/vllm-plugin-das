# vLLM 0.32 DAS RMSNorm Extension Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a fail-closed, independently validated LightOp RMSNorm capability to the vLLM 0.32 DAS adapter extension.

**Architecture:** Extend the existing single adapter extension with an all-or-nothing preflight for SiluAndMul and RMSNorm, then register a guarded vLLM 0.32 RMSNorm OOT implementation. Unsupported vLLM 0.32 inputs retain the exact upstream path; installed-wheel and live-HCU checks prove the expanded boundary.

**Tech Stack:** Python 3.11+, setuptools/PEP 517, hcu-vllm-adapter extension API v1, PyTorch 2.11, vLLM 0.32 OOT custom ops, LightOp 0.6, pytest.

**Spec:** `docs/superpowers/specs/2026-10-26-vllm032-rmsnorm-extension-design.md`

## Global Constraints

- DAS reference commit is exactly `cd412a0d53b9b517fbc3dcdf96121dc3daa1096c`.
- Official vLLM target is exactly `v0.32.0` at `5df63a7203cdb9328d824a27b682e77c7b9928be`.
- `hcu-vllm-adapter` remains the sole HCU `vllm.platform_plugins` and `vllm.general_plugins` owner.
- The plugin keeps exactly one `hcu_vllm_adapter.extensions` entry point and packages no `vllm_hcu` files.
- Plugin version is `0.2.0.dev0`; Adapter extension API remains fixed at v1.
- Discovery imports no Torch, vLLM, LightOp, device runtime, or legacy DAS module.
- Activation preflights both capabilities before importing either product module.
- Both RMSNorm rollback switches and every unsupported vLLM 0.32 semantic use upstream `forward_cuda`.
- Only installed-wheel numerical evidence on `gfx938` can promote this capability; `gfx92a` remains unvalidated.

## Review Focus

- A future Adapter API must still see plugin API v1 and reject it rather than accepting an unported protocol.
- Weightless and variance-override RMSNorm must never reach LightOp.
- Strided, mismatched-dtype, mismatched-shape, narrow, and batch-invariant inputs must use upstream behavior.
- Missing either LightOp RMSNorm symbol or a changed vLLM RMSNorm OOT API must fail before importing either capability module.
- Repeated/concurrent activation must register both modules once and never duplicate Torch/OOT registrations.

---

### Task 1: Version, policy, and atomic activation

**Files:**
- Modify: `adapter_extension/pyproject.toml`
- Modify: `adapter_extension/src/hcu_vllm_das/version.py`
- Modify: `adapter_extension/src/hcu_vllm_das/environment.py`
- Modify: `adapter_extension/src/hcu_vllm_das/extension.py`
- Modify: `tests/vllm032/test_package_boundary.py`
- Modify: `tests/vllm032/test_extension.py`
- Modify: `tests/vllm032/test_silu_and_mul.py`

**Interfaces:**
- Produces: plugin version `0.2.0.dev0`, `rms_norm_enabled() -> bool`, capability `lightop-rms-norm`, and activation import of `hcu_vllm_das.rms_norm` after all dependency checks.
- Consumes: existing extension API v1, SiluAndMul activation, and `VLLM_HCU_USE_CUSTOM_OPS` policy.

- [ ] **Step 1: Write failing version, environment, capability, dependency-order, retry, and concurrency tests.**

Assert the version bump in metadata and import package; the child switch
`VLLM_HCU_USE_CUSTOM_RMS_NORM`; exact two-capability factory result; resolution
of `lightop.norm` and vLLM layernorm APIs before either product import; failure
on either missing RMSNorm symbol; and once-only import of both product modules.

- [ ] **Step 2: Run Task 1 tests and verify RED on the old version/missing policy and capability.**

Run:
`PYTHONPATH=/models/hcu-vllm-adapter/.worktrees/vllm032-hcu-candidate/src python3 -m pytest tests/vllm032/test_package_boundary.py tests/vllm032/test_extension.py tests/vllm032/test_silu_and_mul.py -q`

Expected: failures naming `0.1.0.dev0`, the missing RMSNorm switch/capability,
and absent activation imports.

- [ ] **Step 3: Implement the version, policy, and ordered activation changes.**

Keep `_IMPLEMENTED_EXTENSION_API_VERSION = 1`. Validate callable LightOp RMSNorm
symbols and `vllm.model_executor.layers.layernorm.RMSNorm.register_oot` before
importing first `hcu_vllm_das.silu_and_mul`, then `hcu_vllm_das.rms_norm`.

- [ ] **Step 4: Run Task 1 tests and verify GREEN.**

- [ ] **Step 5: Commit with message `feat: prepare RMSNorm extension activation`.**

### Task 2: Guarded vLLM 0.32 RMSNorm implementation

**Files:**
- Create: `adapter_extension/src/hcu_vllm_das/rms_norm.py`
- Create: `tests/vllm032/test_rms_norm.py`
- Modify: `tests/vllm032/test_vllm_032_contract.py`
- Modify: `adapter_extension/NOTICE`

**Interfaces:**
- Consumes: Task 1 `rms_norm_enabled()`, LightOp categorized norm APIs, vLLM 0.32 `RMSNorm.register_oot`, `forward_cuda`, and `direct_register_custom_op`.
- Produces: `torch.ops.vllm.hcu_rmsnorm_forward_lightop`, `torch.ops.vllm.hcu_fused_add_rms_norm_lightop`, OOT class `HcuRMSNorm`, and minimum optimized width 64.

- [ ] **Step 1: Write failing fake-runtime operator and official-source contract tests.**

Prove ordinary output allocation/call wiring, fused mutation schema and return
identity, OOT registration, and upstream fallback for master/child switches,
narrow width, batch invariance, missing weight, variance override, non-contiguous
input/residual/weight, shape mismatch, and dtype mismatch. Pin official 0.32
`RMSNorm` constructor fields and `forward_cuda` contract.

- [ ] **Step 2: Run Task 2 tests and verify RED because `hcu_vllm_das.rms_norm` is absent.**

Run:
`PYTHONPATH=/models/hcu-vllm-adapter/.worktrees/vllm032-hcu-candidate/src VLLM_TARGET_ROOT=/models/.cache/vllm-0.32.0-source/extracted/vllm-0.32.0 python3 -m pytest tests/vllm032/test_rms_norm.py tests/vllm032/test_vllm_032_contract.py -q`

Expected: import or subprocess failure naming the missing RMSNorm module.

- [ ] **Step 3: Implement the two custom ops and guarded OOT class.**

Use `lightop.norm.rmsnorm_forward_autograd` and `fused_add_rms_norm`. The fused
custom op mutates and returns `x` and `residual`; fallback calls
`super().forward_cuda(x, residual)` without reimplementing upstream semantics.

- [ ] **Step 4: Run Tasks 1-2 tests and verify GREEN.**

- [ ] **Step 5: Commit with message `feat: port LightOp RMSNorm to vLLM 0.32`.**

### Task 3: Installed-wheel and live-HCU RMSNorm evidence

**Files:**
- Modify: `adapter_extension/tools/verify_installed.py`
- Modify: `tests/vllm032/test_verify_installed.py`
- Create: `tests/vllm032/test_rms_norm_hcu.py`
- Modify: `adapter_extension/README.md`
- Create: `docs/validation/v032-adapter-extension-rmsnorm.md`

**Interfaces:**
- Consumes: Tasks 1-2 final wheel, existing Adapter/vLLM wheels, extension report, and both OOT registry entries.
- Produces: installed attestation for both capabilities and a validation record with exact wheel/RECORD hashes plus ordinary/fused/rollback/fallback numerical results.

- [ ] **Step 1: Write failing installed-verifier and opt-in live-HCU tests.**

Require both capability names and exact plugin-owned OOT classes. The live test
instantiates upstream `RMSNorm`, proves `HcuRMSNorm`, checks ordinary and
fused-add BF16/FP16 results against hand-derived float32 references, and checks
master, child, narrow, weightless, and variance-override fallbacks.

- [ ] **Step 2: Run portable Task 3 tests and verify RED on the missing RMSNorm attestation/module.**

- [ ] **Step 3: Implement verifier and operator documentation changes.**

Document install order, `0.2.0.dev0`, eligibility, both rollback switches,
`gfx938` evidence scope, and mandatory RMSNorm disablement on unvalidated
`gfx92a` production deployments.

- [ ] **Step 4: Build and install the `0.2.0.dev0` wheel over the same audited vLLM/Adapter stack.**

Build with `python3 -m pip wheel --no-deps adapter_extension`; install all wheels
with `--no-deps` and run Python with `-E`. Record SHA256 and installed RECORD
hashes without modifying the historical Silu-only record.

- [ ] **Step 5: Run installed verification and one-device gfx938 numerical tests.**

Use `VLLM_PLUGINS=hcu_adapter`, `HSA_VISIBLE_DEVICES=0`, and
`HIP_VISIBLE_DEVICES=0`. Skips do not count as pass.

- [ ] **Step 6: Run all `tests/vllm032` as the release gate and report retained root-reference failures without rerunning known aborting legacy kernels.**

- [ ] **Step 7: Commit with message `validation: verify vLLM 0.32 RMSNorm extension`.**
