# Standard HY4 VL / DFly on unchanged Core

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Run `/models/hy4` with its standard VL tower and checkpoint-native DFly, while keeping the installed OpenDAS v0.33 Core unchanged.

**Architecture:** Plugin owns the missing HY4 configuration, VL model/processor/renderer and model-scoped compatibility. Adapter retains the generic platform/ABI bridge. Reuse stock Core classes through owned subclasses, constructor factories and existing extension registration; do not edit Core files or fabricate `vllm.*` modules.

**Tech Stack:** Core c440c089, Adapter main 2fc1f16, Plugin v0.33.0; Python3.12 / DTK2610 / existing Torch2.11 SDK.

**Spec:** User instruction on 2026-11-28: “core分支不改，直接改到adapter或者vllm plugin，进行适配，另外需要跑下模型”. The prior Core cc54a5e work is reference material only and is superseded for deployment.

## Global constraints

- Keep the checkpoint's `hy_v4_vl` / standard `hy_v4` identity and original INT4, INT8 and BF16 schemes.
- Native DFly, not generic DFlash/MTP; 5 draft layers and default 6 speculative tokens from block size 7.
- Core wheel Python/native bytes must match the original c440 artifact before and after activation/model testing.
- Installed wheels only; no source overlay or PYTHONPATH in the serving environment.
- First functional deployment: eager TP1/DP16/EP16; PCP/DCP/PP remain one, no KV transfer/EPLB/DBO.
- CPU/import success does not close actual weight loading, image/video inference or nonzero DFly acceptance.

## Review focus

- Configuration and registrations survive spawn without changing foreign models or owners.
- M-RoPE reaches both standard attention fast paths and the indexer on stock Core.
- External top-k avoids registering an unused draft indexer/KV cache.
- Logical top-k snapshots precede physical conversion; draft KV aliases remain rank/request local.
- Default container commands select the same installed 0.33 packages as verification commands.

### Task 1: VL configuration and processing

**Files:** `adapter_extension/src/hcu_vllm_das/models/hy4_standard/{configs,mrope,processor,vision,vl_model,mm_preprocess,vl_cudagraph,renderer}.py`, `hy4_vl_activation.py`, corresponding Plugin CPU tests.

- [ ] Demonstrate failure of real model config/processor/registry on stock Core.
- [ ] Port missing model-level modules to the genuine Plugin namespace and register through existing lifecycle/public interfaces.
- [ ] Provide `set_token_rope`, `token_rope`, `apply_mrope`, `build_mrope_positions` for the attention owner.
- [ ] Verify real configuration save/reload, processor, positions, renderer and model registry on installed stock Core.

### Task 2: Standard target attention and native DFly

**Files:** `models/hy4_standard/{attention,model,dfly_config}.py`, existing `hy4_ct.py`, `hy4_dfly.py`, shared DFly compatibility/bindings and their tests.

- [ ] Write failures showing stock Core lacks the required target/draft contracts.
- [ ] Add owned target/attention subclasses and narrow constructor bindings; retain native forward bodies and parameter names.
- [ ] Move standard draft config derivation to Plugin; remove imports of newly added Core modules.
- [ ] Preserve the real DFly config, INT8 loader, snapshots and DP16 ownership tests; verify ordinary CT/MTP/Flash delegation.

### Task 3: Packaging and deployment

**Files:** `adapter_extension/pyproject.toml`, `adapter_extension/README.md`, `docs/hy4_vl_dfly.md`; deployment receipts outside the repository.

- [ ] Restore the original c440 wheel in both owned containers and record whole-wheel/Core-file hashes.
- [ ] Add the required VL entry point and a complete source/import migration ledger.
- [ ] Build clean-commit Plugin wheels and install both nodes; rebuild Adapter only if a generic missing bridge is demonstrated.
- [ ] Validate parent/spawn, ordinary command imports, original Core bytes, CPU regressions and foreign-model ownership.

### Task 4: Actual model and MR update

- [ ] Confirm GPU ownership and use a complete 16-card node for the actual checkpoint.
- [ ] Start target-only text/image/video requests, then native DFly text/image requests; audit weight/scale loading and nonzero draft/accepted counts.
- [ ] Run the applicable accuracy comparison, keeping generation/template settings identical.
- [ ] Independently review; update MR550 around the final Plugin-only implementation and verification. No Core MR.
