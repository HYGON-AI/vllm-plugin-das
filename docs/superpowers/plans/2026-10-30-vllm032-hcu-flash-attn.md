# vLLM 0.32 HCU FlashAttention Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Provide a correct, Graph-capable HCU FLASH_ATTN backend for vLLM 0.32 without copying official orchestration.

**Architecture:** OpenDAS vLLM adds default-preserving protected seams. The Plugin subclasses them to expose dense LBHNC or LBNHC K/V pages to the vendor varlen ABI. The Adapter only routes explicit backend selection to the Plugin class.

**Tech Stack:** Python 3.12, PyTorch/DTK, vLLM 0.32, vendor flash-attn 2.8.x, pytest.

**Spec:** `docs/superpowers/specs/2026-10-30-vllm032-hcu-flash-attn-design.md`

## Global Constraints

- Accept vLLM `>=0.32.0,<0.33.0`; do not exact-lock a wheel or SHA.
- Reuse official allocation, metadata, scheduling, cache writer, and forward logic.
- Keep numerical/provider implementation out of Adapter bridges.
- Use only `flash_attn_varlen_func`; active unsupported FA4-only features fail closed.
- Final serving and accuracy validation must keep Graph enabled.
- Preserve raw HumanEval outputs, finish reasons, token counts, and sample IDs.

## Review Focus

- Padded or cross-layer block strides must not make dense K/V pages overlap.
- K and V storage offsets must remain within the same manager block.
- Active `dynamic_causal`, `mask_mod`, or `aux_tensors` must fail before vendor dispatch.
- Plugin absence must affect only explicit FLASH_ATTN selection, not Adapter startup.
- Graph replay must change output when inputs change while addresses remain fixed.

---

### Task 1: Add default-preserving vLLM seams

**Files:**
- Modify: `vllm/v1/attention/backends/flash_attn.py`
- Test: `tests/v1/attention/test_attention_backends.py`

**Interfaces:**
- Produces: `FlashAttentionImpl._get_kv_cache_views(kv_cache)` and `FlashAttentionImpl._run_flash_attention(**kwargs)`.

- [ ] Write tests asserting the default view seam returns the current packed split views and the kernel seam forwards every keyword unchanged.
- [ ] Run the focused tests and verify they fail because the methods do not exist.
- [ ] Add the two protected methods and route `forward()` plus `do_kv_cache_update()` through them.
- [ ] Run the focused tests and the affected attention-backend suite.
- [ ] Commit the vLLM change.

### Task 2: Implement the Plugin backend

**Files:**
- Create: `adapter_extension/src/hcu_vllm_das/attention/__init__.py`
- Create: `adapter_extension/src/hcu_vllm_das/attention/flash_attn.py`
- Create: `tests/vllm032/test_flash_attention.py`
- Modify: `adapter_extension/README.md`

**Interfaces:**
- Consumes: the two vLLM seams from Task 1.
- Produces: `hcu_vllm_das.attention.flash_attn.HcuFlashAttentionBackend`.

- [ ] Write portable failing tests for LBHNC dense K/V view geometry, block-stride preservation, FA2 version selection, varlen keyword forwarding, and fail-closed FA4 options.
- [ ] Run the focused tests and verify the Plugin backend is missing.
- [ ] Implement the minimal subclass and lazy vendor import needed to pass.
- [ ] Run focused tests and the complete `tests/vllm032` portable suite.
- [ ] Update installation/rollback documentation and commit the Plugin change.

### Task 3: Route Adapter selection to the Plugin

**Files:**
- Modify: `hcu_adapter/bridges/vllm.py`
- Modify: `tests/adapter/test_vllm_bridge.py`

**Interfaces:**
- Consumes: the Plugin backend class path from Task 2.
- Produces: explicit `AttentionBackendEnum.FLASH_ATTN` routing with no eager Plugin import.

- [ ] Write a failing test for the returned Plugin class path and retain the no-eager-import subprocess assertion.
- [ ] Run the focused test and verify it reports the official backend path.
- [ ] Add the class-path-only routing branch.
- [ ] Run Adapter bridge/extension tests and L0.
- [ ] Commit the Adapter change.

### Task 4: Validate native data flow and Graph replay

**Files:**
- Create: `tests/vllm032/test_flash_attention_hcu.py`
- Modify: `docs/validation/vllm032-flash-attn-varlen.md`

**Interfaces:**
- Consumes: the built vLLM, Adapter, and Plugin candidates.
- Produces: live HCU numerical and replay evidence.

- [ ] Add an HCU test that writes dense K/V through the selected official writer and compares prefill/decode vendor output to a Torch reference.
- [ ] Add changing-input Graph capture/replay assertions on fixed tensor addresses.
- [ ] Build isolated wheels and record source SHAs, versions, module origins, and hashes.
- [ ] Run the live HCU tests twice and record tolerances and actual provider paths.
- [ ] Commit the validation test and evidence.

### Task 5: Run model and accuracy gates

**Files:**
- Modify: `docs/validation/vllm032-flash-attn-varlen.md`

**Interfaces:**
- Consumes: candidate wheels from Task 4.
- Produces: Graph serving, TRITON control, HumanEval16, and cleanup evidence.

- [ ] Launch Qwen2-0.5B-Instruct with explicit FLASH_ATTN, block size 64, and Graph enabled.
- [ ] Compare deterministic smoke output and first-token behavior with a matched TRITON control.
- [ ] Run ordered HumanEval16 with matched prompts and generation settings; retain raw responses and finish reasons.
- [ ] Stop only owned services, verify no owned EngineCore remains, and record remaining validation gaps.
- [ ] Review complete diffs before any push or MR update.

### Task 6: Add dual HND/NHD layout support

**Files:**
- Modify: `adapter_extension/src/hcu_vllm_das/attention/flash_attn.py`
- Modify: `tests/vllm032/test_flash_attention.py`
- Modify: `tests/vllm032/test_flash_attention_hcu.py`
- Modify: `adapter_extension/README.md`
- Modify: `docs/validation/vllm032-flash-attn-varlen.md`

**Interfaces:**
- Consumes: vLLM's resolved `KVCacheLayout` and official NHD writer contract.
- Produces: LBHNC/`bhsd` and LBNHC/`bshd` zero-copy read paths.

- [x] Add failing portable tests for both physical stride contracts, vendor
  layout selection, ambiguous geometry, and unsupported-layout failure.
- [x] Implement dual dense views without changing Adapter or vLLM.
- [x] Run BF16/FP16, MHA/GQA, prefill/decode numerical tests on both layouts.
- [x] Run changing-input Graph replay on both layouts.
- [x] Start Graph-enabled model services with explicit HND and NHD and compare
  deterministic responses.
- [x] Build the Plugin wheel and run package gates.
- [ ] Attach final evidence to the MR.
