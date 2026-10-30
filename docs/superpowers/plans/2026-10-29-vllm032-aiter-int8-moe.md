# vLLM 0.32 AITER INT8 MoE Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make explicit `--moe-backend aiter` execute vendor AITER INT8 W8A8 MoE kernels on HCU under official vLLM 0.32, with truthful per-shape fallback evidence and end-to-end accuracy validation.

**Architecture:** Extend only the optimization wheel: a narrow vLLM 0.32 oracle seam returns a Plugin-owned INT8 subclass of official `AiterExperts`, while a separate provider module maps official quant metadata to public `aiter.moe` APIs. Adapter lifecycle ownership and official vLLM quant-config/kernel construction remain unchanged.

**Tech Stack:** Python 3.11+, pytest, official vLLM 0.32 modular MoE APIs, `hcu_adapter.extensions` API v1, PyTorch/DTK, vendor `aiter.moe`, compressed-tensors, OpenAI-compatible serving, HumanEval64.

**Spec:** `docs/superpowers/specs/2026-10-29-vllm032-aiter-int8-moe-design.md`

## Global Constraints

- Actual kernels and tuned configs stay in the installed vendor `aiter` wheel; do not copy kernel code.
- `adapter-das` remains the sole vLLM platform/general plugin and compatibility owner; this change modifies no Adapter source.
- Runtime compatibility is epoch-zero vLLM `>=0.32.0,<0.33.0`, guarded by exact interface capability checks rather than a full-wheel equality lock.
- Reuse official `int8_w8a8_moe_quant_config`, `select_int8_moe_backend`, and `make_int8_moe_kernel`; do not recreate W8A8 math or quant metadata.
- Auto backend priority, BF16, FP8, MXFP4, online INT8, Quark, and non-AITER paths retain official behavior unless their existing consumer explicitly requests `aiter`.
- `--moe-backend aiter` is the capability switch; add no environment variable.
- A fallback shape is logged and reported as Triton fallback, never as an AITER provider hit.
- A compatible AITER config followed by a vendor runtime error fails loudly; do not catch it and change providers.
- Keep Graph enabled for the final model gate and require actual AITER config/kernel evidence plus HumanEval64 artifacts.

## Review Focus

- Late extension activation after a v0.32 INT8 consumer imported oracle aliases must fail before mutation; Task 2 pins this.
- Enum replacement must leave auto selection on the official priority list while explicit `aiter` returns only the Plugin expert; Task 2 pins both arms.
- AITER config misses and unsupported shuffle layouts must use the official functional INT8 Triton fallback with the exact scales/zero points; Task 1 pins both cases.
- Vendor failures after successful config selection must propagate rather than silently fallback; Task 1 pins this.
- Wheel/attestation logic must prove the new modules are Plugin-owned and that no legacy `vllm_hcu` or platform/general entry point leaked in; Task 3 pins this.

---

### Task 1: Public AITER INT8 provider adapter

**Files:**
- Create: `adapter_extension/src/hcu_vllm_das/moe/__init__.py`
- Create: `adapter_extension/src/hcu_vllm_das/moe/fused_moe_aiter.py`
- Create: `tests/vllm032/test_aiter_int8_provider.py`

**Interfaces:**
- Consumes: public `aiter.moe.MoeQuantType.W8A8`, `get_aiter_moe_config(...)`, `aiter_moe(...)`; official `vllm.model_executor.layers.fused_moe.fused_moe.fused_experts_impl(...)`; vLLM `FusedMoEConfig` and `FusedMoEQuantConfig` objects.
- Produces: `apply_aiter_int8_moe(*, hidden_states, w1, w2, topk_weights, topk_ids, moe_config, quant_config, activation, expert_map, output_dtype) -> torch.Tensor` and stable once-per-shape provider/fallback logging.

- [ ] **Step 1: Write failing provider contract tests.**

Assert `MoeQuantType.W8A8`, shape values, channel block size zero, activation,
and `use_shuffle=0` passed to config lookup; assert compatible native-layout
configs call `aiter_moe` with official scales/zero points and expert metadata;
assert config miss and `need_shuffle`/`need_shuffle_scale` call official
`fused_experts_impl` with `use_int8_w8a8=True` and
`per_channel_quant=True`; assert vendor invocation errors propagate.

- [ ] **Step 2: Run `pytest -q tests/vllm032/test_aiter_int8_provider.py` and verify RED on the missing module.**

- [ ] **Step 3: Implement the minimal public-API adapter.**

Derive `M`, `E`, `N1`, `N2`, `K`, and `top_k` from the real tensors; normalize
the vLLM activation enum to the AITER string; use a lock-protected set for
once-per-shape evidence; do not import or call `vllm_hcu`.

- [ ] **Step 4: Run the focused test and verify GREEN.**

- [ ] **Step 5: Commit with message `feat(vllm032): add public AITER INT8 MoE provider`.**

### Task 2: vLLM 0.32 INT8 oracle and expert seam

**Files:**
- Create: `adapter_extension/src/hcu_vllm_das/moe/int8_oracle.py`
- Create: `tests/vllm032/test_aiter_int8_oracle.py`
- Create: `tests/vllm032/test_aiter_int8_contract.py`

**Interfaces:**
- Consumes: Task 1 `apply_aiter_int8_moe`; official v0.32 `Int8MoeBackend`, `backend_to_kernel_cls`, `map_int8_backend`, `select_int8_moe_backend`, `convert_to_int8_moe_kernel_format`, `make_int8_moe_quant_config`, `make_int8_moe_kernel`, and `AiterExperts`.
- Produces: `HcuAiterInt8Experts`, `install() -> bool`, process-local `Int8MoeBackend.AITER`, explicit selector mapping, and canonical-weight conversion.

- [ ] **Step 1: Write failing frozen-source and seam tests.**

Pin the official target symbols and parameter names. Test clean explicit
installation, idempotence, original callable preservation, `aiter` mapping,
Plugin expert selection, canonical weights, official quant-config/kernel
factory reuse, unchanged auto priority, unsupported scheme rejection, and
fail-closed late installation when compressed-tensors, online INT8, or Quark
consumer modules are already loaded.

- [ ] **Step 2: Run `pytest -q tests/vllm032/test_aiter_int8_oracle.py tests/vllm032/test_aiter_int8_contract.py` and verify RED.**

- [ ] **Step 3: Implement the guarded oracle install and expert subclass.**

Create the extended enum from existing members plus `AITER`; wrap only
`backend_to_kernel_cls`, `map_int8_backend`, and
`convert_to_int8_moe_kernel_format`; leave official select/quant/kernel bodies
installed. `HcuAiterInt8Experts._supports_quant_scheme` adds only
`(kInt8StaticChannelSym, kInt8DynamicTokenSym)` and its `apply` delegates to
Task 1 for INT8, otherwise to the official parent.

- [ ] **Step 4: Run Task 2 tests and Task 1 tests and verify GREEN.**

- [ ] **Step 5: Commit with message `feat(vllm032): register AITER INT8 MoE backend`.**

### Task 3: Extension activation, wheel ownership, and attestation

**Files:**
- Modify: `adapter_extension/src/hcu_vllm_das/extension.py`
- Modify: `adapter_extension/README.md`
- Modify: `adapter_extension/tools/verify_installed.py`
- Modify: `tests/vllm032/test_extension.py`
- Modify: `tests/vllm032/test_package_boundary.py`
- Modify: `tests/vllm032/test_verify_installed.py`

**Interfaces:**
- Consumes: Task 2 `install() -> bool`; existing `ExtensionSpec` API v1 and RMSNorm activation.
- Produces: declared capability `aiter-int8-moe`, ordered AITER ABI validation and activation, wheel-content proof, and installed-state selector/ownership evidence.

- [ ] **Step 1: Extend lifecycle and package tests first.**

Assert discovery remains free of Torch/vLLM/LightOp/AITER/legacy imports;
activation validates `aiter.moe` public callables before product import; missing
or changed AITER ABI leaves the shared success latch unset; successful
activation imports and installs the oracle exactly once; the wheel contains
both `hcu_vllm_das/moe` files and no `vllm_hcu`; attestation requires both
capabilities and proves the oracle module owner and explicit selector.

- [ ] **Step 2: Run `pytest -q tests/vllm032/test_extension.py tests/vllm032/test_package_boundary.py tests/vllm032/test_verify_installed.py` and verify RED.**

- [ ] **Step 3: Implement ordered activation and attestation.**

Preserve the single `hcu_adapter.extensions` entry point and existing
success-only lock/latch. Validate all RMSNorm and AITER dependencies before
importing either product module, then import RMSNorm and call the oracle
installer. Document that `--moe-backend aiter` activates the new execution
route and that shape fallback is separately logged.

- [ ] **Step 4: Build the extension wheel and run the complete portable v0.32 suite.**

Run:
`env -u PYTHONPATH HCU_ADAPTER_HOST=none PYTHONPATH=/dev/shm/adapter-das-main-20261029 /dev/shm/adapter-das-pr11-vllm032-qwen2-05b/venv/bin/python -m pytest -q tests/vllm032`

Expected: all portable tests pass; HCU-only tests either run on the owned node
or report their existing explicit skip reason and are not counted as live
evidence.

- [ ] **Step 5: Commit with message `feat(vllm032): activate AITER INT8 MoE capability`.**

### Task 4: HCU numeric, Graph, model, and HumanEval64 gates

**Files:**
- Create: `tests/vllm032/test_aiter_int8_hcu.py`
- Create: `docs/validation/vllm032-aiter-int8-moe.md`
- Modify: `adapter_extension/README.md`

**Interfaces:**
- Consumes: installed wheels from Task 3, official Triton functional fallback, vendor AITER public API, `/model/Qwen3.5-35B-A3B-W8A8`, and the existing isolated HumanEval scorer.
- Produces: exact wheel hashes/import roots, numeric and Graph A/B evidence, server log proving AITER selection/config/kernel, deterministic smoke output, HumanEval64 score/artifacts, throughput, and explicit fallback count.

- [ ] **Step 1: Add the live-HCU numerical and capture/replay test.**

Use legal Qwen3.5 channel/token INT8 tensors for M=1, 16, and 128; compare
AITER output with official vLLM Triton using the repository's audited INT8
tolerances; require nonzero output and actual AITER config hits. Capture and
replay with the same addresses but changed input and compare each replay with
its eager reference.

- [ ] **Step 2: Run the live-HCU test on one explicitly owned idle gfx938 device.**

Expected: numeric predicates pass, all three target shapes report AITER rather
than fallback, and replay changes with input while matching eager output.

- [ ] **Step 3: Build/install the exact candidate wheel in the isolated v0.32 environment and run installed attestation.**

Record plugin, adapter, vLLM, Torch, DTK, and AITER versions; Git SHAs; wheel
SHA256; PCI/device architecture; and every imported module origin. Use
`--no-deps` so the audited stack is not replaced.

- [ ] **Step 4: Launch the target checkpoint non-eager with explicit AITER.**

Launch `/model/Qwen3.5-35B-A3B-W8A8` with the established device/TP/model-length
settings and `--moe-backend aiter`. Require full startup and Graph capture,
`Using AITER Int8 MoE backend`, the tuned AITER config source/solution, at
least one actual AITER kernel execution marker, and zero unreported provider
fallbacks before accepting a deterministic smoke request.

- [ ] **Step 5: Run HumanEval64 through the existing OpenAI-compatible client and isolated scorer.**

Keep the established prompt/template, temperature, token budget, stop policy,
and secure scoring sandbox. Preserve raw generations, HTTP status counts,
sample count, finish reasons, token totals, score, elapsed time, and server
logs. A Triton control may diagnose differences but is a separate row.

- [ ] **Step 6: Update validation documentation and rerun regression gates.**

Run the complete `tests/vllm032` suite, installed verifier, one fresh AITER
server smoke, and artifact completeness checks. State any config miss,
fallback, accuracy difference, or unrun topology as a limitation.

- [ ] **Step 7: Commit with message `test(vllm032): validate AITER INT8 MoE path`.**
