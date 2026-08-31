# HYV4 MTP and boltops iHC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Register and run HYV4 MTP on vLLM v0.25.1 with TP8 MXFP8, and use boltops fused kernels for the HYV4 backbone iHC boundaries.

**Architecture:** Add a device-gated boltops adapter to the existing iHC layers, an exact-import compatibility adapter for HYV4 speculative config rewriting, and a dedicated HYV4 MTP module adapted from PR #54160 to the HCU split-projection and ModelOpt MXFP8 layouts. Preserve the target/draft shared top-k buffer and fail weight loading when any real MTP parameter is unassigned.

**Tech Stack:** Python 3.10, PyTorch/HIP, vLLM v0.25.1, vllm-hcu, boltops, ModelOpt MXFP8, pytest, EvalScope.

**Spec:** `docs/superpowers/plans/2026-08-28-hyv4-mtp-boltops-design.md`

## Global Constraints

- Work in `/models/zb/hy4` and preserve all pre-existing uncommitted changes.
- Target model is `/models/Hy4-preview-FP8-Testing` on TP8 with expert parallel disabled.
- Use `VLLM_MXFP8_EMULATION_DEQUANT_AT_LOAD=0` on gfx938.
- Do not enable iHC inside the MTP block because the checkpoint has no MTP iHC parameters.
- Follow red-green-refactor for every production behavior.

---

### Task 1: boltops iHC dispatch

**Files:**
- Modify: `tests/models/hy_v4/test_hc.py`
- Modify: `vllm_hcu/models/hy_v4/hc.py`

**Interfaces:**
- Consumes: installed `boltops.ihc.ihc_pre`, `ihc_post`, and `ihc_head`.
- Produces: `_use_boltops_ihc(tensor: torch.Tensor) -> bool`; unchanged public HYV4 iHC layer APIs.

- [ ] **Step 1: Write failing dispatch tests**

Add tests that force `_use_boltops_ihc` true, replace each imported boltops
function with a recording function, call the real HYV4 layer, and assert the
exact tensor parameters and scalar eps/magnitude arguments. The production
change caught is bypassing a fused operator or passing a linear module instead
of its FP32 `.weight` tensor.

- [ ] **Step 2: Verify the tests fail for the missing dispatch**

Run: `pytest -q tests/models/hy_v4/test_hc.py -k boltops`

Expected: FAIL because `_use_boltops_ihc` and fused dispatch are absent.

- [ ] **Step 3: Add the minimal device-gated adapter**

Import boltops under `try/except ImportError`; return true only when all three
functions exist and `tensor.device.type == "cuda"`. Call the matching boltops
function at the start of each layer's `forward`, otherwise execute the current
eager body unchanged.

- [ ] **Step 4: Verify fused dispatch and eager numerical behavior**

Run: `pytest -q tests/models/hy_v4/test_hc.py`

Expected: all iHC tests pass, including the pre-existing FP32 reference tests.

### Task 2: HYV4 speculative-config compatibility

**Files:**
- Create: `vllm_hcu/patch/platform/core_fix/patch_hy_v4_mtp_config.py`
- Modify: `vllm_hcu/patch/platform/core_fix/__init__.py`
- Modify: `tests/patch/test_platform_dispatcher.py`
- Create: `tests/models/hy_v4/test_mtp_config.py`

**Interfaces:**
- Consumes: `vllm.config.speculative.SpeculativeConfig.hf_config_override` and `MTPModelTypes`.
- Produces: an idempotent exact-import adapter mapping `hy_v4` to architecture `HYV4MTPModel` and model type `hy_v4_mtp`.

- [ ] **Step 1: Write failing config and inventory tests**

Use a fake speculative module whose original override records delegation. Apply
the adapter and assert HYV4 receives `n_predict`, `hy_v4_mtp`, and
`HYV4MTPModel`, while another model is returned with only the original change.
Assert the platform core inventory contains the new exact target after the
vLLM config adapters.

- [ ] **Step 2: Verify the tests fail because the adapter is absent**

Run: `pytest -q tests/models/hy_v4/test_mtp_config.py`

Expected: FAIL because HYV4 remains `HYV4ForCausalLM`.

- [ ] **Step 3: Implement and register the compatibility adapter**

Validate the target class/method signature, append `hy_v4_mtp` to the runtime
Literal exactly once, wrap the static method exactly once, and register it in
the hand-ordered callback tuple.

- [ ] **Step 4: Verify config behavior**

Run: `pytest -q tests/models/hy_v4/test_mtp_config.py`

Expected: all tests pass with non-HYV4 delegation intact.

### Task 3: HYV4 MTP registry and structural contracts

**Files:**
- Create: `vllm_hcu/models/hy_v4/mtp.py`
- Modify: `vllm_hcu/models/hy_v4/__init__.py`
- Modify: `vllm_hcu/models/__init__.py`
- Modify: `tests/models/hy_v4/test_registration.py`
- Create: `tests/models/hy_v4/test_mtp.py`

**Interfaces:**
- Produces: `HYV4MTP`, `HYV4MultiTokenPredictor`, `HYV4MultiTokenPredictorLayer`, `_extend_layer_types`, `_remap_mtp_quant_exclusions`.
- Consumes: `HYV4DecoderLayer`, target/draft vLLM configs, target top-k buffer, standard vLLM MTP forward/logits/sample signatures.

- [ ] **Step 1: Change the registry test to require the MTP architecture**

Assert registration contains `("HYV4MTPModel", "vllm_hcu.models.hy_v4:HYV4MTP")`
and that lazy package import exports `HYV4MTP` without eagerly importing it
during config-only import.

- [ ] **Step 2: Verify the registry test fails**

Run: `pytest -q tests/models/hy_v4/test_registration.py -k 'registry or import'`

Expected: FAIL because `HYV4MTPModel` is not registered.

- [ ] **Step 3: Add lazy export, registry entry, and minimal MTP classes**

Port the PR structure, force the copied MTP config's `enable_ihc` false, extend
the layer arrays through index 78, use prefix `lm_head`, and keep separate HCU
q/kv projections.

- [ ] **Step 4: Write and run structural tests**

Construct uninitialized/lightweight objects to assert layer extension,
position-zero embedding masking, MTP iHC disablement, spec-step layer reuse,
and `set_topk_indices_buffer` propagation to predictor/indexer/indexer-op/
attention implementation.

Run: `pytest -q tests/models/hy_v4/test_mtp.py -k 'structure or topk or layer'`

Expected before implementation completion: focused assertions fail; after the
minimal implementation: PASS.

### Task 4: ModelOpt MXFP8 MTP weight loading

**Files:**
- Modify: `vllm_hcu/models/hy_v4/mtp.py`
- Modify: `tests/models/hy_v4/test_mtp.py`
- Modify: `tests/models/hy_v4/test_weight_loading.py`

**Interfaces:**
- Produces: `_resolve_fused_expert_param`, `_rewrite_mtp_weight_name`, strict `HYV4MTP.load_weights`.
- Consumes: `_try_load_fp8_indexer_projection`, `_try_load_fp8_router_gate`, `_slice_sink_for_tp`, and fused MoE parameter loaders from the target implementation.

- [ ] **Step 1: Write failing literal mapping tests**

Assert checkpoint `model.mtp_layers.0.eh_proj.weight` maps to
`model.layers.78.eh_proj.weight`, decoder q/kv names gain `.mtp_block`, fused
`gate_up_proj_scale` resolves to `w13_weight_scale`, and wildcard ModelOpt
exclusions preserve `*` after remapping.

- [ ] **Step 2: Verify mapping tests fail**

Run: `pytest -q tests/models/hy_v4/test_mtp.py -k 'weight or exclusion or scale'`

Expected: FAIL for missing helpers/incorrect mappings.

- [ ] **Step 3: Implement minimal strict loader**

Load shared embedding/head, MTP norms/projection, split attention projections,
MXFP8 fused expert weights/scales, FP8/MXFP8 indexer pairs, router gate,
learnable sink TP shard, and generic remaining parameters. Ignore only runtime
`KVCacheScaleParameter`; raise with the complete missing real-parameter list.

- [ ] **Step 4: Verify loader tests and target-loader regression**

Run: `pytest -q tests/models/hy_v4/test_mtp.py tests/models/hy_v4/test_weight_loading.py`

Expected: all tests pass.

### Task 5: Regression and TP8 integration

**Files:**
- Runtime log: `/tmp/hyv4-mtp-tp8.log`
- Eval output: `outputs/hyv4-mtp-humaneval-5/`

**Interfaces:**
- Consumes: complete plugin through `PYTHONPATH=/models/zb/hy4`.
- Produces: an OpenAI-compatible HYV4 service with three-token MTP drafting.

- [ ] **Step 1: Run CPU/static regression**

Run: `python -m compileall -q vllm_hcu/models/hy_v4 vllm_hcu/patch/platform/core_fix`

Run: `pytest -q tests/models/hy_v4 tests/runtime_patch/test_worker_framework_opt.py tests/runtime_patch/test_moe_deepep.py`

Expected: all runnable tests pass; the known source-root-only dispatcher test
may be excluded when `/models/zb/vllm_0251` is absent.

- [ ] **Step 2: Stop the prior non-MTP service and start TP8 MTP**

Use the existing safe PID/port check, then start with
`VLLM_USE_V2_MODEL_RUNNER=1`, `VLLM_MXFP8_EMULATION_DEQUANT_AT_LOAD=0`, TP8,
Triton MoE, `--enforce-eager`, and
`--speculative-config '{"method":"mtp","num_speculative_tokens":3}'`.

- [ ] **Step 3: Validate service behavior and logs**

Poll `/health`, send one no-thinking deterministic arithmetic request, then
five concurrent deterministic requests. Inspect logs for missing parameters,
NaN/Inf, shared-buffer errors, collective errors, and speculative acceptance.

- [ ] **Step 4: Run HumanEval subset**

Run EvalScope against the MTP endpoint with the same first-five case selection
used for the baseline. Expected functional threshold: at least 4/5 pass@1;
record the exact report path and compare with the non-MTP 4/5 result.
