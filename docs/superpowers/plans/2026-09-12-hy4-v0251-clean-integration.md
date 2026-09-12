# HYV4 v0.25.1 Clean Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reconstruct the complete reviewed HYV4 capability set from PR #34 on the current `v0.25.1` branch without replacing the branch's authoritative AITER, SlimQuant, LightOp, FP8 KV, PCP/EP, or DeepEP implementations.

**Architecture:** Start from the clean target and port behavior in vertical slices: registration/parsers, target model, quantized execution, MTP, parallelism, EPLB/P-D, and explicit W4A8 formats. Each slice begins with a failing current-contract test and ends with focused regression, diff review, and a commit. The final artifact is validated against an isolated pinned vLLM wheel and the real Channel-FP8 checkpoint before code review and MR creation.

**Tech Stack:** Python 3.10, PyTorch 2.11 DTK 26.04, vLLM 0.25.1 DAS185, vllm-plugin-das, pytest, Transformers, compressed-tensors, HCU/HIPC, AITER, LightOp, DeepEP, DeepGEMM, SlimQuant, EvalScope, GitHub pull requests.

**Spec:** `docs/superpowers/specs/2026-09-12-hy4-v0251-clean-integration-design.md`

## Global Constraints

- Target base is exactly `bc9329029ac14216b671793d49f5b0f47a3b5c7f` until the final pre-push target refresh.
- Historical source is PR #34 at `c25d6f87381ba8385083a22e9cdab3a2abfba032`; it is a behavior reference, not a branch to merge.
- Framework ABI is `vllm==0.25.1+das185.dtk2604.torch2110.2608171710.g7b108a`.
- Current `v0.25.1` owns AITER, SlimQuant, LightOp, native FP8 KV, PCP/EP, and DeepEP policy.
- Do not edit installed vLLM source or model files.
- Do not set `VLLM_PLUGINS=hcu`; leave it unset so platform, model, and operator entry points all load.
- Every production behavior change requires a test that is observed failing for the intended reason before implementation.
- Localhost probes and evaluations clear upper/lower-case proxy variables and set both `NO_PROXY` forms.
- Runtime validation uses `/models/Hy4-preview-Channel-FP8-w8a8` and never kills unrelated service processes.
- Custom/native GPTQ W4A8 and two-node Mooncake remain explicitly hardware-unvalidated unless matching assets become available.
- Review the full diff before and after every commit; Critical and Important findings block push and MR creation.

---

### Task 1: Freeze and verify the isolated vLLM baseline

**Files:**
- Create outside repository: `/models/artifacts/hy4-v0251-clean/`
- Create outside repository: `/models/.installs/vllm-0.25.1-das185-g7b108a-hy4-clean/`
- Inspect: `pyproject.toml`
- Inspect: `tools/run_patch_tests.py`

**Interfaces:**
- Consumes: frozen target commit and requested package index.
- Produces: `VLLM_TARGET_ROOT` containing the pinned vLLM distribution and a provenance record used by every later command.

- [ ] **Step 1: Download the exact wheel without modifying the active environment**

```bash
mkdir -p /models/artifacts/hy4-v0251-clean
python3 -m pip download --no-deps \
  --dest /models/artifacts/hy4-v0251-clean \
  'vllm==0.25.1+das185.dtk2604.torch2110.2608171710.g7b108a' \
  -i https://pypi.sourcefind.cn/nightly/dtk/
sha256sum /models/artifacts/hy4-v0251-clean/vllm-0.25.1+das185*.whl
```

Expected: exactly one CPython 3.10 Linux wheel and one SHA-256 line.

- [ ] **Step 2: Install the wheel into a new isolated target**

```bash
python3 -m pip install --no-deps \
  --target /models/.installs/vllm-0.25.1-das185-g7b108a-hy4-clean \
  /models/artifacts/hy4-v0251-clean/vllm-0.25.1+das185*.whl
```

- [ ] **Step 3: Prove both import roots and versions**

```bash
export VLLM_TARGET_ROOT=/models/.installs/vllm-0.25.1-das185-g7b108a-hy4-clean
export PLUGIN_ROOT=/models/vllm-plugin-das/.worktrees/feat-hy4-v0251-clean
PYTHONNOUSERSITE=1 PYTHONPATH="$VLLM_TARGET_ROOT:$PLUGIN_ROOT" python3 - <<'PY'
import importlib.metadata as md
import pathlib
import torch
import vllm
import vllm_hcu

target = pathlib.Path('/models/.installs/vllm-0.25.1-das185-g7b108a-hy4-clean').resolve()
plugin = pathlib.Path('/models/vllm-plugin-das/.worktrees/feat-hy4-v0251-clean').resolve()
assert pathlib.Path(vllm.__file__).resolve().is_relative_to(target)
assert pathlib.Path(vllm_hcu.__file__).resolve().is_relative_to(plugin)
assert md.version('vllm') == '0.25.1+das185.dtk2604.torch2110.2608171710.g7b108a'
print('vllm=' + vllm.__file__)
print('vllm_hcu=' + vllm_hcu.__file__)
print('torch=' + torch.__version__)
PY
```

- [ ] **Step 4: Run the clean-branch baseline suites**

```bash
PYTHONNOUSERSITE=1 PYTHONPATH="$VLLM_TARGET_ROOT:$PLUGIN_ROOT" \
  python3 tools/run_patch_tests.py --suite contract -- -q
PYTHONNOUSERSITE=1 PYTHONPATH="$VLLM_TARGET_ROOT:$PLUGIN_ROOT" \
  python3 -m vllm_hcu.doctor
```

Expected: baseline exits zero. If it does not, preserve the full failure and investigate import or base compatibility before porting HYV4.

- [ ] **Step 5: Record immutable model provenance**

```bash
sha256sum \
  /models/Hy4-preview-Channel-FP8-w8a8/config.json \
  /models/Hy4-preview-Channel-FP8-w8a8/model.safetensors.index.json
git rev-parse HEAD origin/v0.25.1 origin/pr/34-current
```

No repository commit is created for this environment-only task.

---

### Task 2: Register HYV4 configuration and serving parsers

**Files:**
- Create: `vllm_hcu/models/hy_v4/__init__.py`
- Create: `vllm_hcu/models/hy_v4/config.py`
- Create: `vllm_hcu/reasoning/__init__.py`
- Create: `vllm_hcu/reasoning/hy_v4_reasoning_parser.py`
- Create: `vllm_hcu/tool_parsers/__init__.py`
- Create: `vllm_hcu/tool_parsers/hy_v4_tool_parser.py`
- Create: `vllm_hcu/patch/platform/core_fix/register_hy_v4_reasoning_parser.py`
- Create: `vllm_hcu/patch/platform/core_fix/register_hy_v4_tool_parser.py`
- Modify: `vllm_hcu/models/__init__.py`
- Modify: `vllm_hcu/patch/platform/core_fix/__init__.py`
- Test: `tests/hy_v4/test_registration.py`
- Test: `tests/hy_v4/test_parsers.py`
- Test: `tests/models/hy_v4/test_registration.py`

**Interfaces:**
- Consumes: current `ModelRegistry`, Transformers auto-config, vLLM reasoning parser manager, and tool parser manager.
- Produces: `HYV4Config`, `register_hy_v4_config()`, lazy model exports, parser names `hy_v4`, and idempotent exact-import registration callbacks.

- [ ] **Step 1: Add registration tests before model code**

```python
def test_hyv4_config_registration_is_idempotent():
    register_hy_v4_config()
    register_hy_v4_config()
    loaded = AutoConfig.for_model("hy_v4")
    assert isinstance(loaded, HYV4Config)


def test_hyv4_architectures_are_registered():
    register_model()
    assert ModelRegistry.is_model_supported("HYV4ForCausalLM")
    assert ModelRegistry.is_model_supported("HYV4MTPModel")
```

Add parser cases for `no_think`, split reasoning boundaries, auto/required/named tools, an ordinary `<` character, multiple tool calls in one delta, and incremental JSON arguments.

- [ ] **Step 2: Run RED registration and parser tests**

```bash
PYTHONNOUSERSITE=1 PYTHONPATH="$VLLM_TARGET_ROOT:$PLUGIN_ROOT" \
  pytest -q tests/hy_v4/test_registration.py \
           tests/hy_v4/test_parsers.py \
           tests/models/hy_v4/test_registration.py
```

Expected: collection/import failures because `vllm_hcu.models.hy_v4` and both parser modules do not exist.

- [ ] **Step 3: Port the final parser/config behavior against current managers**

Port the complete final definitions of `HYV4Config`,
`register_hy_v4_config()`, `HYV4ReasoningExtractor`, `HYV4ReasoningParser`,
`HYV4ToolExtractor`, and `HYV4ToolParser` from the historical source paths
listed above. `HYV4Config.model_type` must equal `"hy_v4"`, and registration
must treat an already-registered identical class as success while rejecting a
different owner. Use PR #34 only for HYV4 token/state semantics. Retain current
target manager registration and exact-import callback conventions. Add only
the two HYV4 registry callbacks to `_ORDERED_ADAPTERS`; preserve existing
adapter order.

- [ ] **Step 4: Run GREEN registration/parser tests and lifecycle neighbors**

```bash
PYTHONNOUSERSITE=1 PYTHONPATH="$VLLM_TARGET_ROOT:$PLUGIN_ROOT" \
  pytest -q tests/hy_v4/test_registration.py \
           tests/hy_v4/test_parsers.py \
           tests/models/hy_v4/test_registration.py \
           tests/patch/test_platform_dispatcher.py \
           tests/patch/test_plugin_lifecycle.py
```

- [ ] **Step 5: Review and commit the registration slice**

```bash
git diff --check
git diff -- vllm_hcu/models vllm_hcu/reasoning vllm_hcu/tool_parsers \
  vllm_hcu/patch/platform/core_fix tests/hy_v4 tests/models/hy_v4
git add vllm_hcu/models vllm_hcu/reasoning vllm_hcu/tool_parsers \
  vllm_hcu/patch/platform/core_fix tests/hy_v4 tests/models/hy_v4
git commit -m "feat(hy4): register config and serving parsers"
git show --check --stat HEAD
```

---

### Task 3: Port the HYV4 target model and strict checkpoint contract

**Files:**
- Create: `vllm_hcu/models/hy_v4/hc.py`
- Create: `vllm_hcu/models/hy_v4/attention.py`
- Create: `vllm_hcu/models/hy_v4/hcu_sparse.py`
- Create: `vllm_hcu/models/hy_v4/moe.py`
- Create: `vllm_hcu/models/hy_v4/model.py`
- Create: `vllm_hcu/patch/platform/core_fix/patch_hy_v4_model_arch_config.py`
- Create: `vllm_hcu/patch/platform/core_fix/patch_hy_v4_model_head_dtype.py`
- Create: `vllm_hcu/patch/platform/core_fix/patch_hy_v4_vllm_config.py`
- Create: `vllm_hcu/patch/platform/core_fix/patch_logits_processor_head_dtype.py`
- Modify: `vllm_hcu/patch/platform/core_fix/__init__.py`
- Test: `tests/models/hy_v4/test_hc.py`
- Test: `tests/models/hy_v4/test_attention.py`
- Test: `tests/models/hy_v4/test_moe.py`
- Test: `tests/models/hy_v4/test_weight_loading.py`
- Test: `tests/models/hy_v4/test_runtime_config.py`
- Test: `tests/integration/models/test_hy_v4_smoke.py`

**Interfaces:**
- Consumes: current `FusedMoE`, compressed-tensors methods, sparse-indexer API, HCU FlashMLA sparse backend, Model Runner V2 and PP interfaces.
- Produces: `HYV4ForCausalLM`, `HYV4Model`, `HYV4DecoderLayer`, HYV4 iHC layers, sink-aware sparse attention, and exact loaded-parameter accounting.

- [ ] **Step 1: Port behavioral tests without production files**

The tests must cover these concrete contracts:

```python
assert HYV4Config(enable_ihc=True).hc_mult == 4
assert compute_skip_topk_layers(config) == expected_full_and_shared_layers
assert require_hyv4_sink_backend(sink_capable_backend) is sink_capable_backend
with pytest.raises(ValueError, match="attention sink"):
    require_hyv4_sink_backend(sink_incapable_backend)
```

Add numerical PyTorch references for iHC pre/post/head, router selection and scaling, sink application, full/shared indexer ownership, TP sink slicing, and strict missing/unexpected/duplicate weight detection.

- [ ] **Step 2: Run RED target-model tests**

```bash
PYTHONNOUSERSITE=1 PYTHONPATH="$VLLM_TARGET_ROOT:$PLUGIN_ROOT" \
  pytest -q tests/models/hy_v4/test_hc.py \
           tests/models/hy_v4/test_attention.py \
           tests/models/hy_v4/test_moe.py \
           tests/models/hy_v4/test_weight_loading.py \
           tests/models/hy_v4/test_runtime_config.py \
           tests/integration/models/test_hy_v4_smoke.py
```

Expected: imports fail on missing target-model classes and core adapters.

- [ ] **Step 3: Port HYV4-owned model modules**

Port the complete final definitions of `HYV4HCPreLayer`,
`HYV4HCPostLayer`, `HYV4HCHeadLayer`, `HYV4FlashMLASparseImpl`,
`HYV4FlashMLASparseBackend`, `Indexer`, `HYV4MLAAttention`,
`HYV4MoEFused`, `HYV4Model`, and `HYV4ForCausalLM` from the historical source
paths listed above. Preserve their constructor and forward signatures from the
pinned vLLM-compatible PR head; change an import or call only when the pinned
wheel proves that the current owner differs. Keep model-specific projections,
iHC, sinks, shared-indexer rules, and weight-name normalization. Construct the
current target's `FusedMoE`; do not bring over PR #34 copies of
`aiter_runtime.py`, `slimquant_w4a8.py`, `router_runtime.py`,
`sparse_attn_indexer.py`, or generic LightOp helpers in this task.

- [ ] **Step 4: Add exact core compatibility adapters**

Each new `apply_to_module(module: ModuleType) -> bool` must validate the pinned owner, preserve explicit user choices, apply once, and expose a marker. Register adapters in deterministic dependency order: model classification, vLLM config, head dtype, then logits processor.

- [ ] **Step 5: Run GREEN target model and current-owner regression tests**

```bash
PYTHONNOUSERSITE=1 PYTHONPATH="$VLLM_TARGET_ROOT:$PLUGIN_ROOT" \
  pytest -q tests/models/hy_v4/test_hc.py \
           tests/models/hy_v4/test_attention.py \
           tests/models/hy_v4/test_moe.py \
           tests/models/hy_v4/test_weight_loading.py \
           tests/models/hy_v4/test_runtime_config.py \
           tests/integration/models/test_hy_v4_smoke.py \
           tests/patch/test_platform_dispatcher.py \
           tests/runtime_patch/test_attention_mla_fla_mamba.py \
           tests/runtime_patch/test_sparse_indexer_loading.py
```

- [ ] **Step 6: Review and commit the target model**

```bash
git diff --check
git diff -- vllm_hcu/models/hy_v4 vllm_hcu/patch/platform/core_fix \
  tests/models/hy_v4 tests/integration/models
git add vllm_hcu/models/hy_v4 vllm_hcu/patch/platform/core_fix \
  tests/models/hy_v4 tests/integration/models
git commit -m "feat(hy4): add HCU target model"
git show --check --stat HEAD
```

---

### Task 4: Integrate Channel/Block FP8 with current AITER and LightOp owners

**Files:**
- Create: `vllm_hcu/model_executor/layers/quantization/group_fp8_runtime.py`
- Create: `vllm_hcu/model_executor/layers/quantization/native_fp8_runtime.py`
- Modify only at audited extension points: `vllm_hcu/models/hy_v4/attention.py`
- Modify only at audited extension points: `vllm_hcu/models/hy_v4/model.py`
- Modify only if RED proves a generic gap: `vllm_hcu/model_executor/layers/quantization/compressed_tensors_moe_runtime.py`
- Modify only if RED proves a generic gap: `vllm_hcu/patch/worker/op_opt/patch_compressed_tensors_moe_w8a8_fp8.py`
- Test: `tests/accuracy/test_hcu_kernel_accuracy.py`
- Test: `tests/runtime_patch/test_fp8_channel_triton_product.py`
- Test: `tests/runtime_patch/test_quant_gemm_aiter.py`
- Test: `tests/models/hy_v4/test_attention.py`
- Test: `tests/models/hy_v4/test_weight_loading.py`

**Interfaces:**
- Consumes: current `aiter_moe_request_context`, AITER config lookup/fallback, categorized LightOp quantization, compressed-tensors channel/block scale contracts, and HIPC FP8 cache writing.
- Produces: HYV4 indexer/query quantization and loader support without a second MoE selector.

- [ ] **Step 1: Add RED numerical and ownership tests**

```python
def test_group_fp8_quant_returns_quantized_values_and_fp32_scales():
    q, scale = per_token_group_quant_fp8(x, group_size=128)
    assert q.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz)
    assert scale.dtype == torch.float32
    torch.testing.assert_close(dequantize(q, scale), x, rtol=0.08, atol=0.08)


def test_hyv4_aiter_uses_current_dispatcher(monkeypatch):
    selected = build_hyv4_moe(moe_backend="aiter")
    assert selected.fused_experts.__module__.startswith(
        "vllm_hcu.model_executor.layers.fused_moe"
    )
```

Cover per-channel `[out]`/`[out,1]`, ModelOpt MXFP8, two-dimensional block scales, raw UE8M0 scale bytes, finite output, and target/indexer/router load isolation.

- [ ] **Step 2: Run RED FP8/AITER tests**

```bash
PYTHONNOUSERSITE=1 PYTHONPATH="$VLLM_TARGET_ROOT:$PLUGIN_ROOT" \
  pytest -q tests/models/hy_v4/test_attention.py \
           tests/models/hy_v4/test_weight_loading.py \
           tests/runtime_patch/test_fp8_channel_triton_product.py \
           tests/runtime_patch/test_quant_gemm_aiter.py -k 'hyv4 or group_fp8'
```

Expected: missing HYV4 group/native FP8 functions or unsupported scale layout.

- [ ] **Step 3: Implement only missing quantization seams**

Add `per_token_group_quant_fp8(x: torch.Tensor, group_size: int)` returning a
quantized tensor plus FP32 scale tensor, and
`dynamic_per_token_quant_fp8(x: torch.Tensor)` returning the current native
per-token result. Port the complete function bodies only after comparing their
called operators with current categorized LightOp wrappers. Dispatch through
categorized LightOp when its audited ABI is available and use the current
target fallback otherwise. HYV4 model loaders normalize only HYV4 scale/name
formats. Leave current AITER config search, expert layout, final implementation
selection, and Triton fallback untouched.

- [ ] **Step 4: Run GREEN FP8/AITER and full neighboring suites**

```bash
PYTHONNOUSERSITE=1 PYTHONPATH="$VLLM_TARGET_ROOT:$PLUGIN_ROOT" \
  pytest -q tests/models/hy_v4 \
           tests/runtime_patch/test_fp8_channel_triton_product.py \
           tests/runtime_patch/test_quant_gemm_aiter.py \
           tests/runtime_patch/test_moe_deepep.py \
           tests/accuracy/test_hcu_kernel_accuracy.py -k 'hyv4 or fp8 or aiter'
```

- [ ] **Step 5: Review for forbidden common-code replacement and commit**

```bash
git diff --check
git diff --stat origin/v0.25.1..HEAD
git diff -- vllm_hcu/model_executor/layers/quantization \
  vllm_hcu/model_executor/layers/fused_moe \
  vllm_hcu/models/hy_v4 tests
git add vllm_hcu/model_executor/layers/quantization \
  vllm_hcu/models/hy_v4 tests
git commit -m "feat(hy4): integrate channel and block FP8"
git show --check --stat HEAD
```

---

### Task 5: Add native HYV4 MTP and preserve current PP ownership

**Files:**
- Create: `vllm_hcu/models/hy_v4/mtp.py`
- Create: `vllm_hcu/patch/platform/core_fix/patch_hy_v4_mtp_config.py`
- Modify: `vllm_hcu/models/hy_v4/__init__.py`
- Modify: `vllm_hcu/models/__init__.py`
- Modify: `vllm_hcu/patch/platform/core_fix/__init__.py`
- Modify only for missing HYV4 state: `vllm_hcu/v1/hcu_model_runner_v2.py`
- Test: `tests/models/hy_v4/test_mtp.py`
- Test: `tests/models/hy_v4/test_mtp_config.py`
- Test: `tests/runtime_patch/test_worker_framework_opt.py`

**Interfaces:**
- Consumes: current vLLM `DraftModelSpeculator`, `Sampler`, PPHandler broadcasts, stable speculator-owned buffers, and HYV4 target model classes.
- Produces: `HYV4MTP`, one checkpoint-backed draft layer, target/draft shared top-k buffers, exact MTP quantization config inheritance, and current-contract PP support.

- [ ] **Step 1: Add RED MTP configuration, load, and sampling tests**

```python
def test_mtp_config_extends_one_layer_and_preserves_activation_scheme():
    draft = _make_mtp_layer_config(target_config, target_config.num_hidden_layers)
    assert len(draft.layer_types) == target_config.num_hidden_layers + 1
    assert draft.quantization_config["activation_scheme"] == expected_scheme


def test_target_and_draft_share_topk_storage(model, draft):
    assert model.model.topk_indices.data_ptr() == draft.model.topk_indices.data_ptr()
```

Also cover target/draft loaded parameter completeness, MTP head dtype, raw scale normalization, PP missing-layer handling, and graph-stable input buffer addresses.

- [ ] **Step 2: Run RED MTP tests**

```bash
PYTHONNOUSERSITE=1 PYTHONPATH="$VLLM_TARGET_ROOT:$PLUGIN_ROOT" \
  pytest -q tests/models/hy_v4/test_mtp.py \
           tests/models/hy_v4/test_mtp_config.py \
           tests/runtime_patch/test_worker_framework_opt.py -k 'hyv4 or mtp or pp'
```

- [ ] **Step 3: Implement HYV4-owned draft model and narrow config adapter**

Port the complete final `HYV4SharedHead`,
`HYV4MultiTokenPredictorLayer`, `HYV4MultiTokenPredictor`, and `HYV4MTP`
definitions from `c25d6f8:vllm_hcu/models/hy_v4/mtp.py`. Preserve current
PPHandler ownership. Do not rebind speculator `temperature` or `seeds`; keep
owned buffers and current ordered copies. Add runner code only when the RED
test demonstrates HYV4 top-k/PP state is absent from the current
implementation.

- [ ] **Step 4: Run GREEN MTP and PP regression tests**

```bash
PYTHONNOUSERSITE=1 PYTHONPATH="$VLLM_TARGET_ROOT:$PLUGIN_ROOT" \
  pytest -q tests/models/hy_v4/test_mtp.py \
           tests/models/hy_v4/test_mtp_config.py \
           tests/runtime_patch/test_worker_framework_opt.py \
           tests/runtime_patch/test_glm52_pcp_runner.py
```

- [ ] **Step 5: Review and commit MTP**

```bash
git diff --check
git diff -- vllm_hcu/models/hy_v4 vllm_hcu/v1/hcu_model_runner_v2.py \
  vllm_hcu/patch/platform/core_fix tests/models/hy_v4 \
  tests/runtime_patch/test_worker_framework_opt.py
git add vllm_hcu/models/hy_v4 vllm_hcu/v1/hcu_model_runner_v2.py \
  vllm_hcu/patch/platform/core_fix tests/models/hy_v4 \
  tests/runtime_patch/test_worker_framework_opt.py
git commit -m "feat(hy4): add native MTP"
git show --check --stat HEAD
```

---

### Task 6: Add fail-closed HYV4 PCP/EP and PP2+PCP4

**Files:**
- Modify: `vllm_hcu/patch/platform/core_fix/patch_vllm_config.py`
- Modify: `vllm_hcu/v1/hcu_model_runner_v2.py`
- Modify: `vllm_hcu/v1/pcp_manager.py`
- Modify only for HYV4 ownership: `vllm_hcu/model_executor/layers/sparse_attn_indexer.py`
- Modify: `vllm_hcu/models/hy_v4/hcu_sparse.py`
- Modify: `vllm_hcu/models/hy_v4/model.py`
- Test: `tests/runtime_patch/test_glm52_pcp_config.py`
- Test: `tests/runtime_patch/test_glm52_pcp_manager.py`
- Test: `tests/runtime_patch/test_glm52_pcp_runner.py`
- Test: `tests/runtime_patch/test_sparse_indexer_loading.py`
- Test: `tests/models/hy_v4/test_attention.py`

**Interfaces:**
- Consumes: current `HcuPCPManager`, PCP+EP communicator, batch partition/restore helpers, current sparse-indexer implementation, and PP sampling/broadcast behavior.
- Produces: explicit HYV4 PCP capability, exact topology validation, batch-only restore on non-final PP stages, and a local full-indexer producer boundary.

- [ ] **Step 1: Add RED topology matrix tests**

```python
@pytest.mark.parametrize("override", [
    {"pipeline_parallel_size": 3},
    {"tensor_parallel_size": 2},
    {"prefill_context_parallel_size": 2},
    {"data_parallel_size": 2},
    {"decode_context_parallel_size": 2},
    {"enable_expert_parallel": False},
    {"enforce_eager": False},
])
def test_hyv4_pp2_pcp4_rejects_nearby_topologies(override):
    values = {
        "pipeline_parallel_size": 2,
        "tensor_parallel_size": 1,
        "prefill_context_parallel_size": 4,
        "enable_expert_parallel": True,
        "enforce_eager": True,
    }
    values.update(override)
    config = make_pcp_config(
        architecture="HYV4ForCausalLM",
        **values,
    )
    with pytest.raises(ValueError, match="Hy4 PP2.*PCP4"):
        patch_vllm_config._validate_hcu_pcp_scope(config)


def test_nonfinal_pp_stage_restores_batch_without_hidden_gather():
    assert runner.execute_model_state.hidden_states is None
    runner.sample_tokens(grammar_output=None)
    manager.restore_global_batch.assert_called_once()
    manager.restore_hidden_states.assert_not_called()
```

Add acceptance for exact `PP2/TP1/PCP4/DP1/DCP1/EP/eager/no-spec`, rejection for other architectures and P/D, and indexer tests preserving local PCP slot/top-k ordering.

- [ ] **Step 2: Run RED PCP/PP tests**

```bash
PYTHONNOUSERSITE=1 PYTHONPATH="$VLLM_TARGET_ROOT:$PLUGIN_ROOT" \
  pytest -q tests/runtime_patch/test_glm52_pcp_config.py \
           tests/runtime_patch/test_glm52_pcp_manager.py \
           tests/runtime_patch/test_glm52_pcp_runner.py \
           tests/runtime_patch/test_sparse_indexer_loading.py \
           tests/models/hy_v4/test_attention.py -k 'hyv4 or pp or pcp'
```

- [ ] **Step 3: Implement the narrow HYV4 exception on current PCP code**

Add HYV4 to the existing audited MLA PCP architecture set. Declare
`HYV4FlashMLASparseImpl.supports_pcp = True`. Add a checked global-batch
accessor to `HcuPCPManager` and choose batch-only restoration when a non-final
PP stage has no hidden states. Preserve the exact `VLLM_PP_LAYER_PARTITION=41,37`
requirement and leave every nearby topology fail closed.

- [ ] **Step 4: Run GREEN PCP/PP and entire changed PCP suite**

```bash
PYTHONNOUSERSITE=1 PYTHONPATH="$VLLM_TARGET_ROOT:$PLUGIN_ROOT" \
  pytest -q tests/runtime_patch/test_glm52_pcp_config.py \
           tests/runtime_patch/test_glm52_pcp_manager.py \
           tests/runtime_patch/test_glm52_pcp_runner.py \
           tests/runtime_patch/test_patch_base_communicator_pcp.py \
           tests/runtime_patch/test_sparse_indexer_loading.py \
           tests/models/hy_v4
```

- [ ] **Step 5: Review and commit PCP/PP**

```bash
git diff --check
git diff -- vllm_hcu/patch/platform/core_fix/patch_vllm_config.py \
  vllm_hcu/v1 vllm_hcu/model_executor/layers/sparse_attn_indexer.py \
  vllm_hcu/models/hy_v4 tests/runtime_patch tests/models/hy_v4
git add vllm_hcu/patch/platform/core_fix/patch_vllm_config.py \
  vllm_hcu/v1 vllm_hcu/model_executor/layers/sparse_attn_indexer.py \
  vllm_hcu/models/hy_v4 tests/runtime_patch tests/models/hy_v4
git commit -m "feat(hy4): support fail-closed PCP and PP2"
git show --check --stat HEAD
```

---

### Task 7: Rebuild static EPLB and offline-map support on current MoE contracts

**Files:**
- Create: `vllm_hcu/model_executor/layers/fused_moe/eplb_dispatch.py`
- Create: `vllm_hcu/model_executor/layers/fused_moe/static_eplb.py`
- Create: `vllm_hcu/patch/worker/framework_opt/patch_eplb_communicator.py`
- Create: `vllm_hcu/patch/worker/framework_opt/patch_model_loader_static_eplb.py`
- Create: `vllm_hcu/patch/worker/framework_opt/patch_model_loader_static_eplb_gate.py`
- Create: `vllm_hcu/patch/worker/framework_opt/patch_offline_eplb.py`
- Create: `vllm_hcu/patch/worker/framework_opt/patch_routing_simulator.py`
- Modify: `vllm_hcu/patch/config.py`
- Modify: `vllm_hcu/patch/platform/core_fix/patch_engine_args.py`
- Modify: `vllm_hcu/patch/worker/__init__.py`
- Modify: `vllm_hcu/patch/worker/framework_opt/__init__.py`
- Modify: `vllm_hcu/models/hy_v4/model.py`
- Modify: `vllm_hcu/models/hy_v4/mtp.py`
- Test: `tests/model_executor/layers/fused_moe/test_static_eplb.py`
- Test: `tests/models/static_eplb_test_utils.py`
- Test: `tests/models/test_common_static_eplb_models.py`
- Test: `tests/models/hy_v4/test_eplb.py`
- Test: `tests/runtime_patch/test_eplb_locality_fair_dispatch.py`
- Test: `tests/runtime_patch/test_eplb_routing_simulation.py`
- Test: `tests/runtime_patch/test_model_loader_static_eplb.py`
- Test: `tests/runtime_patch/test_offline_eplb.py`

**Interfaces:**
- Consumes: current `MixtureOfExperts`, `RoutedExperts.weight_loader`, model loader boundary, EPLB state, EP process groups, and HCU sidecar configuration.
- Produces: immutable `StaticEplbPlan`, model-neutral direct load, HYV4 fused-tensor hooks, record/load CLI fields under `--eplb-config`, locality-fair dispatch, and zero post-load rearrangement in static mode.

- [ ] **Step 1: Add RED plan, direct-load, and runtime-state tests**

```python
def test_static_plan_loads_duplicate_logical_experts_into_physical_slots():
    plan = load_static_eplb_plan(path, "HYV4ForCausalLM", 2, 4, 3)
    assert plan.physical_to_logical.tolist() == [[0, 1, 2, 1], [2, 0, 1, 2]]


def test_static_add_model_commits_without_rearrange(state, model):
    state.add_model(model)
    state.rearrange_expert_weights_inplace.assert_not_called()
    state._commit_eplb_maps.assert_called_once()
```

Cover malformed IDs/counts, missing model keys, cache invalidation, cross-rank fingerprint mismatch, two layer maps, split/fused weights and scales, generic models, HYV4 target/MTP, record mode, dynamic mode, and static-step no-op.

- [ ] **Step 2: Run RED EPLB suites**

```bash
PYTHONNOUSERSITE=1 PYTHONPATH="$VLLM_TARGET_ROOT:$PLUGIN_ROOT" \
  pytest -q tests/model_executor/layers/fused_moe/test_static_eplb.py \
           tests/models/test_common_static_eplb_models.py \
           tests/models/hy_v4/test_eplb.py \
           tests/runtime_patch/test_eplb_locality_fair_dispatch.py \
           tests/runtime_patch/test_eplb_routing_simulation.py \
           tests/runtime_patch/test_model_loader_static_eplb.py \
           tests/runtime_patch/test_offline_eplb.py
```

- [ ] **Step 3: Implement common direct load without replacing current AITER/SlimQuant**

Implement the final immutable `StaticEplbPlan` with fields `model_key`,
`source_path`, `source_sha256`, `_map_values`, `num_logical_experts`,
`num_physical_experts`, and `num_redundant_experts`. Retain these exact public
signatures from the historical final design:

```text
load_static_eplb_plan(path, *, model_key, expected_shape,
                      num_logical_experts, num_redundant_experts)
bind_static_eplb_plan(vllm_config, model)
load_static_logical_expert(routed_experts, original_weight_loader, *, param,
                           loaded_weight, weight_name, shard_id,
                           logical_expert_id, return_success)
build_locality_fair_replica_order(logical_to_physical_map, *, ep_rank,
                                  ep_size, num_nodes,
                                  num_physical_experts, seed=42,
                                  layer_offset=0)
```

Wrap the current model-loader boundary before `load_weights`. Attach standard
rows to current `RoutedExperts`; use thin HYV4 hooks only for pre-fused expert
tensors. Preserve current quantization method objects and current AITER layout
installation.

- [ ] **Step 4: Run GREEN EPLB suites and generic MoE neighbors**

```bash
PYTHONNOUSERSITE=1 PYTHONPATH="$VLLM_TARGET_ROOT:$PLUGIN_ROOT" \
  pytest -q tests/model_executor/layers/fused_moe/test_static_eplb.py \
           tests/models/test_common_static_eplb_models.py \
           tests/models/hy_v4/test_eplb.py \
           tests/runtime_patch/test_eplb_locality_fair_dispatch.py \
           tests/runtime_patch/test_eplb_routing_simulation.py \
           tests/runtime_patch/test_model_loader_static_eplb.py \
           tests/runtime_patch/test_offline_eplb.py \
           tests/runtime_patch/test_moe_deepep.py \
           tests/runtime_patch/test_quant_gemm_aiter.py
```

- [ ] **Step 5: Review and commit EPLB**

```bash
git diff --check
git diff -- vllm_hcu/model_executor/layers/fused_moe \
  vllm_hcu/patch vllm_hcu/models/hy_v4 tests
git add vllm_hcu/model_executor/layers/fused_moe vllm_hcu/patch \
  vllm_hcu/models/hy_v4 tests
git commit -m "feat(hy4): add static EPLB direct loading"
git show --check --stat HEAD
```

---

### Task 8: Add HYV4 Mooncake producer-only PCP/P-D contracts

**Files:**
- Modify: `vllm_hcu/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py`
- Modify: `vllm_hcu/patch/platform/core_fix/patch_vllm_config.py`
- Test: `tests/runtime_patch/test_hcu_mooncake_contract.py`
- Test: `tests/runtime_patch/test_glm52_pcp_config.py`

**Interfaces:**
- Consumes: current Mooncake connector scheduler/worker, PP partition metadata, PCP group ownership, and HCU topology validation.
- Produces: checked PP stage ranges, producer-only PCP bootstrap decisions, asymmetric transfer region validation, and fail-closed unsupported PP+PCP+P/D combinations.

- [ ] **Step 1: Add RED Mooncake transfer and topology tests**

```python
def test_hyv4_producer_only_pcp_launches_one_bootstrap_per_global_first_rank():
    assert should_launch_bootstrap_server(producer_config)
    assert not should_launch_bootstrap_server(nonproducer_config)


def test_hyv4_pp_pcp_pd_is_rejected():
    with pytest.raises(ValueError, match="PP.*PCP.*P/D"):
        _validate_hcu_pcp_scope(unsupported_config)
```

Cover custom PP partition strings, stage-local ranges, sender/receiver region lengths, nonnegative block IDs, dense flags, and unchanged PP1 Mooncake behavior.

- [ ] **Step 2: Run RED Mooncake tests**

```bash
PYTHONNOUSERSITE=1 PYTHONPATH="$VLLM_TARGET_ROOT:$PLUGIN_ROOT" \
  pytest -q tests/runtime_patch/test_hcu_mooncake_contract.py \
           tests/runtime_patch/test_glm52_pcp_config.py -k 'mooncake or pd or hyv4'
```

- [ ] **Step 3: Implement narrow connector helpers on the current file**

Retain the current connector as the base. Add only validated helpers for PP
partition normalization, PP stage ranges, producer-only PCP validation, and
bootstrap ownership. Do not replace the whole connector with PR #34's file.

- [ ] **Step 4: Run GREEN Mooncake and PCP regression suites**

```bash
PYTHONNOUSERSITE=1 PYTHONPATH="$VLLM_TARGET_ROOT:$PLUGIN_ROOT" \
  pytest -q tests/runtime_patch/test_hcu_mooncake_contract.py \
           tests/runtime_patch/test_glm52_pcp_config.py \
           tests/runtime_patch/test_glm52_pcp_manager.py
```

- [ ] **Step 5: Review and commit Mooncake support**

```bash
git diff --check
git diff -- vllm_hcu/distributed/kv_transfer \
  vllm_hcu/patch/platform/core_fix/patch_vllm_config.py tests/runtime_patch
git add vllm_hcu/distributed/kv_transfer \
  vllm_hcu/patch/platform/core_fix/patch_vllm_config.py tests/runtime_patch
git commit -m "feat(hy4): add Mooncake P-D topology contracts"
git show --check --stat HEAD
```

---

### Task 9: Adapt explicit HYV4 W4A8 formats to current SlimQuant

**Files:**
- Create: `vllm_hcu/model_executor/layers/quantization/hyv4_w4a8.py`
- Create: `vllm_hcu/model_executor/layers/quantization/hyv4_w4a8_kernels.py`
- Create: `vllm_hcu/model_executor/layers/quantization/hyv4_w4a8_weights.py`
- Create: `vllm_hcu/model_executor/layers/quantization/hyv4_native.py`
- Modify: `vllm_hcu/patch/platform/core_fix/patch_slimquant_registry.py`
- Modify only for name/loader hooks: `vllm_hcu/models/hy_v4/model.py`
- Modify only for name/loader hooks: `vllm_hcu/models/hy_v4/mtp.py`
- Create: `tools/hy_v4/serve_custom_w4a8.sh`
- Create: `tools/hy_v4/serve_native_w4a8.sh`
- Test: `tests/models/hy_v4/test_custom_w4a8.py`
- Test: `tests/accuracy/test_hyv4_native_format.py`
- Test: `tests/accuracy/test_hyv4_w4a8_kernels.py`

**Interfaces:**
- Consumes: current `SlimQuantW4A8Int8Config`, linear method, AITER MoE method, and current SlimQuant registry callback.
- Produces: explicit `hy4-w4a8-custom-v1` and `hy4_w4a8_v1` checkpoint adapters that do not alter ordinary SlimQuant selection.

- [ ] **Step 1: Add RED format isolation and signed INT4 numerical tests**

```python
def test_custom_format_swaps_nibbles_without_changing_signed_values():
    adapted = to_aiter_packing(packed)
    torch.testing.assert_close(unpack_int4(adapted), expected_signed_matrix)


def test_channel_fp8_does_not_select_hyv4_w4a8():
    config = get_quant_config(channel_fp8_hf_config)
    assert not isinstance(config, HYV4W4A8Config)
```

Cover conversion manifests, identity smoothing vectors, nonidentity rejection,
fused expert scale shape, target and MTP names, native UINT8 packing, linear
and routed MoE arithmetic, and explicit checkpoint-format gates.

- [ ] **Step 2: Run RED W4A8 tests**

```bash
PYTHONNOUSERSITE=1 PYTHONPATH="$VLLM_TARGET_ROOT:$PLUGIN_ROOT" \
  pytest -q tests/models/hy_v4/test_custom_w4a8.py \
           tests/accuracy/test_hyv4_native_format.py \
           tests/accuracy/test_hyv4_w4a8_kernels.py
```

- [ ] **Step 3: Implement subclasses over current SlimQuant**

Implement `HYV4W4A8Config` as a subclass of the current
`SlimQuantW4A8Int8Config`, `HYV4W4A8LinearMethod` as a subclass of the current
SlimQuant linear method, `HYV4W4A8MoEMethod` as a subclass of the current
AITER MoE method, and `HYV4NativeW4A8Config` as the native-format subclass.
Port only HYV4 packing, manifest, and name/scale behavior from the historical
files. Keep current SlimQuant behavior for every other format. Reject nonidentity
input smoothing and unsupported checkpoint metadata before kernel execution.
Scripts must leave `VLLM_PLUGINS` unset and make all model-specific overrides
visible on the command line.

- [ ] **Step 4: Run GREEN W4A8, SlimQuant, and HYV4 suites**

```bash
PYTHONNOUSERSITE=1 PYTHONPATH="$VLLM_TARGET_ROOT:$PLUGIN_ROOT" \
  pytest -q tests/models/hy_v4/test_custom_w4a8.py \
           tests/accuracy/test_hyv4_native_format.py \
           tests/accuracy/test_hyv4_w4a8_kernels.py \
           tests/runtime_patch/test_hy_v3_slimquant_weight_loading.py \
           tests/runtime_patch/test_quant_gemm_aiter.py \
           tests/models/hy_v4
```

- [ ] **Step 5: Review and commit W4A8 adapters**

```bash
git diff --check
git diff -- vllm_hcu/model_executor/layers/quantization \
  vllm_hcu/patch/platform/core_fix/patch_slimquant_registry.py \
  vllm_hcu/models/hy_v4 tools/hy_v4 tests
git add vllm_hcu/model_executor/layers/quantization \
  vllm_hcu/patch/platform/core_fix/patch_slimquant_registry.py \
  vllm_hcu/models/hy_v4 tools/hy_v4 tests
git commit -m "feat(hy4): adapt explicit W4A8 checkpoint formats"
git show --check --stat HEAD
```

---

### Task 10: Close the complete automated test surface and document commands

**Files:**
- Create: `docs/hy4_v0251_clean_validation.md`
- Modify: `.github/workflows/configs/hcu-test-map.yaml`
- Modify: `tests/hcu_ci_registry.py`
- Modify: `tests/integration/model_runtime.py`
- Modify: `tests/integration/test_model_runtime_cli.py`

**Interfaces:**
- Consumes: every prior task's committed implementation and test inventory.
- Produces: one final validation document, current CI routing, and complete changed-test closure.

- [ ] **Step 1: Enumerate every test file changed from target**

```bash
git diff --name-only origin/v0.25.1..HEAD | rg '^tests/.*\.py$' \
  > /tmp/hy4-v0251-clean-changed-tests.txt
sed -n '1,240p' /tmp/hy4-v0251-clean-changed-tests.txt
```

- [ ] **Step 2: Run every changed test module, not a filtered subset**

```bash
PYTHONNOUSERSITE=1 PYTHONPATH="$VLLM_TARGET_ROOT:$PLUGIN_ROOT" \
  pytest -q $(tr '\n' ' ' < /tmp/hy4-v0251-clean-changed-tests.txt)
```

Expected: zero collection, setup, fixture, import, and test failures.

- [ ] **Step 3: Run full contract and static gates**

```bash
PYTHONNOUSERSITE=1 PYTHONPATH="$VLLM_TARGET_ROOT:$PLUGIN_ROOT" \
  python3 tools/run_patch_tests.py --suite contract -- -q
PYTHONNOUSERSITE=1 PYTHONPATH="$VLLM_TARGET_ROOT:$PLUGIN_ROOT" \
  python3 -m vllm_hcu.doctor
python3 -m compileall -q vllm_hcu tests
git diff --check origin/v0.25.1..HEAD
```

- [ ] **Step 4: Write the validation document before hardware runs**

Document exact environment prefixes, target-only/MTP/FP8-KV/PP-PCP/DP-EP
serve commands, proxy-free curl and EvalScope commands, expected log evidence,
teardown checks, and explicit W4A8/two-node P-D gaps. Leave result tables with
the state `not run` rather than fabricated numbers; Task 11 replaces each state
with evidence immediately after its run.

- [ ] **Step 5: Add current CI routing and verify registry tests**

```bash
PYTHONNOUSERSITE=1 PYTHONPATH="$VLLM_TARGET_ROOT:$PLUGIN_ROOT" \
  pytest -q tests/hcu_ci_registry.py \
           tests/integration/test_model_runtime_cli.py \
           tests/patch/test_hcu_ci_selector.py
```

- [ ] **Step 6: Review and commit automated closure/documentation**

```bash
git diff --check
git diff -- docs/hy4_v0251_clean_validation.md \
  .github/workflows/configs/hcu-test-map.yaml tests
git add docs/hy4_v0251_clean_validation.md \
  .github/workflows/configs/hcu-test-map.yaml tests
git commit -m "test(hy4): define v0.25.1 validation gates"
git show --check --stat HEAD
```

---

### Task 11: Run the real HYV4 model validation matrix

**Files:**
- Modify after each run: `docs/hy4_v0251_clean_validation.md`
- Create outside repository: `/models/eval-results/hy4-v0251-clean-*/`
- Create outside repository: `/models/validation-logs/hy4-v0251-clean-*/`

**Interfaces:**
- Consumes: final candidate branch, isolated vLLM, eight free HCUs, model checkpoint, EvalScope HumanEval data.
- Produces: service logs, API responses, metrics, accuracy reports, teardown evidence, and a committed validation record.

- [ ] **Step 1: Wait for all eight devices without terminating unrelated jobs**

```bash
rocm-smi --showmeminfo vram --showuse
ps -eo pid,ppid,pgid,user,stat,lstart,cmd --sort=start_time \
  | rg -i 'vllm|EngineCore|Worker' || true
```

Proceed only when the required devices are free. If another user's service
owns memory, wait and report the resource conflict.

- [ ] **Step 2: Run TP8 AITER target-only with default CUDA Graph**

Launch `vllm serve` as the foreground of a PTY session with:

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    -u http_proxy -u https_proxy -u all_proxy \
  NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
  PYTHONNOUSERSITE=1 PYTHONPATH="$VLLM_TARGET_ROOT:$PLUGIN_ROOT" \
  HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  vllm serve /models/Hy4-preview-Channel-FP8-w8a8 \
    --served-model-name hy4-v0251-target \
    --tensor-parallel-size 8 \
    --moe-backend aiter \
    --enable-prefix-caching \
    --gpu-memory-utilization 0.95 \
    --max-model-len 4096 \
    --max-num-seqs 16 \
    --max-num-batched-tokens 4096 \
    --default-chat-template-kwargs '{"reasoning_effort":"no_think"}' \
    --reasoning-parser hy_v4 \
    --enable-auto-tool-choice \
    --tool-call-parser hy_v4 \
    --port 8000
```

Require module-root logs, `HYV4ForCausalLM`, Channel-FP8 linear selection,
AITER config lookup and final implementation, target graph capture, health,
short chat, long prefill, and clean shutdown.

- [ ] **Step 3: Run TP8 AITER MTP3 with default CUDA Graph**

Repeat Step 2 using served name `hy4-v0251-mtp3` and:

```bash
--speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

Require `HYV4MTPModel`, target/draft graph capture, HTTP 200, and nonzero
drafted and accepted tokens with per-position acceptance rates.

- [ ] **Step 4: Run TP8 MTP3 with native FP8 KV and repeated prefix**

Repeat Step 3 with:

```bash
--kv-cache-dtype fp8_e4m3
```

Send the same long semantic prefix at least three times with controlled
suffixes. Require coherent outputs, later nonzero prefix-cache hits, a healthy
server after reuse, and no cache shape/dtype/slot error.

- [ ] **Step 5: Run PP2+PCP4+EP4 DeepEP HT + DeepGEMM**

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    -u http_proxy -u https_proxy -u all_proxy \
  NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
  PYTHONNOUSERSITE=1 PYTHONPATH="$VLLM_TARGET_ROOT:$PLUGIN_ROOT" \
  HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  VLLM_PP_LAYER_PARTITION=41,37 \
  vllm serve /models/Hy4-preview-Channel-FP8-w8a8 \
    --served-model-name hy4-v0251-pp2-pcp4 \
    --tensor-parallel-size 1 \
    --pipeline-parallel-size 2 \
    --prefill-context-parallel-size 4 \
    --enable-expert-parallel \
    --all2all-backend deepep_high_throughput \
    --moe-backend deep_gemm \
    --kv-cache-dtype fp8_e4m3 \
    --enforce-eager \
    --gpu-memory-utilization 0.95 \
    --max-model-len 4096 \
    --max-num-seqs 16 \
    --max-num-batched-tokens 4096 \
    --default-chat-template-kwargs '{"reasoning_effort":"no_think"}' \
    --reasoning-parser hy_v4 \
    --enable-auto-tool-choice \
    --tool-call-parser hy_v4 \
    --port 8000
```

Require two PP stages, four stage-local PCP/EP ranks per stage, DeepEP HT,
DeepGEMM, FP8 KV, short request, 3000+ token prefill, and clean teardown.

- [ ] **Step 6: Run representative DP8+EP8/EPLB validation**

Use explicit DeepEP/DeepGEMM and the documented `--eplb-config` record/load
syntax. Record mode must produce a valid map. Load mode must bind the same map
before weight loading, return HTTP 200, and log zero post-load expert
rearrangements. If current memory or installed DeepEP rejects the topology,
preserve the exact error as a blocking validation result rather than silently
changing backend or model settings.

- [ ] **Step 7: Run HumanEval-32 target-only and MTP3**

Prepare the exact first 32 numeric tasks once and retain the generated JSONL:

```bash
python3 - <<'PY'
import gzip
import json
import pathlib
import urllib.request

root = pathlib.Path('/tmp/hy4-humaneval32')
root.mkdir(exist_ok=True)
url = 'https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz'
with urllib.request.urlopen(url, timeout=60) as response:
    rows = [json.loads(line) for line in gzip.decompress(response.read()).splitlines()]
rows.sort(key=lambda row: int(row['task_id'].split('/')[-1]))
selected = rows[:32]
assert [row['task_id'] for row in selected] == [f'HumanEval/{index}' for index in range(32)]
(root / 'test.jsonl').write_text(
    ''.join(json.dumps(row) + '\n' for row in selected), encoding='utf-8'
)
(root / 'README.md').write_text(
    '---\nconfigs:\n- config_name: openai_humaneval\n'
    '  data_files:\n  - split: test\n    path: test.jsonl\n---\n',
    encoding='utf-8',
)
PY
sha256sum /tmp/hy4-humaneval32/test.jsonl
```

```bash
NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
HTTP_PROXY= HTTPS_PROXY= ALL_PROXY= \
http_proxy= https_proxy= all_proxy= \
evalscope eval \
  --model "$SERVED_MODEL" \
  --api-url http://127.0.0.1:8000/v1 \
  --api-key EMPTY \
  --eval-type openai_api \
  --generation-config '{"temperature":0,"do_sample":false,"max_tokens":2048,"extra_body":{"chat_template_kwargs":{"reasoning_effort":"no_think"}}}' \
  --eval-batch-size 1 \
  --timeout 1800 \
  --limit 32 \
  --datasets humaneval \
  --dataset-args '{"humaneval":{"local_path":"/tmp/hy4-humaneval32"}}' \
  --work-dir "$FRESH_RESULT_DIR" \
  --no-timestamp
```

Use a fresh service and output directory for target-only and MTP3. Verify 32
predictions and 32 reviews, then record Accuracy, Pass@1, errors, finish
reasons, output lengths, per-sample flips, and MTP acceptance.

- [ ] **Step 8: Update the validation record and rerun static checks**

```bash
git diff --check
python3 -m compileall -q vllm_hcu tests
git add docs/hy4_v0251_clean_validation.md
git commit -m "docs(hy4): record v0.25.1 hardware validation"
git show --check --stat HEAD
```

---

### Task 12: Perform final code review, refresh target, and open the MR

**Files:**
- Review: every file in `origin/v0.25.1..HEAD`
- Update only for review fixes: affected production/tests/docs files
- Update through GitHub: new pull request targeting `v0.25.1`

**Interfaces:**
- Consumes: complete committed candidate and all test/model evidence.
- Produces: no unresolved Critical/Important review findings, verified remote branch, and a clean replacement MR.

- [ ] **Step 1: Refresh the target and inspect integration drift**

```bash
git fetch origin v0.25.1
git log --oneline --left-right --cherry-pick origin/v0.25.1...HEAD
git diff --check origin/v0.25.1..HEAD
git diff --stat origin/v0.25.1..HEAD
```

If the target moved, merge the target into the feature branch without
rewriting published history, rerun all changed tests, and repeat any runtime
gate whose owning code changed.

- [ ] **Step 2: Review the exact full diff**

Review for correctness, security, scope, current vLLM/API alignment,
AITER/SlimQuant/LightOp ownership, graph and distributed collectives,
checkpoint loading, numerical semantics, error behavior, performance
regressions, tests, and unsupported validation claims. Report findings ordered
Critical, Important, Minor, with file and line references.

- [ ] **Step 3: Fix review findings through RED/GREEN cycles**

For every Critical or Important issue, add or strengthen a reproducing test,
observe failure, apply the minimal fix, rerun focused tests, rerun the complete
changed-test set, and commit the fix. Re-review the exact new commit.

- [ ] **Step 4: Run the final verification gate fresh**

```bash
PYTHONNOUSERSITE=1 PYTHONPATH="$VLLM_TARGET_ROOT:$PLUGIN_ROOT" \
  pytest -q $(tr '\n' ' ' < /tmp/hy4-v0251-clean-changed-tests.txt)
PYTHONNOUSERSITE=1 PYTHONPATH="$VLLM_TARGET_ROOT:$PLUGIN_ROOT" \
  python3 tools/run_patch_tests.py --suite contract -- -q
PYTHONNOUSERSITE=1 PYTHONPATH="$VLLM_TARGET_ROOT:$PLUGIN_ROOT" \
  python3 -m vllm_hcu.doctor
python3 -m compileall -q vllm_hcu tests
git diff --check origin/v0.25.1..HEAD
git status --short --branch
```

- [ ] **Step 5: Push the reviewed branch and verify the remote SHA**

```bash
git push -u origin feat/hy4-v0251-clean
git rev-parse HEAD
git ls-remote origin refs/heads/feat/hy4-v0251-clean
```

- [ ] **Step 6: Create the replacement MR**

Create one non-draft GitHub pull request with:

```text
head: feat/hy4-v0251-clean
base: v0.25.1
title: feat(hy4): add clean v0.25.1 runtime support
```

The body must link PR #34, summarize preserved features and discarded legacy
ownership, list exact test counts, link/quote result paths, include all real
serve and EvalScope commands, state W4A8/two-node P-D gaps, and include the
final review findings. Obtain credentials only through `git credential fill`;
never print or persist the token.

- [ ] **Step 7: Review the exact remote MR range**

```bash
git fetch origin feat/hy4-v0251-clean v0.25.1
git diff --check origin/v0.25.1..origin/feat/hy4-v0251-clean
git diff --stat origin/v0.25.1..origin/feat/hy4-v0251-clean
```

Query GitHub for `state=open`, `draft=false`, head SHA equal to local `HEAD`,
base `v0.25.1`, and a clean/mergeable state. If mergeability is dirty, resolve
against the latest target and repeat the review and verification gate before
updating the MR.
