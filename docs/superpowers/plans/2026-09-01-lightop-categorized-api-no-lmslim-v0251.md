# LightOp Categorized API and LMSlim Removal for v0.25.1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the v0.25.1 HCU plugin to LightOp 0.6 categorized APIs, remove production imports and calls of external LMSlim, add a `VLLM_HCU_*` environment bridge, and validate the result with tests and the requested Qwen3.5 W8A8 model.

**Architecture:** Reconcile the already-tested PR #27 migration onto the latest `v0.25.1`, then tighten each existing ownership boundary so required kernels use only categorized LightOp exports. A dependency-light bootstrap translates plugin-owned environment names before LightOp import; exact ABI tests and a final AST boundary audit prevent legacy paths from returning.

**Tech Stack:** Python 3.10, PyTorch 2.11/HIP, vLLM 0.25.1, LightOp 0.6, pytest, Git worktrees, OpenAI-compatible vLLM server.

**Spec:** `docs/superpowers/specs/2026-09-01-lightop-categorized-api-no-lmslim-v0251-design.md`

## Global Constraints

- Base the branch on `origin/v0.25.1` commit `85a4ad5` and target pull requests to `v0.25.1`.
- Treat installed LightOp `0.6.0+das.dtk2604.torch2110.2608171227.g8c835c` and each category's `__all__` as the callable contract.
- Preserve terminal LMSlim call names when the mapping document requires it; in particular use `lightop.gemm_ops.hipblaslt_w8a8_gemm`, not the PR #27 `channelwise` substitution.
- Production code must not import or call external `lmslim` and must not fall back to it.
- Required categorized APIs fail closed; kernel runtime failures never trigger a legacy retry.
- The only allowed top-level LightOp calls are existing `fuse_silu_mul_clamp_quant` and `fuse_silu_mul_clamp_quant_ep`, because installed LightOp 0.6 has no categorized exports for them.
- Do not remove `das-install lmslim`, rename LightOp's internal `_lmslim_native` tree, or edit files under `/usr/local/lib/python3.10/dist-packages`.
- User-facing tuning names use `VLLM_HCU_*`; bridge only to aliases already recognized by installed LightOp.
- Preserve lazy import and module-exchange boundaries; do not pull HCU libraries into CPU-only plugin discovery.
- Follow red-green-refactor for every new behavior and run the stated focused test before each commit.
- Baseline contract evidence is `1162 passed, 49 deselected, 14 warnings` in 266.64 seconds.

## File Structure

### New focused units

- `vllm_hcu/lightop_env.py`: dependency-light normalization, conflict checking, legacy warning, and alias translation before LightOp import.
- `tests/patch/test_lightop_environment.py`: isolated tests for the environment bridge and plugin bootstrap ordering.
- `tests/patch/test_lightop_api_boundary.py`: AST-level production import/call policy with exactly two top-level clamp exceptions.
- `tests/runtime_patch/test_lightop_categorized_api.py`: installed-wheel public export contract.
- `tests/runtime_patch/test_lightop_attention_api.py`: attention/MQA categorized ABI tests.
- `tests/runtime_patch/test_lightop_deepseek_v4_api.py`: main DeepSeek V4 normalization and insertion ABI tests.
- `tests/runtime_patch/test_lightop_deepseek_v4_latest_api.py`: DSpark and current core-fix patch tests added after PR #27.
- `tests/runtime_patch/test_lightop_ops_api.py`: activation, GEMM, norm, quant, tensor, and strict-failure tests.
- `docs/lightop_categorized_api_validation.md`: exact commands and final portable/HCU/model evidence.

### Existing production owners

- Attention: `vllm_hcu/model_executor/layers/attention_runtime.py`, `vllm_hcu/v1/attention/ops/rocm_aiter_mla_sparse.py`.
- Activation/GEMM: `vllm_hcu/ops/{silu_and_mul,fuse_silu_mul_quant}.py` and the three DeepGEMM expert modules.
- MoE: `deep_gemm_utils.py`, `router_runtime.py`, `fuse_moe_gate.py`, `compressed_tensors_moe_marlin.py`, and `patch_moe_align_block_size.py`.
- Norm/quant/tensor: `rms_norm.py`, `gemma_rms_norm.py`, `fuse_rms_norm_quant.py`, `lightop_fp8_runtime.py`, `int8_runtime.py`, and `test_concat.py`.
- DeepSeek V4: `deepseek_v4_attention.py`, `models/deepseek_v4_dspark.py`, and `patch/worker/core_fix/patch_deepseek_v4_attention.py`.
- Bootstrap/config/docs: `vllm_hcu/__init__.py`, `vllm_hcu/platforms/envs.py`, `docker/image-citest-conf-hcu.yaml`, and `docs/pd分离mooncake使用.md`.

---

### Task 1: Reconcile PR #27 onto the current v0.25.1 baseline

**Files:**
- Merge source: `origin/pr-27` at `571bc4c`
- Preserve: `docs/superpowers/specs/2026-09-01-lightop-categorized-api-no-lmslim-v0251-design.md`
- Exclude from result: `docs/superpowers/specs/2026-08-25-lightop-categorized-api-v0251-design.md`
- Exclude from result: `docs/superpowers/plans/2026-08-25-lightop-categorized-api-v0251.md`
- Reconcile: every conflicted production/test file reported by Git

**Interfaces:**
- Consumes: current `v0.25.1` ownership from `85a4ad5` and the categorized ABI work from PR #27.
- Produces: one compilable tree containing the current branch features plus PR #27 categorized modules and tests; strict LMSlim removal is intentionally deferred to later tasks.

- [ ] **Step 1: Record the two parents and start a non-committing merge**

```bash
git rev-parse HEAD
git rev-parse origin/pr-27
git merge --no-commit --no-ff origin/pr-27
```

Expected: either a staged clean merge or conflicts; never resolve a conflict by discarding the latest `v0.25.1` behavior wholesale.

- [ ] **Step 2: Resolve conflicts owner by owner**

For each conflict, inspect both stages before editing:

```bash
git status --short
git show :2:path/to/conflicted.py
git show :3:path/to/conflicted.py
```

Keep current `v0.25.1` additions from PRs #29, #36, #39, and #40. Apply the categorized imports and ABI adaptations from PR #27 around that logic. Remove the superseded 2026-08-25 spec/plan from the merge result; this plan and its 2026-09-01 spec replace them.

- [ ] **Step 3: Verify the reconciled reference layer**

```bash
python -m compileall -q vllm_hcu tests/runtime_patch
python -m pytest -q \
  tests/runtime_patch/test_lightop_categorized_api.py \
  tests/runtime_patch/test_lightop_attention_api.py \
  tests/runtime_patch/test_lightop_deepseek_v4_api.py \
  tests/runtime_patch/test_lightop_ops_api.py
```

Expected: compilation succeeds and all imported PR #27 focused tests pass. Failures caused by a current-branch ABI conflict are fixed here; strict fallback expectations are changed only in their owning later task.

- [ ] **Step 4: Commit the reconciled reference**

```bash
git add -A
git commit -m "refactor: reconcile categorized LightOp API reference"
```

### Task 2: Add the VLLM_HCU LightOp environment bridge

**Files:**
- Create: `vllm_hcu/lightop_env.py`
- Create: `tests/patch/test_lightop_environment.py`
- Modify: `vllm_hcu/__init__.py`
- Modify: `vllm_hcu/platforms/envs.py`
- Modify: `docker/image-citest-conf-hcu.yaml`
- Modify: `docs/pd分离mooncake使用.md`

**Interfaces:**
- Consumes: `MutableMapping[str, str]` compatible with `os.environ`.
- Produces: `configure_lightop_environment(environ: MutableMapping[str, str] | None = None) -> None` and `LightOpEnvironmentError`.
- Produces mappings from four `VLLM_HCU_*` names to installed-LightOp neutral aliases.

- [ ] **Step 1: Write failing normalization, conflict, legacy, and ordering tests**

```python
def test_new_hcu_names_populate_lightop_supported_aliases():
    env = {
        "VLLM_HCU_FUSED_MOE_CHUNK_SIZE": "8192",
        "VLLM_HCU_USE_GLOBAL_MOE_CACHE": "true",
        "VLLM_HCU_USE_FUSED_RMS_QUANT": "1",
        "VLLM_HCU_USE_FUSE_SILU_AND_MUL": "yes",
    }
    configure_lightop_environment(env)
    assert env["VLLM_FUSED_MOE_CHUNK_SIZE"] == "8192"
    assert env["VLLM_USE_GLOBAL_CACHE13"] == "1"
    assert env["USE_FUSED_RMS_QUANT"] == "1"
    assert env["VLLM_USE_FUSE_SILU_AND_MUL"] == "1"


def test_conflicting_hcu_and_dependency_values_fail_closed():
    env = {
        "VLLM_HCU_USE_GLOBAL_MOE_CACHE": "1",
        "VLLM_USE_GLOBAL_CACHE13": "0",
    }
    with pytest.raises(LightOpEnvironmentError, match="conflicting"):
        configure_lightop_environment(env)


def test_legacy_lmslim_value_warns_once_and_bridges(caplog):
    env = {"LMSLIM_USE_GLOBAL_MOE_CACHE": "true"}
    configure_lightop_environment(env)
    configure_lightop_environment(env)
    assert env["VLLM_HCU_USE_GLOBAL_MOE_CACHE"] == "1"
    assert env["VLLM_USE_GLOBAL_CACHE13"] == "1"
    assert caplog.text.count("LMSLIM_USE_GLOBAL_MOE_CACHE is deprecated") == 1
```

Add a subprocess test that sets only `VLLM_HCU_USE_GLOBAL_MOE_CACHE=1`, imports `vllm_hcu`, then imports `lightop.envs`, and asserts `LMSLIM_USE_GLOBAL_MOE_CACHE is True`. This proves bootstrap ordering rather than only helper behavior.

- [ ] **Step 2: Run the tests and verify RED**

```bash
python -m pytest -q tests/patch/test_lightop_environment.py
```

Expected: FAIL because `vllm_hcu.lightop_env` and the new names do not exist.

- [ ] **Step 3: Implement the minimal bridge**

Use one table with a parser kind so semantically equal booleans such as `1` and `true` do not conflict:

```python
_MAPPINGS = (
    ("VLLM_HCU_FUSED_MOE_CHUNK_SIZE", "VLLM_FUSED_MOE_CHUNK_SIZE", "LMSLIM_FUSED_MOE_CHUNK_SIZE", "int"),
    ("VLLM_HCU_USE_GLOBAL_MOE_CACHE", "VLLM_USE_GLOBAL_CACHE13", "LMSLIM_USE_GLOBAL_MOE_CACHE", "bool"),
    ("VLLM_HCU_USE_FUSED_RMS_QUANT", "USE_FUSED_RMS_QUANT", "LMSLIM_USE_FUSED_RMS_QUANT", "bool"),
    ("VLLM_HCU_USE_FUSE_SILU_AND_MUL", "VLLM_USE_FUSE_SILU_AND_MUL", "LMSLIM_USE_FUSE_SILU_AND_MUL", "bool"),
)


def configure_lightop_environment(environ=None):
    environ = os.environ if environ is None else environ
    for hcu_name, alias_name, legacy_name, kind in _MAPPINGS:
        configured = {
            name: _normalize(name, environ[name], kind)
            for name in (hcu_name, alias_name, legacy_name)
            if environ.get(name, "") != ""
        }
        if len(set(configured.values())) > 1:
            raise LightOpEnvironmentError(_conflict_message(configured))
        if legacy_name in configured:
            _warn_legacy_once(legacy_name, hcu_name)
        if configured:
            canonical = next(iter(configured.values()))
            environ.setdefault(hcu_name, canonical)
            environ.setdefault(alias_name, canonical)
```

Call `configure_lightop_environment()` in `vllm_hcu/__init__.py` before importing `vllm_hcu.compatibility`. Register the new values in `platforms/envs.py`. Replace repository-owned LMSlim environment names in the image config/docs with `VLLM_HCU_*`; remove the redundant documented `LMSLIM_USE_LIGHTOP=1` line.

- [ ] **Step 4: Verify GREEN and bootstrap isolation**

```bash
python -m pytest -q \
  tests/patch/test_lightop_environment.py \
  tests/patch/test_plugin_lifecycle.py \
  tests/patch/test_clean_process_bootstrap.py
```

Expected: PASS; CPU-only plugin discovery still does not import LightOp.

- [ ] **Step 5: Commit**

```bash
git add vllm_hcu/lightop_env.py vllm_hcu/__init__.py vllm_hcu/platforms/envs.py \
  tests/patch/test_lightop_environment.py docker/image-citest-conf-hcu.yaml \
  docs/pd分离mooncake使用.md
git commit -m "feat: bridge HCU LightOp environment settings"
```

### Task 3: Make attention and sparse MLA strictly categorized

**Files:**
- Modify: `tests/runtime_patch/test_lightop_attention_api.py`
- Modify: `tests/runtime_patch/test_sparse_indexer_loading.py`
- Modify: `vllm_hcu/model_executor/layers/attention_runtime.py`
- Modify: `vllm_hcu/v1/attention/ops/rocm_aiter_mla_sparse.py`

**Interfaces:**
- Consumes: `lightop.attention.{split_qkv_rms_rotary_embedding_fuse_with_kv_store_quant,mqa_logits,paged_mqa_logits,get_paged_mqa_logits_metadata,top_k_per_row_prefill,top_k_per_row_decode}`.
- Produces: the same vLLM adapter outputs and buffers with no `lightop.op`, `lightop.gemmopt`, or top-level fallback.

- [ ] **Step 1: Change tests to reject all legacy attention fallbacks**

```python
def test_sparse_mla_does_not_retry_legacy_namespace(monkeypatch):
    runtime = _runtime()
    legacy = SimpleNamespace(mqa_logits=lambda *_: pytest.fail("legacy called"))
    monkeypatch.setattr(runtime, "lightop_attention", SimpleNamespace())
    monkeypatch.setattr(runtime, "lightop", legacy, raising=False)
    with pytest.raises(AttributeError):
        runtime.rocm_fp8_mqa_logits(
            torch.ones((1, 1, 2)),
            (torch.ones((1, 2)), torch.ones(1)),
            torch.ones((1, 1)),
            torch.zeros(1, dtype=torch.int32),
            torch.ones(1, dtype=torch.int32),
        )
```

Keep the PR #27 ABI assertions: MQA receives six logical inputs, FP32 contiguous weights, paged metadata, and exact preallocated output buffers.

- [ ] **Step 2: Run focused tests and verify RED**

```bash
python -m pytest -q \
  tests/runtime_patch/test_lightop_attention_api.py \
  tests/runtime_patch/test_sparse_indexer_loading.py -k lightop
```

Expected: FAIL because PR #27 still retries top-level, `lightop.op`, or `lightop.gemmopt` symbols.

- [ ] **Step 3: Remove legacy imports and retries**

Use direct categorized imports at the existing lazy/eager boundary:

```python
from lightop.attention import (
    split_qkv_rms_rotary_embedding_fuse_with_kv_store_quant,
)
```

and in sparse MLA:

```python
from lightop import attention as lightop_attention

return lightop_attention.mqa_logits(
    q, k_fp8, weights.float().contiguous(), cu_seqlen_ks, cu_seqlen_ke, scale
)
```

Do not wrap kernel execution in compatibility `try/except` blocks.

- [ ] **Step 4: Verify GREEN**

```bash
python -m pytest -q \
  tests/runtime_patch/test_lightop_attention_api.py \
  tests/runtime_patch/test_sparse_indexer_loading.py
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/runtime_patch/test_lightop_attention_api.py \
  tests/runtime_patch/test_sparse_indexer_loading.py \
  vllm_hcu/model_executor/layers/attention_runtime.py \
  vllm_hcu/v1/attention/ops/rocm_aiter_mla_sparse.py
git commit -m "refactor: require categorized LightOp attention APIs"
```

### Task 4: Make activation and grouped GEMM owners strictly categorized

**Files:**
- Modify: `tests/runtime_patch/test_lightop_ops_api.py`
- Modify: `tests/runtime_patch/test_moe_deepep.py`
- Modify: `vllm_hcu/ops/silu_and_mul.py`
- Modify: `vllm_hcu/ops/fuse_silu_mul_quant.py`
- Modify: `vllm_hcu/model_executor/layers/fused_moe/experts/deep_gemm_moe.py`
- Modify: `vllm_hcu/model_executor/layers/fused_moe/experts/batched_deep_gemm_moe.py`
- Modify: `vllm_hcu/model_executor/layers/fused_moe/experts/dpsk_v4_deep_gemm_moe.py`

**Interfaces:**
- Consumes: `lightop.activation` activation/quant exports and `lightop.gemm_ops` grouped W8A8 exports.
- Produces: unchanged expert outputs and import timing.
- Preserves: top-level `fuse_silu_mul_clamp_quant(_ep)` only; all other resolver names use `lightop.activation`.

- [ ] **Step 1: Write strict resolver tests**

```python
def test_deepseek_expert_resolver_uses_activation_category(monkeypatch):
    called = []
    activation = ModuleType("lightop.activation")
    activation.fuse_silu_mul_fp8_quant = lambda *a, **k: called.append((a, k))
    _install_category(monkeypatch, "activation", activation)
    module._lightop_activation("fuse_silu_mul_fp8_quant")(object())
    assert len(called) == 1


def test_only_clamp_resolver_uses_top_level_lightop(monkeypatch):
    top = ModuleType("lightop")
    top.fuse_silu_mul_clamp_quant = lambda *a, **k: (a, k)
    monkeypatch.setitem(sys.modules, "lightop", top)
    assert module._lightop_clamp("fuse_silu_mul_clamp_quant") is top.fuse_silu_mul_clamp_quant
```

Also assert categorized import failure does not call same-named top-level symbols for `silu_and_mul_opt`, fused quant, or grouped GEMM.

- [ ] **Step 2: Run focused tests and verify RED**

```bash
python -m pytest -q \
  tests/runtime_patch/test_lightop_ops_api.py \
  tests/runtime_patch/test_moe_deepep.py -k 'lightop or deep_gemm or activation'
```

Expected: FAIL on PR #27 compatibility fallbacks and the current top-level dynamic resolver.

- [ ] **Step 3: Implement categorized resolvers**

```python
@functools.lru_cache(maxsize=None)
def _lightop_activation(name: str):
    from lightop import activation
    return getattr(activation, name)


@functools.lru_cache(maxsize=None)
def _lightop_clamp(name: str):
    import lightop
    if name not in {"fuse_silu_mul_clamp_quant", "fuse_silu_mul_clamp_quant_ep"}:
        raise AttributeError(name)
    return getattr(lightop, name)
```

Use `lightop.activation` and `lightop.gemm_ops` directly in the other owners. Keep current local/lazy timing and do not move HCU imports to package discovery.

- [ ] **Step 4: Verify GREEN and module exchange contracts**

```bash
python -m pytest -q \
  tests/runtime_patch/test_lightop_ops_api.py \
  tests/runtime_patch/test_moe_deepep.py \
  tests/patch/test_module_exchange.py -k 'lightop or deep_gemm or activation'
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/runtime_patch/test_lightop_ops_api.py tests/runtime_patch/test_moe_deepep.py \
  vllm_hcu/ops/silu_and_mul.py vllm_hcu/ops/fuse_silu_mul_quant.py \
  vllm_hcu/model_executor/layers/fused_moe/experts/deep_gemm_moe.py \
  vllm_hcu/model_executor/layers/fused_moe/experts/batched_deep_gemm_moe.py \
  vllm_hcu/model_executor/layers/fused_moe/experts/dpsk_v4_deep_gemm_moe.py
git commit -m "refactor: require categorized LightOp activation APIs"
```

### Task 5: Remove LMSlim from MoE and route through lightop.moe

**Files:**
- Modify: `tests/runtime_patch/test_moe_deepep.py`
- Modify: `tests/runtime_patch/test_quant_gemm_aiter.py`
- Modify: `vllm_hcu/model_executor/layers/fused_moe/deep_gemm_utils.py`
- Modify: `vllm_hcu/model_executor/layers/fused_moe/router_runtime.py`
- Modify: `vllm_hcu/ops/fuse_moe_gate.py`
- Modify: `vllm_hcu/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe_marlin.py`
- Modify: `vllm_hcu/patch/worker/op_opt/moe/patch_moe_align_block_size.py`

**Interfaces:**
- Consumes: `lightop.moe.{ep_gather,ep_scatter,moe_fused_gate,moe_align_block_size_out,fused_experts_impl_fp8_marlin,fused_experts_impl_int8_marlin}`.
- Produces: unchanged MoE results and exact existing Marlin call expressions without LMSlim fallback.

- [ ] **Step 1: Replace fallback tests with strict categorized-only tests**

```python
def test_marlin_moe_never_imports_lmslim(monkeypatch):
    calls = []
    _install_lightop_moe(
        monkeypatch,
        fused_experts_impl_fp8_marlin=lambda **kw: calls.append(("fp8", kw)),
        fused_experts_impl_int8_marlin=lambda **kw: calls.append(("int8", kw)),
    )
    _reject_import_prefix(monkeypatch, "lmslim")
    _run_fp8_and_int8_marlin_paths()
    assert [kind for kind, _ in calls] == ["fp8", "int8"]


def test_missing_lightop_marlin_export_fails_without_lmslim_retry(monkeypatch):
    _install_lightop_moe(monkeypatch)
    _reject_import_prefix(monkeypatch, "lmslim")
    with pytest.raises((ImportError, AttributeError)):
        _run_fp8_marlin_path()
```

Retain exact MoE-align assertions for preallocated outputs, `is_ep=False`, and `is_fuse_fill=False`.

- [ ] **Step 2: Run focused tests and verify RED**

```bash
python -m pytest -q \
  tests/runtime_patch/test_moe_deepep.py \
  tests/runtime_patch/test_quant_gemm_aiter.py \
  -k 'lightop or marlin or moe_align or ep_scatter or fused_gate'
```

Expected: FAIL because PR #27 retains LMSlim and legacy LightOp fallbacks.

- [ ] **Step 3: Remove fallback code and stale LMSlim wording**

Use only:

```python
from lightop.moe import fused_experts_impl_fp8_marlin
from lightop.moe import fused_experts_impl_int8_marlin
```

Keep every keyword argument and return contract unchanged. Replace `lightop.op` routing/EP imports with `lightop.moe`; require `moe_align_block_size_out` and propagate execution errors.

- [ ] **Step 4: Verify GREEN**

```bash
python -m pytest -q \
  tests/runtime_patch/test_moe_deepep.py \
  tests/runtime_patch/test_quant_gemm_aiter.py \
  -k 'lightop or marlin or moe_align or ep_scatter or fused_gate'
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/runtime_patch/test_moe_deepep.py tests/runtime_patch/test_quant_gemm_aiter.py \
  vllm_hcu/model_executor/layers/fused_moe/deep_gemm_utils.py \
  vllm_hcu/model_executor/layers/fused_moe/router_runtime.py \
  vllm_hcu/ops/fuse_moe_gate.py \
  vllm_hcu/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe_marlin.py \
  vllm_hcu/patch/worker/op_opt/moe/patch_moe_align_block_size.py
git commit -m "refactor: remove LMSlim MoE runtime calls"
```

### Task 6: Migrate norm, quant, W8A8 GEMM, and tensor helpers

**Files:**
- Modify: `tests/runtime_patch/test_lightop_ops_api.py`
- Modify: `tests/runtime_patch/test_quant_gemm_aiter.py`
- Modify: `tests/accuracy/test_portable_operator_accuracy.py`
- Modify: `vllm_hcu/ops/rms_norm.py`
- Modify: `vllm_hcu/ops/gemma_rms_norm.py`
- Modify: `vllm_hcu/ops/fuse_rms_norm_quant.py`
- Modify: `vllm_hcu/model_executor/layers/quantization/lightop_fp8_runtime.py`
- Modify: `vllm_hcu/model_executor/layers/quantization/int8_runtime.py`
- Modify: `vllm_hcu/ops/test_concat.py`

**Interfaces:**
- Consumes: `lightop.norm`, `lightop.quant`, `lightop.gemm_ops.hipblaslt_w8a8_gemm`, and `lightop.tensor.ds_cat`.
- Produces: existing adapter shapes/dtypes/status checks and the optional `torch.cat` algorithmic fallback.

- [ ] **Step 1: Write exact ABI and no-LMSlim tests**

```python
def test_int8_linear_uses_documented_lightop_names(monkeypatch):
    quant_calls, gemm_calls = [], []
    _install_lightop_quant_gemm(
        monkeypatch,
        per_token_quant_int8=lambda x: quant_calls.append(x) or _cpu_quant(x),
        hipblaslt_w8a8_gemm=lambda *a: gemm_calls.append(a) or (True, torch.zeros(2, 3)),
    )
    _reject_import_prefix(monkeypatch, "lmslim")
    output = apply_int8_linear(_input(), _weight(), _scale(), torch.bfloat16)
    assert output.shape == (2, 3)
    assert len(quant_calls) == len(gemm_calls) == 1


def test_fp8_quant_uses_keyword_output_abi(monkeypatch):
    calls = []
    _install_lightop_quant(
        per_token_quant_fp8=lambda x, **kw: calls.append((x, kw)) or (kw["out_q"], kw["out_scale"])
    )
    _lightop_per_token_quant_fp8(torch.ones((2, 4)))
    assert set(calls[0][1]) == {"dtype", "out_q", "out_scale"}
```

Keep PR #27 tests for returned dynamic-RMS tensors and Gemma `out=out`. Change concat tests so `lightop.tensor.ds_cat` is preferred and `torch.cat` is the only fallback.

- [ ] **Step 2: Run focused tests and verify RED**

```bash
python -m pytest -q \
  tests/runtime_patch/test_lightop_ops_api.py \
  tests/runtime_patch/test_quant_gemm_aiter.py \
  tests/accuracy/test_portable_operator_accuracy.py \
  -k 'lightop or rms or gemma or int8_linear or w8a8 or concat'
```

Expected: FAIL on LMSlim retries, legacy ABI calls, and PR #27's `channelwise` GEMM name.

- [ ] **Step 3: Implement the categorized calls**

The W8A8 path must preserve the documented terminal name:

```python
from lightop.quant import per_token_quant_int8
from lightop.gemm_ops import hipblaslt_w8a8_gemm

x_q, x_scale = per_token_quant_int8(input)
status, output = hipblaslt_w8a8_gemm(
    x_q_2d, weight, x_scale_2d, weight_scale, m, n, k, "NT", out_dtype
)
```

Use the installed FP8, Gemma, and dynamic RMS signatures exactly. Remove compatibility retries, but keep existing validation and `torch.cat` fallback.

- [ ] **Step 4: Verify GREEN**

```bash
python -m pytest -q \
  tests/runtime_patch/test_lightop_ops_api.py \
  tests/runtime_patch/test_quant_gemm_aiter.py \
  tests/accuracy/test_portable_operator_accuracy.py \
  -k 'lightop or rms or gemma or int8_linear or w8a8 or concat'
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/runtime_patch/test_lightop_ops_api.py \
  tests/runtime_patch/test_quant_gemm_aiter.py tests/accuracy/test_portable_operator_accuracy.py \
  vllm_hcu/ops/rms_norm.py vllm_hcu/ops/gemma_rms_norm.py \
  vllm_hcu/ops/fuse_rms_norm_quant.py vllm_hcu/ops/test_concat.py \
  vllm_hcu/model_executor/layers/quantization/lightop_fp8_runtime.py \
  vllm_hcu/model_executor/layers/quantization/int8_runtime.py
git commit -m "refactor: remove LMSlim quantization runtime calls"
```

### Task 7: Adapt the main DeepSeek V4 data flow to the KVNorm-aware kernel

**Files:**
- Modify: `tests/runtime_patch/test_lightop_deepseek_v4_api.py`
- Modify: `tests/runtime_patch/test_attention_mla_fla_mamba.py`
- Modify: `vllm_hcu/model_executor/layers/deepseek_v4_attention.py`

**Interfaces:**
- Consumes: `lightop.attention.fused_deepseek_v4_qnorm_rope_kvnorm_rope_quant_insert_int32`.
- Produces: Q normalized once, KV normalized exactly once inside LightOp, contiguous int32 slots, and unchanged cache/output ownership.

- [ ] **Step 1: Strengthen the existing PR #27 tests**

```python
assert qnorm_inputs == [raw_qr]
assert projection_inputs == [normalized_qr]
assert inserted[0][1] is raw_kv
assert args[:4] == (q, raw_kv, kv_norm_weight, cache_2d)
assert args[4].dtype is torch.int32
assert args[4].is_contiguous()
```

Keep a fake obsolete `lightop.op` symbol that fails if invoked, and assert missing categorized API raises a targeted `RuntimeError`.

- [ ] **Step 2: Run and verify RED against the latest reconciled owner**

```bash
python -m pytest -q \
  tests/runtime_patch/test_lightop_deepseek_v4_api.py \
  tests/runtime_patch/test_attention_mla_fla_mamba.py -k 'deepseek_v4 or lightop'
```

Expected: FAIL until the latest current-branch flow and PR #27 flow are reconciled without legacy calls.

- [ ] **Step 3: Implement exactly-once normalization and new ABI**

```python
from lightop.attention import (
    fused_deepseek_v4_qnorm_rope_kvnorm_rope_quant_insert_int32,
)

slot_mapping = swa_metadata.slot_mapping.to(torch.int32).contiguous()
fused_deepseek_v4_qnorm_rope_kvnorm_rope_quant_insert_int32(
    q,
    raw_kv,
    self.kv_norm.weight.data,
    swa_kv_cache_2d,
    slot_mapping,
    positions.to(torch.int64),
    self.rotary_emb.cos_sin_cache,
    self.eps,
    swa_metadata.block_size,
)
```

Upstream of this call, normalize QR only; never pass already-normalized KV to the KVNorm-aware kernel.

- [ ] **Step 4: Verify GREEN**

```bash
python -m pytest -q \
  tests/runtime_patch/test_lightop_deepseek_v4_api.py \
  tests/runtime_patch/test_attention_mla_fla_mamba.py \
  tests/patch/test_module_exchange.py -k 'deepseek_v4 or lightop'
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/runtime_patch/test_lightop_deepseek_v4_api.py \
  tests/runtime_patch/test_attention_mla_fla_mamba.py \
  vllm_hcu/model_executor/layers/deepseek_v4_attention.py
git commit -m "refactor: adapt DeepSeek V4 categorized LightOp kernel"
```

### Task 8: Cover post-PR-27 DSpark and core-fix owners

**Files:**
- Create: `tests/runtime_patch/test_lightop_deepseek_v4_latest_api.py`
- Modify: `tests/accuracy/test_deepseek_v4_dspark_ops.py`
- Modify: `vllm_hcu/models/deepseek_v4_dspark.py`
- Modify: `vllm_hcu/patch/worker/core_fix/patch_deepseek_v4_attention.py`

**Interfaces:**
- Consumes: the same categorized KVNorm-aware int32 kernel as Task 7.
- Produces: DSpark context insertion and patched upstream insertion with raw KV and exact new ABI.
- Preserves: two top-level fused clamp helpers in the accuracy test and production resolver.

- [ ] **Step 1: Write failing post-PR-27 ABI tests**

```python
def test_dspark_context_passes_raw_kv_and_kv_norm_weight(monkeypatch):
    calls = []
    _install_lightop_attention(monkeypatch, lambda *args: calls.append(args))
    _run_dspark_context_precompute()
    q, raw_kv, kv_weight, _cache, slots, *_ = calls[0]
    assert raw_kv is expected_raw_kv
    assert kv_weight is expected_kv_norm_weight
    assert slots.dtype is torch.int32 and slots.is_contiguous()


def test_core_fix_patch_requires_categorized_insert(monkeypatch):
    _install_lightop_attention(monkeypatch, kernel=None, legacy_kernel=pytest.fail)
    with pytest.raises(RuntimeError, match="lightop.attention"):
        patched_insert(instance, q, raw_kv, positions, metadata)
```

- [ ] **Step 2: Run and verify RED**

```bash
python -m pytest -q tests/runtime_patch/test_lightop_deepseek_v4_latest_api.py
```

Expected: FAIL because both latest owners still call `lightop.op` and one DSpark caller pre-normalizes KV.

- [ ] **Step 3: Implement both latest owners**

In DSpark precompute, pass `qr_kv[..., attn.q_lora_rank:]` without `attn.kv_norm`. In both insertion functions, resolve the categorized kernel, pass `kv_norm.weight.data`, and convert slot mapping to contiguous int32. Keep positions int64 as required by the installed ABI.

- [ ] **Step 4: Verify portable GREEN and HCU collection**

```bash
python -m pytest -q \
  tests/runtime_patch/test_lightop_deepseek_v4_latest_api.py \
  tests/patch/test_module_exchange.py -k 'deepseek_v4 or dspark'
python -m pytest -q --collect-only tests/accuracy/test_deepseek_v4_dspark_ops.py
```

Expected: portable tests pass and HCU tests collect. The two clamp imports remain top-level and are named in the boundary exception.

- [ ] **Step 5: Commit**

```bash
git add tests/runtime_patch/test_lightop_deepseek_v4_latest_api.py \
  tests/accuracy/test_deepseek_v4_dspark_ops.py \
  vllm_hcu/models/deepseek_v4_dspark.py \
  vllm_hcu/patch/worker/core_fix/patch_deepseek_v4_attention.py
git commit -m "refactor: migrate latest DeepSeek V4 LightOp owners"
```

### Task 9: Enforce the final API boundary and run repository/HCU verification

**Files:**
- Create: `tests/patch/test_lightop_api_boundary.py`
- Modify: `tests/runtime_patch/test_lightop_categorized_api.py`
- Modify: `.github/workflows/configs/hcu-test-map.yaml` only if a current live-HCU file needs explicit selection
- Create: `docs/lightop_categorized_api_validation.md`

**Interfaces:**
- Consumes: final production tree and installed category `__all__`.
- Produces: a fail-closed policy test and recorded verification evidence.

- [ ] **Step 1: Write the final AST boundary test**

```python
ALLOWED_TOP_LEVEL = {
    ("vllm_hcu/model_executor/layers/fused_moe/experts/dpsk_v4_deep_gemm_moe.py", "fuse_silu_mul_clamp_quant"),
    ("vllm_hcu/model_executor/layers/fused_moe/experts/dpsk_v4_deep_gemm_moe.py", "fuse_silu_mul_clamp_quant_ep"),
}


def test_production_uses_public_lightop_categories_only():
    violations = scan_lightop_imports(REPOSITORY / "vllm_hcu")
    assert violations == []


def test_installed_category_exports_cover_production_symbols():
    used = categorized_symbols(REPOSITORY / "vllm_hcu")
    exported = installed_public_exports()
    assert used - exported == set()
```

The scanner must reject `lmslim`, `lightop.op`, `lightop.gemmopt`, moved top-level imports, and attribute calls such as `lightop.op.foo`; it must report file and line. Exact clamp allowlist entries are checked for existence so stale exceptions fail too.

- [ ] **Step 2: Run the boundary test and verify RED if anything remains**

```bash
python -m pytest -q tests/patch/test_lightop_api_boundary.py
```

Expected: FAIL with an exact residual list, or PASS only if every prior task already removed all residuals. If it passes immediately, temporarily add `import lmslim` to a copied temporary fixture and prove the scanner rejects it before accepting the test.

- [ ] **Step 3: Remove every reported residual and update the export contract**

The installed export table must include `lightop.gemm_ops.hipblaslt_w8a8_gemm` and all actual production category symbols. Do not add a fallback to make the scanner pass.

- [ ] **Step 4: Run fresh static and portable verification**

```bash
git diff --check origin/v0.25.1...HEAD
python -m compileall -q vllm_hcu tests
python tools/check_production_boundary.py
python tools/check_patch_test_coverage.py
python -m pytest -q tests/patch/test_lightop_api_boundary.py
python -m pytest -q tests/runtime_patch/test_lightop_categorized_api.py
python tools/run_patch_tests.py --suite contract
python tools/run_patch_tests.py --suite integration-smoke
```

Expected: all commands exit 0. Contract results must be no worse than the 1162-pass baseline; record the exact fresh counts rather than copying this expectation.

- [ ] **Step 5: Run live HCU numerical tests**

```bash
python tools/run_patch_tests.py --suite accuracy-hcu -- \
  -k 'lightop or int8 or deepseek_v4 or dspark'
python tools/run_patch_tests.py --suite contract-hcu -- \
  -k 'lightop or moe_align or deepseek_v4'
```

Expected: selected HCU tests execute and pass on the available BW1100 devices. Any skip must be listed by nodeid and reason; skips are not reported as passes.

- [ ] **Step 6: Record evidence and commit**

Write exact package versions, commands, exit codes, pass/skip counts, and residual-audit output to `docs/lightop_categorized_api_validation.md`.

```bash
git add tests/patch/test_lightop_api_boundary.py \
  tests/runtime_patch/test_lightop_categorized_api.py \
  .github/workflows/configs/hcu-test-map.yaml \
  docs/lightop_categorized_api_validation.md
git commit -m "test: enforce categorized LightOp API boundary"
```

If the workflow file did not change, omit it from `git add`.

### Task 10: Run Qwen3.5 W8A8 and prepare the new pull request

**Files:**
- Modify: `docs/lightop_categorized_api_validation.md`
- Inspect: `/models/Qwen3.5-35B-A3B-W8A8`
- Runtime logs: `/tmp/vllm-hcu-lightop-qwen35/server.log`
- Runtime response: `/tmp/vllm-hcu-lightop-qwen35/response.json`

**Interfaces:**
- Consumes: local worktree through `PYTHONPATH`, installed vLLM/LightOp, one free BW1100, and the OpenAI-compatible server API.
- Produces: startup/generation evidence and a PR targeting `v0.25.1`.

- [ ] **Step 1: Select a free device and prepare an isolated runtime directory**

Use read-only GPU process/memory inspection and choose one device with enough free memory. Create only the explicit case directory; never kill unrelated GPU processes.

```bash
rocm-smi --showmeminfo vram --showpids
SELECTED_HCU=7  # Set this integer to the free device selected from the command above.
mkdir -p /tmp/vllm-hcu-lightop-qwen35
```

Keep `SELECTED_HCU` in the same shell session for the remaining runtime steps.

- [ ] **Step 2: Start the documented slimquant_marlin path from the worktree**

```bash
export PLUGIN_ROOT=/models/.worktrees/vllm-plugin-das-lightop-no-lmslim
export HIP_VISIBLE_DEVICES="$SELECTED_HCU"
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_KV_CACHE_LAYOUT=HND
export VLLM_HCU_USE_FLASH_ATTN_UNIFIED=1
export VLLM_HCU_USE_GLOBAL_MOE_CACHE=1
export VLLM_CACHE_ROOT=/tmp/vllm-cache-qwen35-lightop
export PYTHONPATH="${PLUGIN_ROOT}"

vllm serve /models/Qwen3.5-35B-A3B-W8A8 \
  --served-model-name qwen35-int8-lightop \
  --tensor-parallel-size 1 \
  --max-model-len 65536 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.90 \
  --trust-remote-code \
  --quantization slimquant_marlin \
  --port 18012 \
  > /tmp/vllm-hcu-lightop-qwen35/server.log 2>&1 &
SERVER_PID=$!
```

Preserve `SERVER_PID` in the same shell session.

- [ ] **Step 3: Wait for health and make a deterministic request**

Poll `/health` for up to 15 minutes while also checking that `SERVER_PID` is alive. Then run:

```bash
curl -fsS http://127.0.0.1:18012/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen35-int8-lightop","prompt":"Write one sentence explaining why interface stability matters.","temperature":0,"max_tokens":64}' \
  -o /tmp/vllm-hcu-lightop-qwen35/response.json

python - <<'PY'
import json
p = "/tmp/vllm-hcu-lightop-qwen35/response.json"
data = json.load(open(p, encoding="utf-8"))
text = data["choices"][0]["text"]
assert text.strip(), data
print(text)
PY
```

Expected: health succeeds, request exits 0, and completion text is non-empty.

- [ ] **Step 4: Stop only the captured server and inspect logs**

```bash
kill -TERM "$SERVER_PID"
wait "$SERVER_PID" || true
rg -n 'ERROR|Traceback|lmslim|LightOp|lightop|slimquant|Marlin|W8A8' \
  /tmp/vllm-hcu-lightop-qwen35/server.log
```

Expected: no migration-related traceback or external LMSlim fallback message. Record the LightOp/Marlin route evidence and response in the validation document.

- [ ] **Step 5: Commit final validation evidence and run final fresh checks**

```bash
git add docs/lightop_categorized_api_validation.md
git commit -m "docs: record LightOp model validation"
python tools/run_patch_tests.py --suite contract
git diff --check origin/v0.25.1...HEAD
git status --short
```

Expected: contract suite passes, diff check is clean, and worktree has no uncommitted files.

- [ ] **Step 6: Review, push, and create the PR**

Request an independent code review against `origin/v0.25.1`, fix every critical/important finding, and rerun affected plus final verification. Then push without force and create a PR targeting `v0.25.1` with:

- categorized old-to-new interface table;
- explicit statement that plugin production code no longer imports/calls external LMSlim;
- the two unchanged top-level clamp exceptions and why they remain;
- `VLLM_HCU_*` environment bridge and precedence;
- exact portable/HCU/model commands and results;
- references to `/models/Lightop&Lmslim.md` and PR #27;
- model response and server-log evidence location.

Use the configured Git author `zhangzbb`; do not place an access token in a remote URL, command argument, repository file, PR body, or log.

## Completion Criteria

- Production `vllm_hcu` has zero external LMSlim imports/calls.
- Every mapped LightOp symbol uses its categorized public path and installed ABI.
- Only the two approved top-level clamp calls remain, enforced by test.
- User-facing repository configuration uses `VLLM_HCU_*` names and imports LightOp only after the bridge.
- Focused, contract, integration-smoke, and selected HCU suites pass with exact counts recorded.
- `/models/Qwen3.5-35B-A3B-W8A8` loads and returns a non-empty deterministic completion on the slimquant_marlin path.
- The branch is pushed and a new PR targets `v0.25.1`.
