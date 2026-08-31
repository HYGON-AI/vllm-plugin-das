# DeepEP Auto Correctness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve DeepSeek-V4 Channel-INT8 numerics and make DeepEP auto HT/LL selection, dispatch, buffer reuse, and supported concurrency behavior explicit and testable.

**Architecture:** Keep paired HT/LL delegates, but select the mode before each modular MoE call and reject concurrent ubatching until invocation-local state exists. Preserve quantization metadata at the factory boundary, require explicit phase evidence, clean the shared buffer before each LL dispatch, and delegate unsupported DSpark cache layouts upstream.

**Tech Stack:** Python 3.10, PyTorch, pytest, vLLM v0.25.1 runtime adapters, HCU DeepEP/DeepGEMM.

**Spec:** `docs/superpowers/specs/2026-08-31-deepep-auto-correctness-design.md`

## Global Constraints

- Preserve pinned vLLM v0.25.1 behavior outside HCU-owned paths.
- Do not claim deepep_auto DBO or concurrent ubatching support.
- Missing or ambiguous phase metadata must select HT.
- Every DeepEP rank must clean LL state with identical arguments.

---

### Task 1: Preserve the Channel-INT8 SwiGLU clamp

**Files:**
- Modify: `tests/runtime_patch/test_quant_gemm_aiter.py`
- Modify: `tests/runtime_patch/test_moe_deepep.py`
- Modify: `vllm_hcu/patch/worker/op_opt/moe/patch_int8_oracle.py`
- Modify: `vllm_hcu/patch/worker/op_opt/moe/patch_config.py`
- Modify: `vllm_hcu/model_executor/layers/fused_moe/config_runtime.py`

**Interfaces:**
- Consumes: `layer.swiglu_limit: float | None`.
- Produces: `int8_w8a8_moe_quant_config(..., gemm1_clamp_limit=None)` and a factory result with the layer clamp.

- [ ] **Step 1: Write the failing factory and expert-path tests**

Call the patched factory with a real layer value:

```python
quant_config = target.make_int8_moe_quant_config(
    target.Int8MoeBackend.HCU_DEEPGEMM,
    w1_scale,
    w2_scale,
    per_act_token_quant=True,
    layer=SimpleNamespace(swiglu_limit=10.0),
)
assert quant_config.gemm1_clamp_limit == 10.0
```

Use that factory result in the existing contiguous and masked DeepGEMM tests; each must observe the clamped LightOp and reject the unclamped operation.

- [ ] **Step 2: Run RED**

```bash
VLLM_V0251_SOURCE_ROOT=/models/zb/vllm_025/vllm pytest -q tests/runtime_patch/test_quant_gemm_aiter.py -k int8_oracle
```

Expected: the clamp assertion fails.

- [ ] **Step 3: Implement the minimum plumbing**

Extend the HCU W8A8 wrapper and runtime helper with `gemm1_clamp_limit=None`. When a clamp or block shape is present, build via:

```python
return module.FusedMoEQuantConfig.make(
    module.torch.int8,
    w1_scale=w1_scale,
    w2_scale=w2_scale,
    a1_scale=a1_scale,
    a2_scale=a2_scale,
    w1_bias=w1_bias,
    w2_bias=w2_bias,
    per_act_token_quant=per_act_token_quant,
    per_out_ch_quant=False,
    block_shape=block_shape,
    gemm1_clamp_limit=gemm1_clamp_limit,
)
```

Pass `gemm1_clamp_limit=getattr(layer, "swiglu_limit", None)` from the oracle. Keep the upstream helper for calls with neither HCU extension.

- [ ] **Step 4: Run GREEN and commit**

```bash
VLLM_V0251_SOURCE_ROOT=/models/zb/vllm_025/vllm pytest -q tests/runtime_patch/test_quant_gemm_aiter.py tests/runtime_patch/test_moe_deepep.py -k 'clamp or int8_oracle'
git add tests/runtime_patch/test_quant_gemm_aiter.py tests/runtime_patch/test_moe_deepep.py vllm_hcu/patch/worker/op_opt/moe/patch_int8_oracle.py vllm_hcu/patch/worker/op_opt/moe/patch_config.py vllm_hcu/model_executor/layers/fused_moe/config_runtime.py
git commit -m "fix(hcu): preserve DeepSeek V4 SwiGLU clamp"
```

### Task 2: Propagate INT8 dispatch through the auto factory

**Files:**
- Modify: `tests/runtime_patch/test_moe_deepep.py`
- Modify: `vllm_hcu/patch/worker/op_opt/moe/patch_all2all_utils.py`

**Interfaces:**
- Consumes: `quant_config.quant_dtype`.
- Produces: mutually exclusive `use_fp8_dispatch` and `use_int8_dispatch` LL flags.

- [ ] **Step 1: Parameterize the factory test for FP8 and INT8**

For INT8 require:

```python
assert result.ll_prepare_finalize.kwargs["use_fp8_dispatch"] is False
assert result.ll_prepare_finalize.kwargs["use_int8_dispatch"] is True
```

- [ ] **Step 2: Run RED**

```bash
VLLM_V0251_SOURCE_ROOT=/models/zb/vllm_025/vllm pytest -q tests/runtime_patch/test_moe_deepep.py -k all2all_auto_builds
```

Expected: the INT8 flag is absent.

- [ ] **Step 3: Derive, validate, and pass both flags**

```python
use_fp8_dispatch = quant_config.quant_dtype == target.current_platform.fp8_dtype()
use_int8_dispatch = quant_config.quant_dtype == target.torch.int8
if use_fp8_dispatch and use_int8_dispatch:
    raise RuntimeError("DeepEP auto cannot enable FP8 and INT8 dispatch together")
```

- [ ] **Step 4: Run GREEN and commit**

```bash
VLLM_V0251_SOURCE_ROOT=/models/zb/vllm_025/vllm pytest -q tests/runtime_patch/test_moe_deepep.py -k 'all2all_auto_builds or hcu_int8_dispatch_contract'
git add tests/runtime_patch/test_moe_deepep.py vllm_hcu/patch/worker/op_opt/moe/patch_all2all_utils.py
git commit -m "fix(hcu): enable INT8 dispatch in deepep auto"
```

### Task 3: Require explicit non-prefill phase evidence

**Files:**
- Modify: `tests/runtime_patch/test_worker_framework_opt.py`
- Modify: `vllm_hcu/forward_context_runtime.py`

**Interfaces:**
- Consumes: nested metadata with lengths and `is_prefilling`.
- Produces: metadata fallback selects LL only when all contributing rows explicitly say decode.

- [ ] **Step 1: Add cached-prefill, mixed, decode, speculative-decode, and missing-field cases**

```python
cached_prefill = SimpleNamespace(max_query_len=2, max_seq_len=100, is_prefilling=torch.tensor([True]))
mixed = SimpleNamespace(max_query_len=2, max_seq_len=100, is_prefilling=torch.tensor([False, True]))
decode = SimpleNamespace(max_query_len=1, max_seq_len=100, is_prefilling=torch.tensor([False]))
spec_decode = SimpleNamespace(max_query_len=8, max_seq_len=100, is_prefilling=torch.tensor([False]))
unknown = SimpleNamespace(max_query_len=2, max_seq_len=100)
```

Require HT, HT, LL, LL, and HT respectively.

- [ ] **Step 2: Run RED**

```bash
VLLM_V0251_SOURCE_ROOT=/models/zb/vllm_025/vllm pytest -q tests/runtime_patch/test_worker_framework_opt.py -k dspark_deepep_auto_uses_ht_for_prefill
```

- [ ] **Step 3: Add conservative phase normalization**

Flatten tensors/arrays through `.tolist()` and return false unless the value is present, non-empty, and every element is explicitly false. Require this helper before accepting each metadata length pair.

- [ ] **Step 4: Run GREEN and commit**

```bash
VLLM_V0251_SOURCE_ROOT=/models/zb/vllm_025/vllm pytest -q tests/runtime_patch/test_worker_framework_opt.py -k deepep_auto
git add tests/runtime_patch/test_worker_framework_opt.py vllm_hcu/forward_context_runtime.py
git commit -m "fix(hcu): keep cached prefills on DeepEP HT"
```

### Task 4: Clean the shared buffer before every LL dispatch

**Files:**
- Modify: `tests/runtime_patch/test_moe_deepep.py`
- Modify: `vllm_hcu/model_executor/layers/fused_moe/deepep_runtime.py`

**Interfaces:**
- Consumes: max tokens, hidden size, global experts, and computed quant group size.
- Produces: an ordered cleanup immediately before every LL dispatch.

- [ ] **Step 1: Write cleanup-order, consecutive-LL, FP8-group, and missing-API tests**

```python
assert calls == [
    ("clean", max_tokens, hidden, num_experts, 0),
    ("dispatch", 1),
    ("clean", max_tokens, hidden, num_experts, 0),
    ("dispatch", 1),
]
```

- [ ] **Step 2: Run RED**

```bash
VLLM_V0251_SOURCE_ROOT=/models/zb/vllm_025/vllm pytest -q tests/runtime_patch/test_moe_deepep.py -k 'cleanup or int8_dispatch_contract'
```

- [ ] **Step 3: Clean after group-size calculation and before dispatch**

```python
cleanup = getattr(self.buffer, "clean_low_latency_buffer", None)
if not callable(cleanup):
    raise RuntimeError("HCU DeepEP LL buffer does not expose clean_low_latency_buffer")
cleanup(self.max_tokens_per_rank, hidden_size, num_experts, quant_group_size)
```

- [ ] **Step 4: Run GREEN and commit**

```bash
VLLM_V0251_SOURCE_ROOT=/models/zb/vllm_025/vllm pytest -q tests/runtime_patch/test_moe_deepep.py -k 'deepep_ll or cleanup'
git add tests/runtime_patch/test_moe_deepep.py vllm_hcu/model_executor/layers/fused_moe/deepep_runtime.py
git commit -m "fix(hcu): clean shared buffer before DeepEP LL"
```

### Task 5: Snapshot mode before modular contract queries and reject ubatching

**Files:**
- Modify: `tests/runtime_patch/test_moe_deepep.py`
- Modify: `tests/runtime_patch/test_platform_hcu_config.py`
- Modify: `vllm_hcu/model_executor/layers/fused_moe/prepare_finalize/deepep_auto.py`
- Modify: `vllm_hcu/model_executor/layers/fused_moe/modular_kernel.py`
- Modify: `vllm_hcu/patch/platform/core_fix/patch_vllm_config.py`

**Interfaces:**
- Consumes: forward mode and `parallel_config.use_ubatching`.
- Produces: `begin_moe_call() -> bool`, one stable supported invocation snapshot, and startup rejection for ubatching.

- [ ] **Step 1: Add failing first-transition and configuration tests**

Use HT and LL fakes with different activation/quantization contracts. Assert `begin_moe_call` runs before contract reads. Set `enable_dbo=True` and separately expose `use_ubatching=True`; both must raise `ValueError` matching `deepep_auto.*ubatching.*not supported`.

- [ ] **Step 2: Run RED**

```bash
VLLM_V0251_SOURCE_ROOT=/models/zb/vllm_025/vllm pytest -q tests/runtime_patch/test_moe_deepep.py -k 'begin_moe_call or snapshots_mode'
VLLM_V0251_SOURCE_ROOT=/models/zb/vllm_025/vllm pytest -q tests/runtime_patch/test_platform_hcu_config.py -k 'deepep_auto and ubatch'
```

- [ ] **Step 3: Implement `begin_moe_call` and remove prepare-time resampling**

```python
def begin_moe_call(self) -> bool:
    self._use_low_latency_snapshot = (
        _forward_uses_low_latency()
        if self._fixed_use_low_latency is None
        else self._fixed_use_low_latency
    )
    if self._auto_experts is not None:
        self._auto_experts.set_deepep_auto_use_low_latency(self._use_low_latency_snapshot)
    return self._use_low_latency_snapshot
```

- [ ] **Step 4: Call the optional hook at the start of modular `_prepare`**

```python
begin_moe_call = getattr(self.prepare_finalize, "begin_moe_call", None)
if begin_moe_call is not None:
    begin_moe_call()
```

- [ ] **Step 5: Reject deepep_auto when `use_ubatching` is true**

Add the guard inside existing deepep_auto validation, before model loading.

- [ ] **Step 6: Run GREEN and commit**

```bash
VLLM_V0251_SOURCE_ROOT=/models/zb/vllm_025/vllm pytest -q tests/runtime_patch/test_moe_deepep.py tests/runtime_patch/test_platform_hcu_config.py
git add tests/runtime_patch/test_moe_deepep.py tests/runtime_patch/test_platform_hcu_config.py vllm_hcu/model_executor/layers/fused_moe/prepare_finalize/deepep_auto.py vllm_hcu/model_executor/layers/fused_moe/modular_kernel.py vllm_hcu/patch/platform/core_fix/patch_vllm_config.py
git commit -m "fix(hcu): latch deepep auto mode before MoE prepare"
```

### Task 6: Fall back for non-uint8 DSpark context KV caches

**Files:**
- Modify: `tests/runtime_patch/test_deepseek_v4_dspark_model.py`
- Modify: `vllm_hcu/models/deepseek_v4_dspark.py`

**Interfaces:**
- Consumes: actual cache tensor dtype and upstream `_dspark._insert_context_kv`.
- Produces: LightOp for uint8 only; upstream behavior for all other dtypes.

- [ ] **Step 1: Use uint8 in the LightOp test and add a failing BF16 fallback test**

```python
fallback_calls = []
dspark_module._dspark._insert_context_kv = lambda *args: fallback_calls.append(args)
dspark_module._insert_context_kv(attn, kv, positions, slot_mapping)
assert fallback_calls == [(attn, kv, positions, slot_mapping)]
```

- [ ] **Step 2: Run RED**

```bash
VLLM_V0251_SOURCE_ROOT=/models/zb/vllm_025/vllm pytest -q tests/runtime_patch/test_deepseek_v4_dspark_model.py -k context_insert
```

- [ ] **Step 3: Delegate non-uint8 caches upstream**

```python
if swa_cache.dtype != torch.uint8:
    return _dspark._insert_context_kv(attn, kv, positions, slot_mapping)
```

- [ ] **Step 4: Run GREEN and commit**

```bash
VLLM_V0251_SOURCE_ROOT=/models/zb/vllm_025/vllm pytest -q tests/runtime_patch/test_deepseek_v4_dspark_model.py tests/runtime_patch/test_deepseek_v4_dspark_patches.py
git add tests/runtime_patch/test_deepseek_v4_dspark_model.py vllm_hcu/models/deepseek_v4_dspark.py
git commit -m "fix(hcu): preserve non-uint8 DSpark KV insertion"
```

### Task 7: Full verification and review

**Files:**
- Verify: every file changed in Tasks 1-6.

**Interfaces:**
- Consumes: complete branch diff from PR 36 head.
- Produces: software evidence and an explicit hardware-test status.

- [ ] **Step 1: Run all affected tests**

```bash
VLLM_V0251_SOURCE_ROOT=/models/zb/vllm_025/vllm pytest -q tests/runtime_patch/test_moe_deepep.py tests/runtime_patch/test_quant_gemm_aiter.py tests/runtime_patch/test_worker_framework_opt.py tests/runtime_patch/test_platform_hcu_config.py tests/runtime_patch/test_deepseek_v4_dspark_model.py tests/runtime_patch/test_deepseek_v4_dspark_patches.py
```

- [ ] **Step 2: Run static validation**

```bash
python -m compileall -q vllm_hcu tests
git diff --check pr/36...HEAD
```

- [ ] **Step 3: Inspect scope**

```bash
git status --short --branch
git diff --stat pr/36...HEAD
git log --oneline pr/36..HEAD
```

- [ ] **Step 4: Report hardware status honestly**

If multi-rank HCU is unavailable, state that HT -> LL -> LL -> HT -> LL hang and numerical comparison remain an external merge gate; do not label them passed.
