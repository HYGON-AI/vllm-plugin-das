# Unified AITER MoE Routing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Route every plugin-owned AITER MoE path through AITER's public multi-solution selector and fall back to vLLM Triton only when AITER explicitly returns no solution.

**Architecture:** Add one HCU adapter that owns the AITER problem description, config/weight caches, solution-aware EP map, and `aiter_moe()` invocation. W16A16, W4A16, compressed-tensors W8A8, and SlimQuant/Marlin W8A8 call the adapter and retain their vLLM-specific Triton fallback closures. Registered expert weights remain canonical; AITER-specific layouts are bounded derived caches.

**Tech Stack:** Python 3.10, PyTorch/ROCm, vLLM 0.25.1 modular MoE, HCU AITER `aiter.moe`, pytest.

**Spec:** `docs/superpowers/specs/2026-08-31-unified-aiter-moe-routing-design.md`

## Global Constraints

- Never pass `spec_sol_type` from plugin-owned AITER MoE code.
- Pass one unified `use_shuffle` hint to every plugin-owned `get_aiter_moe_config()` call; its default is true.
- Fall back only for `status=False`; imports, invalid configs, shuffles, ABI mismatches, and kernel failures must raise.
- Preserve canonical registered weights and the original vLLM expert map for framework fallback.
- Do not change explicit non-AITER MoE backends.
- Do not load a checkpoint, start a server, or run a large-model test.
- Do not store the supplied GitLab access token in source, Git config/remotes, documentation, command output, or test artifacts.

---

## File Map

- Create `vllm_hcu/model_executor/layers/fused_moe/aiter_moe_dispatch.py`: public HCU/AITER adapter, caches, solution-aware layouts/maps, execution.
- Modify `vllm_hcu/platforms/envs.py`: unified shuffle flag and deprecated aliases.
- Modify `vllm_hcu/model_executor/layers/fused_moe/aiter_runtime.py`: W16A16 adapter call and vLLM Triton fallback; remove direct ASM dispatch.
- Modify `vllm_hcu/model_executor/layers/fused_moe/unquantized_fused_moe_method.py`: canonical W16A16 post-load setup.
- Modify `vllm_hcu/model_executor/layers/quantization/compressed_tensors_moe_runtime.py`: compressed-tensors INT8/FP8 integration.
- Modify `vllm_hcu/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe_marlin.py`: SlimQuant/Marlin INT8 integration.
- Modify `vllm_hcu/patch/worker/op_opt/moe/patch_fused_moe.py`: W4A16 integration.
- Modify `tests/runtime_patch/test_quant_gemm_aiter.py`: behavioral regression coverage.
- Replace `tests/gemma4_test/test_aiter_w16a16_moe_asm_guards.py` with unified-routing static guards.
- Create `tests/accuracy/test_unified_aiter_moe_operator.py`: small synthetic HCU operator comparisons.
- Modify `tests/integration/parallel/README.md` and `tests/integration/parallel/test_tp_ep_models.py`: rename obsolete direct-ASM path labels without adding model runs to verification.

---

### Task 1: Unified shuffle configuration

**Files:**
- Modify: `vllm_hcu/platforms/envs.py:1-60,318-337`
- Test: `tests/runtime_patch/test_quant_gemm_aiter.py`

**Interfaces:**
- Produces: `resolve_aiter_moe_shuffle() -> bool`
- Produces: dynamic attribute `VLLM_HCU_USE_AITER_MOE_SHUFFLE: bool`
- Preserves: `VLLM_HCU_USE_AITER_W16A16_MOE_SHUFFLE` as a one-release alias

- [ ] **Step 1: Write failing environment-resolution tests**

Add tests that clear the resolver cache around each environment combination:

```python
@pytest.mark.parametrize(
    ("new_value", "legacy_value", "expected"),
    [(None, None, True), ("0", None, False),
     ("1", "0", True), (None, "0", False)],
)
def test_unified_aiter_moe_shuffle_env_precedence(
    monkeypatch, new_value, legacy_value, expected
):
    for name, value in (
        ("VLLM_HCU_USE_AITER_MOE_SHUFFLE", new_value),
        ("VLLM_HCU_USE_AITER_W16A16_MOE_SHUFFLE", legacy_value),
    ):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    henvs.resolve_aiter_moe_shuffle.cache_clear()
    assert henvs.VLLM_HCU_USE_AITER_MOE_SHUFFLE is expected
```

Add `caplog` assertions that explicit legacy use warns once and that the new
variable wins when both are set.

- [ ] **Step 2: Run the new tests and verify RED**

Run:

```bash
python -m pytest tests/runtime_patch/test_quant_gemm_aiter.py \
  -k unified_aiter_moe_shuffle_env -vv
```

Expected: FAIL because `resolve_aiter_moe_shuffle` and
`VLLM_HCU_USE_AITER_MOE_SHUFFLE` do not exist.

- [ ] **Step 3: Implement the resolver and compatibility aliases**

Add an `@functools.lru_cache(maxsize=1)` resolver that reads the new variable
first, falls back to the legacy variable, defaults to true, and logs the agreed
deprecation/precedence warning once. Register the new name in
`hcu_vllm_environment_variables` and add it to the `TYPE_CHECKING` declarations.
Retain the legacy entry unchanged for direct compatibility access.

Make the obsolete config flag resolve true and warn when explicitly set false:

```python
@functools.lru_cache(maxsize=1)
def resolve_aiter_moe_config_compat() -> bool:
    raw = os.environ.get("VLLM_HCU_USE_AITER_MOE_CONFIG")
    if raw is not None and raw.lower() not in ("true", "1"):
        logger.warning(
            "VLLM_HCU_USE_AITER_MOE_CONFIG=0 is deprecated and ignored; "
            "unified AITER MoE routing always uses AiterMoeConfig"
        )
    return True
```

- [ ] **Step 4: Verify GREEN**

Run the command from Step 2 and then:

```bash
python -m pytest tests/patch/test_config.py tests/runtime_patch/test_platform_hcu_config.py -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit Task 1**

```bash
git add vllm_hcu/platforms/envs.py tests/runtime_patch/test_quant_gemm_aiter.py
git commit -m "feat(hcu): unify AITER MoE shuffle configuration"
```

---

### Task 2: Central AITER MoE dispatch adapter

**Files:**
- Create: `vllm_hcu/model_executor/layers/fused_moe/aiter_moe_dispatch.py`
- Test: `tests/runtime_patch/test_quant_gemm_aiter.py`

**Interfaces:**
- Produces: immutable `AiterMoeProblem`
- Produces: `select_aiter_moe_config(problem, cache_owner) -> object | None`
- Produces: `prepare_aiter_moe_weights(w1, w2, config, cache_owner, block_shape=None) -> tuple[Tensor, Tensor]`
- Produces: `prepare_aiter_moe_scales(scale1, scale2, config, cache_owner) -> tuple[Tensor | None, Tensor | None]`
- Produces: `aiter_expert_map_for_solution(expert_map, config, global_num_experts) -> Tensor | None`
- Produces: `execute_aiter_moe(selection, *, tensors/scales/metadata) -> Tensor`

- [ ] **Step 1: Write failing selector tests**

Add tests using a fake `aiter.moe` module. Assert the exact selector kwargs:

```python
problem = AiterMoeProblem(
    M=2, E=4, N1=16, N2=8, K=8, top_k=2,
    block_size=0, dtype=torch.bfloat16, device=torch.device("cpu"),
    quant_type="w16a16", activation="silu", use_shuffle=True,
)
selection = select_aiter_moe_config(problem, cache_owner=torch.empty(1))
assert captured == {
    "M": 2, "E": 4, "N1": 16, "N2": 8, "K": 8,
    "top_k": 2, "block_size": 0, "dtype": torch.bfloat16,
    "quant_type": "w16a16", "activation": "silu", "use_shuffle": 1,
}
assert "spec_sol_type" not in captured
```

Cover a true valid config, `status=False -> None`, and
`status=True/config=None -> HcuAiterMoeDispatchError`.

- [ ] **Step 2: Run selector tests and verify RED**

```bash
python -m pytest tests/runtime_patch/test_quant_gemm_aiter.py \
  -k aiter_dispatch_selector -vv
```

Expected: collection/import FAIL because the adapter module does not exist.

- [ ] **Step 3: Implement `AiterMoeProblem` and selection**

Create the module with a frozen dataclass, a context-rich error class, bounded
cache ownership helpers, and selector behavior. Use the problem fields as the
cache key, but do not pass `device` to AITER. Do not catch AITER import or
execution exceptions.

- [ ] **Step 4: Verify selector GREEN**

Run the command from Step 2. Expected: all selector tests pass.

- [ ] **Step 5: Write failing layout/cache/map tests**

Add parameterized tests for `ASM`, `MOE_C`, `TRITON`, and `CK`:

```python
@pytest.mark.parametrize(
    ("solution", "need_shuffle", "should_shuffle"),
    [("asm", True, True), ("moe_c", True, True),
     ("triton", False, False), ("ck", False, False)],
)
def test_aiter_dispatch_prepares_solution_layout(
    monkeypatch, solution, need_shuffle, should_shuffle
):
    calls = []
    monkeypatch.setitem(
        sys.modules,
        "aiter.moe",
        _module(
            "aiter.moe",
            aiter_moe_shfl_weight=lambda w1, w2, config: (
                calls.append(config) or w1.clone(),
                w2.clone(),
            ),
        ),
    )
    config = SimpleNamespace(
        quant_type="w16a16",
        solution_type=solution,
        need_shuffle=need_shuffle,
        config={"layout": solution},
    )
    w1 = torch.ones((2, 8, 4))
    w2 = torch.ones((2, 4, 4))
    actual_w1, actual_w2 = prepare_aiter_moe_weights(
        w1, w2, config, cache_owner=w1
    )
    assert bool(calls) is should_shuffle
    assert (actual_w1 is not w1) is should_shuffle
    assert (actual_w2 is not w2) is should_shuffle
```

Assert repeated calls reuse derived tensors, an in-place `w1.add_(1)` causes a
new shuffle, and config data that changes layout does not reuse the old entry.
Assert ASM converts `[-1, 0, 1, -1]` to an `int32` mask while other solutions
preserve the original global-to-local map object.

- [ ] **Step 6: Run layout tests and verify RED**

```bash
python -m pytest tests/runtime_patch/test_quant_gemm_aiter.py \
  -k 'aiter_dispatch_prepares or aiter_dispatch_expert_map' -vv
```

Expected: FAIL because layout preparation and map normalization are absent.

- [ ] **Step 7: Implement preparation and execution helpers**

Use `aiter_moe_shfl_weight` only for `need_shuffle=True`, validate returned
tensors and shapes, and bound the per-owner cache at eight entries. Use
`aiter_moe_shfl_scale` only for `need_shuffle_scale=True`. Implement ASM-only
map conversion and pass all public AITER arguments through `execute_aiter_moe`.
Wrap only local contract-validation errors with the formatted problem and
solution; allow AITER exceptions to remain the cause.

- [ ] **Step 8: Verify Task 2 GREEN and regressions**

```bash
python -m pytest tests/runtime_patch/test_quant_gemm_aiter.py \
  -k 'aiter_dispatch' -vv
python -m pytest tests/runtime_patch/test_moe_deepep.py -q
```

Expected: all selected tests pass.

- [ ] **Step 9: Commit Task 2**

```bash
git add vllm_hcu/model_executor/layers/fused_moe/aiter_moe_dispatch.py \
  tests/runtime_patch/test_quant_gemm_aiter.py
git commit -m "feat(hcu): add unified AITER MoE dispatcher"
```

---

### Task 3: W16A16 unified routing and canonical weights

**Files:**
- Modify: `vllm_hcu/model_executor/layers/fused_moe/aiter_runtime.py:610-955`
- Modify: `vllm_hcu/model_executor/layers/fused_moe/unquantized_fused_moe_method.py`
- Replace tests: `tests/gemma4_test/test_aiter_w16a16_moe_asm_guards.py`
- Test: `tests/runtime_patch/test_quant_gemm_aiter.py:1840-2285`

**Interfaces:**
- Consumes: Task 2 adapter APIs
- Produces: W16A16 AITER dynamic selection with vLLM Triton no-solution fallback
- Preserves: `HcuUnquantizedFusedMoEMethod.process_weights_after_loading(layer)`

- [ ] **Step 1: Replace direct-ASM tests with failing unified-route tests**

Change tests to require `aiter_moe()` for an AITER-selected config and require
vLLM `fused_experts_impl` for `status=False`. Assert both receive canonical
weights. Add a fault test where `get_aiter_moe_config` raises
`RuntimeError("aiter config fault")`; assert that exception propagates and the
fallback call list stays empty.

Replace the static guard assertions with checks that
`spec_sol_type=MoeSolutionType.ASM`, `fused_experts_asm_impl`, and
`_hcu_aiter_moe_asm_packed` are absent.

- [ ] **Step 2: Run W16 tests and verify RED**

```bash
python -m pytest tests/runtime_patch/test_quant_gemm_aiter.py \
  tests/gemma4_test/test_aiter_w16a16_moe_asm_guards.py \
  -k 'w16a16 or configured_w16 or unquantized_moe' -vv
```

Expected: FAIL because W16 still forces/directly calls ASM and pre-shuffles
registered parameters.

- [ ] **Step 3: Implement runtime selection and fallback**

Delete `get_w16a16_moe_config()` and `get_w16a16_moe_solution_id()`. In the
unquantized branch build `AiterMoeProblem` with `M=hidden_states.shape[0]`,
`N1=w1.shape[1]`, `N2=w2.shape[1]`, `K=w1.shape[2]`, and the unified shuffle
flag. On selection, use adapter-prepared weights and `execute_aiter_moe`. On
`None`, import vLLM `fused_experts_impl` and call it with unquantized flags,
canonical weights, original top-k data, global expert count, and original map.

- [ ] **Step 4: Preserve canonical weights at load time**

Replace the ASM-specific post-load implementation with one that initializes
`moe_quant_config` and `moe_kernel` for the AITER backend without replacing
`layer.w13_weight` or `layer.w2_weight`. Retain the upstream implementation for
non-AITER backends and retain routing-table compatibility.

- [ ] **Step 5: Verify W16 GREEN**

Run the command from Step 2 plus:

```bash
python -m pytest tests/runtime_patch/test_quant_gemm_aiter.py \
  -k 'aiter_feature_off or explicit_aiter_backend' -vv
```

Expected: all selected tests pass and feature-off delegation is unchanged.

- [ ] **Step 6: Commit Task 3**

```bash
git add vllm_hcu/model_executor/layers/fused_moe/aiter_runtime.py \
  vllm_hcu/model_executor/layers/fused_moe/unquantized_fused_moe_method.py \
  tests/runtime_patch/test_quant_gemm_aiter.py \
  tests/gemma4_test/test_aiter_w16a16_moe_asm_guards.py
git commit -m "refactor(hcu): route W16A16 MoE through AITER"
```

---

### Task 4: compressed-tensors INT8/FP8 W8A8 integration

**Files:**
- Modify: `vllm_hcu/model_executor/layers/quantization/compressed_tensors_moe_runtime.py:31-475`
- Test: `tests/runtime_patch/test_quant_gemm_aiter.py:2880-3725`
- Test: `tests/runtime_patch/test_moe_deepep.py:820-900`

**Interfaces:**
- Consumes: Task 2 adapter APIs
- Produces: `apply_aiter_quantized_moe(...)` with no-solution vLLM Triton fallback
- Produces: `apply_aiter_w8a8_fp8_moe(...)` with the same routing contract

- [ ] **Step 1: Write failing W8A8 fallback and shuffle-hint tests**

Update the existing no-solution case to assert output from a patched vLLM
`fused_experts_impl`, rather than expecting `HcuCompressedTensorsMoeError`.
Assert `use_shuffle=1` is present for INT8 and FP8 config calls. Assert a raised
AITER config exception leaves the fallback call list empty.

- [ ] **Step 2: Run W8A8 tests and verify RED**

```bash
python -m pytest tests/runtime_patch/test_quant_gemm_aiter.py \
  -k 'quantized_aiter_runtime or aiter_w8a8_runtime' -vv
```

Expected: FAIL because the current helpers raise on no solution and omit the
unified shuffle hint.

- [ ] **Step 3: Replace duplicate config/layout helpers with the adapter**

Remove `_get_aiter_quantized_runtime_config`,
`_get_aiter_quantized_weights`, `get_aiter_w8a8_runtime_config`, and
`get_aiter_weights_for_solution` after their call sites use the adapter.
Continue applying BoltOps INT8/FP8 quantization contexts only when the selected
solution token is `ASM`.

- [ ] **Step 4: Add the vLLM Triton fallback**

For selection `None`, call
`vllm.model_executor.layers.fused_moe.fused_moe.fused_experts_impl` with the
original weights/map, correct `use_int8_w8a8` or `use_fp8_w8a8` flag, channel
quantization flag, scales, zero points, activation, global expert count, and
block shape. Do not enter ASM quantization contexts on fallback.

- [ ] **Step 5: Verify W8A8 GREEN and DeepEP compatibility**

```bash
python -m pytest tests/runtime_patch/test_quant_gemm_aiter.py \
  -k 'quantized_aiter_runtime or aiter_w8a8_runtime' -vv
python -m pytest tests/runtime_patch/test_moe_deepep.py -q
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit Task 4**

```bash
git add vllm_hcu/model_executor/layers/quantization/compressed_tensors_moe_runtime.py \
  tests/runtime_patch/test_quant_gemm_aiter.py tests/runtime_patch/test_moe_deepep.py
git commit -m "refactor(hcu): unify quantized AITER MoE routing"
```

---

### Task 5: SlimQuant/Marlin INT8 and W4A16 integration

**Files:**
- Modify: `vllm_hcu/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe_marlin.py:540-750`
- Modify: `vllm_hcu/patch/worker/op_opt/moe/patch_fused_moe.py:77-155`
- Test: `tests/runtime_patch/test_quant_gemm_aiter.py`

**Interfaces:**
- Consumes: Task 2 adapter APIs
- Produces: consistent INT8 W8A8 and W4A16 AITER routing/fallback

- [ ] **Step 1: Write failing SlimQuant/Marlin tests**

Extend the existing method tests to assert that the class no longer owns
`_get_aiter_moe_runtime_config` or `_get_aiter_weights_for_solution`, passes
`use_shuffle=1`, calls vLLM Triton on `status=False`, and does not invoke its
LMSlim Marlin kernel for that fallback.

- [ ] **Step 2: Write failing W4A16 tests**

Patch AITER selection to return each solution type and assert the adapter path
is used. For `status=False`, assert the existing `original()` receives the
original packed weights/scales. Add a config-exception case that proves
`original()` is not called.

- [ ] **Step 3: Run Task 5 tests and verify RED**

```bash
python -m pytest tests/runtime_patch/test_quant_gemm_aiter.py \
  -k 'marlin_aiter_moe or w4a16' -vv
```

Expected: FAIL because both paths own independent selector/layout logic and do
not implement the agreed fallback contract.

- [ ] **Step 4: Integrate SlimQuant/Marlin INT8**

Delete its local AITER config/weight methods, use Task 2 selection/preparation,
and call vLLM `fused_experts_impl` for selection `None`. Keep canonical weights
in the existing AITER post-load branch and leave the non-AITER LMSlim Marlin
branch unchanged.

- [ ] **Step 5: Integrate W4A16**

Replace direct imports of selector/shuffle/execute functions with Task 2
adapter calls. Use `prepare_aiter_moe_scales` for `need_shuffle_scale`. Preserve
the current `original()` call exactly as the no-solution and feature-disabled
path.

- [ ] **Step 6: Verify Task 5 GREEN**

```bash
python -m pytest tests/runtime_patch/test_quant_gemm_aiter.py \
  -k 'marlin_aiter_moe or w4a16' -vv
python -m pytest tests/runtime_patch/test_quant_gemm_aiter.py -q
```

Expected: the complete focused file passes.

- [ ] **Step 7: Commit Task 5**

```bash
git add \
  vllm_hcu/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe_marlin.py \
  vllm_hcu/patch/worker/op_opt/moe/patch_fused_moe.py \
  tests/runtime_patch/test_quant_gemm_aiter.py
git commit -m "refactor(hcu): share AITER routing across MoE quantizers"
```

---

### Task 6: Route labels, source guards, and scoped regression suite

**Files:**
- Modify: `tests/integration/parallel/test_tp_ep_models.py:24-60`
- Modify: `tests/integration/parallel/README.md:12-20`
- Modify: `tests/gemma4_test/test_aiter_w16a16_moe_asm_guards.py`

**Interfaces:**
- Produces: route names that describe AITER automatic routing, not direct ASM

- [ ] **Step 1: Write/adjust source guard assertions**

Require every plugin-owned `get_aiter_moe_config(` call to be adapter-owned,
and assert production source contains neither `spec_sol_type=` nor direct
`fused_experts_asm_impl(` calls for MoE dispatch.

- [ ] **Step 2: Run source guards and verify RED if stale paths remain**

```bash
python -m pytest tests/gemma4_test/test_aiter_w16a16_moe_asm_guards.py -vv
rg -n 'spec_sol_type|fused_experts_asm_impl' \
  vllm_hcu/model_executor/layers/fused_moe \
  vllm_hcu/model_executor/layers/quantization \
  vllm_hcu/patch/worker/op_opt/moe
```

Expected before cleanup: the test or search identifies obsolete direct-ASM
dispatch references. Quantization-context imports of the ASM module are allowed
only when they do not execute the MoE kernel directly.

- [ ] **Step 3: Rename integration matrix labels**

Replace `aiter-tuned-shuffle`, `aiter-asm-shuffle`, and
`aiter-asm-nonshuffle` with `aiter-auto-shuffle` and
`aiter-auto-nonshuffle`. Use `VLLM_HCU_USE_AITER_MOE_SHUFFLE` in the matrix.
Do not execute these model cases in this task.

- [ ] **Step 4: Run scoped CPU/static regression tests**

```bash
python -m pytest \
  tests/gemma4_test/test_aiter_w16a16_moe_asm_guards.py \
  tests/runtime_patch/test_quant_gemm_aiter.py \
  tests/runtime_patch/test_moe_deepep.py \
  tests/patch/test_config.py \
  tests/patch/test_module_exchange.py -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit Task 6**

```bash
git add tests/gemma4_test/test_aiter_w16a16_moe_asm_guards.py \
  tests/integration/parallel/test_tp_ep_models.py \
  tests/integration/parallel/README.md
git commit -m "test(hcu): align MoE coverage with AITER auto routing"
```

---

### Task 7: Synthetic HCU operator verification

**Files:**
- Create: `tests/accuracy/test_unified_aiter_moe_operator.py`
- Modify: `docs/aiter_quantized_moe_validation.md`

**Interfaces:**
- Consumes: production adapter from Task 2 and integrated call sites
- Produces: device-gated synthetic W16A16, INT8 W8A8, and FP8 W8A8 comparisons

- [ ] **Step 1: Write the device-gated operator tests**

Create tests that skip unless PyTorch exposes a ROCm/HCU device and the public
AITER MoE API imports. Use deterministic seeds and small aligned tensors. For
each case, query the adapter at `M in (1, 16, 64)`, record `solution_type`, run
unified dispatch, and compare against vLLM `fused_experts_impl`.

Use these initial tolerances, tightening them if the observed kernels permit:

```python
TOLERANCES = {
    "w16a16": {"rtol": 2e-2, "atol": 2e-2},
    "int8_w8a8": {"rtol": 8e-2, "atol": 8e-2},
    "fp8_w8a8": {"rtol": 5e-2, "atol": 5e-2},
}
```

The test must assert finite outputs and print the selected route only under
pytest verbose diagnostics. Optional unavailable quantized kernels use
`pytest.skip()` with the missing capability in the message.

- [ ] **Step 2: Run the new operator test**

```bash
HIP_VISIBLE_DEVICES=0 python -m pytest \
  tests/accuracy/test_unified_aiter_moe_operator.py -vv -s
```

Expected: each available case passes numerically; unavailable optional cases
are explicit skips. Any failure is investigated with the systematic-debugging
skill before changing tolerances or production code.

- [ ] **Step 3: Record the operator-only validation contract**

Add the exact command, tensor shapes, selected solution types, tolerances, and
pass/skip counts to `docs/aiter_quantized_moe_validation.md`. State explicitly
that no checkpoint or server was used.

- [ ] **Step 4: Run the complete scoped verification**

```bash
python -m pytest \
  tests/gemma4_test/test_aiter_w16a16_moe_asm_guards.py \
  tests/runtime_patch/test_quant_gemm_aiter.py \
  tests/runtime_patch/test_moe_deepep.py \
  tests/patch/test_config.py \
  tests/patch/test_module_exchange.py -q
HIP_VISIBLE_DEVICES=0 python -m pytest \
  tests/accuracy/test_unified_aiter_moe_operator.py -vv -s
git diff --check
```

Expected: zero failures, explicit device/capability skips only, and clean diff
whitespace.

- [ ] **Step 5: Audit acceptance criteria and commit Task 7**

```bash
rg -n 'get_aiter_moe_config\(' vllm_hcu
rg -n 'spec_sol_type|fused_experts_asm_impl\(' vllm_hcu
rg -n 'VLLM_HCU_USE_AITER_(W16A16_)?MOE_SHUFFLE' vllm_hcu tests docs
```

Every selector call must be centralized, every intended compatibility alias
must be explained, and no direct ASM MoE execution may remain.

```bash
git add tests/accuracy/test_unified_aiter_moe_operator.py \
  docs/aiter_quantized_moe_validation.md
git commit -m "test(hcu): validate unified AITER MoE operators"
```

---

## Final Verification Gate

Before reporting completion, invoke `superpowers:verification-before-completion`
and freshly run Task 7 Step 4. Report exact pass/skip/failure counts, selected
AITER solution types observed by the operator test, and the final commit list.
Do not claim model-level accuracy or throughput; neither is in scope.
