# vLLM v0.25 HCU AITER Quantized W8A8 MoE Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make compressed-tensors FP8 and INT8 W8A8 MoE select the HCU public AITER MoE API through `--moe-backend aiter` on vLLM v0.25.1.

**Architecture:** Patch the v0.25 INT8/FP8 oracles and `AiterExperts` compatibility boundary while preserving target-owned modular kernels. A quantized runtime translates `FusedMoEQuantConfig` to the installed `aiter.moe` ABI and owns configuration and solution-specific weight-shuffle caches.

**Tech Stack:** Python 3.10, PyTorch, vLLM v0.25.1, HCU AITER `aiter.moe`, pytest, EvalScope.

**Spec:** `docs/superpowers/specs/2026-08-21-v025-aiter-quantized-w8a8-moe-design.md`

## Global Constraints

- Implement on `fix/hcu-v025-aiter-w8a8-quantized` based directly on `origin/fix/glm51-pp-mtp-mrv2`.
- Do not modify `/models/zb/vllm_025/vllm` or copy v0.21 compressed-tensors methods.
- Preserve v0.25 modular prepare/finalize, routing, shared-expert, output-workspace, and graph lifecycles.
- `--moe-backend aiter` is explicit and must never silently fall back.
- Keep INT8 automatic backend priority unchanged.
- Keep `slimquant_marlin` available as the INT8 accuracy baseline.
- Keep quantized public-API logic out of `aiter_runtime.py`.
- Use test-first red-green-refactor for every production behavior.

---

### Task 1: Register Explicit INT8 AITER Backend Selection

**Files:**
- Create: `vllm_hcu/patch/worker/op_opt/moe/patch_int8_oracle.py`
- Modify: `vllm_hcu/patch/worker/__init__.py`
- Test: `tests/runtime_patch/test_quant_gemm_aiter.py`

**Interfaces:**
- Consumes: target `Int8MoeBackend`, `backend_to_kernel_cls`,
  `map_int8_backend`, and `convert_to_int8_moe_kernel_format`.
- Produces: `Int8MoeBackend.AITER`, explicit `aiter` mapping to target
  `AiterExperts`, and canonical AITER INT8 weights.

- [ ] **Step 1: Write failing oracle behavior tests**

Build a complete synthetic target oracle with the original enum and functions,
apply `patch_int8_oracle.apply_to_module`, and assert:

```python
assert module.map_int8_backend("aiter") == module.Int8MoeBackend.AITER
assert module.backend_to_kernel_cls(module.Int8MoeBackend.AITER) == [AiterExperts]
assert module.convert_to_int8_moe_kernel_format(
    module.Int8MoeBackend.AITER, w13, w2
) == (w13, w2)
assert module.map_int8_backend("triton").value == "TRITON"
```

Add a worker registry assertion proving the oracle patch is in
`_MOE_FOUNDATION_CALLBACKS` before compressed-tensors modules construct their
methods.

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
python3 -m pytest -q tests/runtime_patch/test_quant_gemm_aiter.py \
  -k 'int8_aiter_oracle or worker_registers_int8_oracle'
```

Expected: collection/import failure because `patch_int8_oracle` and the AITER
enum member do not exist.

- [ ] **Step 3: Implement the minimal oracle sidecar**

Follow `patch_fp8_oracle.py` compatibility validation. Replace the target enum
with a value-preserving enum plus `AITER = "AITER"`; wrap only these branches:

```python
def hcu_map_int8_backend(runner_backend):
    if runner_backend == "aiter":
        return hcu_enum.AITER
    return original_map(runner_backend)

def hcu_backend_to_kernel_cls(backend):
    if backend == hcu_enum.AITER:
        from vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe import AiterExperts
        return [AiterExperts]
    return original_backend_to_cls(backend)

def hcu_convert_to_int8_moe_kernel_format(
    int8_backend, w13, w2, layer=None, w13_scale=None
):
    if int8_backend == hcu_enum.AITER:
        return w13, w2
    return original_convert(int8_backend, w13, w2, layer, w13_scale)
```

Register it in `_MOE_FOUNDATION_CALLBACKS` without changing automatic target
priority.

- [ ] **Step 4: Run focused and neighboring oracle tests**

Run:

```bash
VLLM_V0251_SOURCE_ROOT=/models/zb/vllm_025/vllm \
python3 -m pytest -q tests/runtime_patch/test_quant_gemm_aiter.py \
  -k 'int8_aiter_oracle or worker_registers_int8_oracle or fp8_oracle'
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit INT8 backend registration**

```bash
git add vllm_hcu/patch/worker/op_opt/moe/patch_int8_oracle.py \
  vllm_hcu/patch/worker/__init__.py tests/runtime_patch/test_quant_gemm_aiter.py
git commit -m "feat(hcu): register AITER INT8 MoE backend"
```

### Task 2: Generalize the Public AITER Quantized Runtime

**Files:**
- Modify: `vllm_hcu/model_executor/layers/quantization/compressed_tensors_moe_runtime.py`
- Test: `tests/runtime_patch/test_quant_gemm_aiter.py`

**Interfaces:**
- Consumes: v0.25 `FusedMoEQuantConfig` and installed
  `MoeQuantType.{FP8_W8A8,W8A8}`.
- Produces: `apply_aiter_quantized_moe(...) -> torch.Tensor`, cached runtime
  configuration, and solution-specific canonical/shuffled weight selection.

- [ ] **Step 1: Write failing quant-type and argument-forwarding tests**

Use real tensor fixtures and a complete fake `aiter.moe` module. Parameterize
FP8 and INT8 quant configs so the expected quant types are hand-written:

```python
@pytest.mark.parametrize(
    ("use_fp8", "use_int8", "expected"),
    [(True, False, "fp8_w8a8"), (False, True, "int8_w8a8")],
)
def test_quantized_aiter_runtime_selects_exact_quant_type(...):
    output = apply_aiter_quantized_moe(...)
    assert captured["config_quant_type"] == expected
    assert captured["inplace"] is False
    assert captured["topk_weights"].dtype == torch.float32
    assert captured["topk_ids"].dtype == torch.int32
    assert captured["expert_map"] is expert_map
    assert output is expected_output
```

The fake API must mirror the installed signatures and return an independent
output tensor.

- [ ] **Step 2: Run quant-type test and verify RED**

Run:

```bash
python3 -m pytest -q tests/runtime_patch/test_quant_gemm_aiter.py \
  -k quantized_aiter_runtime_selects_exact_quant_type
```

Expected: failure because `apply_aiter_quantized_moe` does not exist.

- [ ] **Step 3: Implement minimal runtime translation**

Add a function with the target expert boundary explicitly represented:

```python
def apply_aiter_quantized_moe(
    hidden_states, w1, w2, topk_weights, topk_ids, vllm_moe_config,
    activation, apply_router_weight_on_input, expert_map, quant_config,
    a1q_scale=None, output_dtype=None,
) -> torch.Tensor:
    ...
```

Select only `FP8_W8A8` or `W8A8`, validate required scales and layouts, call
`get_aiter_moe_config`, prepare weights with `aiter_moe_shfl_weight` when
`need_shuffle`, and call `aiter_moe` with `inplace=False`. Pass
`a1q_scale` ahead of the configured a1 scale, preserve output dtype, and use
`vllm_moe_config.num_experts` plus the supplied expert map.

- [ ] **Step 4: Write and verify failing cache/error tests**

Add separate tests proving:

- identical shape/config calls query AITER configuration once;
- a different token count queries a new configuration;
- `need_shuffle=True` shuffles once for an unchanged tensor generation;
- an in-place weight mutation causes a new shuffle;
- missing scales, mismatched top-k tensors, unavailable enum/API, and
  `apply_router_weight_on_input=True` raise `HcuCompressedTensorsMoeError`.

Run the new tests before completing cache/validation code and confirm each
fails on the missing behavior, not fixture setup.

- [ ] **Step 5: Complete cache and validation behavior**

Use a bounded function cache for shape-only AITER configs. Store the shuffled
weight cache on the canonical w1 tensor, keyed by both tensor generations and
the AITER quant/solution identity. Do not mutate the canonical parameters.

- [ ] **Step 6: Run all compressed-tensors runtime tests**

Run:

```bash
python3 -m pytest -q tests/runtime_patch/test_quant_gemm_aiter.py \
  -k 'quantized_aiter_runtime or aiter_w8a8_runtime or aiter_weights'
```

Expected: all selected tests pass.

- [ ] **Step 7: Commit the generalized runtime**

```bash
git add vllm_hcu/model_executor/layers/quantization/compressed_tensors_moe_runtime.py \
  tests/runtime_patch/test_quant_gemm_aiter.py
git commit -m "feat(hcu): add public AITER quantized MoE runtime"
```

### Task 3: Route v0.25 AiterExperts Through the Quantized Runtime

**Files:**
- Modify: `vllm_hcu/patch/worker/op_opt/moe/patch_rocm_aiter_moe.py`
- Test: `tests/runtime_patch/test_moe_deepep.py`

**Interfaces:**
- Consumes: `apply_aiter_quantized_moe(...)` from Task 2 and target
  `rocm_aiter_fused_experts`.
- Produces: FP8/INT8 interception while preserving target behavior for
  unquantized and unrelated quant schemes.

- [ ] **Step 1: Write failing expert support and routing tests**

Extend the synthetic target with `kInt8StaticChannelSym` and
`kInt8DynamicTokenSym`. Assert the patched class reports this pair supported.
Exercise the wrapped function with complete FP8 and INT8 quant configs and
assert the real plugin runtime function returns the result while the audited
target function is not entered. Exercise an unquantized config and assert the
target result is retained.

- [ ] **Step 2: Run expert tests and verify RED**

Run:

```bash
python3 -m pytest -q tests/runtime_patch/test_moe_deepep.py \
  -k 'aiter_experts_support_int8 or quantized_aiter_experts_route'
```

Expected: INT8 support is false and quantized calls enter the target function.

- [ ] **Step 3: Implement minimal expert-boundary interception**

Wrap `AiterExperts._supports_quant_scheme` to add only:

```python
(target.kInt8StaticChannelSym, target.kInt8DynamicTokenSym)
```

In `hcu_fused_experts`, detect `quant_config.use_fp8_w8a8` or
`quant_config.use_int8_w8a8` and call `apply_aiter_quantized_moe` with all
target arguments. Retain the existing GELU-tanh rebuild and request context
for every other scheme.

- [ ] **Step 4: Run full AITER expert adapter tests**

Run:

```bash
VLLM_V0251_SOURCE_ROOT=/models/zb/vllm_025/vllm \
python3 -m pytest -q tests/runtime_patch/test_moe_deepep.py
```

Expected: all tests pass, including cold replacement and shared-expert
contracts.

- [ ] **Step 5: Commit expert routing**

```bash
git add vllm_hcu/patch/worker/op_opt/moe/patch_rocm_aiter_moe.py \
  tests/runtime_patch/test_moe_deepep.py
git commit -m "feat(hcu): route quantized experts through AITER"
```

### Task 4: Enable the Explicit FP8 AITER Load Path

**Files:**
- Modify: `vllm_hcu/patch/worker/op_opt/moe/patch_fp8_oracle.py`
- Modify: `vllm_hcu/patch/worker/op_opt/patch_compressed_tensors_moe_w8a8_fp8.py`
- Test: `tests/runtime_patch/test_quant_gemm_aiter.py`

**Interfaces:**
- Consumes: target `Fp8MoeBackend.AITER` and channel/token compressed-tensors
  method constructor.
- Produces: explicit AITER or Triton validation and canonical AITER FP8 weights.

- [ ] **Step 1: Write failing FP8 selection tests**

Update the constructor fixture to assert:

```python
assert construct(moe_backend="aiter").fp8_backend.value == "AITER"
assert construct(moe_backend="triton").fp8_backend.value == "TRITON"
with pytest.raises(RuntimeError, match="explicit.*aiter.*triton"):
    construct(moe_backend="auto")
```

Add an FP8 oracle test passing canonical tensors to AITER conversion and assert
object identity for all four returned tensors. Its fake legacy
`shuffle_weights` must raise if called.

- [ ] **Step 2: Run FP8 tests and verify RED**

Run:

```bash
python3 -m pytest -q tests/runtime_patch/test_quant_gemm_aiter.py \
  -k 'channel_fp8_moe or fp8_aiter_keeps_canonical_weights'
```

Expected: explicit AITER is rejected by the current target-Triton policy and
the oracle invokes legacy shuffle.

- [ ] **Step 3: Implement explicit route validation**

Replace the target-Triton-only policy with a mapping:

```python
expected = {"aiter": "AITER", "triton": "TRITON"}
requested = getattr(moe, "moe_backend", None)
if requested not in expected:
    raise RuntimeError(
        "Channel-FP8 MoE requires explicit --moe-backend aiter or triton"
    )
original_init(...)
if _selected_backend_name(self) != expected[requested]:
    raise RuntimeError("target selected a backend different from the request")
```

In `patch_fp8_oracle.hcu_convert_to_fp8_moe_kernel_format`, return canonical
weights/scales for `hcu_enum.AITER`; preserve every other branch.

- [ ] **Step 4: Run focused and complete quantization tests**

Run:

```bash
VLLM_V0251_SOURCE_ROOT=/models/zb/vllm_025/vllm \
python3 -m pytest -q tests/runtime_patch/test_quant_gemm_aiter.py \
  tests/runtime_patch/test_platform_hcu_config.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit FP8 selection and conversion**

```bash
git add vllm_hcu/patch/worker/op_opt/moe/patch_fp8_oracle.py \
  vllm_hcu/patch/worker/op_opt/patch_compressed_tensors_moe_w8a8_fp8.py \
  tests/runtime_patch/test_quant_gemm_aiter.py
git commit -m "feat(hcu): enable explicit AITER FP8 MoE"
```

### Task 5: Static and CPU-safe Regression Verification

**Files:**
- Modify only if a focused failing test demonstrates a defect in Tasks 1-4.

**Interfaces:**
- Consumes: all production and test changes.
- Produces: evidence that the plugin stays compatible with vLLM v0.25.1.

- [ ] **Step 1: Run compilation and diff checks**

```bash
python3 -m compileall -q vllm_hcu tests/runtime_patch
git diff --check
```

Expected: both commands exit zero.

- [ ] **Step 2: Run the relevant runtime suite**

```bash
VLLM_V0251_SOURCE_ROOT=/models/zb/vllm_025/vllm \
python3 -m pytest -q \
  tests/runtime_patch/test_quant_gemm_aiter.py \
  tests/runtime_patch/test_moe_deepep.py \
  tests/runtime_patch/test_platform_hcu_config.py
```

Expected: all tests pass. Existing dependency warnings are recorded separately
and no new warnings are accepted without explanation.

- [ ] **Step 3: Review scope**

Confirm `git diff origin/fix/glm51-pp-mtp-mrv2...HEAD` contains no upstream
vLLM edits, no `aiter_runtime.py` edit, no automatic INT8 priority change, and
no unrelated cleanup.

### Task 6: Hardware Operator and Full-Model Accuracy Validation

**Files:**
- Modify: `docs/aiter_quantized_w8a8_validation.md`

**Interfaces:**
- Consumes: installed HCU AITER, target Triton kernels, both Qwen3.5 model
  directories, vLLM OpenAI server, and EvalScope HumanEval.
- Produces: reproducible commands, logs, operator errors, model outputs, and
  paired HumanEval scores.

- [ ] **Step 1: Run FP8 and INT8 operator comparisons**

Use the same hidden states, top-k tensors, canonical weights, and scales for
each pair. Seed PyTorch deterministically. Record output dtype/shape,
`torch.isfinite(...).all()`, maximum absolute error, and mean absolute error
for AITER versus Triton. Run once without graph capture so operator errors are
not hidden by server initialization.

- [ ] **Step 2: Start the INT8 AITER graph-enabled service**

Run from `/models/zb/vllm_025/vllm` with the plugin worktree on `PYTHONPATH`:

```bash
VLLM_USE_V2_MODEL_RUNNER=1 \
PYTHONPATH=/models/.worktrees/vllm-plugin-das-aiter-quantized:$PYTHONPATH \
vllm serve /models/Qwen3.5-35B-A3B-W8A8 \
  --served-model-name qwen35-int8-aiter --tensor-parallel-size 1 \
  --max-model-len 65536 --gpu-memory-utilization 0.90 \
  --moe-backend aiter --port 8011
```

Adjust only tensor parallelism and GPU visibility to fit available physical
cards; do not change generation parameters between paired runs. Confirm logs
contain `Using AITER Int8 MoE backend` and the public `aiter_moe` path.

- [ ] **Step 3: Start the INT8 SlimQuant baseline**

```bash
VLLM_USE_V2_MODEL_RUNNER=1 \
PYTHONPATH=/models/.worktrees/vllm-plugin-das-aiter-quantized:$PYTHONPATH \
vllm serve /models/Qwen3.5-35B-A3B-W8A8 \
  --served-model-name qwen35-int8-slimquant --tensor-parallel-size 1 \
  --max-model-len 65536 --gpu-memory-utilization 0.90 \
  --quantization slimquant_marlin --port 8012
```

- [ ] **Step 4: Start paired FP8 services**

Run `/models/Qwen3.5-35B-A3B-CHANNEL-FP8` first with
`--moe-backend aiter --port 8013`, then with
`--moe-backend triton --port 8014`, using otherwise identical settings. Verify
the logs report `AITER` and `TRITON` respectively.

- [ ] **Step 5: Verify deterministic non-garbled generation**

For every service, submit the same OpenAI chat request with
`temperature=0`, thinking disabled in the template/request, and
`max_tokens=4096`. Store response JSON and verify finish reason, token count,
UTF-8 text, and absence of repeated replacement characters or NaN-related
errors.

- [ ] **Step 6: Run paired EvalScope HumanEval**

Run HumanEval for the INT8 AITER service with:

```bash
evalscope eval \
  --model qwen35-int8-aiter --api-url http://127.0.0.1:8011/v1 \
  --api-key EMPTY --eval-type openai_api --datasets humaneval \
  --limit 32 --eval-batch-size 1 --generation-config \
  '{"temperature":0,"max_tokens":32768,"extra_body":{"chat_template_kwargs":{"enable_thinking":false}}}' \
  --dataset-args '{"humaneval":{"few_shot_random":false,"metrics_list":["Pass@1"]}}' \
  --work-dir /tmp/vllm-hcu-evalscope/qwen35-int8-aiter --no-timestamp
```

Repeat with these exact model, URL, and work-directory triples while retaining
the same dataset and generation arguments:

```text
qwen35-int8-slimquant | http://127.0.0.1:8012/v1 | /tmp/vllm-hcu-evalscope/qwen35-int8-slimquant
qwen35-fp8-aiter      | http://127.0.0.1:8013/v1 | /tmp/vllm-hcu-evalscope/qwen35-fp8-aiter
qwen35-fp8-triton     | http://127.0.0.1:8014/v1 | /tmp/vllm-hcu-evalscope/qwen35-fp8-triton
```

Record raw pass@1 and paired difference; do not round before comparison.

- [ ] **Step 7: Document actual results**

Create `docs/aiter_quantized_w8a8_validation.md` containing exact environment,
GPU mapping, service commands, client commands, backend log evidence, operator
error metrics, example output checks, EvalScope work directories, raw scores,
and any external runtime limitation encountered.

- [ ] **Step 8: Commit validation documentation**

```bash
git add docs/aiter_quantized_w8a8_validation.md
git commit -m "docs(hcu): record quantized AITER MoE validation"
```

### Task 7: Final Branch and MR Readiness

**Files:**
- Modify only when final verification finds a tested defect or missing
  validation evidence.

**Interfaces:**
- Consumes: all commits and validation artifacts.
- Produces: a clean branch ready for a separate MR.

- [ ] **Step 1: Re-run final verification**

Run Task 5 commands from a clean shell and confirm hardware services are
stopped and GPU memory released.

- [ ] **Step 2: Inspect commit and author compliance**

Confirm every commit has author and committer name `zhangzbb`, email
`1414695739@qq.com`, a conventional commit subject, and no forbidden identity
token in author, committer, email, or message.

- [ ] **Step 3: Review final diff and branch state**

```bash
git status --short --branch
git log --format='%h %an <%ae> | %cn <%ce> | %s' \
  origin/fix/glm51-pp-mtp-mrv2..HEAD
git diff --stat origin/fix/glm51-pp-mtp-mrv2...HEAD
git diff --check origin/fix/glm51-pp-mtp-mrv2...HEAD
```

Expected: clean worktree, only scoped commits/files, and zero diff-check errors.

- [ ] **Step 4: Push and create the separate MR**

Push `fix/hcu-v025-aiter-w8a8-quantized` and create an MR targeting
`fix/glm51-pp-mtp-mrv2`. Include design summary, exact AITER and baseline
commands, unit-test result, operator metrics, HumanEval scores, and known
limitations.
