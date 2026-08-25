# Final fix report: LightOp categorized API v0.25.1

## Status and commit

All actionable final-review code findings are addressed in this fix wave.

- Commit: this commit, `fix: address final LightOp API review findings`
- Starting HEAD: `dabd46b2cb40d104821f61b4e2781ddfd7ee1f76`
- Branch: `feat/lightop-categorized-api-v0251`
- Source compatibility root used by tests:
  `/models/zb/vllm_025/vllm`

The commit SHA is reported by the final handoff because a file cannot contain
the SHA of the same commit that contains it without changing that SHA.

## Finding disposition

### Important 1: paged MQA clean-logit behavior

Fixed. `rocm_fp8_paged_mqa_logits` again passes `False` as the final
`clean_logits` argument to `lightop.attention.paged_mqa_logits`. This restores
the behavior on `origin/v0.25.1`; only the weights conversion to FP32 contiguous
layout remains part of the categorized API migration.

The runtime test now executes the real owner function against an exact fake
kernel signature and observes all eight arguments. It asserts that schedule
metadata remains `None`, weights are FP32 contiguous, the returned object is
preserved, and cleanup remains disabled.

### Important 2: broad LMSlim fallback import catches

Fixed. Both LMSlim compatibility imports in `int8_runtime.py` now catch only
`(ImportError, AttributeError)`:

- `lmslim.layers.gemm.int8_utils.per_token_quant_int8`
- `lmslim.quant_ops`

Two runtime-routing tests execute the real `apply_int8_linear` function while
the categorized export is absent and the fallback import raises a synthetic
binary-initialization `RuntimeError`. Each test asserts that the exact original
exception propagates instead of being translated into dependency absence. The
existing kernel-execution translation remains unchanged except for neutral
wording.

### Minor 1: AST-only module-exchange detector

Fixed by removal. The AST token detector
`test_deep_gemm_replacements_keep_categorized_lightop_boundaries` was deleted
and was not replaced by another source-string/import-token assertion. Existing
module-exchange surface tests remain, while the stronger tests in
`test_lightop_ops_api.py` execute the real DeepGEMM import boundaries and
methods and assert categorized routing, output consumption, legacy fallback,
and warnings.

### Dynamic RMS fake contract

Fixed. The fake now has the exact four positional arguments plus keyword-only
`residual` and `update_input`. The test verifies input and weight identity,
epsilon, dtype, `residual=` identity, `update_input is False` after
`bool(None)`, and identity of both returned tensors.

### INT8 wording

Fixed. The module docstring and categorized/compatibility kernel execution and
invalid-status errors now refer neutrally to HCU W8A8 hipBLASLt rather than
mislabeling every selected kernel as LMSlim. Messages specifically describing
the LMSlim fallback import or deprecation remain LMSlim-specific.

### Task 2 top-k buffer

Verified with no production change. The existing tests confirm that the
chunked sparse path passes the caller's exact `topk_indices_buffer` to the
categorized top-k kernel and that the standalone prefill/decode helpers route
through `lightop.attention`.

### Task 6 TDD chronology

Unresolved process Minor. The historical Task 6 implementation/test chronology
cannot be repaired by changing the current code or rewriting history in this
fix wave. It remains explicitly recorded here and does not represent an open
runtime behavior defect.

## TDD record

### RED

Before production changes, the paged-MQA expectation and two real int8 import
routing tests were added. Command:

```text
VLLM_V0251_SOURCE_ROOT=/models/zb/vllm_025/vllm \
HIP_VISIBLE_DEVICES='' ROCR_VISIBLE_DEVICES='' CUDA_VISIBLE_DEVICES='' \
python -m pytest -q \
  tests/runtime_patch/test_sparse_indexer_loading.py::test_rocm_lightop_paged_mqa_preserves_disabled_clean_logits_and_builds_schedule_internally \
  tests/runtime_patch/test_quant_gemm_aiter.py::test_int8_quant_fallback_propagates_lmslim_runtime_import_failure \
  tests/runtime_patch/test_quant_gemm_aiter.py::test_int8_gemm_fallback_propagates_lmslim_runtime_import_failure
```

Actual result: `3 failed, 14 warnings in 6.01s`.

- paged MQA observed `True`, not expected `False`;
- quant fallback produced `HcuInt8LinearError: ... unavailable`, masking the
  injected LMSlim binary-initialization `RuntimeError`;
- GEMM fallback produced the analogous masked-unavailable error.

These were the intended failure reasons.

### GREEN

After the minimal production fixes, the same three behavior paths passed:

```text
VLLM_V0251_SOURCE_ROOT=/models/zb/vllm_025/vllm \
HIP_VISIBLE_DEVICES='' ROCR_VISIBLE_DEVICES='' CUDA_VISIBLE_DEVICES='' \
python -m pytest -q \
  tests/runtime_patch/test_sparse_indexer_loading.py::test_rocm_lightop_paged_mqa_keeps_clean_logits_disabled \
  tests/runtime_patch/test_quant_gemm_aiter.py::test_int8_quant_fallback_propagates_lmslim_runtime_import_failure \
  tests/runtime_patch/test_quant_gemm_aiter.py::test_int8_gemm_fallback_propagates_lmslim_runtime_import_failure
```

Actual result: `3 passed, 14 warnings in 6.38s`.

The strengthened dynamic RMS test also passed separately:

```text
VLLM_V0251_SOURCE_ROOT=/models/zb/vllm_025/vllm \
python -m pytest -q \
  tests/runtime_patch/test_lightop_ops_api.py::test_dynamic_rms_quant_consumes_returned_tensors
```

Actual result: `1 passed, 14 warnings in 18.73s`.

## Focused verification

All commands used `VLLM_V0251_SOURCE_ROOT=/models/zb/vllm_025/vllm`.

1. Sparse/indexer owner, hidden devices:

   ```text
   HIP_VISIBLE_DEVICES='' ROCR_VISIBLE_DEVICES='' CUDA_VISIBLE_DEVICES='' \
   python -m pytest -q tests/runtime_patch/test_sparse_indexer_loading.py
   ```

   Result: `13 passed, 14 warnings in 5.76s`.

2. INT8 owner and portable categorized/compatibility paths, hidden devices:

   ```text
   HIP_VISIBLE_DEVICES='' ROCR_VISIBLE_DEVICES='' CUDA_VISIBLE_DEVICES='' \
   python -m pytest -q \
     tests/runtime_patch/test_quant_gemm_aiter.py::test_int8_hcu_owned_kernel_validates_and_computes_shapes \
     tests/runtime_patch/test_quant_gemm_aiter.py::test_int8_linear_prefers_categorized_lightop_quant_and_gemm \
     tests/runtime_patch/test_quant_gemm_aiter.py::test_int8_quant_fallback_propagates_lmslim_runtime_import_failure \
     tests/runtime_patch/test_quant_gemm_aiter.py::test_int8_gemm_fallback_propagates_lmslim_runtime_import_failure \
     tests/accuracy/test_portable_operator_accuracy.py::test_w8a8_linear_matches_dequantized_reference \
     tests/accuracy/test_portable_operator_accuracy.py::test_w8a8_linear_lmslim_fallback_warns_once_and_matches_reference
   ```

   Result: `9 passed in 2.90s` (the parameterized reference test contributes
   four cases).

3. Dynamic RMS and real DeepGEMM routing behavior, visible configured device:

   ```text
   python -m pytest -q tests/runtime_patch/test_lightop_ops_api.py \
     -k 'dynamic_rms_quant or deep_gemm'
   ```

   Result: `9 passed, 16 deselected, 14 warnings in 20.05s`.

4. Remaining module-exchange DeepGEMM selectors, hidden devices:

   ```text
   HIP_VISIBLE_DEVICES='' ROCR_VISIBLE_DEVICES='' CUDA_VISIBLE_DEVICES='' \
   python -m pytest -q tests/patch/test_module_exchange.py -k 'deep_gemm'
   ```

   Result: `5 passed, 16 deselected in 0.10s`.

5. Task 2 top-k buffer/routing regression coverage, hidden devices:

   ```text
   HIP_VISIBLE_DEVICES='' ROCR_VISIBLE_DEVICES='' CUDA_VISIBLE_DEVICES='' \
   python -m pytest -q \
     tests/runtime_patch/test_lightop_attention_api.py::test_chunked_sparse_mla_uses_new_abi_and_categorized_topk \
     tests/runtime_patch/test_lightop_attention_api.py::test_sparse_mla_topk_helpers_use_categorized_attention_kernels
   ```

   Result: `2 passed, 14 warnings in 5.45s`.

The 14 repeated warnings are Torch JIT `script_method` deprecation warnings
from installed dependencies. No target test emitted a LightOp routing warning
outside the cases that explicitly assert compatibility warnings.

Immediately before commit, all hidden-device focused cases above were rerun in
one fresh pytest invocation, with the complete `test_module_exchange.py` file
rather than only its DeepGEMM selection. Result:
`45 passed, 14 warnings in 6.66s`. The visible-device dynamic RMS/DeepGEMM
selection was also rerun fresh and produced
`9 passed, 16 deselected, 14 warnings in 20.01s`.

## Environment notes and static checks

- A combined GREEN attempt with all devices hidden could not collect
  `test_lightop_ops_api.py`: its import path queried CUDA capability and raised
  `RuntimeError: No CUDA GPUs are available`. The focused dynamic RMS/DeepGEMM
  command therefore used the configured visible device and passed. No
  visible-device full suite was run.
- `ruff format --check ...` could not run because `ruff` is not installed.
- `python -m black --check ...` could not run because the `black` module is not
  installed.
- `python -m py_compile` over all six changed Python files passed with exit 0.
- `git diff --check` passed with exit 0.
- Real HCU numerical kernel validation was not run; these portable tests prove
  routing, exception, argument, and output-ownership contracts only.

## Self-review

- Compared paged MQA with `origin/v0.25.1` and confirmed only the unintended
  clean-logit boolean is restored; schedule metadata and FP32-contiguous weight
  migration remain unchanged.
- Checked both compatibility selectors catch exactly `ImportError` and
  `AttributeError`; kernel execution still translates errors once and never
  retries another API.
- Searched tests for the old LMSlim-only execution wording; no assertion or
  stale production execution label remains.
- Confirmed the AST-only module-exchange test is removed without introducing a
  replacement source-token detector.
- Mutation check: changing paged cleanup back to `True`, broadening either
  LMSlim import catch, loosening the dynamic RMS call shape, or changing its
  returned tensors would fail the new/strengthened behavior tests.
- Reviewed the complete six-file code diff plus this report and found no
  unrelated production changes.

## Remaining concerns

- Historical Task 6 TDD chronology remains the one unresolved process Minor.
- Hardware numerical behavior remains outside this CPU/mock-focused final fix
  wave and requires a validated HCU environment.
