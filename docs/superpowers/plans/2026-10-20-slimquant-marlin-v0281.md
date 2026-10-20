# SlimQuant Marlin vLLM 0.28.1 Compatibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix `-q slimquant_marlin` activation-dtype discovery against the DTK 26.10 vLLM 0.28.1 wheel.

**Architecture:** Keep lazy registry discovery in the facade and adapt the target vLLM method shape in the concrete SlimQuant Marlin configuration class. Verify the consumer-visible facade result using the real implementation.

**Tech Stack:** Python 3.12, vLLM 0.28.1 DTK 26.10, pytest

**Spec:** `docs/superpowers/specs/2026-10-20-slimquant-marlin-v0281-design.md`

## Global Constraints

- Do not eagerly import concrete SlimQuant kernels during registry setup.
- Do not alter the existing SlimQuant W4A8 facade behavior.
- Do not claim runtime quantization support for an unquantized BF16 checkpoint.
- Preserve current activation dtypes: FP32, FP16, and BF16.

## Review Focus

- Removing the concrete compatibility method must reproduce the original missing-argument failure through the facade; Task 1's regression test covers it.
- Facade registration must remain lazy; Task 1 reruns the existing registry test.
- The concrete configuration must return exactly FP32, FP16, and BF16; Task 1 asserts literal values.
- W4A8 activation-dtype discovery must not regress; Task 1 reruns all SlimQuant platform-config tests.
- Direct calls on the concrete configuration class must remain valid; Task 1 exercises the class-level API through the facade.

---

### Task 1: Adapt SlimQuant Marlin activation-dtype discovery

**Files:**
- Modify: `vllm_hcu/model_executor/layers/quantization/compressed_tensors/compressed_tensors_marlin.py`
- Modify: `tests/runtime_patch/test_platform_hcu_config.py`

**Interfaces:**
- Consumes: `_SlimQuantFacade.get_supported_act_dtypes()` class-level dispatch and target `CompressedTensorsConfig.get_supported_act_dtypes`.
- Produces: `SlimQuantCompressedTensorsMarlinConfig.get_supported_act_dtypes() -> list[torch.dtype]` callable on the class.

- [x] **Step 1: Write the failing regression test**

Add `test_slimquant_marlin_facade_resolves_supported_activation_dtypes` and
assert the real facade returns `[torch.float32, torch.float16, torch.bfloat16]`.

- [x] **Step 2: Run the regression test and verify RED**

Run: `python3.12 -m pytest -q tests/runtime_patch/test_platform_hcu_config.py::test_slimquant_marlin_facade_resolves_supported_activation_dtypes`

Expected: FAIL with `TypeError` saying `get_supported_act_dtypes()` is missing `cls`.

- [x] **Step 3: Add the concrete class-method adapter**

Implement `SlimQuantCompressedTensorsMarlinConfig.get_supported_act_dtypes()`
as a class method that invokes the target vLLM implementation with the
concrete class argument.

- [x] **Step 4: Verify GREEN and existing SlimQuant contracts**

Run: `python3.12 -m pytest -q tests/runtime_patch/test_platform_hcu_config.py -k slimquant`

Expected: all selected tests pass.

- [x] **Step 5: Commit**

Commit the implementation, regression test, spec, and plan with message
`fix(quantization): adapt slimquant dtype discovery`.
