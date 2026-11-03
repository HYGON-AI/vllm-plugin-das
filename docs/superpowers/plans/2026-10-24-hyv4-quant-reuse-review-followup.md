# HYV4 Quant Reuse Review Follow-up Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the two remaining review findings without changing routed DeepEP quantization semantics or regressing the verified HYV4 W4A8 accuracy and decode performance.

**Architecture:** Promote the persistent activation kernel's expert index before every expert-stride multiplication. Add a separate shared-expert-only quantized pair capability from `MoERunner` through SlimQuant into the HCU modular kernel; routed dispatch continues to consume the original BF16 tensor, while unsupported quant methods keep using the public `forward_modular` fallback.

**Tech Stack:** Python, PyTorch, Triton, vLLM 0.25.1, pytest, ROCm/HCU

**Spec:** `docs/superpowers/specs/2026-10-24-hyv4-int8-quant-reuse-design.md`

## Global Constraints

- Keep all new parameters optional and preserve behavior when fused RMS quantization is disabled or unsupported.
- Do not change the upstream `RoutedExperts.forward_modular` interface.
- Do not send pre-quantized INT8 activations through routed DeepEP dispatch.
- HumanEval cases 0--63 must remain at least 63/64.
- Performance validation is DP8, EP8, MTP3, DeepEP low latency, FP8 E4M3 KV cache, FULL_AND_PIECEWISE, input 1, output 1024, concurrency 128, 16 warmups, and 128 measured requests.
- Investigate any mean throughput regression greater than 2% against the current 1308.22 tok/s W4A8 reference.

## Review Focus

- An expert base offset exactly at `2**31` elements must stay positive and correct; Task 1 tests this boundary.
- Token-count, input, output, and scale expert offsets must all derive from an already-promoted expert index; Task 1 inspects the executable kernel AST and evaluates its address expressions.
- Internal overlap must deliver the shared quantized pair exactly once while routed input stays BF16; Task 2 tests the real runner-to-SlimQuant-to-modular boundary.
- NO_OVERLAP must not gain a second shared invocation; Task 2 retains and extends the order-sensitive contract test.
- A quant method without shared-only support must continue through public `forward_modular`; Task 2 pins this fallback.

---

### Task 1: Make persistent expert offsets 64-bit safe

**Files:**
- Modify: `vllm_hcu/model_executor/layers/fused_moe/experts/w4a8_persistent_activation.py`
- Test: `tests/runtime_patch/test_quant_gemm_aiter.py`

**Interfaces:**
- Consumes: Triton's `tl.program_id(0)` and tensor element strides.
- Produces: `_persistent_masked_m_silu_mul_clamp_int8_quant` whose `expert` value is `tl.int64` before all expert-stride products.

- [x] **Step 1: Write the failing large-offset semantic test**

  Add `test_persistent_w4a8_expert_offsets_promote_before_stride_multiply`. Load the real kernel function body through the existing AST test utilities, execute the address expressions with `expert=16` and `stride_i_e=134217728`, and assert the input offset is the literal `2147483648`; also assert every expert-stride expression consumes the promoted value.

- [x] **Step 2: Run the focused test and confirm RED**

  Run: `pytest -q tests/runtime_patch/test_quant_gemm_aiter.py::test_persistent_w4a8_expert_offsets_promote_before_stride_multiply`

  Expected: FAIL because `expert` is currently an int32 division result and the simulated input offset becomes `-2147483648`.

- [x] **Step 3: Promote the expert index in the Triton kernel**

  Change the expert assignment to cast `(pid // PARALLEL_TOKENS)` to `tl.int64` before it is used by token-count, input, output, or scale address arithmetic. Keep `token_slot` and the rest of the kernel unchanged.

- [x] **Step 4: Run the focused and existing persistent-kernel tests**

  Run: `pytest -q tests/runtime_patch/test_quant_gemm_aiter.py -k 'persistent and (w4a8 or int8)' tests/accuracy/test_unified_aiter_moe_operator.py -k persistent_int8`

  Expected: all selected tests PASS; on an HCU host, the accuracy entry compiles and runs the kernel.

- [x] **Step 5: Commit the independently reviewable fix**

  Run: `git add vllm_hcu/model_executor/layers/fused_moe/experts/w4a8_persistent_activation.py tests/runtime_patch/test_quant_gemm_aiter.py && git commit -m "fix(moe): use 64-bit persistent expert offsets"`

### Task 2: Preserve shared quant reuse in internal overlap

**Files:**
- Modify: `vllm_hcu/model_executor/layers/fused_moe/moe_runner.py`
- Modify: `vllm_hcu/model_executor/layers/quantization/slimquant_w4a8.py`
- Modify: `vllm_hcu/model_executor/layers/fused_moe/modular_kernel.py`
- Test: `tests/models/hy_v4/test_quant_reuse.py`
- Test: `tests/runtime_patch/test_moe_deepep.py`

**Interfaces:**
- Consumes: `shared_experts_x_and_scale_quanted: tuple[torch.Tensor, torch.Tensor] | None` and the existing BF16 `x`.
- Produces: `supports_shared_quanted_inputs() -> bool` on opted-in quant methods; an optional `shared_experts_x_and_scale_quanted` keyword on SlimQuant and `FusedMoEKernelModularImpl.apply`; unchanged routed `i_q`/`i_s` rejection.

- [x] **Step 1: Write failing runner and SlimQuant contract tests**

  Add tests proving: an opted-in method receives the exact shared pair while `x` remains the original floating tensor and routed `i_q`/`i_s` remain absent; an unsupported method still calls public `forward_modular`; incomplete shared pairs are rejected; SlimQuant forwards the complete tuple to its modular kernel but still rejects routed pre-quantization.

- [x] **Step 2: Write the failing modular-kernel order test**

  Exercise the real `FusedMoEKernelModularImpl` finalize path with a minimal async prepare/finalize double and real order selection. Assert `MK_INTERNAL_OVERLAPPED` receives the exact tuple once, while NO_OVERLAP remains handled only by `MoERunner`.

- [x] **Step 3: Run focused tests and confirm RED**

  Run: `pytest -q tests/models/hy_v4/test_quant_reuse.py tests/runtime_patch/test_moe_deepep.py -k 'shared_quanted or internal_overlap or prequantized_dispatch'`

  Expected: new cases FAIL because the runner falls back to `forward_modular` and SlimQuant currently discards the shared-only tuple.

- [x] **Step 4: Add the opt-in capability and runner branch**

  Implement cached `MoERunner._quant_method_supports_shared_quanted_inputs() -> bool`. When routed pre-quantization is unsupported but shared-only support is declared, call the quant method directly with BF16 `x` and `shared_experts_x_and_scale_quanted`; otherwise retain the existing public-forward fallback.

- [x] **Step 5: Thread the shared-only tuple through SlimQuant and the modular kernel**

  Make `SlimQuantW4A8Int8AiterMoEMethod.supports_shared_quanted_inputs() -> bool` return true, validate the tuple, and pass it to `moe_kernel.apply`. Add the optional keyword to `FusedMoEKernelModularImpl.apply`, use it only during `_finalize`, and leave `_prepare` on BF16 unless the separate routed pair is present.

- [x] **Step 6: Run focused tests and confirm GREEN**

  Run: `pytest -q tests/models/hy_v4/test_quant_reuse.py tests/runtime_patch/test_moe_deepep.py -k 'shared_quanted or internal_overlap or prequantized_dispatch'`

  Expected: all selected tests PASS, with one shared quantization consumer call in every supported overlap order.

- [x] **Step 7: Run the related regression suites**

  Run: `pytest -q tests/models/hy_v4/test_quant_reuse.py tests/runtime_patch/test_moe_deepep.py tests/runtime_patch/test_quant_gemm_aiter.py`

  Expected: all tests PASS.

- [x] **Step 8: Commit the independently reviewable fix**

  Run: `git add vllm_hcu/model_executor/layers/fused_moe/moe_runner.py vllm_hcu/model_executor/layers/quantization/slimquant_w4a8.py vllm_hcu/model_executor/layers/fused_moe/modular_kernel.py tests/models/hy_v4/test_quant_reuse.py tests/runtime_patch/test_moe_deepep.py && git commit -m "fix(moe): reuse shared quant in internal overlap"`

### Task 3: Verify accuracy, performance, and delivery

**Files:**
- Modify: PR description or benchmark note only if measured results change.

**Interfaces:**
- Consumes: Tasks 1 and 2 plus the existing DP8/EP8/MTP3 launch and benchmark commands.
- Produces: reproducible test, HumanEval, and two-run performance evidence on the final HEAD.

- [x] **Step 1: Run portable repository validation**

  Run the established portable contract suite used for the prior 2571-pass gate, then `git diff --check`.

  Expected: zero failures and no whitespace errors.

- [x] **Step 2: Run the full 64-case HumanEval gate**

  Launch the final HEAD with the requested DP8/EP8/MTP3, low-latency DeepEP, FP8 E4M3 KV cache, and FULL_AND_PIECEWISE settings, then run cases 0--63.

  Expected: at least 63/64 pass.

- [x] **Step 3: Measure final performance twice**

  With the same server, run two controlled concurrency-128 benchmarks at input length 1 and output length 1024 with 16 warmups and 128 measured requests.

  Expected: report both runs and their mean; investigate before delivery if the mean is below `1282.06 tok/s` (2% below 1308.22).

- [x] **Step 4: Review and deliver through the existing single PR**

  Request a fresh whole-branch review, fix any substantiated findings, push with the user-supplied access token without interactive username/password prompts, and update the existing PR rather than opening another one.
