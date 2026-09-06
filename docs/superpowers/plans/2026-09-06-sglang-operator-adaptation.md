# sglang-das Operator Adaptation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Adapt numerically correct and measurably faster LightOp/AITER LLM operators from sglang-das into the vLLM HCU plugin without violating plugin patch boundaries.

**Architecture:** Build a complete operator decision matrix, screen candidates with reproducible live-HCU accuracy and timing, and add only accepted routes through explicit vLLM 0.25.1 patch adapters and HCU-owned runtime modules. Every new route is gated by `VLLM_HCU_USE_CUSTOM_OPS` plus a leaf environment variable and preserves the existing plugin/vLLM path for disabled or ineligible inputs.

**Tech Stack:** Python 3.10, PyTorch/HCU, vLLM 0.25.1, LightOp 0.6.x, AITER, pytest, EvalScope, Git worktrees.

**Spec:** `docs/superpowers/specs/2026-09-06-sglang-operator-adaptation-design.md`

## Global Constraints

- Do not edit `/models/zb/vllm_025`, installed vLLM files, or canonical `vllm.*` source.
- Base every change on `origin/v0.25.1` and keep unrelated local branch history out of the MR.
- Keep patch registration explicit, ordered, exact-target validated, idempotent, and failure-latched.
- Put substantial HCU behavior in `vllm_hcu`-owned modules, not patch callbacks.
- Gate every new route with `VLLM_HCU_USE_CUSTOM_OPS` and a leaf switch.
- Select layout-mutating backends before weight conversion; never execute a fallback against incompatible packed weights.
- Use an independent numerical reference and record seed, shape, dtype, tolerance, maximum errors, NaN/Inf, contiguity, and mutation.
- Require at least 5% operator speedup on target shapes and no more than 2% end-to-end throughput regression.
- Model coverage is not required for an operator that passes independent live-HCU accuracy.
- Do not write tokens or credentials into files, logs, Git remotes, or command arguments.

---

### Task 1: Reproducible operator audit and benchmark reporting

**Files:**
- Create: `docs/operator_adaptation_audit_v0251.md`
- Create: `tools/benchmark_sglang_operator_candidates.py`
- Create: `tests/accuracy/test_operator_benchmark_reporting.py`
- Modify: `tests/README.md`

**Interfaces:**
- Produces: `TimingSummary(samples_us: tuple[float, ...], median_us: float, minimum_us: float, maximum_us: float)`.
- Produces: `summarize_timings(samples_us: Sequence[float]) -> TimingSummary`.
- Produces: `speedup_percent(baseline_us: float, candidate_us: float) -> float`.
- Produces: CLI selectors `sqrtsoftplus-gate`, `silu-and-mul`, and `w16a16-moe` with `--warmup`, `--iterations`, `--repeats`, `--json-output`, and model-derived shape arguments.
- Produces: an audit table with the fixed dispositions `already-covered`, `adapt-candidate`, `dependency-unavailable`, `no-vllm-seam`, `accuracy-rejected`, `performance-rejected`, and `accepted`.

- [ ] **Step 1: Write failing pure reporting tests**

Add literal, hand-derived cases:

```python
def test_summarize_timings_uses_median_and_extrema():
    result = summarize_timings([7.0, 3.0, 5.0])
    assert result.median_us == 5.0
    assert result.minimum_us == 3.0
    assert result.maximum_us == 7.0


def test_speedup_percent_is_relative_to_baseline():
    assert speedup_percent(10.0, 8.0) == pytest.approx(20.0)
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
python3 -m pytest -q tests/accuracy/test_operator_benchmark_reporting.py
```

Expected: collection fails because `tools.benchmark_sglang_operator_candidates`
does not exist.

- [ ] **Step 3: Implement the reporting core and CLI skeleton**

Implement immutable timing summaries, positive finite sample validation,
speedup calculation, device synchronization around each timed batch, warmup,
repeat collection, JSON serialization, deterministic seeds, and explicit
dependency errors. Do not add an operator route in this task.

- [ ] **Step 4: Run focused and static checks**

Run:

```bash
python3 -m pytest -q tests/accuracy/test_operator_benchmark_reporting.py
python3 tools/benchmark_sglang_operator_candidates.py --help
python3 tools/check_production_boundary.py
```

Expected: tests pass, CLI lists all three selectors, production boundary is
clean.

- [ ] **Step 5: Complete the initial audit matrix**

Enumerate LightOp/AITER call sites from the reference checkout. For each
unique public operator family, record installed ABI, plugin coverage, vLLM
seam, layout semantics, fallback, and disposition. Do not mark an operator
accepted before Task 3 or Task 5 records live-HCU evidence.

- [ ] **Step 6: Commit**

```bash
git add docs/operator_adaptation_audit_v0251.md tools/benchmark_sglang_operator_candidates.py tests/accuracy/test_operator_benchmark_reporting.py tests/README.md
git commit -m "test: add operator adaptation audit harness"
```

### Task 2: LightOp sqrt-softplus router contract

**Files:**
- Create: `vllm_hcu/model_executor/layers/fused_moe/sqrtsoftplus_routing.py`
- Create: `tests/runtime_patch/test_lightop_sqrtsoftplus_routing.py`
- Modify: `vllm_hcu/platforms/envs.py`
- Modify: `vllm_hcu/patch/worker/op_opt/moe/patch_fused_topk_bias_router.py`
- Modify: `tests/runtime_patch/test_environment_routing.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: LightOp `moe_fused_gate_sqrtsoftplus(x, bias, topk, num_fused_shared_experts=0, renormalize=True, routed_scaling_factor=1.0, apply_routed_scaling_factor_on_output=False)`.
- Produces: `can_use_lightop_sqrtsoftplus(gating_output, correction_bias, topk, input_tokens, hash_indices_table) -> bool`.
- Produces: `run_lightop_sqrtsoftplus(gating_output, correction_bias, topk, renormalize, routed_scaling_factor, indices_dtype) -> tuple[torch.Tensor, torch.Tensor]`.
- Produces: `VLLM_HCU_USE_LIGHTOP_SQRTSOFTPLUS_GATE`, default `True`, subordinate to the master switch.
- Preserves: original `vllm_topk_softplus_sqrt` for hash routing, missing bias, unsupported dtype/layout/shape, or disabled switches.

- [ ] **Step 1: Write failing route-selection tests**

Cover the observable behavior that catches a wrong branch:

```python
def test_non_hash_sqrtsoftplus_uses_lightop_when_both_switches_are_enabled(...):
    weights, ids = patched(..., input_tokens=None, hash_indices_table=None)
    assert weights.tolist() == [[0.75, 0.25]]
    assert ids.dtype is requested_indices_dtype


@pytest.mark.parametrize("master, leaf", [(False, True), (True, False)])
def test_sqrtsoftplus_switches_preserve_official_fallback(master, leaf, ...):
    assert patched(...) == official_sentinel


def test_hash_routing_never_calls_lightop(...):
    assert patched(..., hash_indices_table=hash_table) == official_sentinel
```

Use complete module doubles only for the external LightOp/vLLM modules; assert
returned values from the real patch wrapper, not mock call counts.

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```bash
python3 -m pytest -q tests/runtime_patch/test_lightop_sqrtsoftplus_routing.py
```

Expected: failures show the new environment getter/helper and route are absent.

- [ ] **Step 3: Implement the minimal HCU runtime helper**

Eligibility requires HCU tensors, FP32 contiguous two-dimensional logits,
FP32 contiguous one-dimensional bias, matching expert dimension, supported
expert count, `0 < topk <= 16`, no hash table/input-token hash route, and an
installed callable LightOp symbol. Convert returned IDs to the allocated vLLM
index dtype when required. Let malformed explicit backend ABIs raise an
actionable runtime error.

- [ ] **Step 4: Wire the existing exact patch adapter**

Keep `PATCH_ID`, target module, target symbol, signature guard, original
callable and idempotence marker. The wrapper selects the helper only when both
switches are true and eligibility passes; otherwise it performs the existing
hash dtype normalization and calls the original function.

- [ ] **Step 5: Run GREEN and surrounding contracts**

Run:

```bash
python3 -m pytest -q tests/runtime_patch/test_lightop_sqrtsoftplus_routing.py tests/runtime_patch/test_moe_deepep.py -k 'sqrtsoftplus or fused_topk_bias or hash_router'
python3 -m pytest -q tests/runtime_patch/test_environment_routing.py -k sqrtsoftplus
python3 tools/check_patch_test_coverage.py --json
python3 tools/check_production_boundary.py
```

Expected: all selected tests pass and both static gates report clean.

- [ ] **Step 6: Commit**

```bash
git add vllm_hcu/model_executor/layers/fused_moe/sqrtsoftplus_routing.py vllm_hcu/platforms/envs.py vllm_hcu/patch/worker/op_opt/moe/patch_fused_topk_bias_router.py tests/runtime_patch/test_lightop_sqrtsoftplus_routing.py tests/runtime_patch/test_environment_routing.py README.md
git commit -m "feat: route sqrt-softplus gating through LightOp"
```

### Task 3: Live-HCU sqrt-softplus accuracy and performance gate

**Files:**
- Create: `tests/accuracy/test_lightop_sqrtsoftplus_gate.py`
- Modify: `tools/benchmark_sglang_operator_candidates.py`
- Modify: `docs/operator_adaptation_audit_v0251.md`

**Interfaces:**
- Consumes: Task 2 `run_lightop_sqrtsoftplus`.
- Produces: independent FP32 torch reference implementing softplus, square
  root, correction-bias selection, gather, optional renormalization, and
  routed scaling.
- Produces: benchmark JSON for expert counts 256 and 384, top-k values 6 and
  8, and token counts `1, 2, 4, 8, 16, 32, 64, 128, 256, 1024` where supported.

- [ ] **Step 1: Write the live-HCU numerical tests before changing benchmark execution**

Parameterize deterministic random, zero, large positive/negative and tied
selection inputs. Assert exact expert IDs when selection is unambiguous,
`torch.testing.assert_close` for weights with explicit tolerances, finite
outputs, and unchanged inputs.

- [ ] **Step 2: Run on one idle HCU and verify the test detects an intentionally unsupported route**

Run the eligible matrix, then temporarily select one deliberately unsupported
expert count in the test and verify it exercises the official fallback. Remove
no assertions after RED; separate eligible-kernel and ineligible-fallback
cases.

```bash
HIP_VISIBLE_DEVICES=0 python3 -m pytest -q -s tests/accuracy/test_lightop_sqrtsoftplus_gate.py
```

- [ ] **Step 3: Add the benchmark runner for the accepted semantic matrix**

Time LightOp against the current patched-off vLLM implementation using the
same preallocated inputs. Report warmup, iterations, repeats, per-shape median,
dispersion and speedup; never assert timing inside pytest.

- [ ] **Step 4: Execute the performance gate**

```bash
HIP_VISIBLE_DEVICES=0 python3 tools/benchmark_sglang_operator_candidates.py sqrtsoftplus-gate --warmup 50 --iterations 200 --repeats 7 --json-output /tmp/vllm-hcu-sqrtsoftplus-benchmark.json
```

If all target shapes fail the 5% threshold, revert Task 2's runtime route and
retain only the audit/accuracy/benchmark evidence with disposition
`performance-rejected`. Otherwise record the accepted shape predicate and
keep the route limited to it.

- [ ] **Step 5: Re-run accuracy after final shape predicate and commit**

```bash
HIP_VISIBLE_DEVICES=0 python3 -m pytest -q -s tests/accuracy/test_lightop_sqrtsoftplus_gate.py
git add tests/accuracy/test_lightop_sqrtsoftplus_gate.py tools/benchmark_sglang_operator_candidates.py docs/operator_adaptation_audit_v0251.md vllm_hcu tests/runtime_patch README.md
git commit -m "test: validate LightOp sqrt-softplus routing"
```

### Task 4: LightOp W16A16 Marlin MoE backend

**Files:**
- Create: `vllm_hcu/model_executor/layers/fused_moe/experts/lightop_w16a16_moe.py`
- Create: `vllm_hcu/model_executor/layers/fused_moe/lightop_w16a16_runtime.py`
- Create: `vllm_hcu/patch/worker/op_opt/moe/patch_unquantized_oracle.py`
- Create: `tests/runtime_patch/test_lightop_w16a16_moe.py`
- Create: `tests/accuracy/test_lightop_w16a16_moe.py`
- Modify: `vllm_hcu/platforms/envs.py`
- Modify: `vllm_hcu/patch/worker/__init__.py`
- Modify: `vllm_hcu/model_executor/layers/fused_moe/unquantized_fused_moe_method.py`
- Modify: `tools/benchmark_sglang_operator_candidates.py`
- Modify: `docs/operator_adaptation_audit_v0251.md`
- Modify: `README.md`

**Interfaces:**
- Produces: `VLLM_HCU_USE_LIGHTOP_W16A16_MOE`, default `False`, subordinate to
  the master switch.
- Produces: `LightopW16A16Layout(logical_w13_shape, logical_w2_shape, packed_w13_shape, packed_w2_shape)`.
- Produces: `select_lightop_w16a16_config(moe_config, expected_m, device) -> tuple[dict[str, object], dict[str, object]] | None`.
- Produces: `pack_lightop_w16a16_weights(w13, w2) -> tuple[torch.Tensor, torch.Tensor, LightopW16A16Layout]`.
- Produces: `LightopW16A16Experts`, a standard-activation-format unquantized
  expert implementation which owns alignment, two LightOp GEMMs, fused SiLU,
  top-k weighting and reduction.
- Extends: the exact vLLM unquantized oracle with enum member
  `HCU_LIGHTOP_W16A16`, selected only in auto mode when both switches and the
  strict support predicate pass.

- [ ] **Step 1: Write failing oracle and layout-selection contracts**

Tests must catch wrong backend ownership, master/leaf gating, unsupported
activation, EP/EPLB/shared-expert rejection until supported, missing LightOp
symbol, unsupported alignment, no-config status, idempotent packing, and the
rule that Triton/AITER fallback occurs before any parameter replacement.

```python
def test_w16a16_auto_selects_lightop_only_when_master_and_leaf_enabled(...):
    backend, experts = patched_selector(config)
    assert backend.name == "HCU_LIGHTOP_W16A16"
    assert experts is LightopW16A16Experts


def test_w16a16_no_config_preserves_original_selector_and_weights(...):
    before = (layer.w13_weight, layer.w2_weight)
    assert patched_selector(config) == official_result
    assert (layer.w13_weight, layer.w2_weight) == before
```

- [ ] **Step 2: Run contracts and verify RED**

```bash
python3 -m pytest -q tests/runtime_patch/test_lightop_w16a16_moe.py
```

Expected: the backend adapter/runtime symbols are absent.

- [ ] **Step 3: Implement the strict selection and packing boundary**

Use LightOp configuration lookup before mutation. Accept only BF16, SiLU,
standard activation format, contiguous 3-D expert weights, aligned K/N,
matching expert counts and top-k, and supported TP-only topology for the first
accepted version. Preserve loader attributes when replacing parameters. Mark
logical shape, physical shape and generation on both packed parameters.

- [ ] **Step 4: Implement the expert runtime**

Allocate scratch through the vLLM workspace mechanism where possible. Use the
plugin's LightOp MoE alignment helper, call the two W16A16 GEMMs, use
`lightop.activation.fuse_silu_and_mul`, apply router weights exactly once, and
reduce with `lightop.moe.moe_sum`. Return the shape/dtype required by vLLM's
standard prepare/finalize contract. Fail closed if packed layout markers are
missing or inconsistent.

- [ ] **Step 5: Wire explicit registration and make contracts GREEN**

Register `patch_unquantized_oracle` in the MoE foundation group before layer
construction. Preserve old enum values and original selector/converter/kernel
factories exactly as the existing FP8/INT8 oracle adapters do.

```bash
python3 -m pytest -q tests/runtime_patch/test_lightop_w16a16_moe.py tests/runtime_patch/test_quant_gemm_aiter.py -k 'unquantized or w16a16'
python3 tools/check_patch_test_coverage.py --json
python3 tools/check_production_boundary.py
```

- [ ] **Step 6: Write and run live-HCU numerical accuracy**

Compare against a float32 per-expert torch reference at the Qwen3.6 shape
family `E=256, K=2048, N=512, topk=8` with smaller expert fixtures where
memory permits, token counts from decode through prefill, nonuniform routing
weights, repeated experts, empty experts, zeros and extremes. Record explicit
BF16 tolerances and verify input weights are not modified after packing.

```bash
HIP_VISIBLE_DEVICES=0 python3 -m pytest -q -s tests/accuracy/test_lightop_w16a16_moe.py
```

- [ ] **Step 7: Benchmark against AITER and Triton**

```bash
HIP_VISIBLE_DEVICES=0 python3 tools/benchmark_sglang_operator_candidates.py w16a16-moe --experts 256 --hidden-size 2048 --intermediate-size 512 --top-k 8 --token-counts 1 2 4 8 16 32 64 128 256 --warmup 20 --iterations 100 --repeats 7 --json-output /tmp/vllm-hcu-w16a16-benchmark.json
```

If LightOp is not at least 5% faster on an accepted target range, remove the
oracle/runtime route and record `performance-rejected`. Keep the benchmark and
accuracy evidence. If accepted, encode the measured range in the selector;
the leaf switch remains opt-in because packing changes weight layout.

- [ ] **Step 8: Commit**

```bash
git add vllm_hcu tests/runtime_patch/test_lightop_w16a16_moe.py tests/accuracy/test_lightop_w16a16_moe.py tools/benchmark_sglang_operator_candidates.py docs/operator_adaptation_audit_v0251.md README.md
git commit -m "feat: add LightOp W16A16 MoE backend"
```

### Task 5: Screen and adapt remaining operator candidates

**Files:**
- Modify: `docs/operator_adaptation_audit_v0251.md`
- Modify: `tools/benchmark_sglang_operator_candidates.py`
- Modify when accepted: the smallest matching `vllm_hcu` runtime/adapter and
  focused `tests/runtime_patch/` plus `tests/accuracy/` files.

**Interfaces:**
- Consumes: the audit dispositions and timing helpers from Task 1.
- Candidate families: AITER SiLU-and-multiply, LightOp
  `ep_build_m_indices`, `lm_faster_rmsquant`,
  `rms_norm_per_token_fp8_quant`, model-agnostic KV-store fusions,
  metadata/concat kernels, and sampling operators.
- Produces: an explicit final disposition and evidence reference for every
  candidate family; an accepted family receives one leaf switch and one exact
  plugin integration seam.

- [ ] **Step 1: Characterize each vLLM seam without production changes**

For each family, identify the canonical vLLM callable, current plugin path,
input/output and mutation contract. Mark `no-vllm-seam` when SGLang-specific
scheduler state or cache layout has no equivalent vLLM boundary. Mark
`dependency-unavailable` when the installed callable is absent or its ABI
cannot express vLLM semantics.

- [ ] **Step 2: Write a failing numerical test for each remaining viable candidate**

Each test names the concrete wrong result it catches and uses literal or
independently derived expected output. Existing equivalent plugin operators
are tested as the comparison baseline; a candidate that adds no new behavior
or performance is classified `already-covered`.

- [ ] **Step 3: Run live-HCU accuracy before writing a route**

When a viable test file exists, run its exact command:

```bash
HIP_VISIBLE_DEVICES=0 python3 -m pytest -q -s tests/accuracy/test_aiter_silu_and_mul.py
HIP_VISIBLE_DEVICES=0 python3 -m pytest -q -s tests/accuracy/test_lightop_ep_build_m_indices.py
HIP_VISIBLE_DEVICES=0 python3 -m pytest -q -s tests/accuracy/test_lightop_fused_rms_quant.py
```

Do not run a filename for a family already classified as `no-vllm-seam` or
`dependency-unavailable`; its audit evidence is the deliverable instead.

Numerical failure produces `accuracy-rejected` and no production integration.

- [ ] **Step 4: Benchmark every numerically viable candidate**

Use model-derived shapes from DeepSeek V4, Qwen3.6, Qwen3.5 W8A8, HY4,
Qwen3.6 dense and Gemma4 as applicable. A candidate below 5% is marked
`performance-rejected` and receives no route.

- [ ] **Step 5: For each accepted candidate, perform one complete TDD cycle**

Add the route test first, observe RED, implement the minimum HCU-owned helper
and exact adapter, observe GREEN, then add master-off/leaf-off/fallback and
strict-dynamic tests. Never combine two unrelated accepted operators in one
adapter.

- [ ] **Step 6: Commit each accepted operator separately, then commit audit-only rejections**

For an accepted AITER SiLU route:

```bash
git add vllm_hcu/ops/silu_and_mul.py vllm_hcu/platforms/envs.py tests/accuracy/test_aiter_silu_and_mul.py tests/runtime_patch/test_environment_routing.py README.md docs/operator_adaptation_audit_v0251.md
git commit -m "feat: add AITER SiLU routing for HCU"
```

For an accepted LightOp EP m-indices route:

```bash
git add vllm_hcu/model_executor/layers/fused_moe/ep_m_indices.py vllm_hcu/platforms/envs.py tests/accuracy/test_lightop_ep_build_m_indices.py tests/runtime_patch/test_moe_deepep.py README.md docs/operator_adaptation_audit_v0251.md
git commit -m "feat: add LightOp EP m-indices routing"
```

For an accepted LightOp fused RMS/quant route:

```bash
git add vllm_hcu/ops/fuse_rms_norm_quant.py vllm_hcu/platforms/envs.py tests/accuracy/test_lightop_fused_rms_quant.py tests/runtime_patch/test_environment_routing.py README.md docs/operator_adaptation_audit_v0251.md
git commit -m "feat: extend LightOp fused RMS quant routing"
```

Record all non-accepted families separately:

```bash
git add docs/operator_adaptation_audit_v0251.md tools/benchmark_sglang_operator_candidates.py tests/accuracy
git commit -m "docs: record remaining operator screening"
```

### Task 6: Primary model and HumanEval-32 acceptance

**Files:**
- Create: `tests/models/deepseek_v4_int8_humaneval_evalscope.yaml`
- Create: `tests/models/qwen36_35b_a3b_humaneval_evalscope.yaml`
- Create: `tests/integration/server/test_evalscope_operator_adaptation_humaneval.py`
- Create: `docs/sglang_operator_adaptation_validation.md`
- Modify: `tests/models/README.md`
- Modify: `tests/integration/server/README.md`

**Interfaces:**
- Consumes: `/models/DeepSeek-V4-Flash-Channel-INT8-w8a8` and
  `/models/Qwen3.6-35B-A3B`.
- Produces: deterministic feature-off and feature-on server cases for exactly
  32 HumanEval samples.
- Produces: route-evidence assertions, exact prediction/review counts, Pass@1
  comparison, throughput comparison, peak device memory and environment
  fingerprint.

- [ ] **Step 1: Write failing collection-safe config contracts**

Assert both configs contain existing local model paths, unique ports/work
directories, deterministic temperature, a 32-sample limit, explicit server
topology, feature-off/on environment maps, and pass criteria comparing on to
off. Assert the runner rejects reports with 31 or 33 predictions.

- [ ] **Step 2: Run config contracts and verify RED**

```bash
python3 -m pytest -q tests/integration/server/test_evalscope_operator_adaptation_humaneval.py -k contract
```

Expected: missing configs/runner behavior fail before a server starts.

- [ ] **Step 3: Implement configs and comparison runner**

Reuse `tests/integration/server/evalscope_server.py`. Keep server lifecycle,
timeouts and process-group cleanup there; add only the smallest reusable
feature-off/on comparison and exact-32 report validation. Do not put model
checkpoints in the repository.

- [ ] **Step 4: Run DeepSeek V4 INT8 feature-off/on**

Before starting, confirm assigned devices are idle. Use the same TP/EP and
generation settings in both runs. Save logs and EvalScope reports below a
unique `/tmp/vllm-hcu-evalscope/` case directory. Require on Pass@1 not lower
than off and throughput regression no worse than 2%.

- [ ] **Step 5: Run Qwen3.6-35B-A3B feature-off/on**

Repeat the exact acceptance logic with the W16A16 leaf explicitly on only if
Task 4 accepted it. If W16A16 was rejected, exercise the other accepted generic
operators and record that the model does not cover W16A16 LightOp.

- [ ] **Step 6: Run supplemental models only where they add operator coverage**

Choose from DeepSeek V4 FP8, Qwen3.5 W8A8, HY4 BF16, Qwen3.6 dense and Gemma4.
Record skipped models and the coverage reason; do not block an independently
validated operator solely for lack of a model hit.

- [ ] **Step 7: Commit acceptance assets**

```bash
git add tests/models tests/integration/server docs/sglang_operator_adaptation_validation.md
git commit -m "test: validate adapted operators on primary models"
```

### Task 7: Final verification, review, and standalone MR

**Files:**
- Modify only if results require corrections: files already listed above.
- Verify: all changed production, test, documentation and benchmark files.

**Interfaces:**
- Produces: a clean standalone branch, complete validation summary and one MR
  against `v0.25.1`.

- [ ] **Step 1: Run focused tests for every accepted route**

Run all new runtime-patch and HCU-accuracy node IDs, confirming positive route
evidence and both switch-off fallbacks.

- [ ] **Step 2: Run repository gates**

```bash
python3 tools/check_production_boundary.py
python3 tools/check_patch_test_coverage.py --json
python3 -m compileall -q vllm_hcu tests tools
python3 tools/run_patch_tests.py --suite inventory
python3 tools/run_patch_tests.py --suite contract
```

Expected: zero failures, clean boundary, no untested adapters.

- [ ] **Step 3: Re-run final HCU and model acceptance**

Run the accepted operator accuracy suite, both primary model feature-off/on
cases, and HumanEval-32 after the final code commit. Record exact commands,
commit, environment fingerprint, JSON report paths and results in the
validation document.

- [ ] **Step 4: Inspect diff and credential safety**

```bash
git diff --check origin/v0.25.1...HEAD
git status --short
git log --oneline origin/v0.25.1..HEAD
git grep -n -E 'ghp_[A-Za-z0-9]+' origin/v0.25.1..HEAD -- .
```

Expected: clean diff, intentional commits only, and no credential match.

- [ ] **Step 5: Request code review and address findings with TDD**

Use `superpowers:requesting-code-review`. For every valid finding, first add or
identify the failing test, then implement the correction and rerun the focused
plus surrounding suites.

- [ ] **Step 6: Commit final validation updates**

```bash
git add docs/sglang_operator_adaptation_validation.md
git commit -m "docs: record operator adaptation validation"
```

- [ ] **Step 7: Push without embedding credentials and create one MR**

Push the standalone branch through the configured secure credential mechanism.
Create one MR targeting `v0.25.1` with the audit table, accepted/rejected
operators, switches, accuracy, performance, model and HumanEval evidence. Do
not place a token in the remote URL or shell history.
