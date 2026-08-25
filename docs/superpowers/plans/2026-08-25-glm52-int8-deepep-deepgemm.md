# GLM-5.2 INT8 DeepEP and DeepGEMM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Connect v0.25.1's HCU INT8 MoE oracle to the existing DeepGEMM experts so the GLM-5.2 channel-W8A8 checkpoint runs with both DeepEP high-throughput and low-latency modes.

**Architecture:** Extend only the HCU runtime adapter for vLLM's INT8 oracle. Explicit `dpsk_deep_gemm` selection exposes `DeepGemmExperts` for standard/HT activation layout and `BatchedDeepGemmExperts` for batched/LL activation layout, while the target v0.25.1 oracle continues to build quant configs, prepare/finalize objects, and modular kernels.

**Tech Stack:** Python 3.10, vLLM 0.25.1, PyTorch 2.11 HCU, DeepEP 1.1, DeepGEMM 2.1, LightOP, pytest, EvalScope 1.10.

**Spec:** `docs/superpowers/specs/2026-08-25-glm52-int8-deepep-deepgemm-design.md`

## Global Constraints

- Base every code change on remote `v0.25.1` commit `7fb6ea6ed7235e282de96ee7e61157315b4e115a`.
- Keep `dpsk_deep_gemm` opt-in; do not alter `moe_backend=auto` selection.
- Preserve current AITER, Triton, Humming, and CPU INT8 behavior.
- Keep DeepGEMM and LightOP imports lazy.
- Do not modify external DeepEP, DeepGEMM, LightOP, vLLM, or model files.
- Use `/models/GLM-5.2-Channel-INT8-w8a8` for hardware validation.
- Require deterministic correctness before accepting performance measurements.

---

### Task 1: Register DeepGEMM in the HCU INT8 Oracle

**Files:**
- Modify: `tests/runtime_patch/test_quant_gemm_aiter.py`
- Modify: `vllm_hcu/patch/worker/op_opt/moe/patch_int8_oracle.py`

**Interfaces:**
- Consumes: `HcuFeatureConfig.moe_backend`, target `select_int8_moe_backend(config, weight_key, activation_key)`, and `FusedMoEParallelConfig.use_batched_activation_format`.
- Produces: enum member `Int8MoeBackend.DPSK_DEEPGEMM`, mapping for `dpsk_deep_gemm`, and support-checked selection of `DeepGemmExperts` or `BatchedDeepGemmExperts`.

- [ ] **Step 1: Read the complete target and reference implementations**

Read these files without editing them:

```bash
sed -n '1,380p' /models/zb/vllm_025/vllm/vllm/model_executor/layers/fused_moe/oracle/int8.py
sed -n '1,520p' /models/zb/vllm_hcu_021/vllm-hcu/vllm_hcu/model_executor/layers/fused_moe/experts/deep_gemm_moe.py
sed -n '1,520p' vllm_hcu/model_executor/layers/fused_moe/experts/deep_gemm_moe.py
sed -n '1,520p' vllm_hcu/model_executor/layers/fused_moe/experts/batched_deep_gemm_moe.py
```

Expected: confirm both current expert classes accept
`(kInt8StaticChannelSym, kInt8DynamicTokenSym)` on HCU and differ by standard
versus batched activation format.

- [ ] **Step 2: Add a failing selection contract test**

Extend the existing fake INT8 oracle test with a target
`select_int8_moe_backend` and a fake modular activation-format namespace. Add
fake DeepGEMM expert modules whose support checks accept only their intended
format. The essential assertions are:

```python
config = SimpleNamespace(
    moe_backend="auto",
    moe_parallel_config=SimpleNamespace(
        use_batched_activation_format=False,
    ),
    _hcu_vllm_config=SimpleNamespace(
        additional_config={
            "hcu": {
                "moe_backend": "dpsk_deep_gemm",
            }
        }
    ),
)

backend, experts = target.select_int8_moe_backend(config, "weight", "activation")
assert backend is target.Int8MoeBackend.DPSK_DEEPGEMM
assert experts is DeepGemmExperts
assert target.map_int8_backend("dpsk_deep_gemm") is backend

config.moe_parallel_config.use_batched_activation_format = True
backend, experts = target.select_int8_moe_backend(config, "weight", "activation")
assert experts is BatchedDeepGemmExperts

converted_w13, converted_w2 = target.convert_to_int8_moe_kernel_format(
    backend,
    w13,
    w2,
)
assert converted_w13 is w13
assert converted_w2 is w2
```

Also assert that an HCU sidecar with `moe_backend="auto"` delegates to the
original selector and that explicit `dpsk_deep_gemm` rejects an official
`config.moe_backend` other than `auto`.

- [ ] **Step 3: Run the new test and verify RED**

Run:

```bash
VLLM_V0251_SOURCE_ROOT=/models/zb/vllm_025/vllm \
PYTHONPATH=$PWD \
python -m pytest -q \
  tests/runtime_patch/test_quant_gemm_aiter.py::test_int8_aiter_oracle_maps_explicit_backend_and_keeps_canonical_weights
```

Expected: FAIL because `DPSK_DEEPGEMM` and the wrapped selector do not exist.

- [ ] **Step 4: Implement the minimal oracle adapter**

In `patch_int8_oracle.py`:

1. Add `select_int8_moe_backend` to `TARGETS`, validate its exact parameter
   names, and retain the original callable.
2. Add a sidecar helper using `get_current_vllm_config_or_none()` with the same
   config fallback used by `patch_fp8_oracle.py`.
3. Add both HCU enum values without changing the existing patch ID or marker.
4. Lazily return the existing expert classes for `DPSK_DEEPGEMM`.
5. Wrap selection only when the sidecar explicitly requests the backend.

The selector structure must be:

```python
@functools.wraps(select_backend)
def hcu_select_int8_moe_backend(config, weight_key, activation_key):
    if _sidecar_backend(config) != "dpsk_deep_gemm":
        return select_backend(config, weight_key, activation_key)
    if getattr(config, "moe_backend", "auto") != "auto":
        raise ValueError(
            "HCU sidecar selects dpsk_deep_gemm but official FusedMoEConfig "
            f"selects {config.moe_backend!r}; official backend must remain 'auto'"
        )
    activation_format = (
        target.mk.FusedMoEActivationFormat.BatchedExperts
        if config.moe_parallel_config.use_batched_activation_format
        else target.mk.FusedMoEActivationFormat.Standard
    )
    reasons = []
    for kernel_cls in hcu_backend_to_kernel_cls(hcu_enum.DPSK_DEEPGEMM):
        supported, reason = kernel_cls.is_supported_config(
            kernel_cls,
            config,
            weight_key,
            activation_key,
            activation_format,
        )
        if supported:
            return hcu_enum.DPSK_DEEPGEMM, kernel_cls
        reasons.append(f"{kernel_cls.__name__}: {reason or 'unsupported'}")
    raise ValueError(
        "dpsk_deep_gemm is required by HCU sidecar but does not support "
        "this INT8 MoE configuration: " + "; ".join(reasons)
    )
```

Treat both AITER and DPSK canonical INT8 weight conversion as no-ops. Do not
special-case quant-config or modular-kernel construction for DPSK.

- [ ] **Step 5: Run the focused test and verify GREEN**

Run the Step 3 command again.

Expected: PASS with AITER assertions still passing and the new DeepGEMM
selection assertions green.

- [ ] **Step 6: Run adjacent oracle and DeepEP contracts**

Run:

```bash
VLLM_V0251_SOURCE_ROOT=/models/zb/vllm_025/vllm \
PYTHONPATH=$PWD \
python -m pytest -q \
  tests/runtime_patch/test_quant_gemm_aiter.py \
  tests/runtime_patch/test_moe_deepep.py \
  tests/patch/test_worker_dispatcher.py
```

Expected: all selected tests pass; only existing PyTorch deprecation warnings
may remain.

- [ ] **Step 7: Commit the oracle change**

```bash
git add \
  tests/runtime_patch/test_quant_gemm_aiter.py \
  vllm_hcu/patch/worker/op_opt/moe/patch_int8_oracle.py
git diff --cached --check
git commit -m "feat(hcu): route INT8 DeepEP through DeepGEMM"
```

---

### Task 2: Add GLM-5.2 DeepEP Nightly Coverage

**Files:**
- Create: `tests/integration/test_model_runtime_cli.py`
- Modify: `tests/integration/model_runtime.py`
- Modify: `tests/integration/parallel/test_tp_ep_models.py`

**Interfaces:**
- Consumes: existing `run_vllm_case("tp-ep-smoke", ...)` subprocess harness.
- Produces: optional `--data-parallel-size` and `--all2all-backend` test-case arguments plus an eight-HCU GLM-5.2 HT/LL nightly test.

- [ ] **Step 1: Add a failing CPU-safe CLI forwarding test first**

Create `tests/integration/test_model_runtime_cli.py` and replace the expensive
case function with a recorder before calling `_main`:

```python
from pathlib import Path

from tests.integration import model_runtime


def test_tp_ep_cli_forwards_data_parallel_and_all2all(monkeypatch, capsys):
    captured = {}

    def fake_case(model_path, **kwargs):
        captured["model_path"] = model_path
        captured.update(kwargs)
        return {"output": []}

    monkeypatch.setattr(model_runtime, "_case_tp_ep_smoke", fake_case)
    assert model_runtime._main(
        [
            "tp-ep-smoke",
            "--model", "/models/fake",
            "--tensor-parallel-size", "1",
            "--data-parallel-size", "8",
            "--all2all-backend", "deepep_low_latency",
            "--moe-backend", "dpsk_deep_gemm",
        ]
    ) == 0
    assert captured == {
        "model_path": Path("/models/fake"),
        "tensor_parallel_size": 1,
        "data_parallel_size": 8,
        "gpu_memory_utilization": 0.6,
        "all2all_backend": "deepep_low_latency",
        "moe_backend": "dpsk_deep_gemm",
    }
    assert "VLLM_HCU_RESULT=" in capsys.readouterr().out
```

- [ ] **Step 2: Run the CLI test and verify RED**

Run:

```bash
PYTHONPATH=$PWD python -m pytest -q \
  tests/integration/test_model_runtime_cli.py::test_tp_ep_cli_forwards_data_parallel_and_all2all
```

Expected: FAIL with argparse rejecting `--data-parallel-size` and
`--all2all-backend`.

- [ ] **Step 3: Add the real-model result assertions and mode matrix**

Extend the current TP/EP result assertions so the new fields are observable:

```python
assert result["requested_data_parallel_size"] == expected_dp
assert result["requested_all2all_backend"] == expected_all2all
assert parallel_config["data_parallel_size"] == expected_dp
assert parallel_config["all2all_backend"] == expected_all2all
```

Add the GLM mode matrix:

```python
GLM52_CHANNEL_INT8 = "GLM-5.2-Channel-INT8-w8a8"
GLM52_DEEPEP_MODES = (
    pytest.param("deepep_high_throughput", id="high-throughput"),
    pytest.param("deepep_low_latency", id="low-latency"),
)
```

The test must request TP=1, DP=8, EP enabled,
`moe_backend="dpsk_deep_gemm"`, and `VLLM_USE_DEEP_GEMM=1`.

- [ ] **Step 4: Extend the subprocess harness minimally**

Update `_case_tp_ep_smoke` and its parser with:

```python
parser.add_argument("--data-parallel-size", type=int, default=1)
parser.add_argument("--all2all-backend", default=None)
```

Pass both values through `_llm_kwargs`, return them in the result payload, and
include `all2all_backend` in `_parallel_config_summary`. Existing callers keep
their current behavior through the defaults.

- [ ] **Step 5: Run the CLI test and verify GREEN**

Run the Step 2 command again.

Expected: PASS and the recorder receives the exact DP and all-to-all values.

- [ ] **Step 6: Add the real-model nightly test**

The new parameterized test calls:

```python
result = run_vllm_case(
    "tp-ep-smoke",
    model_path,
    timeout_s=5400,
    extra_env={"VLLM_USE_DEEP_GEMM": "1"},
    log_label=f"glm52-int8-{all2all_backend}",
    extra_args=[
        "--tensor-parallel-size", "1",
        "--data-parallel-size", "8",
        "--all2all-backend", all2all_backend,
        "--gpu-memory-utilization", "0.9",
        "--moe-backend", "dpsk_deep_gemm",
    ],
)
```

Mark it `hcu`, `model`, `multi_hcu`, `hcu_count(8)`, `slow`, and `nightly`.

- [ ] **Step 7: Verify collection and CPU-safe integration contracts**

Run:

```bash
VLLM_V0251_SOURCE_ROOT=/models/zb/vllm_025/vllm \
PYTHONPATH=$PWD \
python -m pytest --collect-only -q \
  tests/integration/parallel/test_tp_ep_models.py -k glm52
VLLM_V0251_SOURCE_ROOT=/models/zb/vllm_025/vllm \
PYTHONPATH=$PWD \
python -m pytest -q tests/patch/test_worker_dispatcher.py
```

Expected: two GLM cases collected and worker contracts pass.

- [ ] **Step 8: Commit the nightly coverage**

```bash
git add \
  tests/integration/test_model_runtime_cli.py \
  tests/integration/model_runtime.py \
  tests/integration/parallel/test_tp_ep_models.py
git diff --cached --check
git commit -m "test(hcu): cover GLM-5.2 DeepEP DeepGEMM modes"
```

---

### Task 3: Diagnose and Verify the Real GLM-5.2 Runtime

**Files:**
- Read: `/tmp/vllm-hcu-integration/logs/*.log`
- Modify only if a reproduced root cause requires a new regression test and a scoped production fix.

**Interfaces:**
- Consumes: Task 1 oracle registration and Task 2 nightly harness.
- Produces: successful HT and LL deterministic generations on eight HCUs.

- [ ] **Step 1: Confirm the HCU software and device baseline**

```bash
python -m pip list | rg -i 'vllm|deep-ep|deepgemm|lightop|torch'
rocm-smi --showmeminfo vram
```

Expected: vLLM 0.25.1, DeepEP 1.1, DeepGEMM 2.1, LightOP installed, and
all eight HCUs idle before launch.

- [ ] **Step 2: Run both nightly cases against the real model**

```bash
VLLM_HCU_GLM52_CHANNEL_INT8_MODEL=/models/GLM-5.2-Channel-INT8-w8a8 \
VLLM_USE_DEEP_GEMM=1 \
VLLM_V0251_SOURCE_ROOT=/models/zb/vllm_025/vllm \
PYTHONPATH=$PWD \
python -m pytest -q -s \
  --strict-test-resources \
  tests/integration/parallel/test_tp_ep_models.py -k glm52
```

Expected: both mode cases pass and logs contain the intended DeepEP manager
plus `DeepGemmExperts` for HT and `BatchedDeepGemmExperts` for LL.

- [ ] **Step 3: If a case fails, use systematic debugging before editing**

Read the full traceback and worker logs, reproduce one mode at a time, trace
the failing value across oracle, quant-config, prepare/finalize, and expert
boundaries, then state one root-cause hypothesis. Add the smallest automated
failing regression test, verify RED, implement one scoped fix, and verify
GREEN. If three hypotheses fail, stop and revisit the architecture with the
user instead of stacking a fourth fix.

- [ ] **Step 4: Verify deterministic API parity and concurrency**

For each running mode, issue temperature-zero completions for
`"The capital of France is"` with 32 output tokens, seed 0, and logprobs 5.
Compare text, token IDs, and first-token logprob with the already recorded
v0.25.1 default result. Then run four simultaneous 24-token requests and
require four HTTP 200 responses.

- [ ] **Step 5: Re-run the focused automated suite after runtime fixes**

```bash
VLLM_V0251_SOURCE_ROOT=/models/zb/vllm_025/vllm \
PYTHONPATH=$PWD \
python -m pytest -q \
  tests/runtime_patch/test_quant_gemm_aiter.py \
  tests/runtime_patch/test_moe_deepep.py \
  tests/runtime_patch/test_platform_hcu_config.py \
  tests/patch/test_worker_dispatcher.py \
  tests/patch/test_module_exchange.py
```

Expected: all selected tests pass.

---

### Task 4: Measure Performance and HumanEval Accuracy

**Files:**
- Write benchmark output only under `/tmp/glm52-deepep-deepgemm-v0251/`.
- Do not commit generated benchmark or evaluation artifacts.

**Interfaces:**
- Consumes: OpenAI-compatible servers for default, HT, and LL configurations.
- Produces: comparable JSON benchmark results and two HumanEval reports.

- [ ] **Step 1: Start three configurations sequentially**

Use the same model, port, maximum model length, and memory utilization. The
reference uses DeepEP HT with `moe_backend=auto`; the two candidates use
`dpsk_deep_gemm` with HT and LL respectively:

```bash
glm52_all2all_backend=deepep_high_throughput
glm52_moe_backend=dpsk_deep_gemm
glm52_run_label=ht-deepgemm
HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
PYTHONPATH=$PWD \
VLLM_USE_DEEP_GEMM=1 \
python -m vllm.entrypoints.openai.api_server \
  --model /models/GLM-5.2-Channel-INT8-w8a8 \
  --served-model-name glm52-deepep-test \
  --tensor-parallel-size 1 \
  --data-parallel-size 8 \
  --enable-expert-parallel \
  --all2all-backend "$glm52_all2all_backend" \
  --moe-backend "$glm52_moe_backend" \
  --gpu-memory-utilization 0.90 \
  --max-model-len 4096 \
  --max-num-seqs 32 \
  --default-chat-template-kwargs '{"enable_thinking":false}' \
  --port 10154
```

Run the block three times with these exact variable assignments:

```bash
# Default-backend reference
glm52_all2all_backend=deepep_high_throughput
glm52_moe_backend=auto
glm52_run_label=ht-auto-reference

# DeepGEMM high-throughput candidate
glm52_all2all_backend=deepep_high_throughput
glm52_moe_backend=dpsk_deep_gemm
glm52_run_label=ht-deepgemm

# DeepGEMM low-latency candidate
glm52_all2all_backend=deepep_low_latency
glm52_moe_backend=dpsk_deep_gemm
glm52_run_label=ll-deepgemm
```

Stop and fully release all workers between configurations.

- [ ] **Step 2: Warm up and benchmark concurrency 1**

```bash
vllm bench serve \
  --backend openai \
  --base-url http://127.0.0.1:10154 \
  --model glm52-deepep-test \
  --served-model-name glm52-deepep-test \
  --dataset-name random \
  --num-prompts 32 \
  --random-input-len 256 \
  --random-output-len 128 \
  --request-rate inf \
  --max-concurrency 1 \
  --percentile-metrics ttft,tpot,itl,e2el \
  --save-result \
  --result-dir "/tmp/glm52-deepep-deepgemm-v0251/$glm52_run_label-c1"
```

Expected: all 32 requests complete and JSON reports request throughput,
output-token throughput, TTFT, and TPOT.

- [ ] **Step 3: Benchmark throughput load**

Repeat Step 2 with `--num-prompts 128 --max-concurrency 32` for the reference
and HT candidate. Run LL with `--num-prompts 32 --max-concurrency 4` to
characterize its small-batch behavior without claiming it is a throughput
configuration.

- [ ] **Step 4: Run HumanEval 32 for HT**

With the HT DeepGEMM server running:

```bash
PYTHONPATH=/tmp/evalscope-target-v025 \
python -m evalscope.cli.cli eval \
  --model glm52-deepep-test \
  --api-url http://127.0.0.1:10154/v1 \
  --api-key EMPTY \
  --eval-type openai_api \
  --datasets humaneval \
  --dataset-hub modelscope \
  --limit 32 \
  --eval-batch-size 1 \
  --generation-config '{"max_tokens":2048,"temperature":0.0,"do_sample":false,"stream":true,"extra_body":{"chat_template_kwargs":{"enable_thinking":false}}}' \
  --seed 0 \
  --work-dir /tmp/glm52-deepep-deepgemm-v0251/humaneval-ht \
  --no-timestamp
```

Expected: 32 reviewed samples and a `mean_acc_pass@1` report.

- [ ] **Step 5: Run HumanEval 32 for LL**

Restart in LL DeepGEMM mode and repeat Step 4 with work directory
`/tmp/glm52-deepep-deepgemm-v0251/humaneval-ll`.

Expected: 32 reviewed samples and a `mean_acc_pass@1` report. Compare both
scores with the existing DCP2 reference of 32/32.

- [ ] **Step 6: Shut down and verify resource release**

```bash
rocm-smi --showmeminfo vram
pgrep -af 'vllm|EngineCore|Worker_TP|Worker_DP' || true
```

Expected: no test server or worker remains, and every HCU returns to its idle
VRAM level.

---

### Task 5: Final Verification, Review, and Merge Request

**Files:**
- Review every file changed from `origin/v0.25.1`.
- Use the repository's existing MR template if present.

**Interfaces:**
- Consumes: all implementation commits and `/tmp` validation artifacts.
- Produces: one reviewed branch and one merge request targeting `v0.25.1`.

- [ ] **Step 1: Run the full plugin test suite fresh**

```bash
VLLM_V0251_SOURCE_ROOT=/models/zb/vllm_025/vllm \
PYTHONPATH=$PWD \
python -m pytest -q tests
```

Expected: zero failures. Environment-only skips are reported separately and
must not be described as passes.

- [ ] **Step 2: Run formatting and static checks available in the repository**

```bash
git diff --check origin/v0.25.1...HEAD
python -m compileall -q vllm_hcu tests
```

Expected: both commands exit zero.

- [ ] **Step 3: Perform a line-by-line code review**

```bash
git diff --stat origin/v0.25.1...HEAD
git diff --find-renames origin/v0.25.1...HEAD
git log --oneline origin/v0.25.1..HEAD
```

Review for target-signature drift, enum identity errors, silent fallback,
eager optional imports, mutation of default paths, missing test isolation,
unrelated changes, secrets, generated artifacts, and claims not supported by
fresh command output. Fix every Critical or Important issue and re-run the
affected verification.

- [ ] **Step 4: Verify branch cleanliness and commit final scoped fixes**

```bash
git status --short
git diff --check origin/v0.25.1...HEAD
```

Expected: clean worktree and no whitespace errors.

- [ ] **Step 5: Push and create the MR requested by the user**

```bash
git push -u origin feat/glm52-deepep-deepgemm-v0251
gh pr create \
  --repo HYGON-AI/vllm-plugin-das \
  --base v0.25.1 \
  --head feat/glm52-deepep-deepgemm-v0251 \
  --title "feat(hcu): enable GLM-5.2 INT8 DeepEP DeepGEMM" \
  --body-file /tmp/glm52-deepep-deepgemm-v0251/pr-body.md
```

The MR body must state the root cause, explain why this is a v0.25.1-native
oracle change rather than a v0.21 class backport, list exact test commands and
counts, include HT/LL benchmark metrics and both HumanEval scores, disclose AI
assistance if required by repository policy, and contain no credential.

- [ ] **Step 6: Report the MR and remaining review risks**

Provide the MR URL, commits, changed files, verification counts, HumanEval
Pass@1 for HT and LL, performance comparison, code-review findings, and any
non-blocking limitation such as first-run JIT cost.
