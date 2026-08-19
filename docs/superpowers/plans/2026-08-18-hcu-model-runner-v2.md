# HCU Model Runner V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Route the HCU worker's Model Runner V2 path through a plugin-owned thin adapter and prove it with `/models/Qwen3-8B` inference.

**Architecture:** `HcuGPUModelRunnerV2` subclasses upstream v0.25.1 MRV2 without copying it. A small worker factory selects this adapter for V2 and the existing HCU runner for V1, preserving `VLLM_USE_V2_MODEL_RUNNER=0` as rollback.

**Tech Stack:** Python 3, vLLM v0.25.1, PyTorch, HCU runtime, pytest.

**Spec:** `docs/superpowers/specs/2026-08-18-hcu-model-runner-v2-design.md`

## Global Constraints

- Keep Model Runner V1 and the explicit `VLLM_USE_V2_MODEL_RUNNER=0` fallback.
- Do not copy upstream MRV2 or mechanically port the V1 HCU runner.
- Add an HCU-specific override only when a focused test or Qwen3-8B failure demonstrates the need.
- Preserve upstream MRV2 validation and fail explicitly instead of silently falling back.
- Target the vLLM v0.25.1 module contract already used by this plugin.

---

### Task 1: Add the HCU MRV2 Integration Boundary

**Files:**
- Create: `vllm_hcu/v1/hcu_model_runner_v2.py`
- Modify: `vllm_hcu/v1/worker.py`
- Test: `tests/patch/test_plugin_lifecycle.py`

**Interfaces:**
- Consumes: `vllm.v1.worker.gpu.model_runner.GPUModelRunner(VllmConfig, torch.device)`.
- Produces: `HcuGPUModelRunnerV2` and `_create_model_runner(vllm_config, device, *, use_v2_model_runner)`.

- [ ] **Step 1: Write failing runner-selection tests**

Add CPU-safe tests that inject fake modules for both runner paths, call the
factory, and assert V2 constructs `HcuGPUModelRunnerV2` while V1 constructs the
existing `vllm_hcu.v1.hcu_model_runner.GPUModelRunner`:

```python
@pytest.mark.parametrize(
    ("use_v2", "expected_module"),
    [(True, "vllm_hcu.v1.hcu_model_runner_v2"),
     (False, "vllm_hcu.v1.hcu_model_runner")],
)
def test_worker_selects_plugin_owned_model_runner(
    monkeypatch, cpu_safe_hcu_worker_module, use_v2, expected_module
):
    events = []
    runner_module = ModuleType(expected_module)

    class Runner:
        def __init__(self, config, device):
            events.append((config, device))

    runner_module.HcuGPUModelRunnerV2 = Runner
    runner_module.GPUModelRunner = Runner
    monkeypatch.setitem(sys.modules, expected_module, runner_module)
    config = object()
    device = object()
    result = cpu_safe_hcu_worker_module._create_model_runner(
        config, device, use_v2_model_runner=use_v2
    )
    assert isinstance(result, Runner)
    assert events == [(config, device)]
```

Also add a contract test that imports the adapter against a fake upstream
module and proves it is a strict subclass with no copied execution methods.

- [ ] **Step 2: Run the focused tests and verify they fail**

Run:

```bash
pytest -q tests/patch/test_plugin_lifecycle.py -k 'plugin_owned_model_runner or hcu_model_runner_v2_is_thin_adapter'
```

Expected: failure because `_create_model_runner` and
`vllm_hcu.v1.hcu_model_runner_v2` do not exist.

- [ ] **Step 3: Implement the thin adapter and selection factory**

Create:

```python
from vllm.v1.worker.gpu.model_runner import GPUModelRunner


class HcuGPUModelRunnerV2(GPUModelRunner):
    """HCU integration boundary for upstream v0.25.1 Model Runner V2."""

    pass
```

Add to `worker.py`:

```python
def _create_model_runner(vllm_config, device, *, use_v2_model_runner):
    if use_v2_model_runner:
        from vllm_hcu.v1.hcu_model_runner_v2 import HcuGPUModelRunnerV2
        return HcuGPUModelRunnerV2(vllm_config, device)
    from vllm_hcu.v1.hcu_model_runner import GPUModelRunner
    return GPUModelRunner(vllm_config, device)
```

Replace the inline constructor branch in `HcuGPUWorker.init_device` with this
factory call and keep the existing V2 log message.

- [ ] **Step 4: Run focused and lifecycle tests**

Run `pytest -q tests/patch/test_plugin_lifecycle.py`.

Expected: all tests pass.

- [ ] **Step 5: Commit the integration boundary**

```bash
git add vllm_hcu/v1/hcu_model_runner_v2.py vllm_hcu/v1/worker.py tests/patch/test_plugin_lifecycle.py
git commit -m "feat: route HCU through model runner v2 adapter"
```

### Task 2: Add Reproducible Qwen3-8B MRV2 Coverage

**Files:**
- Create: `tests/integration/models/test_qwen3_8b_mrv2.py`
- Modify: `tests/integration/models/README.md`

**Interfaces:**
- Consumes: `require_model_runtime(...)` and `run_vllm_case(...)` from `tests.integration.model_runtime`.
- Produces: a one-HCU eager and graph MRV2 smoke test selected with `VLLM_HCU_QWEN3_8B_MODEL`.

- [ ] **Step 1: Write the MRV2 integration test**

Create a test resolving `/models/Qwen3-8B` through the environment override,
running existing `smoke` and `graph-parity` cases with
`extra_env={"VLLM_USE_V2_MODEL_RUNNER": "1"}`, and checking token output:

```python
def test_qwen3_8b_mrv2_eager_and_graph(hcu_test_resources):
    model_path = require_model_runtime(
        hcu_test_resources,
        env_name="VLLM_HCU_QWEN3_8B_MODEL",
        relative_path="Qwen3-8B",
        label="Qwen3-8B MRV2",
    )
    eager = run_vllm_case(
        "smoke", model_path, timeout_s=1800,
        extra_env={"VLLM_USE_V2_MODEL_RUNNER": "1"},
        log_label="qwen3-8b-mrv2-eager",
    )
    graph = run_vllm_case(
        "graph-parity", model_path, timeout_s=1800,
        extra_env={"VLLM_USE_V2_MODEL_RUNNER": "1"},
        log_label="qwen3-8b-mrv2-graph",
    )
    assert all(item["token_ids"] for item in eager["first"])
    assert graph["eager"] == graph["graph"]
```

Document the exact command and environment override in the models README.

- [ ] **Step 2: Collect the test and verify resource routing**

Run:

```bash
VLLM_HCU_QWEN3_8B_MODEL=/models/Qwen3-8B pytest --collect-only -q tests/integration/models/test_qwen3_8b_mrv2.py
```

Expected: one collected test and no import-time accelerator failure.

- [ ] **Step 3: Run eager Qwen3-8B MRV2 first**

Run:

```bash
VLLM_USE_V2_MODEL_RUNNER=1 VLLM_HCU_USE_FLASH_ATTN_UNIFIED=1 python -m tests.integration.model_runtime smoke --model /models/Qwen3-8B
```

Expected: output contains `Using V2 Model Runner`, a `VLLM_HCU_RESULT=` record,
and non-empty token IDs. A failure may add only the smallest HCU adapter
override plus a focused unit test reproducing that failure.

- [ ] **Step 4: Run graph parity**

Run:

```bash
VLLM_USE_V2_MODEL_RUNNER=1 VLLM_HCU_USE_FLASH_ATTN_UNIFIED=1 python -m tests.integration.model_runtime graph-parity --model /models/Qwen3-8B
```

Expected: eager and graph token records match. A graph-only failure becomes a
separate focused adapter contract.

- [ ] **Step 5: Run reusable integration and unit tests**

Run:

```bash
VLLM_HCU_QWEN3_8B_MODEL=/models/Qwen3-8B pytest -q tests/integration/models/test_qwen3_8b_mrv2.py
pytest -q tests/patch/test_plugin_lifecycle.py tests/runtime_patch/test_platform_hcu_config.py
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit MRV2 integration coverage**

```bash
git add tests/integration/models/test_qwen3_8b_mrv2.py tests/integration/models/README.md
git commit -m "test: cover Qwen3-8B with HCU model runner v2"
```

### Task 3: Final Compatibility Verification

**Files:**
- Modify only if verification demonstrates a specific HCU MRV2 defect.

**Interfaces:**
- Consumes: the adapter, worker factory, and integration test from Tasks 1-2.
- Produces: proof that V2 works and V1 remains selectable.

- [ ] **Step 1: Verify source and formatting contracts**

Run:

```bash
python -m compileall -q vllm_hcu tests/patch/test_plugin_lifecycle.py tests/integration/models/test_qwen3_8b_mrv2.py
git diff --check
```

Expected: both commands exit zero.

- [ ] **Step 2: Verify V1 selection remains intact**

Run the focused factory selection test with both parametrized cases. Expected:
both pass and construct different plugin-owned runner classes.

- [ ] **Step 3: Review the final diff against the spec**

Confirm there is no upstream MRV2 copy, no V1 deletion, no automatic fallback,
and no unrelated refactor. Record exact unit and hardware results in handoff.
