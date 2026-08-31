# HY V4 HCU Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add accuracy-first HY V4 ordinary inference support to the vLLM HCU plugin, with HCU sparse MLA/indexer/sink execution and Triton compressed-tensors FP8 W8A8 MoE on eight BW1100 devices.

**Architecture:** The adapter is a plugin-owned `vllm_hcu.models.hy_v4` package. It preserves the HY-specific configuration, iHC, gated sparse MLA, indexer, sink, MoE routing, and checkpoint mapping while reusing the plugin's existing HCU runtime and quantization layers. New Triton code is introduced only behind an HY V4-specific boundary and only after a numerical reference test fails against the available HCU implementation.

**Tech Stack:** Python 3.10, PyTorch/HIP, Triton, vLLM 0.25.1, compressed-tensors FP8 W8A8, pytest.

**Spec:** `docs/superpowers/specs/2026-08-24-hy-v4-hcu-adapter-design.md`

## Global Constraints

- Plugin baseline is v0.25.1 commit `f1f7a06489e7535bd63711027e876ea1d3301b23`.
- The validation checkpoint is `/models/Hy4-preview-Testing-Channel-FP8-w8a8-v2`.
- The vLLM API reference is `/models/zb/vllm_025/vllm` on v0.25.1.
- Initial full-model topology is TP=8 and EP=8 on eight BW1100 devices.
- Initial MoE backend is exactly `triton`.
- Sparse attention, indexer behavior, and the learnable sink must fail closed; none may silently fall back or be disabled.
- Accuracy validation precedes performance work.
- First delivery excludes MTP, reasoning parsing, and tool-call parsing.
- The adapter has no runtime dependency on `/models/zb/Hy4-p_vLLM`.

---

### Task 1: Establish the baseline and register HY V4 configuration/model names

**Files:**
- Create: `tests/models/hy_v4/__init__.py`
- Create: `tests/models/hy_v4/test_registration.py`
- Create: `vllm_hcu/models/hy_v4/__init__.py`
- Create: `vllm_hcu/models/hy_v4/config.py`
- Modify: `vllm_hcu/models/__init__.py`

**Interfaces:**
- Produces: `HYV4Config`, `register_hy_v4_config() -> None`, and lazy registry path `vllm_hcu.models.hy_v4:HYV4ForCausalLM`.
- Consumes: vLLM's `_CONFIG_REGISTRY`, Transformers `AutoConfig`, and `ModelRegistry.register_model`.

- [ ] **Step 1: Run a clean baseline test selection**

```bash
VLLM_V0251_SOURCE_ROOT=/models/zb/vllm_025/vllm \
PYTHONPATH=/models/zb/vllm_025/vllm:. \
/usr/bin/python3.10 -m pytest \
  tests/patch/test_plugin_lifecycle.py \
  tests/runtime_patch/test_sparse_indexer_loading.py -q
```

Expected: the existing selection passes. If it does not, record the exact
pre-existing failure before changing production code.

- [ ] **Step 2: Write failing registration tests**

```python
def test_register_hy_v4_config_is_idempotent(monkeypatch):
    registry: dict[str, object] = {}
    monkeypatch.setattr(vllm_config, "_CONFIG_REGISTRY", registry)
    register_hy_v4_config()
    register_hy_v4_config()
    assert registry["hy_v4"] is HYV4Config


def test_hy_v4_config_derives_architecture_defaults():
    config = HYV4Config(num_hidden_layers=3, qk_nope_head_dim=192,
                       qk_rope_head_dim=64)
    assert config.qk_head_dim == 256
    assert config.mlp_layer_types == ["dense", "sparse", "sparse"]
    assert config.indexer_types == ["full", "shared", "shared"]


def test_hy_v4_registry_is_backbone_only(monkeypatch):
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(ModelRegistry, "register_model",
                        lambda name, path: calls.append((name, path)))
    register_model()
    assert ("HYV4ForCausalLM",
            "vllm_hcu.models.hy_v4:HYV4ForCausalLM") in calls
    assert all(name != "HYV4MTPModel" for name, _ in calls)
```

- [ ] **Step 3: Run the tests and verify RED**

```bash
VLLM_V0251_SOURCE_ROOT=/models/zb/vllm_025/vllm \
PYTHONPATH=/models/zb/vllm_025/vllm:. \
/usr/bin/python3.10 -m pytest tests/models/hy_v4/test_registration.py -q
```

Expected: collection fails because `vllm_hcu.models.hy_v4` does not exist.

- [ ] **Step 4: Implement configuration and registration**

Port the field-for-field `HYV4Config` from the approved architecture delta and
add this idempotent boundary:

```python
def register_hy_v4_config() -> None:
    from transformers import AutoConfig
    from vllm.transformers_utils import config as vllm_config

    vllm_config._CONFIG_REGISTRY[HYV4Config.model_type] = HYV4Config
    AutoConfig.register(HYV4Config.model_type, HYV4Config, exist_ok=True)
```

Call `register_hy_v4_config()` at the start of `register_model()`, then add:

```python
ModelRegistry.register_model(
    "HYV4ForCausalLM",
    "vllm_hcu.models.hy_v4:HYV4ForCausalLM",
)
```

During this task the package `__init__.py` exports only `HYV4Config` and
`register_hy_v4_config`, so importing the config does not require the not-yet
implemented model. Task 5 adds the `HYV4ForCausalLM` export. The package must
not contain the CUDA/ROCm platform dispatch from the NVIDIA-only delta.

- [ ] **Step 5: Run registration and lifecycle tests and verify GREEN**

```bash
VLLM_V0251_SOURCE_ROOT=/models/zb/vllm_025/vllm \
PYTHONPATH=/models/zb/vllm_025/vllm:. \
/usr/bin/python3.10 -m pytest \
  tests/models/hy_v4/test_registration.py \
  tests/patch/test_plugin_lifecycle.py -q
```

Expected: all selected tests pass and plugin registration remains idempotent.

- [ ] **Step 6: Commit**

```bash
git add tests/models/hy_v4 vllm_hcu/models/hy_v4 \
  vllm_hcu/models/__init__.py
git commit -m "feat: register HY V4 model configuration"
```

---

### Task 2: Port and numerically validate independent Hyper-Connections

**Files:**
- Create: `vllm_hcu/models/hy_v4/hc.py`
- Create: `tests/models/hy_v4/test_hc.py`

**Interfaces:**
- Produces: `HYV4HCPreLayer`, `HYV4HCPostLayer`, `HYV4HCHeadLayer`, and `HYV4HCLayer`.
- Consumes: `HYV4Config`, PyTorch tensors, and vLLM `ReplicatedLinear`.

- [ ] **Step 1: Write failing pure-reference tests**

```python
def reference_post(branch, residual, gates):
    return residual.float() + gates.float().unsqueeze(-1) * branch.float().unsqueeze(1)


def test_hc_post_matches_fp32_reference():
    branch = torch.tensor([[0.5, -1.0]], dtype=torch.float32)
    residual = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    gates = torch.tensor([[0.25, 0.75]])
    actual = HYV4HCPostLayer(SimpleNamespace())(branch, residual, gates)
    expected = reference_post(branch, residual, gates)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
```

Cover disabled iHC, 2D-to-3D channel reshape, gate ordering, FP32 gate math,
head merging, and invalid channel dimensions.

- [ ] **Step 2: Run the tests and verify RED**

```bash
PYTHONPATH=/models/zb/vllm_025/vllm:. \
/usr/bin/python3.10 -m pytest tests/models/hy_v4/test_hc.py -q
```

Expected: import fails because `hc.py` is absent.

- [ ] **Step 3: Port the minimal iHC implementation**

Port the four classes from the architecture delta without platform dispatch.
Keep FP32-sensitive gates explicit:

```python
gate_logits = self.hc_fn(hidden_states).float()
pre_gates, post_gates = gate_logits.chunk(2, dim=-1)
pre_gates = torch.sigmoid(pre_gates).to(hidden_states.dtype)
post_gates = torch.sigmoid(post_gates).to(hidden_states.dtype)
```

Preserve checkpoint-visible names `hc_fn` and `hc_head_fn`. Reject a last
dimension that is neither `hidden_size` nor `hc_mult * hidden_size`.

- [ ] **Step 4: Run numerical tests and verify GREEN**

```bash
PYTHONPATH=/models/zb/vllm_025/vllm:. \
/usr/bin/python3.10 -m pytest tests/models/hy_v4/test_hc.py -q
```

Expected: all tests pass. If an HCU operation fails, add a reproducing test
before introducing an HY V4-scoped Triton kernel.

- [ ] **Step 5: Commit**

```bash
git add vllm_hcu/models/hy_v4/hc.py tests/models/hy_v4/test_hc.py
git commit -m "feat: add HY V4 independent hyper-connections"
```

---

### Task 3: Add accuracy-checked Triton MoE routing policy

**Files:**
- Create: `vllm_hcu/models/hy_v4/moe.py`
- Create: `tests/models/hy_v4/test_moe.py`
- Modify only after a numerical RED case: `vllm_hcu/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe_runtime.py`

**Interfaces:**
- Produces: `HYV4FeedForward` and `HYV4MoEFused`.
- Consumes: `FusedMoE`, `GateLinear`, compressed-tensors W8A8 FP8 methods, and `VllmConfig.kernel_config.moe_backend`.

- [ ] **Step 1: Write failing router and backend-policy tests**

```python
def reference_sigmoid_topk(logits, *, top_k, scale):
    scores = logits.float().sigmoid()
    weights, ids = torch.topk(scores, top_k, dim=-1)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    return weights * scale, ids


def test_hy_v4_rejects_non_triton_moe_backend():
    config = fake_vllm_config(moe_backend="auto")
    with pytest.raises(RuntimeError, match="--moe-backend triton"):
        HYV4MoEFused(config=config.hf_config, vllm_config=config,
                     prefix="model.layers.1.mlp")
```

Also assert FP32 router weights/logits, expert correction bias, shared expert,
normalized sigmoid top-k, routed scale `2.827`, and `swiglu_limit=10.0`.

- [ ] **Step 2: Run the tests and verify RED**

```bash
PYTHONPATH=/models/zb/vllm_025/vllm:. \
/usr/bin/python3.10 -m pytest tests/models/hy_v4/test_moe.py -q
```

Expected: import fails because `moe.py` is absent.

- [ ] **Step 3: Port feed-forward and MoE modules**

Use the v0.25.1 `FusedMoE` interface and enforce the backend before allocating
expert weights:

```python
moe_backend = vllm_config.kernel_config.moe_backend
if moe_backend != "triton":
    raise RuntimeError(
        "HY V4 FP8 W8A8 currently requires --moe-backend triton; "
        f"got {moe_backend!r}."
    )
```

Construct `FusedMoE` with sigmoid scoring, grouped top-k values of one, the
correction bias, shared expert, routed scale, EPLB fields, and SwiGLU clamp.

- [ ] **Step 4: Verify the compressed-tensors target Triton route**

```bash
PYTHONPATH=/models/zb/vllm_025/vllm:. \
/usr/bin/python3.10 -m pytest \
  tests/models/hy_v4/test_moe.py \
  tests/runtime_patch/test_quant_gemm_aiter.py \
  -k 'hy_v4 or moe_fp8 or compressed_tensors' -q
```

Expected: the HY V4 tests and existing target-Triton route tests pass.

- [ ] **Step 5: Commit**

```bash
git add vllm_hcu/models/hy_v4/moe.py tests/models/hy_v4/test_moe.py \
  vllm_hcu/model_executor/layers/quantization/compressed_tensors/
git commit -m "feat: add HY V4 Triton MoE path"
```

If the existing target-Triton path exposes an unsupported HY V4 shape, stop
this task at RED, amend this plan with the tested kernel interface, and then
continue. Do not add speculative kernel code before observing that failure.

---

### Task 4: Adapt gated sparse MLA, lightning indexer, and attention sink

**Files:**
- Create: `vllm_hcu/models/hy_v4/attention.py`
- Create only after a backend RED case: `vllm_hcu/models/hy_v4/hcu_sparse.py`
- Create: `tests/models/hy_v4/test_attention.py`

**Interfaces:**
- Produces: `HYV4MLAAttention`, `Indexer`, `compute_skip_topk_layers()`, `is_skip_topk_indexer_weight()`, and `require_hyv4_sink_backend()`.
- Consumes: vLLM `MLAAttention`, HCU sparse MLA/indexer operations, scheduler top-k buffers, and plugin-patched attention runtime.

- [ ] **Step 1: Write failing layer-pattern and sink tests**

```python
def test_full_and_shared_indexer_pattern():
    config = SimpleNamespace(
        num_hidden_layers=6,
        layer_types=["deepseek_sparse_attention"] * 6,
        indexer_types=["full", "shared", "shared", "full", "shared", "shared"],
    )
    assert compute_skip_topk_layers(config) == {1, 2, 4, 5}


def test_sink_incapable_backend_fails_closed(monkeypatch):
    monkeypatch.setattr(
        FakeSparseBackend,
        "supports_sink",
        classmethod(lambda cls: False),
    )
    with pytest.raises(RuntimeError, match="learnable sink"):
        require_hyv4_sink_backend(FakeSparseBackend)
```

Define `FakeSparseBackend` in the test with `is_sparse() -> True`. The wished-for
production helper `require_hyv4_sink_backend()` returns a sink-capable sparse
backend or raises. Add monkeypatch-based kernel forwarding tests that capture
the exact `attn_sink` object passed during prefill and decode. Also cover
interleaved indexer RoPE, shared top-k reuse, FP32 sink dtype, TP sink slicing,
elementwise gate shape, and prevention of dense fallback.

- [ ] **Step 2: Run the tests and verify RED**

```bash
PYTHONPATH=/models/zb/vllm_025/vllm:. \
/usr/bin/python3.10 -m pytest tests/models/hy_v4/test_attention.py -q
```

Expected: import fails because `attention.py` is absent.

- [ ] **Step 3: Port HY-specific projections and indexer logic**

Port q/kv LoRA projections, output gate, interleaved RoPE, full/shared indexer
selection, and scheduler-sized shared top-k buffer. Remove NVIDIA imports and
replace warning-based sink disablement with:

```python
if self.learnable_sink and not backend.supports_sink():
    raise RuntimeError(
        "HY V4 learnable sink requires a sink-capable sparse MLA backend "
        "for both prefill and decode."
    )
```

- [ ] **Step 4: Exercise and bind the HCU sink-capable implementation**

First run the plugin's HCU sparse MLA functions. If generic `MLAAttention`
does not forward sinks, add a failing adapter test and create `hcu_sparse.py`.
The adapter passes `attn_sink=self.sinks` in prefill and decode and does not
alter global backend selection.

- [ ] **Step 5: Run component numerical tests and verify GREEN**

```bash
VLLM_V0251_SOURCE_ROOT=/models/zb/vllm_025/vllm \
PYTHONPATH=/models/zb/vllm_025/vllm:. \
/usr/bin/python3.10 -m pytest \
  tests/models/hy_v4/test_attention.py \
  tests/runtime_patch/test_sparse_indexer_loading.py \
  tests/runtime_patch/test_mla_target_ownership.py -q
```

Expected: all tests pass, including both sink phases and import ownership.

- [ ] **Step 6: Commit**

```bash
git add vllm_hcu/models/hy_v4/attention.py \
  tests/models/hy_v4/test_attention.py
git add vllm_hcu/models/hy_v4/hcu_sparse.py 2>/dev/null || true
git commit -m "feat: add HY V4 sparse MLA on HCU"
```

---

### Task 5: Assemble the backbone and prove exact weight coverage

**Files:**
- Create: `vllm_hcu/models/hy_v4/model.py`
- Create: `tests/models/hy_v4/test_weight_loading.py`
- Modify: `vllm_hcu/models/hy_v4/__init__.py`

**Interfaces:**
- Produces: `HYV4DecoderLayer`, `HYV4Model`, and `HYV4ForCausalLM`.
- Consumes: Tasks 1-4, vLLM PP/TP/EP helpers, compressed-tensors config, and checkpoint iterators.

- [ ] **Step 1: Write a failing synthetic weight-coverage test**

Build a three-layer configuration with one dense layer, two MoE layers, iHC,
sparse attention, full/shared indexers, gated MLA, and sinks:

```python
def assert_fully_loaded(model, checkpoint):
    expected = {name for name, _ in model.named_parameters()}
    loaded = model.load_weights(iter(checkpoint))
    assert loaded == expected


@pytest.mark.hcu
def test_backbone_loads_every_synthetic_parameter(hcu_device, small_config):
    model = build_small_hyv4_model(small_config, device=hcu_device)
    checkpoint = synthetic_checkpoint_for(model, small_config)
    assert_fully_loaded(model, checkpoint)
```

Add cases for `wk` plus `weights_proj`, TP sink slices, expert bias, iHC names
without `.weight`, packed gate/up projections, quantization scales, and
unknown required names.

- [ ] **Step 2: Run the tests and verify RED**

```bash
PYTHONPATH=/models/zb/vllm_025/vllm:. \
/usr/bin/python3.10 -m pytest tests/models/hy_v4/test_weight_loading.py -q
```

Expected: import fails because `model.py` is absent.

- [ ] **Step 3: Port decoder, backbone, and causal LM wrapper**

Port ordinary/iHC forward paths, PP intermediate tensors, embedding,
normalization, LM head, and EPLB metadata. Pass `VllmConfig` into
`HYV4MoEFused` so backend policy is checked before expert allocation.

- [ ] **Step 4: Implement exact weight mapping**

```python
stacked_params_mapping = [
    (".gate_up_proj", ".gate_proj", 0),
    (".gate_up_proj", ".up_proj", 1),
    ("wk_weights_proj", "wk", 0),
    ("wk_weights_proj", "weights_proj", 1),
]
```

Use vLLM's expert mapping generator, buffer quantized indexer weight/scale
pairs, slice sinks by TP rank, rewrite iHC names explicitly, and delegate
remaining tensors to attached loaders. Return the exact loaded name set.

- [ ] **Step 5: Run all HY V4 tests and verify GREEN**

```bash
VLLM_V0251_SOURCE_ROOT=/models/zb/vllm_025/vllm \
PYTHONPATH=/models/zb/vllm_025/vllm:. \
/usr/bin/python3.10 -m pytest tests/models/hy_v4 -q
```

Expected: all HY V4 component and weight tests pass.

- [ ] **Step 6: Commit**

```bash
git add vllm_hcu/models/hy_v4/model.py \
  vllm_hcu/models/hy_v4/__init__.py \
  tests/models/hy_v4/test_weight_loading.py
git commit -m "feat: assemble HY V4 HCU backbone"
```

---

### Task 6: Add fail-closed integration contracts and run regressions

**Files:**
- Create: `tests/integration/models/test_hy_v4_smoke.py`
- Modify: `tests/integration/models/README.md`
- Modify: `tests/models/README.md`

**Interfaces:**
- Produces: automated startup contract and documented invocation.
- Consumes: registered HY V4 model and model-test resource fixtures.

- [ ] **Step 1: Write a failing eight-device integration contract**

```python
pytestmark = [
    pytest.mark.hcu,
    pytest.mark.model,
    pytest.mark.multi_hcu,
    pytest.mark.hcu_count(8),
    pytest.mark.slow,
]


def test_hy_v4_short_generation(hcu_test_resources):
    model = require_model_runtime(
        hcu_test_resources,
        env_name="VLLM_HCU_HY_V4_MODEL",
        relative_path="/models/Hy4-preview-Testing-Channel-FP8-w8a8-v2",
        label="HY V4 FP8 W8A8",
        hcu_count=8,
    )
    result = run_vllm_case(
        "tp-ep-smoke",
        model,
        timeout_s=7200,
        extra_args=[
            "--tensor-parallel-size", "8",
            "--gpu-memory-utilization", "0.9",
            "--moe-backend", "triton",
        ],
    )
    assert result["parallel_config"]["tensor_parallel_size"] == 8
    assert result["parallel_config"]["enable_expert_parallel"] is True
    assert all(record["token_ids"] for record in result["output"])
```

- [ ] **Step 2: Run collection and verify RED**

```bash
PYTHONPATH=/models/zb/vllm_025/vllm:. \
/usr/bin/python3.10 -m pytest \
  tests/integration/models/test_hy_v4_smoke.py --collect-only -q
```

Expected: collection exposes missing fixture fields or runtime arguments.
Correct the harness contract before launching the checkpoint.

- [ ] **Step 3: Complete harness and documentation**

Add only fixture surface needed for TP=8, EP, Triton MoE, eager mode,
deterministic sampling, token IDs, and finite-logit reporting. Document model
path and pytest markers.

- [ ] **Step 4: Run fast HY V4 and regression suites**

```bash
VLLM_V0251_SOURCE_ROOT=/models/zb/vllm_025/vllm \
PYTHONPATH=/models/zb/vllm_025/vllm:. \
/usr/bin/python3.10 -m pytest \
  tests/models/hy_v4 \
  tests/patch/test_plugin_lifecycle.py \
  tests/runtime_patch/test_sparse_indexer_loading.py \
  tests/runtime_patch/test_quant_gemm_aiter.py \
  tests/runtime_patch/test_mla_target_ownership.py -q
```

Expected: zero failures.

- [ ] **Step 5: Run source checks**

```bash
git diff --check
/usr/bin/python3.10 -m compileall -q \
  vllm_hcu/models/hy_v4 tests/models/hy_v4
```

Expected: both commands exit zero.

- [ ] **Step 6: Commit**

```bash
git add tests/integration/models/test_hy_v4_smoke.py \
  tests/integration/models/README.md tests/models/README.md
git commit -m "test: add HY V4 HCU integration coverage"
```

---

### Task 7: Validate the checkpoint on eight BW1100 devices

**Files:**
- Create: `docs/validation/hy-v4-fp8-w8a8-bw1100.md`
- Modify production/test files only through a new RED-GREEN cycle for each observed failure.

**Interfaces:**
- Produces: launch command, load report, generation result, and limitations.
- Consumes: all prior tasks, eight free devices, and supplied checkpoint.

- [ ] **Step 1: Confirm all eight devices are free**

```bash
/opt/hyhal/bin/hy-smi --showmeminfo vram
```

- [ ] **Step 2: Verify checkout import provenance**

```bash
PYTHONPATH=/models/zb/vllm_025/vllm:. \
/usr/bin/python3.10 -c \
  'import vllm,vllm_hcu; print(vllm.__file__); print(vllm_hcu.__file__)'
```

Expected: vLLM resolves under `/models/zb/vllm_025/vllm` and the plugin under
the isolated HY V4 checkout.

- [ ] **Step 3: Launch deterministic generation**

```bash
HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
PYTHONPATH=/models/zb/vllm_025/vllm:. \
/usr/bin/python3.10 -m vllm.entrypoints.openai.api_server \
  --model /models/Hy4-preview-Testing-Channel-FP8-w8a8-v2 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --moe-backend triton \
  --enforce-eager \
  --max-model-len 4096 \
  --max-num-seqs 1 \
  --port 8014
```

Expected: all 131 shards load without required missing parameters and the
server becomes ready.

- [ ] **Step 4: Send one short request**

```bash
curl -sS http://127.0.0.1:8014/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"/models/Hy4-preview-Testing-Channel-FP8-w8a8-v2","prompt":"请用一句话介绍海洋。","max_tokens":8,"temperature":0}'
```

Expected: non-empty completion and no worker NaN/Inf or kernel error.

- [ ] **Step 5: Diagnose runtime failures test-first**

For each failure, capture the smallest reproducer in the owning test file,
verify RED, implement the minimum correction, verify GREEN, rerun the owning
suite, and retry the checkpoint.

- [ ] **Step 6: Record evidence and run final verification**

Write commit, versions, command, topology, load result, first output,
numerical checks, timing, and limitations to the validation document. Run:

```bash
VLLM_V0251_SOURCE_ROOT=/models/zb/vllm_025/vllm \
PYTHONPATH=/models/zb/vllm_025/vllm:. \
/usr/bin/python3.10 -m pytest tests/models/hy_v4 -q
git diff --check
git status --short
```

Expected: tests pass, diff check exits zero, and only intended files remain.

- [ ] **Step 7: Commit validation evidence**

```bash
git add docs/validation/hy-v4-fp8-w8a8-bw1100.md vllm_hcu tests
git commit -m "feat: validate HY V4 FP8 inference on BW1100"
```
