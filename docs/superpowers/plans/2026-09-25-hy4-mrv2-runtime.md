# Hy4 MRV2 HCU Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Serve /models/Hy4-preview-Channel-FP8-w8a8-v2 on the v0.28.1-dev HCU stack using Model Runner V2, TP8, native MTP3, and FP8 E4M3 KV, with HumanEval/0-7 evidence.

**Architecture:** Keep target OpenDAS vLLM's Hy4 config, parser, MTP conversion, scheduler, and MRV2. Register an HCU-owned Hy4 model and adapt the v0.25.1 final model, MLA/indexer, iHC, MoE, and checkpoint loader to the current target APIs; reject unsupported ABIs and V1 rather than routing around them.

**Tech Stack:** Python 3.10, installed OpenDAS vLLM 0.28.1 (77acaf6 lineage), PyTorch 2.11/HIP 6.3, HCU plugin, pytest, eight HCU cards, OpenAI-compatible API, local HumanEval JSONL.

**Spec:** docs/superpowers/specs/2026-09-25-hy4-v0281-runtime-alignment-design.md

## Global Constraints

- Start from plugin origin/v0.28.1-dev@93021650a5c768121507d5b8241d0e848756e44e; refresh both plugin remote refs before code and before MR publication.
- Source behavior is origin/v0.25.1@6ea7b12f3d77d4564d613aac31c5e6c5e7a8641d; adapt final behavior, never bulk cherry-pick.
- Use explicit VLLM_USE_V2_MODEL_RUNNER=1 in every server process. Assert VllmConfig.use_v2_model_runner and HcuGPUModelRunnerV2; V1 is not a fallback.
- Target-only TP8 first, then MTP3, then MTP3 plus --kv-cache-dtype fp8_e4m3. Preserve default CUDA Graph policy and --moe-backend aiter.
- Never set VLLM_PLUGINS=hcu. Never install over the global vLLM/Torch/HIP stack. Use apply_patch for edits.
- Do not claim W4A8, PCP/EP, dynamic EPLB, or two-node behavior from this Channel-FP8 checkpoint.
- The historical v0.25.1 run and any startup-only smoke are not target accuracy evidence.

## Review Focus

- VLLM_USE_V2_MODEL_RUNNER unset or 0: config/worker must fail closed rather than silently switch Hy4 to V1 (Task 1 test).
- Tied lm_head and missing/duplicate Channel-FP8 shard or scale: loader must reject incomplete/ambiguous weights without silent defaults (Task 5 tests).
- AITER tuned config missing for one capture shape: choose the target Triton fallback before Graph capture and record which path executed (Task 3 test and Task 7 logs).
- Sparse indexer with shared producer or FP8 E4M3 cache: preserve causal indices and native HIPC cache writes under MRV2 (Task 4 tests).
- MTP layer count or quantization mismatch: reject invalid draft metadata and leave backbone quantization unchanged (Task 6 tests).

---

### Task 1: Freeze provenance and pin MRV2

**Files:**
- Modify: tests/patch/test_plugin_lifecycle.py
- Create: docs/hy4_v0281_validation.md

**Interfaces:**
- Consumes: vllm_hcu.v1.worker._create_model_runner(config, device, *, use_v2_model_runner: bool)
- Produces: a tested V1 rejection and explicit MRV2 run contract; no product-code change.

- [ ] **Step 1: Refresh and record exact source, target, installed vLLM, model config, and GPU state.**

~~~bash
git fetch origin v0.25.1 v0.28.1-dev
git rev-parse origin/v0.25.1 origin/v0.28.1-dev HEAD
python3 -m pip show vllm torch vllm-plugin-das
sha256sum /models/Hy4-preview-Channel-FP8-w8a8-v2/config.json
rocm-smi
~~~

Compare fetched SHAs with the spec; if either moved, update the disposition before coding. Record installed import roots and proprietary provider version in docs/hy4_v0281_validation.md using apply_patch.

- [ ] **Step 2: Add the V1-rejection test.**

~~~python
def test_hcu_worker_rejects_model_runner_v1(cpu_safe_hcu_worker_module):
    with pytest.raises(RuntimeError, match="only Model Runner V2"):
        cpu_safe_hcu_worker_module._create_model_runner(
            object(), object(), use_v2_model_runner=False
        )


def test_explicit_model_runner_v2_config(monkeypatch):
    from vllm import envs
    from vllm.config import VllmConfig
    monkeypatch.setattr(envs, "VLLM_USE_V2_MODEL_RUNNER", True)
    config = object.__new__(VllmConfig)
    assert config.use_v2_model_runner is True
~~~

- [ ] **Step 3: Run the new test and existing V2-selection test.**

~~~bash
PYTHONNOUSERSITE=1 PYTHONPATH=. pytest -q tests/patch/test_plugin_lifecycle.py -k 'model_runner or worker_rejects'
~~~

Expected: both pass against the existing worker. A failure is a baseline defect to diagnose before changing the worker.

- [ ] **Step 4: Add the explicit launch contract to the validation record.**

~~~bash
export VLLM_USE_V2_MODEL_RUNNER=1
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
~~~

Record that the vLLM config property must be true and live worker must construct vllm_hcu.v1.hcu_model_runner_v2.HcuGPUModelRunnerV2; the plugin's GLM-only normalization is not a Hy4 selector.

- [ ] **Step 5: Commit this independently testable baseline.**

~~~bash
git add tests/patch/test_plugin_lifecycle.py docs/hy4_v0281_validation.md
git commit -m "test: pin Hy4 MRV2 and baseline provenance"
~~~

### Task 2: HCU Hy4 registration against target config ownership

**Files:**
- Create: vllm_hcu/models/hy_v4/__init__.py
- Modify: vllm_hcu/models/__init__.py
- Create: tests/models/hy_v4/test_registration.py

**Interfaces:**
- Consumes: target vllm.transformers_utils.configs.hy_v4.HYV4Config and target ModelRegistry.
- Produces: HYV4ForCausalLM and HYV4MTPModel plugin model routes; Task 5/6 supply the classes.

- [ ] **Step 1: Write a failing registry/owner test.**

~~~python
def test_hcu_hyv4_registration_uses_target_config(monkeypatch):
    from vllm.transformers_utils.configs.hy_v4 import HYV4Config
    from vllm_hcu.models import register_model
    from vllm import ModelRegistry
    calls = []
    monkeypatch.setattr(ModelRegistry, "register_model",
                        lambda architecture, implementation: calls.append(
                            (architecture, implementation)))
    register_model()
    assert ("HYV4ForCausalLM",
            "vllm_hcu.models.hy_v4:HYV4ForCausalLM") in calls
    assert ("HYV4MTPModel", "vllm_hcu.models.hy_v4:HYV4MTP") in calls
    assert HYV4Config.model_type == "hy_v4"
~~~

- [ ] **Step 2: Run the test; expect the two registration assertions to fail.**

~~~bash
PYTHONNOUSERSITE=1 PYTHONPATH=. pytest -q tests/models/hy_v4/test_registration.py
~~~

- [ ] **Step 3: Add only the two plugin model registrations and lazy exports.**

~~~python
ModelRegistry.register_model(
    "HYV4ForCausalLM", "vllm_hcu.models.hy_v4:HYV4ForCausalLM"
)
ModelRegistry.register_model(
    "HYV4MTPModel", "vllm_hcu.models.hy_v4:HYV4MTP"
)
~~~

The new package __getattr__ lazily imports .model.HYV4ForCausalLM and .mtp.HYV4MTP; import HYV4Config from target vLLM, never copy v0.25.1 config.py or its config/parser registration patches.

- [ ] **Step 4: Rerun registration plus plugin lifecycle tests.**

~~~bash
PYTHONNOUSERSITE=1 PYTHONPATH=. pytest -q tests/models/hy_v4/test_registration.py tests/patch/test_plugin_lifecycle.py
~~~

- [ ] **Step 5: Commit.**

~~~bash
git add vllm_hcu/models/__init__.py vllm_hcu/models/hy_v4/__init__.py tests/models/hy_v4/test_registration.py
git commit -m "feat: register HCU Hy4 models on target config"
~~~

### Task 3: Adapt Hy4 MoE and iHC to target owners

**Files:**
- Create: vllm_hcu/models/hy_v4/moe.py
- Create: vllm_hcu/models/hy_v4/hc.py
- Create: tests/models/hy_v4/test_moe.py
- Create: tests/models/hy_v4/test_hc.py

**Interfaces:**
- Consumes: target FusedMoEFactory(..., gate: nn.Module | None, shared_experts: nn.Module | None, scoring_func: str, e_score_correction_bias: Tensor | None) -> MoERunner.
- Produces: HYV4FeedForward, HYV4MoEFused, HYV4HCLayer and HYV4HCHeadLayer for Task 5.

- [ ] **Step 1: Bring source tests test_hy_v4_moe_preserves_router_and_clamp_contract, test_hy_v4_pure_tp_keeps_all_experts_in_local_metadata, and the iHC numerical/layout cases into the new target test files using apply_patch.** Replace the old FusedMoE monkeypatch with target FusedMoEFactory and import HYV4Config from target vLLM. Pin routing with these assertions:

~~~python
assert factory_kwargs["scoring_func"] == "sigmoid"
assert factory_kwargs["e_score_correction_bias"] is module.expert_bias
assert factory_kwargs["gate"] is module.gate
assert factory_kwargs["shared_experts"] is module.shared_experts
~~~

- [ ] **Step 2: Run those tests; expect import/API failure while the Hy4 modules are absent.**

~~~bash
PYTHONNOUSERSITE=1 PYTHONPATH=. pytest -q tests/models/hy_v4/test_moe.py tests/models/hy_v4/test_hc.py
~~~

- [ ] **Step 3: Port the final v0.25.1 moe.py and hc.py with apply_patch, replacing the removed constructor with the target factory.**

~~~python
from vllm.model_executor.layers.fused_moe import FusedMoEFactory, GateLinear

self.experts = FusedMoEFactory(
    num_experts=self.n_routed_experts,
    top_k=config.num_experts_per_tok,
    hidden_size=config.hidden_size,
    intermediate_size=config.expert_hidden_dim,
    scoring_func="sigmoid",
    e_score_correction_bias=self.expert_bias,
    gate=self.gate,
    shared_experts=self.shared_experts,
    quant_config=quant_config,
    prefix=f"{prefix}.experts",
)
~~~

Retain source routing arguments for renormalization, grouped TopK,
scaling, SwiGLU limit, and expert counts when the target signature and tests
prove their semantics. Keep default AITER/Triton choice in target MoE owner;
do not use upstream FlyDSL. Run the existing target MoE backend tests for
per-shape configuration and fallback.

- [ ] **Step 4: Rerun tests and commit.**

~~~bash
PYTHONNOUSERSITE=1 PYTHONPATH=. pytest -q tests/models/hy_v4/test_moe.py tests/models/hy_v4/test_hc.py tests/runtime_patch/test_moe_deepep.py
git add vllm_hcu/models/hy_v4/moe.py vllm_hcu/models/hy_v4/hc.py tests/models/hy_v4/test_moe.py tests/models/hy_v4/test_hc.py
git commit -m "feat: adapt Hy4 MoE and iHC to v0.28 interfaces"
~~~

### Task 4: Sparse MLA/indexer and native FP8 KV under MRV2

**Files:**
- Create: vllm_hcu/models/hy_v4/attention.py
- Create: vllm_hcu/models/hy_v4/hcu_sparse.py
- Create: vllm_hcu/models/hy_v4/fp8_kv_dequant.py
- Create: tests/models/hy_v4/test_attention.py

**Interfaces:**
- Consumes: target sparse MLA backend, MRV2 attention metadata, HCU native reshape-and-cache writer.
- Produces: HYV4MLAAttention, HYV4FlashMLASparseBackend and accuracy-safe FP8 E4M3 KV route for Task 5/6.

- [ ] **Step 1: Add failing source-derived tests for skip-TopK shared producer, causal sparse indices, BF16/FP8 KV layout, invalid dtype, and writer ownership.**

~~~python
def test_hy_v4_normalizes_fp8_e4m3_for_sparse_flashmla_selection():
    from vllm_hcu.models.hy_v4.attention import _normalize_hy_v4_kv_cache_dtype
    assert _normalize_hy_v4_kv_cache_dtype(
        "fp8_e4m3", use_sparse=True
    ) == "fp8_ds_mla"
    assert _normalize_hy_v4_kv_cache_dtype(
        "fp8_e4m3", use_sparse=False
    ) == "fp8_e4m3"
~~~

Port source tests test_full_and_shared_indexer_pattern,
test_shared_indexer_pattern_requires_a_preceding_full_producer,
test_hy_v4_mla_cache_spec_marks_fp8_as_quantized, and
test_hy_v4_rejects_accuracy_unsafe_kv_cache_dtype. Also check the target
native FlashAttention writer directly:

~~~python
def test_hcu_flash_cache_writer_is_native():
    import inspect
    from vllm_hcu.v1.attention.backends.fa_utils import reshape_and_cache_flash
    assert "torch.ops.hcu_ops.reshape_and_cache_flash" in inspect.getsource(
        reshape_and_cache_flash
    )
~~~

The live FP8 sparse route still needs its own writer trace in Task 7; this
unit check alone does not prove which writer Hy4 selected.

- [ ] **Step 2: Run focused tests; expect missing Hy4 attention modules.**

~~~bash
PYTHONNOUSERSITE=1 PYTHONPATH=. pytest -q tests/models/hy_v4/test_attention.py
~~~

- [ ] **Step 3: Adapt final source attention.py, hcu_sparse.py, and fp8_kv_dequant.py through apply_patch.**

~~~python
def _normalize_hy_v4_kv_cache_dtype(
    kv_cache_dtype: str, *, use_sparse: bool
) -> str:
    _require_accuracy_safe_kv_cache_dtype(kv_cache_dtype)
    if use_sparse and kv_cache_dtype == "fp8_e4m3":
        return "fp8_ds_mla"
    return kv_cache_dtype
~~~

Preserve the source's exact semantics for this helper, then align target metadata and writer signatures. AITER selects MoE only. Do not route FP8 cache writes through AITER, force eager mode, or bypass target Graph checks.

- [ ] **Step 4: Run attention tests, relevant existing sparse/FP8 runtime tests, and commit.**

~~~bash
PYTHONNOUSERSITE=1 PYTHONPATH=. pytest -q tests/models/hy_v4/test_attention.py tests/runtime_patch/test_sparse_indexer_loading.py tests/runtime_patch/test_hcu_model_runner_v2_api.py
git add vllm_hcu/models/hy_v4/attention.py vllm_hcu/models/hy_v4/hcu_sparse.py vllm_hcu/models/hy_v4/fp8_kv_dequant.py tests/models/hy_v4/test_attention.py
git commit -m "feat: adapt Hy4 sparse MLA and native FP8 KV for MRV2"
~~~

### Task 5: Channel-FP8 checkpoint loader and backbone

**Files:**
- Create: vllm_hcu/models/hy_v4/model.py
- Create: tests/models/hy_v4/test_weight_loading.py

**Interfaces:**
- Consumes: Task 3 layers, Task 4 attention, target AutoWeightsLoader(module, *, ignore_unexpected_prefixes=...) and target compressed-tensors quantization.
- Produces: HYV4Model, HYV4DecoderLayer, HYV4ForCausalLM and strict load_weights(...) -> set[str].

- [ ] **Step 1: Port test cases from source test_weight_loading.py for tied head, duplicate name, missing channel scale, packed shard extent, and strict unexpected weights.** Add the target ABI assertion:

~~~python
def test_target_auto_loader_uses_ignore_unexpected_prefixes():
    import inspect
    from vllm.model_executor.models.utils import AutoWeightsLoader
    parameters = inspect.signature(AutoWeightsLoader).parameters
    assert "ignore_unexpected_prefixes" in parameters
    assert "skip_prefixes" not in parameters
~~~

- [ ] **Step 2: Run the tests; expect absent model/loader failures.**

~~~bash
PYTHONNOUSERSITE=1 PYTHONPATH=. pytest -q tests/models/hy_v4/test_weight_loading.py
~~~

- [ ] **Step 3: Port final source model.py with apply_patch, retaining the checkpoint accounting and weight-name rewrite but adapting loader arguments.**

~~~python
loader = _HYV4AutoWeightsLoader(
    self,
    ignore_unexpected_prefixes=(
        ["lm_head."] if self.config.tie_word_embeddings else None
    ),
)
~~~

Keep Channel-FP8 on target compressed-tensors. Packed W4A8 is outside this
MR; reject that format explicitly rather than import an absent adapter or
pretend this Channel-FP8 run certifies it. Reject incomplete scales before any
fallback.

~~~python
checkpoint_format = getattr(self.quant_config, "checkpoint_format", None)
if checkpoint_format in ("hy4-w4a8-custom-v1", "hy4_w4a8_v1"):
    raise NotImplementedError("HYV4 packed W4A8 is outside this MR")
~~~

This explicit rejection replaces the unconditional import of the absent
v0.25.1 W4A8 adapter.

- [ ] **Step 4: Run focused loader and backbone tests, then commit.**

~~~bash
PYTHONNOUSERSITE=1 PYTHONPATH=. pytest -q tests/models/hy_v4/test_weight_loading.py tests/models/hy_v4/test_attention.py
git add vllm_hcu/models/hy_v4/model.py tests/models/hy_v4/test_weight_loading.py
git commit -m "feat: adapt Hy4 backbone and strict Channel-FP8 loading"
~~~

### Task 6: Native MTP3 and target interface gate

**Files:**
- Create: vllm_hcu/models/hy_v4/mtp.py
- Create: tests/models/hy_v4/test_mtp.py
- Create: tests/models/hy_v4/test_mtp_config.py

**Interfaces:**
- Consumes: target SpeculativeConfig.hf_config_override for hy_v4_mtp, Task 5 backbone, target MRV2 sampler.
- Produces: HYV4MTP, strict one-layer draft loading and checkpoint-native Channel-FP8 quant selection.

- [ ] **Step 1: Port source test_mtp.py and test_mtp_config.py through apply_patch, deleting tests that require the obsolete plugin MTP config patch.** Add a target owner assertion:

~~~python
def test_target_hyv4_mtp_conversion():
    from vllm.config.speculative import SpeculativeConfig
    from vllm.transformers_utils.configs.hy_v4 import HYV4Config
    target = HYV4Config(num_nextn_predict_layers=1)
    draft = SpeculativeConfig.hf_config_override(target)
    assert draft.model_type == "hy_v4_mtp"
    assert draft.architectures == ["HYV4MTPModel"]
~~~

- [ ] **Step 2: Run tests; expect absent HYV4MTP implementation, while target conversion passes.**

~~~bash
PYTHONNOUSERSITE=1 PYTHONPATH=. pytest -q tests/models/hy_v4/test_mtp.py tests/models/hy_v4/test_mtp_config.py
~~~

- [ ] **Step 3: Port final source mtp.py with apply_patch and adapt target MRV2 sampling/quant interfaces.**

~~~python
if getattr(hf_config, "mtp_quant_algo", "NONE").upper() == "NONE":
    return backbone_quant_config
~~~

Preserve explicit BF16/FP16 draft opt-out and fail closed on unknown quantization or non-single-layer extension. Do not register a second hy_v4_mtp config conversion patch.

- [ ] **Step 4: Run all Hy4 tests and the relevant installed-target bootstrap tests, then commit.**

~~~bash
PYTHONNOUSERSITE=1 PYTHONPATH=. pytest -q tests/models/hy_v4 tests/patch/test_clean_process_bootstrap.py tests/patch/test_plugin_lifecycle.py
git add vllm_hcu/models/hy_v4/mtp.py tests/models/hy_v4/test_mtp.py tests/models/hy_v4/test_mtp_config.py
git commit -m "feat: adapt Hy4 native MTP to MRV2"
~~~

### Task 7: Live TP8, MTP3, FP8 KV, and HumanEval/0-7

**Files:**
- Modify: docs/hy4_v0281_validation.md
- Create: tests/integration/server/test_evalscope_hy4_humaneval.py

**Interfaces:**
- Consumes: Tasks 1-6 implementation and the local /models/datasets/humaneval/HumanEval.jsonl.gz source.
- Produces: exact commands, log paths, eight-prompt target/MTP prediction records, pass@1 and per-item comparison. This is a validation gate, not an accuracy claim in advance.

- [ ] **Step 1: Add a server-test wrapper that builds the same arguments for target, MTP3, and MTP3+FP8, varying only the two documented switches.**

~~~python
BASE_ARGS = [
    "--tensor-parallel-size", "8", "--moe-backend", "aiter",
    "--enable-prefix-caching", "--max-model-len", "4096",
    "--max-num-seqs", "16", "--max-num-batched-tokens", "4096",
    "--default-chat-template-kwargs", '{"reasoning_effort":"no_think"}',
    "--seed", "0",
]
MTP_ARGS = ["--speculative-config",
            '{"method":"mtp","num_speculative_tokens":3}']
FP8_KV_ARGS = ["--kv-cache-dtype", "fp8_e4m3"]
~~~

The wrapper must set VLLM_USE_V2_MODEL_RUNNER=1, eight visible cards, NO_PROXY, and a fresh result/log directory per run. It must assert exactly HumanEval/0-7 and eight unique predictions.

~~~bash
VLLM_USE_V2_MODEL_RUNNER=1 HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  env -u VLLM_PLUGINS python3 -m vllm.entrypoints.cli.main serve \
  /models/Hy4-preview-Channel-FP8-w8a8-v2 \
  --host 127.0.0.1 --port 8000 --trust-remote-code \
  --tensor-parallel-size 8 --moe-backend aiter \
  --enable-prefix-caching --max-model-len 4096 \
  --max-num-seqs 16 --max-num-batched-tokens 4096 \
  --default-chat-template-kwargs '{"reasoning_effort":"no_think"}' \
  --seed 0
~~~

MTP3 adds --speculative-config '{"method":"mtp","num_speculative_tokens":3}';
the FP8 run adds --kv-cache-dtype fp8_e4m3. Record the actual final command
and resolved runner/Graph/backend logs; this plan command is not test evidence.

- [ ] **Step 2: Run the wrapper first as argument-only tests, then each fresh foreground server one at a time.**

~~~bash
PYTHONNOUSERSITE=1 PYTHONPATH=. pytest -q tests/integration/server/test_evalscope_hy4_humaneval.py -k args
~~~

Run target-only, MTP3, and MTP3+FP8 E4M3 with the same BASE_ARGS and the model path. Capture resolved MRV2 class, Graph mode/capture, AITER config hit or Triton fallback, sparse backend, request metrics, draft/accepted tokens, cache-hit evidence, responses, and teardown. Do not force Graph mode or eager.

- [ ] **Step 3: Evaluate HumanEval/0-7 against target-only and MTP3 (and FP8 KV if it is an accepted default) with temp=0, seed=0, max_tokens=2048, reasoning_effort=no_think.**

~~~bash
PYTHONNOUSERSITE=1 PYTHONPATH=. pytest -q tests/integration/server/test_evalscope_hy4_humaneval.py -k humaneval
~~~

Use isolated EvalScope dependencies if available; otherwise document a reproducible local evaluator with the same eight task IDs and code-execution safety boundary. Store fresh outputs outside the repository and summarize exact paths and per-item flips in docs/hy4_v0281_validation.md. Fail the gate on truncation or evaluator mismatch pending investigation.

- [ ] **Step 4: Rerun all changed test files and commit only factual evidence.**

~~~bash
PYTHONNOUSERSITE=1 PYTHONPATH=. pytest -q tests/models/hy_v4 tests/patch/test_plugin_lifecycle.py tests/patch/test_clean_process_bootstrap.py tests/runtime_patch/test_hcu_model_runner_v2_api.py
git add docs/hy4_v0281_validation.md tests/integration/server/test_evalscope_hy4_humaneval.py
git commit -m "test: record Hy4 MRV2 device and HumanEval evidence"
~~~

If any required hardware gate fails, retain logs and report the MR as unverified; do not weaken topology, runner, Graph, backend, or quantization.
