# HY V4 DCP with DeepEP Low-Latency Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Make HY V4 serve correctly with TP8, DCP2, EP8, DeepEP low-latency, DeepGEMM, and the existing FP8 KV dequantization path, then retain the established 8/8 HumanEval result.

**Architecture:** Add a phase boundary to worker terminal-patch validation so the lazy DeepEP runtime callback is required after warmup instead of immediately after model loading. Merge each HCU DCP rank's local sparse-indexer candidates into a global TopK. Add a HY V4-specific DCP sparse-MLA path that localizes those global cache indices, dequantizes selected FP8 cache rows, forwards real kernel LSE, gathers the TP-sharded sink once, and normalizes the replicated sink contribution before DCP reduction.

**Tech Stack:** Python 3.10, PyTorch, vLLM 0.25.1, vLLM-HCU, FlashMLA sparse kernels, Triton DCP index conversion, pytest, EvalScope.

**Spec:** `docs/superpowers/specs/2026-09-19-hy4-dcp-deepep-low-latency-design.md`

## Global Constraints

- Work on `feat/hy4-lightop-mask-topk-adapt` and stack commits on its existing remote MR.
- Do not add or use the rejected direct Aiter paged-MQA optimization.
- Preserve the existing DCP-size-one HY V4 path and public return shapes.
- Keep feature-off terminal callbacks eligible to remain armed.
- Full runtime validation after warmup must fail closed for enabled callbacks that are armed, skipped, or failed.
- FlashMLA's raw sparse LSE excludes `attn_sink`; fold the normalized sink into
  the returned DCP LSE with `torch.logaddexp`.
- DCP empty local sparse rows return zero output. They return negative-infinity
  LSE without a sink and preserve normalized sink LSE when a sink is present.
- HCU DCP indexer ranks exchange compact score/global-ID candidate pairs and
  run a global TopK before the attention backend localizes indices.
- DCP decode uses the logits-producing HCU indexer route because the fused
  LightOp mask-TopK route does not expose candidate scores. DCP size one keeps
  the LightOp route unchanged.
- Count the virtual attention sink exactly once across DCP ranks by subtracting `log(dcp_world_size)` from each rank's sink logit.
- Hardware acceptance uses `/models/Hy4-preview-Channel-FP8-w8a8`, TP8, DCP2, EP8, `deepep_low_latency`, DeepGEMM, `fp8_ds_mla`, block size 64, and `VLLM_HCU_HYV4_FP8_KV_DEQUANT=1`.
- Accuracy acceptance is HumanEval items 0 through 7 with 8/8 correct and Pass@1 100%.

---

### Task 1: Move DeepEP Terminal Validation to the Runtime Boundary

**Files:**
- Modify: `vllm_hcu/patch/worker/__init__.py`
- Modify: `vllm_hcu/patch/worker/framework_opt/patch_base_communicator_pcp.py`
- Modify: `vllm_hcu/v1/worker.py`
- Test: `tests/patch/test_worker_dispatcher.py`
- Test: `tests/patch/test_plugin_lifecycle.py`
- Test: `tests/runtime_patch/test_patch_base_communicator_pcp.py`

**Interfaces:**
- Consumes: `ExactImportCoordinator.registrations()`, `IMPORT_COORDINATOR.drain_ready_callbacks()`, and the existing `validate_worker_patches(require_applied=True, coordinator=None)` API.
- Produces: `validate_worker_patches(require_applied: bool = True, *, phase: Literal["model_load", "runtime"] = "runtime", coordinator: ExactImportCoordinator | None = None) -> None`; `HcuGPUWorker.load_model()` validates phase `"model_load"`; `HcuGPUWorker.compile_or_warm_up_model()` drains callbacks and validates phase `"runtime"` only after upstream warmup succeeds; the base communicator enables explicit DeepEP when either PCP or DCP is greater than one with EP and DP1.

- [x] **Step 1: Write failing dispatcher phase tests**

Extend `test_terminal_validation_rejects_enabled_armed_but_allows_feature_off` with a second enabled armed registration whose patch ID is `worker.framework_opt.communicator.deep_ep_runtime`. Assert that model-load validation permits only that registration, runtime validation rejects it, and an invalid phase raises `ValueError`:

```python
worker_dispatcher.validate_worker_patches(
    require_applied=True,
    phase="model_load",
    coordinator=coordinator,
)
with pytest.raises(RuntimeError, match="deep_ep_runtime"):
    worker_dispatcher.validate_worker_patches(
        require_applied=True,
        phase="runtime",
        coordinator=coordinator,
    )
with pytest.raises(ValueError, match="phase"):
    worker_dispatcher.validate_worker_patches(
        phase="invalid",  # type: ignore[arg-type]
        coordinator=coordinator,
    )
```

- [x] **Step 2: Write failing worker lifecycle tests**

Update the load-model lifecycle assertion to expect `("validate", True, "model_load")`. Add a warmup test whose fake parent appends `"parent_warmup"`, whose fake coordinator appends `"drain"`, and whose fake validation appends its phase; assert the return value is preserved and the order is:

```python
assert worker.compile_or_warm_up_model() == "warmup-result"
assert events == [
    "parent_warmup",
    "drain",
    ("validate", True, "runtime"),
]
```

Also make the parent warmup raise `RuntimeError("warmup failed")` and assert neither drain nor runtime validation runs.

- [x] **Step 3: Run the new tests and verify they fail**

Run:

```bash
pytest -q \
  tests/patch/test_worker_dispatcher.py::test_terminal_validation_rejects_enabled_armed_but_allows_feature_off \
  tests/patch/test_plugin_lifecycle.py::test_worker_applies_before_parent_init_and_validates_after_load \
  tests/patch/test_plugin_lifecycle.py -k 'warmup and worker'
```

Expected: failures because `phase` is not accepted and warmup does not drain or validate callbacks.

- [x] **Step 4: Implement phase-aware terminal validation**

In the dispatcher, define the only model-load deferral and validate the phase before inspecting registrations:

```python
_MODEL_LOAD_DEFERRED_TERMINAL_IDS = frozenset(
    {"worker.framework_opt.communicator.deep_ep_runtime"}
)

def validate_worker_patches(
    require_applied: bool = True,
    *,
    phase: Literal["model_load", "runtime"] = "runtime",
    coordinator: ExactImportCoordinator | None = None,
) -> None:
    if phase not in {"model_load", "runtime"}:
        raise ValueError(f"unknown worker patch validation phase: {phase!r}")
    deferred = (
        _MODEL_LOAD_DEFERRED_TERMINAL_IDS
        if phase == "model_load"
        else frozenset()
    )
    # Existing failure checks remain unchanged. Exclude `deferred` only from
    # the pending armed terminal list.
```

In `HcuGPUWorker`, pass `phase="model_load"` after model loading. Refactor both warmup branches to capture the upstream return value, then after successful completion call:

```python
from vllm_hcu.patch.import_coordinator import IMPORT_COORDINATOR
from vllm_hcu.patch.worker import validate_worker_patches

IMPORT_COORDINATOR.drain_ready_callbacks()
validate_worker_patches(require_applied=True, phase="runtime")
return result
```

Extend `patch_base_communicator_pcp.hcu_init` to read both
`prefill_context_parallel_size` and `decode_context_parallel_size`, and set
`use_all2all = True` for an EP communicator when either size exceeds one and
the explicit backend is `deepep_high_throughput` or `deepep_low_latency`.
Add a DCP2/PCP1/DP1 positive test and retain the negative test where both CP
sizes equal one.

- [x] **Step 5: Run dispatcher and lifecycle tests**

Run:

```bash
pytest -q tests/patch/test_worker_dispatcher.py tests/patch/test_plugin_lifecycle.py \
  tests/runtime_patch/test_patch_base_communicator_pcp.py
```

Expected: all tests pass.

- [x] **Step 6: Commit the lifecycle change**

```bash
git add vllm_hcu/patch/worker/__init__.py \
  vllm_hcu/patch/worker/framework_opt/patch_base_communicator_pcp.py \
  vllm_hcu/v1/worker.py tests/patch/test_worker_dispatcher.py \
  tests/patch/test_plugin_lifecycle.py \
  tests/runtime_patch/test_patch_base_communicator_pcp.py
git commit -m "fix: validate DeepEP runtime patches after warmup"
```

### Task 2: Return Correct HY V4 DCP Output and LSE

**Files:**
- Modify: `vllm_hcu/models/hy_v4/attention.py`
- Modify: `vllm_hcu/models/hy_v4/hcu_sparse.py`
- Test: `tests/models/hy_v4/test_attention.py`

**Interfaces:**
- Consumes: `get_dcp_group().all_gather(tensor, dim=0)`, `triton_filter_and_convert_dcp_index(..., return_valid_counts=True)`, `gather_dequantize_fp8_ds_mla_cache(...)`, and `flash_mla_sparse_fwd(...) -> (output, auxiliary, lse)`.
- Produces: `HYV4MLAAttentionLayer.process_weights_after_loading(act_dtype: torch.dtype) -> None` forwards to the backend after MLA projection preparation; `HYV4FlashMLASparseImpl.can_return_lse_for_decode = True`; `_dcp_sinks: torch.Tensor | None`; backend `process_weights_after_loading(act_dtype: torch.dtype) -> None`; `_bf16_flash_mla_kernel_with_lse(...) -> tuple[torch.Tensor, torch.Tensor]`; and `forward_mqa(...) -> tuple[torch.Tensor, torch.Tensor | None]`.

- [x] **Step 1: Write failing sink-gather and DCP capability tests**

Make `_bare_impl` initialize `dcp_world_size`, `dcp_rank`, `kv_cache_dtype`, `head_size`, `kv_lora_rank`, `tokens_per_request`, and `_dcp_sinks`. Add tests that assert the class advertises LSE support, the HY V4 MLA layer invokes the selected backend hook after its parent post-load processing, the backend hook invokes its parent then all-gathers a four-head local sink to eight heads once for DCP2, and `_sinks_for_query` chooses the gathered layout for an eight-head query.

Assert the DCP kernel sink is normalized:

```python
expected = gathered_sinks - math.log(2)
torch.testing.assert_close(captured["attn_sink"][:8], expected)
```

- [x] **Step 2: Write failing BF16 kernel LSE tests**

Change fake sparse kernels to return `(output, torch.empty(0), lse)`. Test `_bf16_flash_mla_kernel_with_lse` with a gathered eight-head query and 64-head kernel padding; assert output is sliced to eight heads and returned LSE equals `torch.logaddexp(raw_lse, normalized_sink)`. Make the fake return `None` for LSE and assert `RuntimeError("did not return LSE")`.

- [x] **Step 3: Write failing FP8 DCP forward tests**

Build a bare DCP2 implementation with two token rows. Stub `triton_filter_and_convert_dcp_index` to return one populated row and one empty row, stub `gather_dequantize_fp8_ds_mla_cache` to record the localized indices, and stub the BF16 kernel to return deterministic output/LSE. Assert the empty-row output is zero and its kernel LSE is preserved because the normalized sink remains in the denominator:

```python
output, lse = impl.forward_mqa(q, fp8_cache, metadata, object())
assert dequant_indices.equal(localized_indices)
assert torch.equal(output[1], torch.zeros_like(output[1]))
assert torch.equal(lse[1], kernel_lse[1])
```

Repeat with `impl.sinks = None` and `impl._dcp_sinks = None`; assert the empty
row LSE is negative infinity in the sink-free case.

Also assert DCP-size-one delegates to `super().forward_mqa` so existing prefill/decode behavior remains unchanged.

- [x] **Step 4: Run the new HY V4 tests and verify they fail**

Run:

```bash
pytest -q tests/models/hy_v4/test_attention.py -k 'dcp or lse or sink'
```

Expected: failures because the HY V4 subclass does not yet advertise LSE, gather sinks, localize DCP indices, or return kernel LSE.

- [x] **Step 5: Implement sink gathering and runtime layout selection**

Override the HY V4 MLA layer's weight hook to call `super()` and then
`self.impl.process_weights_after_loading(act_dtype)`. Initialize
`_dcp_sinks = None`, advertise `can_return_lse_for_decode = True`, and override
the backend weight hook:

```python
def process_weights_after_loading(self, act_dtype: torch.dtype) -> None:
    super().process_weights_after_loading(act_dtype)
    self._dcp_sinks = None
    if self.sinks is not None and self.dcp_world_size > 1:
        from vllm.distributed.parallel_state import get_dcp_group
        self._dcp_sinks = get_dcp_group().all_gather(self.sinks, dim=0)
```

Make `_sinks_for_query` select `_dcp_sinks` when its length matches the runtime query-head count. For DCP calls, subtract `math.log(self.dcp_world_size)` before padding without mutating the loaded sink tensor.

- [x] **Step 6: Preserve BF16 kernel LSE and runtime head counts**

Add `_bf16_flash_mla_kernel_with_lse` and base padding on `q.shape[1]`, rather than `self.num_heads`, because vLLM gathers DCP query heads before this call. Capture output and LSE from positions zero and two of `flash_mla_sparse_fwd`, reject a null LSE, slice both to the runtime head count, and use `torch.logaddexp` to add the normalized sink to LSE because FlashMLA only applies it to output. Keep `_bf16_flash_mla_kernel` as a tensor-only wrapper for existing inherited callers:

```python
def _bf16_flash_mla_kernel(self, *args, **kwargs) -> torch.Tensor:
    output, _ = self._bf16_flash_mla_kernel_with_lse(*args, **kwargs)
    return output
```

- [x] **Step 7: Implement HY V4 DCP forward**

For DCP-size-one, delegate to the parent. Otherwise concatenate tuple queries when necessary, take the active rows from `topk_indices_buffer`, and call `triton_filter_and_convert_dcp_index` with metadata block size, interleave size, rank, world size, and `return_valid_counts=True`. For FP8 cache, gather/dequantize the localized selected rows into compact BF16 storage; for BF16 cache, use the local cache directly. Call `_bf16_flash_mla_kernel_with_lse(..., topk_length=topk_length)`, clear the empty-row output, and apply the sink-free reduction identity:

```python
empty_rows = topk_length == 0
output.masked_fill_(empty_rows.view(-1, 1, 1), 0.0)
if self.sinks is None:
    lse.masked_fill_(empty_rows.view(-1, 1), float("-inf"))
return output, lse
```

- [x] **Step 8: Run the full HY V4 attention test file**

Run:

```bash
pytest -q tests/models/hy_v4/test_attention.py
```

Expected: all tests pass.

- [x] **Step 9: Commit the attention change**

```bash
git add vllm_hcu/models/hy_v4/hcu_sparse.py tests/models/hy_v4/test_attention.py
git commit -m "fix: support HY V4 sparse MLA decode context parallelism"
```

### Task 2B: Merge HCU DCP Sparse-Indexer Candidates Globally

**Files:**
- Modify: `vllm_hcu/model_executor/layers/sparse_attn_indexer.py`
- Modify: `vllm_hcu/v1/attention/ops/rocm_aiter_mla_sparse.py`
- Test: `tests/runtime_patch/test_attention_mla_fla_mamba.py`

**Interfaces:**
- Consumes: each rank's local logits, local TopK indices, DCP rank/world size,
  cache interleave, and `get_dcp_group().all_gather`.
- Produces: the global logical TopK token IDs expected by the DCP attention
  localization step.

- [x] **Step 1: Reproduce the incorrect local-only candidate behavior**

Run TP8/DCP2 with BF16 KV cache to remove FP8 gather/dequantization from the
failure. Verify that TP8 without DCP returns a correct deterministic function
while DCP2 produces invalid text. This isolates the failure to DCP index or
attention handling rather than DeepEP, FP8 KV, or the attention sink.

- [x] **Step 2: Add focused global-merge tests**

Cover HCU score packing, interleaved local-to-global ID conversion, all-gather,
and selection of a remote rank's higher-scoring candidate. Cover the HCU
native decode entry point forwarding DCP rank/world/interleave metadata and
invoking the global merge.

- [x] **Step 3: Implement the HCU global merge**

For DCP, pass the parallel metadata into the native HCU indexer. After local
prefill or decode TopK, gather compact score/global-ID pairs and run device-side
TopK over all candidates. Account for packed prefill row offsets when reading
scores. Keep the existing CuTeDSL implementation on CUDA.

Bypass the fused LightOp mask-TopK decode pair for DCP because it returns only
indices and cannot supply the candidate scores required by the global merge.
Keep its DCP-size-one behavior and public custom-op schema unchanged.

- [x] **Step 4: Run indexer regressions**

```bash
pytest -q \
  tests/runtime_patch/test_sparse_indexer_loading.py \
  tests/models/hy_v4/test_indexer_pcp.py \
  tests/runtime_patch/test_attention_mla_fla_mamba.py
```

Expected: 101 tests pass.

- [x] **Step 5: Commit the indexer change**

```bash
git add vllm_hcu/model_executor/layers/sparse_attn_indexer.py \
  vllm_hcu/v1/attention/ops/rocm_aiter_mla_sparse.py \
  tests/runtime_patch/test_attention_mla_fla_mamba.py
git commit -m "fix: merge HY V4 DCP indexer candidates"
```

### Task 3: Run Software Regression Checks

**Files:**
- Verify: `vllm_hcu/patch/worker/__init__.py`
- Verify: `vllm_hcu/v1/worker.py`
- Verify: `vllm_hcu/models/hy_v4/hcu_sparse.py`
- Verify: all affected tests

**Interfaces:**
- Consumes: the lifecycle and attention APIs produced by Tasks 1 and 2.
- Produces: recorded evidence that focused and adjacent regression suites pass before hardware use.

- [x] **Step 1: Run focused tests**

```bash
pytest -q \
  tests/models/hy_v4/test_attention.py \
  tests/runtime_patch/test_mla_target_ownership.py \
  tests/patch/test_import_coordinator.py \
  tests/patch/test_worker_dispatcher.py \
  tests/patch/test_plugin_lifecycle.py \
  tests/runtime_patch/test_worker_framework_opt.py \
  tests/runtime_patch/test_patch_base_communicator_pcp.py
```

Expected: all tests pass.

- [x] **Step 2: Run adjacent HY V4 and runner regressions**

```bash
pytest -q \
  tests/runtime_patch/test_lightop_attention_api.py \
  tests/runtime_patch/test_sparse_indexer_loading.py \
  tests/runtime_patch/test_attention_mla_fla_mamba.py \
  tests/models/hy_v4/test_indexer_pcp.py \
  tests/runtime_patch/test_hcu_model_runner_spec_decode_contract.py
```

Expected: all tests pass.

- [x] **Step 3: Compile changed Python files and check the diff**

```bash
python -m py_compile \
  vllm_hcu/patch/worker/__init__.py \
  vllm_hcu/v1/worker.py \
  vllm_hcu/models/hy_v4/hcu_sparse.py \
  vllm_hcu/model_executor/layers/sparse_attn_indexer.py \
  vllm_hcu/v1/attention/ops/rocm_aiter_mla_sparse.py
git diff --check 4692b9e..HEAD
git status --short
```

Expected: compilation and whitespace checks succeed; only known local validation artifacts remain untracked.

### Task 4: Validate TP8 DCP2 EP8 DeepEP Low-Latency on Hardware

**Files:**
- Create locally, do not commit: `_smoke_dcp2_ep8_ll.sh`
- Create locally, do not commit: `_dcp2_ep8_ll_validation.log`

**Interfaces:**
- Consumes: all production changes from Tasks 1 and 2 and eight idle HCU devices.
- Produces: server startup/decode logs for the exact requested topology and a cleanly stopped owned server process.

- [x] **Step 1: Confirm devices are idle**

Run the installed HCU management utility and `ps` search used by the existing validation scripts. Expected: all eight devices have only baseline memory use and no vLLM worker owns them. If another process owns the devices, stop this task without killing an unowned process.

- [x] **Step 2: Start the exact service configuration**

Launch a managed background process whose PID is recorded by the script:

```bash
VLLM_HCU_USE_CUSTOM_FLASH_ATTN=1 \
VLLM_HCU_HYV4_FP8_KV_DEQUANT=1 \
VLLM_HCU_USE_LIGHTOP_MASK_TOPK=1 \
vllm serve /models/Hy4-preview-Channel-FP8-w8a8 \
  --served-model-name hy4-dcp2-ep8-ll \
  --trust-remote-code --dtype bfloat16 -q compressed-tensors \
  --tensor-parallel-size 8 --decode-context-parallel-size 2 \
  --enable-expert-parallel --all2all-backend deepep_low_latency \
  --moe-backend deep_gemm --kv-cache-dtype fp8_ds_mla \
  --block-size 64 --max-model-len 8192 --max-num-seqs 8 \
  --max-num-batched-tokens 8192 --gpu-memory-utilization 0.90 \
  --enable-prefix-caching --port 20116 \
  > _dcp2_ep8_ll_validation.log 2>&1
```

- [x] **Step 3: Poll readiness and exercise decode**

Poll `http://127.0.0.1:20116/health` while also checking that the owned PID remains alive. Once healthy, send deterministic chat/completion requests that generate multiple tokens. Expected: HTTP success, non-empty completions, no terminal-patch error, no missing-LSE error, and no worker crash.

- [x] **Step 4: Stop only the owned service tree**

Send `TERM` to the recorded process group, wait for exit, and use `KILL` only for members of that same process group that do not exit. Confirm no worker from this validation remains.

### Task 5: Run HumanEval 8 and Update the Existing Remote MR Branch

**Files:**
- Create locally, do not commit: `_humaneval8_dcp2_ep8_ll_20260919/`
- Verify: current branch commits and remote tracking branch

**Interfaces:**
- Consumes: the healthy Task 4 service and EvalScope's OpenAI-compatible evaluator.
- Produces: exactly eight HumanEval predictions/reviews, 8/8 accuracy, Pass@1 100%, and the stacked remote branch update.

- [x] **Step 1: Restart or retain the validated service**

Use the exact Task 4 command and confirm `/health` succeeds immediately before evaluation.

- [x] **Step 2: Run the eight deterministic HumanEval samples**

```bash
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
python -m evalscope.cli.cli eval \
  --model hy4-dcp2-ep8-ll --model-id hy4-dcp2-ep8-ll \
  --api-url http://127.0.0.1:20116/v1 --api-key EMPTY \
  --eval-type openai_api --datasets humaneval --dataset-hub modelscope \
  --limit 8 --eval-batch-size 1 \
  --generation-config '{"max_tokens":2048,"temperature":0,"do_sample":false,"extra_body":{"chat_template_kwargs":{"reasoning_effort":"no_think"}}}' \
  --work-dir /tmp/vpd25/_humaneval8_dcp2_ep8_ll_20260919 \
  --no-timestamp
```

- [x] **Step 3: Validate evaluation completeness and scores**

Inspect EvalScope outputs programmatically. Assert the sample IDs are exactly `HumanEval/0` through `HumanEval/7`, there are eight predictions and eight reviews, accuracy is `1.0`, and Pass@1 is `1.0`. Treat missing or duplicate samples as failure even if the aggregate file reports 100%.

- [x] **Step 4: Run final repository checks**

```bash
git diff --check 4692b9e..HEAD
git status --short
git log --oneline 4692b9e..HEAD
```

Expected: implementation, tests, design, and plan commits are present; no Aiter operator change is present; evaluation artifacts remain untracked.

- [ ] **Step 5: Push the stacked branch**

```bash
git push -u origin feat/hy4-lightop-mask-topk-adapt
```

Expected: the existing remote MR branch advances to the verified local HEAD.
