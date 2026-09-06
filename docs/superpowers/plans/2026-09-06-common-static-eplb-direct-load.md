# Common Static EPLB Direct-Load Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Direct-load offline EPLB checkpoint experts through the common MoE weight-loading layer for every compatible model, retain thin hooks only for exceptional checkpoint layouts, and validate the original `--eplb-config` CLI syntax.

**Architecture:** A common pre-load binder creates one immutable `StaticEplbPlan` after model construction and attaches its rows to standard `RoutedExperts` layers. Common expert mappings enumerate logical experts, while the routed-expert weight loader fans each checkpoint tensor out to all mapped local physical slots; exceptional HY V4 fused tensors use the same fan-out helper through a thin adapter. Runtime registration requires the pre-load plan and commits only routing metadata, with no compatibility redistribution fallback.

**Tech Stack:** Python 3.10, PyTorch, vLLM 0.25.1 model-loader and EPLB APIs, pytest, HCU/DeepEP runtime.

**Spec:** `docs/superpowers/specs/2026-09-06-common-static-eplb-direct-load-design.md`

## Global Constraints

- Static-map mode must perform zero post-load expert redistribution.
- `expert_map_record_path` and `expert_map_path` remain mutually exclusive and are documented inside `--eplb-config`.
- No-map loading, dynamic EPLB, record mode, and profile mode preserve upstream behavior.
- Static offline EPLB remains incompatible with elastic EP and upstream EP weight filtering.
- Map parsing and validation are model-neutral; model-specific hooks may not parse the JSON themselves.
- Unsupported custom MoE loaders fail before checkpoint loading instead of falling back to startup redistribution.
- Production changes are made test-first and each task ends with a focused passing test set.

---

### Task 1: Bind a static plan at the common pre-load boundary

**Files:**

- Modify: `vllm_hcu/model_executor/layers/fused_moe/static_eplb.py`
- Create: `vllm_hcu/patch/worker/framework_opt/patch_model_loader_static_eplb.py`
- Modify: `vllm_hcu/patch/worker/framework_opt/__init__.py`
- Modify: `vllm_hcu/patch/worker/__init__.py`
- Create: `tests/runtime_patch/test_model_loader_static_eplb.py`
- Modify: `tests/patch/test_worker_dispatcher.py`

**Interfaces:**

- Consumes: `maybe_load_static_eplb_plan(vllm_config, *, model_key, num_moe_layers, num_logical_experts, num_physical_experts, num_redundant_experts) -> StaticEplbPlan | None`.
- Produces: `bind_static_eplb_plan(vllm_config: object, model: object) -> StaticEplbPlan | None` and worker patch `patch_model_loader_static_eplb.apply_to_module(module: ModuleType) -> bool`.
- Publishes: `model._vllm_hcu_static_eplb_plan` and `routed_experts._vllm_hcu_static_eplb_row` before `model.load_weights` can run.

- [ ] **Step 1: Write failing binder tests**

Create fake `MixtureOfExperts`-compatible models with two `moe_layers`, each
holding `routed_experts`, and a two-row map.  Assert:

```python
plan = bind_static_eplb_plan(config, model)
assert model._vllm_hcu_static_eplb_plan is plan
assert model.moe_layers[0].routed_experts._vllm_hcu_static_eplb_row == (2, 1, 0, 2)
assert model.moe_layers[1].routed_experts._vllm_hcu_static_eplb_row == (1, 0, 2, 1)
```

Add cases for no configured path returning `None`, a non-MoE model, layer
count mismatch, a layer without `routed_experts`, and generic error messages
that contain no hard-coded `HY V4` prefix.

- [ ] **Step 2: Run the binder tests and observe RED**

Run:

```bash
pytest -q tests/model_executor/layers/fused_moe/test_static_eplb.py tests/runtime_patch/test_model_loader_static_eplb.py
```

Expected: collection fails because `bind_static_eplb_plan` and the model-loader
patch do not exist.

- [ ] **Step 3: Implement the model-neutral binder**

Add this public contract to `static_eplb.py`:

```python
def bind_static_eplb_plan(
    vllm_config: object,
    model: object,
) -> StaticEplbPlan | None:
    """Validate and bind a static EPLB plan before checkpoint loading."""
```

When a load path exists, validate the `MixtureOfExperts` attributes, load the
plan using `model.__class__.__name__`, normalize each `MoERunner` to its
`routed_experts` object, attach rows in sequence order, and fail before any
attribute is published if the whole model cannot be bound.  Remove HY V4 from
generic validation messages.

- [ ] **Step 4: Implement and register the initialization patch**

Wrap `vllm.model_executor.model_loader.utils.initialize_model` with the exact
audited signature.  Call the original initializer, call
`bind_static_eplb_plan(vllm_config, model)`, and return the same model.  Patch
already-imported aliases only after verifying identity with the original;
otherwise raise `PatchCompatibilityError`.  Register the callback in the
worker dispatcher before model-specific modules are imported.

- [ ] **Step 5: Verify GREEN and commit**

Run:

```bash
pytest -q tests/model_executor/layers/fused_moe/test_static_eplb.py tests/runtime_patch/test_model_loader_static_eplb.py tests/patch/test_worker_dispatcher.py
```

Expected: all tests pass.  Commit:

```bash
git add vllm_hcu/model_executor/layers/fused_moe/static_eplb.py vllm_hcu/patch/worker/framework_opt/patch_model_loader_static_eplb.py vllm_hcu/patch/worker/framework_opt/__init__.py vllm_hcu/patch/worker/__init__.py tests/runtime_patch/test_model_loader_static_eplb.py tests/patch/test_worker_dispatcher.py
git commit -m "feat(eplb): bind static plans before common model loading"
```

### Task 2: Fan logical checkpoint experts out in common `RoutedExperts`

**Files:**

- Modify: `vllm_hcu/model_executor/layers/fused_moe/static_eplb.py`
- Modify: `vllm_hcu/patch/worker/op_opt/moe/patch_layer.py`
- Modify: `tests/runtime_patch/test_moe_deepep.py`
- Modify: `tests/model_executor/layers/fused_moe/test_static_eplb.py`

**Interfaces:**

- Consumes: `routed_experts._vllm_hcu_static_eplb_row: tuple[int, ...]` from Task 1.
- Produces: `load_static_logical_expert(routed_experts, original_weight_loader, *, param, loaded_weight, weight_name, shard_id, logical_expert_id, return_success) -> bool | None`.
- Preserves: original `RoutedExperts.get_expert_mapping`, `RoutedExperts.make_expert_params_mapping`, and `RoutedExperts.weight_loader` behavior when no row is bound.

- [ ] **Step 1: Write failing common-loader tests**

Extend the fake routed-expert class used by `test_moe_deepep.py` with an
original loader that records physical IDs.  Bind rows `(2, 1, 0, 2)` and
`(1, 0, 2, 1)` to two instances.  For logical expert `2`, assert the first
instance loads physical IDs `[0, 3]` and the second loads `[2]`.  Add:

```python
assert patched.weight_loader(
    param, tensor, "w13_weight", "w1", 2, return_success=True
) is True
```

Cover a rank where none of the mapped physical slots are local, duplicate
logical experts, `return_success=False`, split mapping generation, standard
fused tensors, scale names, and no-row byte-for-byte delegation.

- [ ] **Step 2: Run the common-loader tests and observe RED**

Run:

```bash
pytest -q tests/runtime_patch/test_moe_deepep.py -k 'static_eplb or layer_patch'
```

Expected: the patched loader calls only the upstream initial-layout physical
ID and fails the mapped-slot assertions.

- [ ] **Step 3: Implement the shared fan-out helper**

Add a helper that iterates `enumerate(row)`, selects entries equal to
`logical_expert_id`, and calls the original loader once per selected physical
slot with `return_success=True`.  Return `True` if at least one local slot was
loaded, `False` if none was local, and `None` when the caller did not request a
success result.  Validate logical IDs against the bound plan before copying.

- [ ] **Step 4: Patch all three common mapping/loading entry points**

In `patch_layer.py`, audit and wrap:

```python
RoutedExperts.weight_loader
RoutedExperts.get_expert_mapping
RoutedExperts.make_expert_params_mapping
```

With a static row, mapping methods call the existing mapping builder with
`num_redundant_experts=0`, so tuple `expert_id` values represent checkpoint
logical IDs exactly once.  Preserve the upstream fused-prefix tuples from
`include_fused=True`; their shard indices remain shard selectors, while the
logical IDs generated by unbinding the fused tensor are expanded by the
patched weight loader.

- [ ] **Step 5: Run common MoE regression tests and commit**

Run:

```bash
pytest -q tests/runtime_patch/test_moe_deepep.py tests/model_executor/layers/fused_moe/test_static_eplb.py
```

Expected: all tests pass.  Commit:

```bash
git add vllm_hcu/model_executor/layers/fused_moe/static_eplb.py vllm_hcu/patch/worker/op_opt/moe/patch_layer.py tests/runtime_patch/test_moe_deepep.py tests/model_executor/layers/fused_moe/test_static_eplb.py
git commit -m "feat(eplb): direct-load experts in common MoE layer"
```

### Task 3: Reduce HY V4 to exceptional fused-checkpoint hooks

**Files:**

- Modify: `vllm_hcu/models/hy_v4/model.py`
- Modify: `vllm_hcu/models/hy_v4/mtp.py`
- Modify: `vllm_hcu/model_executor/layers/fused_moe/static_eplb.py`
- Modify: `tests/models/hy_v4/test_eplb.py`
- Modify: `tests/models/hy_v4/test_weight_loading.py`
- Modify: `tests/models/hy_v4/test_mtp.py`
- Create: `tests/models/test_common_static_eplb_models.py`

**Interfaces:**

- Consumes: the bound plan/rows and `load_static_logical_expert` from Tasks 1-2.
- Produces: HY V4 target/MTP fused adapters that accept a logical checkpoint expert ID and delegate slot fan-out to the common helper.
- Removes: constructor calls to `maybe_load_static_eplb_plan`, private per-weight layer-name parsing, and duplicated split mapping generation from HY V4 files.

- [ ] **Step 1: Change HY V4 tests to require common binding**

Replace constructor assertions with a fake common-binder invocation and assert
that target and MTP wrappers publish the same plan object as their inner model.
For fused tensors containing distinguishable logical rows, assert each target
physical slot equals `loaded_weight[plan.layer_map(layer)[physical_id]]` for
weights, weight scales, and input scales.

- [ ] **Step 2: Add a representative non-HY-V4 test and observe RED**

Build a lightweight fake around `DeepseekV2MixtureOfExperts` or the audited
DeepSeek loader mapping shape.  Bind a non-default row and assert its standard
parameter loader receives only mapped local physical slots without importing
or calling an HY V4 helper.  Run:

```bash
pytest -q tests/models/test_common_static_eplb_models.py tests/models/hy_v4/test_eplb.py tests/models/hy_v4/test_weight_loading.py tests/models/hy_v4/test_mtp.py
```

Expected: HY V4 still depends on constructor-specific direct-loading branches,
so the new common-binding assertions fail.

- [ ] **Step 3: Refactor HY V4 split loading to the common path**

Remove `static_eplb_layer_map_for_weight` and
`build_expert_params_mapping_for_row` calls from target and MTP split loading.
Let `fused_moe_make_expert_params_mapping` emit logical IDs and let the bound
common parameter loader perform row-specific fan-out.

- [ ] **Step 4: Keep only the pre-fused tensor adapter**

For a pre-fused `[logical_expert, ...]` tensor, iterate logical IDs and invoke
the common fan-out helper through the destination routed-expert loader.  Do not
re-parse the checkpoint layer number or map JSON.  Preserve the original path
when no row is bound.

- [ ] **Step 5: Audit other HCU MoE model loaders**

Inspect DeepSeek V2/V4, HY V3, GLM4 MoE, and their MTP loaders.  For every
loader ending in standard `param.weight_loader(... expert_id=...)`, add the
representative common-path assertion only.  If a loader directly slices a
fused all-expert tensor, add a thin adapter test and hook using the same common
fan-out helper.  A model with neither route receives an explicit pre-load
compatibility error test.

- [ ] **Step 6: Verify tensor equivalence and commit**

Run:

```bash
pytest -q tests/models/test_common_static_eplb_models.py tests/models/hy_v4/test_eplb.py tests/models/hy_v4/test_weight_loading.py tests/models/hy_v4/test_mtp.py
```

Expected: all tests pass, including bitwise equality between direct load and
default load followed by rearrangement.  Commit:

```bash
git add vllm_hcu/models/hy_v4/model.py vllm_hcu/models/hy_v4/mtp.py vllm_hcu/model_executor/layers/fused_moe/static_eplb.py tests/models/test_common_static_eplb_models.py tests/models/hy_v4/test_eplb.py tests/models/hy_v4/test_weight_loading.py tests/models/hy_v4/test_mtp.py
git commit -m "refactor(hy-v4): reuse common static EPLB loading"
```

### Task 4: Remove static-map redistribution fallback and lock the public CLI

**Files:**

- Modify: `vllm_hcu/patch/worker/framework_opt/patch_offline_eplb.py`
- Modify: `tests/runtime_patch/test_offline_eplb.py`
- Modify: `tests/runtime_patch/test_platform_hcu_config.py`

**Interfaces:**

- Consumes: `model._vllm_hcu_static_eplb_plan` bound before checkpoint loading.
- Produces: static runtime registration that either commits that plan with zero transfer or raises `PatchCompatibilityError`.
- Preserves: record-mode policy calculation/file output and no-path dynamic EPLB behavior.

- [ ] **Step 1: Write failing no-fallback runtime tests**

Change the existing non-HY-V4 compatibility test so a configured static path
without an attached plan must raise:

```python
with pytest.raises(PatchCompatibilityError, match="before checkpoint loading"):
    state.add_model(model, model_config)
assert rearrangements == []
```

Retain tests proving a valid generic plan commits metadata, performs the
cross-rank fingerprint check/barrier, and never calls rearrangement.  Retain
dynamic and record-mode pass-through tests.

- [ ] **Step 2: Run runtime tests and observe RED**

Run:

```bash
pytest -q tests/runtime_patch/test_offline_eplb.py
```

Expected: the old compatibility branch calls
`rearrange_expert_weights_inplace` instead of raising.

- [ ] **Step 3: Remove the compatibility redistribution branch**

In `hcu_add_model`, require a `StaticEplbPlan` whenever `load_path` is set.
Use model-neutral path/key/count diagnostics, verify fingerprints, call only
`original_commit`, synchronize, and leave later static steps disabled.  Delete
the configured-static call to `load_offline_expert_map` followed by
`original_rearrange_weights`.

- [ ] **Step 4: Strengthen original CLI syntax tests**

Use fresh vLLM subprocess parsing for both keys:

```python
namespace = parser.parse_args([
    "--eplb-config",
    json.dumps({"window_size": 2, path_key: path_value}),
])
args = EngineArgs.from_cli_args(namespace)
config = args.create_engine_config()
assert args.eplb_config.window_size == 2
assert getattr(get_hcu_config(config), path_key) == path_value
```

Assert the HCU-only field is absent from the official `EPLBConfig` object and
the two paths remain mutually exclusive.

- [ ] **Step 5: Verify runtime/config tests and commit**

Run:

```bash
pytest -q tests/runtime_patch/test_offline_eplb.py tests/runtime_patch/test_platform_hcu_config.py
```

Expected: all tests pass.  Commit:

```bash
git add vllm_hcu/patch/worker/framework_opt/patch_offline_eplb.py tests/runtime_patch/test_offline_eplb.py tests/runtime_patch/test_platform_hcu_config.py
git commit -m "fix(eplb): require direct loading for static maps"
```

### Task 5: Review, full verification, runtime smoke test, and PR update

**Files:**

- Modify if needed: `docs/superpowers/specs/2026-09-06-common-static-eplb-direct-load-design.md`
- Modify if needed: `docs/superpowers/plans/2026-09-06-common-static-eplb-direct-load.md`
- Remote: GitHub PR #69 title, description, and review-thread replies.

**Interfaces:**

- Consumes: all production and test changes from Tasks 1-4.
- Produces: reviewed commits on `perf/deepep-ll-scheduling-bubble`, verified remote head, original-syntax record/load commands, and review evidence on PR #69.

- [ ] **Step 1: Run the focused regression matrix**

Run:

```bash
pytest -q \
  tests/model_executor/layers/fused_moe/test_static_eplb.py \
  tests/models/test_common_static_eplb_models.py \
  tests/models/hy_v4/test_eplb.py \
  tests/models/hy_v4/test_weight_loading.py \
  tests/models/hy_v4/test_mtp.py \
  tests/runtime_patch/test_model_loader_static_eplb.py \
  tests/runtime_patch/test_moe_deepep.py \
  tests/runtime_patch/test_offline_eplb.py \
  tests/runtime_patch/test_platform_hcu_config.py \
  tests/patch/test_worker_dispatcher.py
python -m compileall -q vllm_hcu
git diff --check feat/hy-v4-mtp-blockwise-v0251..HEAD
```

Expected: every test passes, compileall is silent, and diff-check has no
output.

- [ ] **Step 2: Review the complete diff**

Review against the spec with explicit checks for: no-map exact delegation,
all static paths bound before loading, logical/physical ID separation,
duplicate experts, scales, MTP model keys, PP layer order, no redistribution
fallback, record-mode preservation, and public CLI compatibility.  Fix every
Critical or Important finding and rerun Step 1.

- [ ] **Step 3: Record a map with original CLI syntax**

Run the existing DP8/EP8 HY V4 command, placing the path inside the JSON:

```bash
PYTHONPATH=/models/.worktrees/vllm-plugin-das-pr62-kv-fp8 \
vllm serve /models/Hy4-preview-Channel-FP8-w8a8-v2 \
  --served-model-name hy4-eplb-record \
  --data-parallel-size 8 --enable-expert-parallel \
  --all2all-backend deepep_low_latency --enable-eplb \
  --eplb-config '{"window_size":2,"step_interval":100,"num_redundant_experts":0,"log_balancedness":true,"log_balancedness_interval":100,"use_async":false,"policy":"default","expert_map_record_path":"/tmp/hy4-pr69-common.json"}' \
  --enforce-eager --moe-backend deep_gemm \
  --gpu-memory-utilization 0.9 --kv-cache-memory-bytes 67108864 \
  --kv-cache-dtype fp8_e4m3 --max-model-len 256 \
  --max-num-batched-tokens 16 --max-num-seqs 1 \
  --default-chat-template-kwargs '{"reasoning_effort":"no_think"}' \
  --reasoning-parser hy_v4 --enable-auto-tool-choice \
  --tool-call-parser hy_v4 --disable-log-stats --port 10187
```

Require a valid map, successful health/chat requests, and zero transfer,
shared-memory timeout, traceback, or worker failure during recording.

- [ ] **Step 4: Direct-load the recorded map with original CLI syntax**

Run the same command with:

```bash
--served-model-name hy4-eplb-static \
--eplb-config '{"window_size":2,"step_interval":100,"num_redundant_experts":0,"log_balancedness":true,"log_balancedness_interval":100,"use_async":false,"policy":"default","expert_map_path":"/tmp/hy4-pr69-common.json"}'
```

Require all eight ranks to report the same plan digest and zero expert
rearrangement, plus HTTP 200 health/chat and zero shared-memory timeout, RPC
timeout, traceback, or worker failure.

- [ ] **Step 5: Verify, push, and update PR #69**

Invoke verification-before-completion, then push:

```bash
git push origin perf/deepep-ll-scheduling-bubble
git ls-remote origin refs/heads/perf/deepep-ll-scheduling-bubble
```

Update the PR title to describe common MoE direct loading.  Replace all
`--additional-config` examples with the exact original-syntax commands from
Steps 3-4.  Report unit/runtime evidence and the dual-node caveat, and reply to
both review comments with the commit and validation summary.
