# HY V4 Static EPLB Direct-Load Implementation Plan

> Execute this plan task by task with tests failing before the corresponding
> production change.  Keep the legacy post-load rearrangement path for models
> that do not opt in to direct loading.

**Goal:** Make HY V4 target and native MTP load checkpoint experts directly
into the physical slots in a configured offline EPLB map, then register that
map without any cross-rank expert rearrangement.

**Architecture:** A neutral `StaticEplbPlan` parser owns map validation and
fingerprinting.  HY V4 loaders attach a validated plan before consuming
weights and use a layer-specific physical-to-logical row for split and fused
expert tensors.  The EPLB runtime patch detects the plan, verifies it across
EP ranks, commits routing metadata, and skips rearrangement; other model
families retain the compatibility path.

**Tech stack:** Python 3.10, PyTorch, vLLM EPLB APIs, pytest.

---

## Task 1: Add the immutable static-map plan and parser

**Files:**

- Create: `vllm_hcu/model_executor/layers/fused_moe/static_eplb.py`
- Create: `tests/model_executor/layers/fused_moe/test_static_eplb.py`
- Modify: `vllm_hcu/patch/worker/framework_opt/patch_offline_eplb.py`
- Modify: `tests/runtime_patch/test_offline_eplb.py`

1. Write tests that create v2 target/MTP maps and assert the returned plan has
   a canonical path, SHA-256 digest, exact counts, an `int64` CPU tensor, and a
   stable fingerprint.  Add failing cases for missing model keys, booleans or
   floating IDs, shape mismatch, out-of-range IDs, missing logical experts,
   and a redundant-count mismatch.  Add a cache-invalidation test that
   rewrites the file and observes a new digest/map.
2. Run:

   ```bash
   pytest -q tests/model_executor/layers/fused_moe/test_static_eplb.py
   ```

   Confirm collection/import fails because the helper does not exist.
3. Implement this public contract:

   ```python
   @dataclass(frozen=True)
   class StaticEplbPlan:
       model_key: str
       source_path: str
       source_sha256: str
       physical_to_logical_map: torch.Tensor
       num_logical_experts: int
       num_physical_experts: int
       num_redundant_experts: int

       def fingerprint(self) -> tuple[str, str, tuple[int, int], int, int, int]: ...
       def layer_map(self, layer_idx: int) -> tuple[int, ...]: ...

   def load_static_eplb_plan(
       path: str | Path,
       *,
       model_key: str,
       expected_shape: tuple[int, int],
       num_logical_experts: int,
       num_redundant_experts: int,
   ) -> StaticEplbPlan: ...
   ```

   Cache decoded bytes by canonical path plus `(st_dev, st_ino, st_size,
   st_mtime_ns)`; compute the digest from the exact bytes.  Clone the validated
   CPU tensor on plan construction and return tuple rows from `layer_map` so a
   caller cannot mutate cached state.  Keep legacy single-model JSON behavior,
   including selecting the final rows for MTP.
4. Make `load_offline_expert_map` delegate to the neutral parser so existing
   imports and non-HY-V4 compatibility behavior remain intact.
5. Run both focused suites and commit:

   ```bash
   pytest -q tests/model_executor/layers/fused_moe/test_static_eplb.py tests/runtime_patch/test_offline_eplb.py
   git add vllm_hcu/model_executor/layers/fused_moe/static_eplb.py tests/model_executor/layers/fused_moe/test_static_eplb.py vllm_hcu/patch/worker/framework_opt/patch_offline_eplb.py tests/runtime_patch/test_offline_eplb.py
   git commit -m "feat(eplb): add validated static expert plans"
   ```

## Task 2: Attach plans before HY V4 checkpoint loading

**Files:**

- Modify: `vllm_hcu/model_executor/layers/fused_moe/static_eplb.py`
- Modify: `vllm_hcu/models/hy_v4/model.py`
- Modify: `vllm_hcu/models/hy_v4/mtp.py`
- Modify: `tests/models/hy_v4/test_eplb.py`

1. Add failing tests for target and MTP constructors using a minimal fake
   config.  Assert that a configured path attaches `_vllm_hcu_static_eplb_plan`
   to both the EPLB wrapper and the actual object that implements
   `load_weights`, while an absent path leaves the attribute unset.
2. Add a helper which reads `_vllm_hcu_expert_map_path` from
   `vllm_config.parallel_config`, verifies EPLB/expert-parallel prerequisites,
   rejects `enable_ep_weight_filter=True` with a precise compatibility error,
   and loads the plan using the model's observed MoE layer/expert counts.
3. At the end of `HYV4Model` and `HYV4MultiTokenPredictor` construction, attach
   the target (`HYV4ForCausalLM`) or draft (`HYV4MTP`) plan.  The wrappers must
   expose the same object so runtime registration can find it without knowing
   the internal model layout.
4. Run and commit:

   ```bash
   pytest -q tests/models/hy_v4/test_eplb.py tests/models/hy_v4/test_runtime_config.py tests/models/hy_v4/test_mtp_config.py
   git add vllm_hcu/model_executor/layers/fused_moe/static_eplb.py vllm_hcu/models/hy_v4/model.py vllm_hcu/models/hy_v4/mtp.py tests/models/hy_v4/test_eplb.py
   git commit -m "feat(hy-v4): attach static EPLB plans before load"
   ```

## Task 3: Direct-load split and fused expert checkpoints

**Files:**

- Modify: `vllm_hcu/model_executor/layers/fused_moe/static_eplb.py`
- Modify: `vllm_hcu/models/hy_v4/model.py`
- Modify: `vllm_hcu/models/hy_v4/mtp.py`
- Modify: `tests/models/hy_v4/test_eplb.py`
- Modify: `tests/models/hy_v4/test_weight_loading.py`
- Modify: `tests/models/hy_v4/test_mtp.py`

1. Extend the existing fused-loader tests with distinct layer maps such as
   `(3, 2, 1, 0, 3, 2)`.  Assert every physical slot receives the checkpoint
   tensor for the mapped logical ID, for target and MTP, including `w1`, `w2`,
   `w3`, weight scales, and input/activation scales.
2. Add split-format tests for two MoE layers with different rows.  A checkpoint
   name containing `layers.<absolute_layer>.experts.<logical>` must call the
   weight loader once for every matching physical slot in that layer, including
   duplicate replicas, and never use another layer's row.
3. Implement helpers to extract the absolute layer number from a weight name
   and map it to the MoE-row order recorded at construction.  Replace the
   default modulo rule in both fused loaders with:

   ```python
   logical_expert_id = (
       physical_to_logical[physical_expert_id]
       if physical_to_logical is not None
       else physical_expert_id % num_experts
   )
   ```

4. Generate split mappings per map row: for checkpoint logical expert `l`,
   emit a tuple for every physical slot `p` where `row[p] == l`.  Preserve the
   upstream mapping when no static plan is attached.
5. Add a tensor-level equivalence test.  Simulate the previous default-layout
   load followed by rearrangement and compare every final physical-slot tensor
   bit-for-bit with direct load for the same map.
6. Run and commit:

   ```bash
   pytest -q tests/models/hy_v4/test_eplb.py tests/models/hy_v4/test_weight_loading.py tests/models/hy_v4/test_mtp.py
   git add vllm_hcu/model_executor/layers/fused_moe/static_eplb.py vllm_hcu/models/hy_v4/model.py vllm_hcu/models/hy_v4/mtp.py tests/models/hy_v4/test_eplb.py tests/models/hy_v4/test_weight_loading.py tests/models/hy_v4/test_mtp.py
   git commit -m "perf(hy-v4): load static EPLB experts directly"
   ```

## Task 4: Commit direct-loaded maps without expert transfer

**Files:**

- Modify: `vllm_hcu/patch/worker/framework_opt/patch_offline_eplb.py`
- Modify: `tests/runtime_patch/test_offline_eplb.py`

1. Change the static-load runtime regression test to attach a valid plan and
   assert `rearrangements == []`, map metadata is committed, dynamic steps are
   disabled, and a synchronization barrier is reached.  Add tests for a plan
   fingerprint mismatch and for the legacy non-HY-V4 path still rearranging.
2. Implement `_find_static_eplb_plan(model)` and startup-only rank
   synchronization.  Exchange a deterministic UTF-8 fingerprint through the
   EP group's CPU process group when available; otherwise use a fixed-size
   `uint8` tensor on the EP device group.  Raise before commit if any rank
   differs, and call a device-group barrier after commit.
3. In `hcu_add_model`, verify the attached plan matches the configured path,
   model key, and EPLB-state shape/counts.  For a valid direct plan, call only:

   ```python
   original_commit(model_state, new_physical_to_logical_map=plan.map_for(...))
   ```

   Do not invoke `rearrange_expert_weights_inplace`.  Preserve the current
   load-and-rearrange compatibility path when no direct plan is attached.
4. Log one unambiguous message containing `direct-loaded`, the model key, map
   digest, and `zero expert rearrangement`, making runtime verification
   searchable.
5. Run and commit:

   ```bash
   pytest -q tests/runtime_patch/test_offline_eplb.py tests/models/hy_v4/test_eplb.py
   git add vllm_hcu/patch/worker/framework_opt/patch_offline_eplb.py tests/runtime_patch/test_offline_eplb.py
   git commit -m "fix(eplb): commit HY V4 static maps without transfer"
   ```

## Task 5: Focused and broad verification

**Files:** No production changes unless a regression is found.

1. Run focused tests:

   ```bash
   pytest -q tests/runtime_patch/test_offline_eplb.py tests/models/hy_v4/test_eplb.py tests/models/hy_v4/test_weight_loading.py tests/models/hy_v4/test_mtp.py tests/models/hy_v4/test_runtime_config.py tests/models/hy_v4/test_mtp_config.py tests/runtime_patch/test_multiproc_executor.py tests/runtime_patch/test_worker_dispatcher.py
   ```

2. Run syntax and repository hygiene checks:

   ```bash
   python -m compileall -q vllm_hcu
   git diff --check b52da21..HEAD
   ```

3. Review the complete diff against the approved design, with special
   attention to no-map behavior, record mode, dynamic EPLB, MTP map selection,
   and exception behavior.  Fix and retest any issue before claiming success.

## Task 6: One-node full-model validation and PR update

**Files:** Runtime logs and temporary request outputs stay outside git.

1. Start `/models/Hy4-preview-Channel-FP8-w8a8-v2` on all eight local HCUs with
   the existing verified launch settings: DP8/EP8, native MTP2, static EPLB
   map, `fp8_e4m3` KV cache, DeepGEMM, and a deliberately short EPLB
   `step_interval`.
2. Wait for API readiness while continuously checking worker exits.  Send
   deterministic chat/completions requests and require HTTP 200 responses.
3. Inspect the complete log and require:

   - target and MTP direct-load messages,
   - zero `rearrange_expert_weights_inplace` or transfer operations,
   - zero dynamic EPLB map commits after startup,
   - zero `No available shared memory broadcast block` messages,
   - zero traceback or engine/RPC timeout.

4. Compare the generated first tokens with the prior legacy-layout result and
   report exact matches.  If a second node is unavailable, state explicitly
   that the root two-node timeout is structurally removed and unit/single-node
   verified, but dual-node validation remains pending.
5. Push all commits to `perf/deepep-ll-scheduling-bubble`, verify the remote
   head, and update PR #69 with the design, test evidence, actual launch
   command, and remaining dual-node caveat.
