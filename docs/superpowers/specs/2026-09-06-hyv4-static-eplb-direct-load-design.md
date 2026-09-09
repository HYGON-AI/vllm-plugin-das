# HY V4 Static EPLB Direct-Load Design

## Problem

The HCU offline EPLB adapter currently applies a static expert map after model
weights have been loaded with vLLM's default physical-to-logical layout.  Its
`EplbState.add_model` wrapper calls `rearrange_expert_weights_inplace` to move
all target and MTP expert weights into the saved layout and only then commits
the map.

For HY V4 this means transferring 77 target MoE layers plus the MTP layer.
On two nodes, one or more workers can remain inside the NIXL/UCX transfer long
enough that the engine's scheduler-output ring buffer is not consumed.  vLLM
then emits `No available shared memory broadcast block found in 60 seconds`,
and a later RPC or engine deadline can fail the request.  Increasing the
ring-buffer warning interval hides the symptom but does not remove the
transfer or control-plane stall.

## Scope

This change makes static offline EPLB direct-load authoritative for the HCU
HY V4 target model and its native MTP model.  It covers both split per-expert
checkpoints and fused all-expert tensors, including FP8 weights and scales.

Dynamic EPLB, offline recording, elastic-EP rejection, and non-HY-V4 models
retain their existing behavior.  This change does not alter expert-selection
policy, generate a new map, change KV-cache precision, or change DeepEP's
high-throughput/low-latency selection.

## Goals

1. Load each HY V4 physical expert slot directly from the logical expert named
   by the static map.
2. Commit the same map into vLLM EPLB routing state without calling any expert
   weight rearrangement function.
3. Validate the map before it can influence checkpoint loading and verify that
   all EP ranks use identical map content before the service becomes ready.
4. Keep a static map immutable for the lifetime of the loaded model.
5. Preserve current dynamic EPLB and record-mode semantics.
6. Fail clearly instead of silently falling back to a mismatched layout.

## Non-goals

- A generic direct-load implementation for every upstream vLLM MoE model.
- Runtime switching between static and dynamic expert layouts.
- Reformatting or rewriting the checkpoint on disk.
- Masking a genuinely hung worker by making RPC timeouts unbounded.

## Architecture

### Neutral static-map plan

Move map selection and structural validation out of the runtime patch wrapper
into a neutral HCU fused-MoE helper.  The helper returns an immutable CPU plan
containing:

- model key (`HYV4ForCausalLM` or `HYV4MTP`),
- normalized source path,
- full-file SHA-256 digest,
- physical-to-logical tensor with shape
  `[num_moe_layers, num_physical_experts]`, and
- logical/physical/redundant expert counts.

The existing v2 multi-model JSON and legacy single-model JSON formats remain
accepted.  Validation continues to reject non-integer IDs, wrong shapes,
negative/out-of-range IDs, missing logical experts, and incorrect redundant
slot counts.  File content is cached per normalized path and file identity so
the target and MTP loaders do not repeatedly parse the JSON.

When no `expert_map_path` is configured, the helper returns no plan and the
current loader behavior is unchanged.

### HY V4 checkpoint loading

HY V4 target and MTP model construction select their own plan before weights
are consumed and attach it to the model.  MoE layers are paired with map rows
in the same order used by `set_eplb_state`; a mismatch is an initialization
error.

For split checkpoints, the generated expert-parameter mapping becomes
layer-specific.  For each physical slot `p` in layer `l`, checkpoint weights
for logical expert `plan[l, p]` are loaded into physical slot `p`.

For fused checkpoints, the loader selects `loaded_weight[plan[l, p]]` instead
of the current `loaded_weight[p % num_logical_experts]`.  Weight tensors,
weight scales, and activation/input scales all use the same slot mapping.
Only local physical slots are materialized by the existing expert-map manager;
no inter-rank weight transfer is needed.

If upstream EP weight filtering is enabled, the filter must include the union
of logical experts referenced by the local physical slots.  If that contract
cannot be established for the audited vLLM version, startup fails with an
explicit incompatibility error rather than loading incomplete experts.

### EPLB runtime registration

The offline EPLB adapter recognizes a model carrying a validated direct-load
plan.  It lets upstream `EplbState.add_model` allocate normal routing/load
state, then commits the plan's map directly with `_commit_eplb_maps`.
It must not call `rearrange_expert_weights_inplace` for this path because the
weights are already in the committed physical layout.

The adapter clears `should_record_tensor`, forces `is_async=False`, and makes
all later `EplbState.step` calls no-ops while a load path is configured.  The
existing post-load rearrangement remains only as compatibility behavior for
non-HY-V4 models that do not advertise a direct-load plan.

### Cross-rank consistency

Before committing routing metadata, all ranks exchange a compact plan
fingerprint consisting of the model key, shape, counts, and SHA-256 digest.
Every fingerprint must match.  A mismatch raises a descriptive startup error
that includes the local rank and mismatched fingerprints.  A device-group
barrier after commit prevents any rank from entering warmup or inference with
partially committed routing state.

This synchronization is startup-only and is not placed in the inference hot
path.

## Error handling

- Missing model key: report available keys and stop before weight loading.
- Shape/count/ID errors: report path, model key, layer, and expected values.
- Model layer-order mismatch: report the model class and observed layer count.
- Rank fingerprint mismatch: stop all ranks before API readiness.
- Direct-loaded model without an attached plan at EPLB registration: reject
  the inconsistent state rather than rearranging it silently.
- Non-HY-V4 model: retain the current validated post-load compatibility path.

## Compatibility and rollout

Direct load is automatically selected when all of the following are true:

1. `expert_map_path` is configured,
2. EPLB and expert parallelism are enabled,
3. the model is HCU HY V4 target or native MTP, and
4. the map validates for that model.

No new user-facing flag is required.  Removing `expert_map_path` restores the
standard initial layout and dynamic/record behavior.  The existing rejection
of static offline EPLB with elastic EP remains in force.

## Testing

### Unit and patch tests

- Parse and validate target/MTP plans, including cache invalidation when the
  file identity changes.
- Reject missing keys, invalid IDs, missing experts, wrong redundancy, and
  mismatched layer counts before checkpoint loading.
- Verify split checkpoint mappings for two layers with different layouts.
- Verify fused target and MTP weights/scales load into the mapped physical
  slots, including duplicated logical experts.
- Verify local EP filtering uses all logical experts needed by local slots or
  raises an explicit compatibility error.
- Verify direct-loaded registration calls commit but never calls rearrange.
- Verify rank fingerprint mismatch and commit-barrier failure stop startup.
- Verify static `step` remains a no-op.
- Verify record mode and dynamic EPLB still call the original paths.
- Verify a non-HY-V4 model retains the compatibility rearrangement path.

Every behavior change is implemented test-first with a failing regression
test observed before production code is written.

### Runtime verification

1. Run the focused offline-EPLB, HY V4 weight-loading, MTP, worker-dispatch,
   multiprocess-timeout, and platform-config suites.
2. Run the broader patch/model regression set and `compileall`/`diff --check`.
3. On one node, start the 8-HCU DP8/EP8 HY V4 Channel-FP8 model with MTP2,
   `fp8_e4m3` KV cache, DeepGEMM, and a static map using a short
   `step_interval`.  Require all requests to return HTTP 200, no dynamic
   rearrangement, no map write, no shared-memory warning, and no traceback.
4. Compare deterministic prompts against the legacy post-load layout using
   the same map.  Require matching first generated tokens; report any later
   token divergence separately because FP8 kernels can be nondeterministic.
5. On two nodes when available, require startup and inference without a
   shared-memory 60-second warning or RPC timeout, and confirm logs show
   direct load plus metadata commit with zero expert-transfer operations.

## Success criteria

- HY V4 static-map startup performs zero calls to expert rearrangement.
- Target and MTP weights occupy the physical slots specified by the map.
- All ranks commit an identical map before API readiness.
- Static-map inference never schedules a later dynamic rearrangement.
- Dynamic and record modes retain their existing behavior.
- Single-node full-model validation passes, and dual-node validation passes
  when the second node is available.
