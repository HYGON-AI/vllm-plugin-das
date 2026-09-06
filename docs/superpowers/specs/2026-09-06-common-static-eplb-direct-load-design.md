# Common Static EPLB Direct-Load Design

## Problem

PR #69 removes the startup expert redistribution for HY V4 by loading
checkpoint experts directly into the physical slots selected by an offline
EPLB map.  The implementation currently attaches the map in HY V4 target and
MTP constructors and performs the remapping in their private weight loaders.
Other models implementing vLLM's `MixtureOfExperts` interface still load the
initial placement and enter the compatibility redistribution path during
`EplbState.add_model`.  On a multi-node deployment that transfer can block
workers long enough to exhaust the scheduler-output shared-memory broadcast
blocks.

The command examples also put `expert_map_record_path` and `expert_map_path`
in `--additional-config`, even though the established public interface puts
all offline EPLB fields in `--eplb-config`.  The parser already extracts those
plugin fields before constructing vLLM's official `EPLBConfig`, so the code and
documentation currently describe different interfaces.

## Goals

1. Make direct checkpoint loading the required static-map path for every model
   that supports vLLM offline EPLB through `MixtureOfExperts` and the standard
   routed-expert weight-loader contract.
2. Keep model-specific code only for checkpoint layouts that bypass the
   standard routed-expert loader, such as HY V4's pre-fused expert tensors.
3. Remove startup expert redistribution as a fallback for configured static
   maps.  Unsupported models must fail before checkpoint loading with an
   actionable error instead of entering the potentially hanging transfer.
4. Preserve ordinary loading, dynamic EPLB, plan-only recording, and profile
   behavior when no static map is configured.
5. Keep `expert_map_record_path` and `expert_map_path` in `--eplb-config` as
   the documented and validated public CLI interface.

## Non-goals

- Changing the EPLB placement policy or recorded JSON schema.
- Rewriting checkpoint files on disk.
- Supporting static maps together with elastic EP.
- Making DeepEP low-latency accept unsupported redundant-expert counts.
- Changing inference-time kernels, KV-cache formats, or quantization policy.

## Architecture

### 1. Model-independent static plan

`StaticEplbPlan` remains the immutable, cached representation of one model's
physical-to-logical map.  Its validation and diagnostics become model-neutral:
no helper or exception identifies HY V4 unless the supplied model key itself
is HY V4.

The plan continues to contain the canonical path, file digest, model key,
expert counts, and one map row per MoE layer.  It accepts the existing v2
multi-model file and legacy single-model file.  Structural validation occurs
before any checkpoint tensor is consumed.

### 2. Common pre-load binding

A worker patch wraps vLLM's common model-initialization boundary.  After the
model object has been constructed and before any model loader invokes
`model.load_weights`, the wrapper performs the following operation when
`expert_map_path` is configured:

1. Require the returned model to satisfy vLLM's `MixtureOfExperts` contract.
2. Load a plan using `model.__class__.__name__` and the model's observed MoE
   layer and expert counts.
3. Require `len(model.moe_layers) == model.num_moe_layers`.
4. Attach the immutable plan to the model and attach exactly one plan row to
   each layer's standard `RoutedExperts` object in `model.moe_layers` order.
5. Verify that every layer exposes the common expert-loading contract or an
   explicitly registered model hook.

The layer order is intentionally the same order used later by
`MixtureOfExperts.set_eplb_state`.  This avoids parsing architecture-specific
checkpoint layer numbers in the common path.  Pipeline-local models use the
map rows represented by their `moe_layers` sequence; model-specific row
selection is permitted only when a model's public sequence is not sufficient.

The initialization wrapper is installed before vLLM model-loader modules are
imported.  Compatibility checks verify the audited v0.25.1 function signature
and any already-imported aliases, so a stale or partially applied patch fails
at worker setup rather than silently loading the wrong layout.

### 3. Logical-expert loading in `RoutedExperts`

The common MoE weight-loading layer owns static remapping:

- Without a bound static row, `RoutedExperts` delegates unchanged to vLLM.
- With a bound row, expert mappings enumerate each checkpoint logical expert
  once instead of encoding the initial redundant-expert placement.
- The bound parameter `weight_loader` expands one checkpoint logical expert
  to every physical slot whose map-row value equals that logical ID.  The
  existing vLLM global-to-local expert map decides which slots are local to the
  current EP rank, so checkpoint tensors are never transferred between ranks.
- Standard split checkpoints and standard all-expert fused tensors use this
  same path.  Weights, weight scales, input scales, and metadata handled by
  the existing loader receive identical logical-to-physical expansion.

The legacy `fused_moe_make_expert_params_mapping(model, ...)` entry point is
also routed through the common behavior.  This covers models with hand-written
outer `load_weights` loops as long as they delegate the final tensor copy to a
standard `RoutedExperts.weight_loader`.

### 4. Model-specific hooks

A model-specific hook is required only when checkpoint code copies or slices
expert tensors without delegating logical expert IDs to the standard routed
expert loader.  The hook receives the already validated common plan and uses
the common row/fan-out helper; it must not parse JSON or implement its own map
validation.

HY V4 target and MTP retain thin adapters for their pre-fused checkpoint
tensors.  Their constructor-specific plan loading and duplicated split-map
builders are removed.  Other HCU MoE models are audited against the standard
contract.  Any model that needs an adapter gets a focused hook and regression
test; otherwise it inherits direct loading without model-file changes.

### 5. EPLB runtime registration

`EplbState.add_model` remains responsible only for runtime metadata.  When a
static path is configured it requires the common pre-load plan, verifies the
path, model key, counts, shape, and cross-rank fingerprint, then calls
`_commit_eplb_maps` without calling `rearrange_expert_weights_inplace`.

The previous no-plan compatibility redistribution is removed for static-map
mode.  Reaching runtime registration without a plan is a compatibility error,
because the model weights can no longer be proven to match the configured map.
Static `step` remains disabled for the lifetime of the service.  Dynamic EPLB
and record mode continue to use their existing paths.

## Public CLI contract

The public commands place the plugin-owned path in the same JSON object as the
official EPLB options:

```bash
--eplb-config '{"window_size":2,"step_interval":100,"num_redundant_experts":0,"use_async":false,"policy":"default","expert_map_record_path":"/tmp/eplb-map.json"}'
```

or:

```bash
--eplb-config '{"window_size":2,"step_interval":100,"num_redundant_experts":0,"use_async":false,"policy":"default","expert_map_path":"/tmp/eplb-map.json"}'
```

The EngineArgs adapter extracts the two HCU-only keys before calling the
official `EPLBConfig` parser and stores them in the HCU sidecar internally.
That storage detail is not a public migration requirement.  The two paths
remain mutually exclusive.  Existing programmatic sidecar input remains
backward compatible, but PR examples and validation use only
`--eplb-config`.

## Error handling

- A missing or malformed map, wrong model key, invalid expert ID, count/shape
  mismatch, or missing logical expert fails before checkpoint loading.
- A model that does not implement `MixtureOfExperts` while a static path is
  configured fails before checkpoint loading.
- A MoE layer that exposes neither standard `RoutedExperts` loading nor an
  audited model hook fails with the model and layer type in the message.
- `enable_ep_weight_filter` remains rejected because upstream filtering does
  not express a different logical-expert set for every layer.
- A missing plan at `EplbState.add_model` fails closed; it never triggers
  startup redistribution.
- Cross-rank fingerprint disagreement fails before routing metadata commit.
- When no static path is configured, the wrapper and common loader are strict
  no-ops and preserve upstream errors and return values.

## Testing

All behavior changes follow red-green TDD.

1. Plan tests verify model-neutral diagnostics and immutable per-layer rows.
2. Initialization-patch tests verify plan binding before weight loading for a
   generic fake `MixtureOfExperts`, no-map pass-through, incompatible models,
   layer-count mismatches, and idempotent patching.
3. Common `RoutedExperts` tests verify two layers with different maps,
   duplicate logical experts, per-rank slot filtering, split checkpoints,
   fused checkpoints, scales, and unchanged no-map behavior.
4. A representative non-HY-V4 HCU model test verifies it acquires direct load
   through the common path without a private static-map implementation.
5. HY V4 target and MTP tests verify their exceptional fused layouts use the
   shared plan and produce tensors bitwise equal to default load followed by
   rearrangement.
6. EPLB runtime tests require zero rearrangement for every static-map model and
   require a hard failure when the pre-load plan is absent.
7. Real CLI parser tests cover both record and direct-load JSON using only
   `--eplb-config`.

## Runtime validation

Run a single-node DP8/EP8 record-to-load smoke test with
`/models/Hy4-preview-Channel-FP8-w8a8-v2`, `deepep_low_latency`, DeepGEMM,
`fp8_e4m3` KV cache, and the original `--eplb-config` syntax.  Record mode must
write a valid map after a short interval without transferring weights.  Static
mode must load the same map on all ranks, answer health and deterministic chat
requests, log zero expert rearrangements, and show no shared-memory 60-second
warning, RPC timeout, traceback, or worker failure.

Focused unit suites, worker-dispatch subprocess tests, `compileall`, and
`git diff --check` must pass before push.  The PR description will contain the
exact commands used and explicitly retain the dual-node acceptance caveat if a
second node is unavailable.

## Success criteria

- Static-map startup has no post-load expert redistribution path.
- Standard MoE models direct-load mapped experts through the common
  `RoutedExperts` layer without model-specific JSON handling.
- HY V4 target and MTP use only thin hooks for their exceptional fused layout.
- Recorded and loaded map commands work with paths inside `--eplb-config`.
- Static-map routing metadata matches the physical tensors before the service
  becomes ready, and later dynamic rearrangement remains disabled.
- Tests and the single-node full-model smoke validation pass before PR #69 is
  updated.
