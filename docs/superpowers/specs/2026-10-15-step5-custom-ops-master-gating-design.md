# Step5 and HY4 Custom-Op Master Gating Design

## Goal

Make `VLLM_HCU_USE_CUSTOM_OPS` the highest-priority switch for optional HCU
accelerations introduced by Step5 commit `75aa13701`, and for the HY4
PCP-sharded `linear_gate` path being added on this branch.

The change must preserve model registration, checkpoint loading, MTP behavior,
distributed correctness, and model semantics. Turning the master off must not
silently leave a child optimization enabled. If a requested topology has no
safe and usable fallback, configuration must fail before model loading with a
message that names the master and child requirements.

## User-visible contract

`VLLM_HCU_USE_CUSTOM_OPS` continues to default to enabled. Its value is fixed
for a server process at startup; changing it after model construction is not
supported because several choices affect weight layout, graph registration, or
buffer construction.

For every managed optional feature:

```text
effective_enabled = custom_ops_enabled() and child_enabled
```

Consequently, an explicit child value of `1` cannot override
`VLLM_HCU_USE_CUSTOM_OPS=0`. With the master enabled, existing child defaults
and explicit settings remain unchanged unless this document defines a new
topology-aware default.

### HY4 PCP linear-gate sharding

`VLLM_HCU_ENABLE_LINEAR_GATE_PCP_SHARD` becomes a managed child switch.

- With the master enabled and the exact supported HY4 topology TP1/PCP8/PP1,
  an absent child setting resolves to enabled.
- With the master enabled, an explicit child value remains authoritative.
- With the master disabled, effective sharding is disabled even if the child
  is explicitly `1`.
- TP1/PCP8 startup fails early when effective sharding is disabled. The
  unsharded layout is not an acceptable fallback for this topology because it
  replicates the gate weights and exhausts the KV-cache memory budget observed
  during validation.
- Outside TP1/PCP8, an absent child setting remains disabled. An explicit `1`
  may enable the existing path only where its current shape checks accept it.

This keeps the safe PCP8 default while preserving the master as a real global
opt-out instead of allowing topology defaults to bypass it.

## Scope classification

The files added or materially changed by `75aa13701` fall into three groups.
The classification is based on behavior, not on whether the implementation
happens to use Triton, a custom Torch op, or an HCU-owned module.

### 1. Always-on correctness and compatibility

The following remain active when the master is off:

- Step5 config classes, model registration, canonical module aliases, and
  version-compatibility adapters;
- checkpoint parsing, weight-name mapping, base model construction, and MTP
  model loading/sampling;
- runner metadata, block-table mirroring, pending-region ownership, collective
  ordering, and failure propagation;
- EPLB state, mixed-format transfer correctness, and static/live map handling;
- compile-prefix and graph-registration plumbing required by the selected
  baseline implementation;
- Step5 DSA semantics and the baseline DSA implementation.

DSA is deliberately in this group. The existing dense escape hatch is only
equivalent while `topk * region_block_size` covers the complete history; beyond
4096 tokens it represents a different, denser model. The master therefore
must not translate to `VLLM_STEP5_DISABLE_DSA=1`. Optional provider choices
inside DSA are managed only when a semantics-preserving fallback exists.

These paths are model support or correctness machinery, not optional operator
accelerations. Disabling them would either make the checkpoint unloadable or
change serving behavior.

### 2. Managed optional accelerations

The audit and implementation cover the following feature families. Each must
resolve through the shared master policy before importing, registering,
constructing, or executing its optimized provider.

| Area | Child setting or selector | Master-off behavior |
| --- | --- | --- |
| HY4 PCP gate layout | `VLLM_HCU_ENABLE_LINEAR_GATE_PCP_SHARD` | Reject TP1/PCP8 early; otherwise retain `ColumnParallelLinear` |
| Step5 QKV/gate projection fusion | `VLLM_HCU_STEP5_FUSE_QKVG` | Construct the original independent projections |
| Step5 BF16/fused router | `VLLM_HCU_STEP5_BF16_ROUTER`, `VLLM_HCU_STEP5_FUSED_ROUTER_SHARED` | Use the original router and shared-expert composition |
| Step5 load-time FP8 conversions | `VLLM_HCU_STEP5_FP8_DENSE`, `VLLM_HCU_STEP5_FP8_BF16_MOE` | Keep checkpoint/native weight dtypes and methods |
| Step5 SWA fused QK and gate | `VLLM_HCU_STEP5_SWA_FUSED_QK`, `VLLM_HCU_STEP5_FUSED_GATE` | Use the unfused graph operations |
| Step5 skinny LM head | `STEP5_SKINNY_LM_HEAD` | Use the original logits processor head path |
| Step5/HCU token FP8 helper | `VLLM_HCU_TRITON_TOKEN_FP8_QUANT` | Use the existing non-fused quantization path |
| EP fused expert/combine paths | `VLLM_HCU_EP_FUSED_MOE`, `VLLM_HCU_EP_FUSED_COMBINE` | Use the configured regular MoE execution and collective |
| DP IPC all-gather | `VLLM_HCU_DP_CUSTOM_AG` | Use the existing process-group all-gather |
| Host sequence-length shortcut | `VLLM_HCU_HOST_SEQ_LENS` | Use the device/native sequence-length path |
| HCU SWA decode replacement | `VLLM_HCU_SWA_DECODE` | Use the registered baseline attention path |

The implementation audit must also inspect all code introduced by
`75aa13701` for equivalent implicit fast paths that lack a named child flag.
Such a path is added to this managed set only when its baseline callable and
layout boundary are identifiable. Otherwise it is classified explicitly as a
required primitive or startup is rejected; it must not remain silently outside
the policy.

### 3. Existing global-policy exceptions

This work does not reinterpret the repository-wide exceptions already defined
by the HCU custom-op master contract:

- an explicit MoE expert backend remains authoritative; the master controls
  optional helpers around it, not `--moe-backend` itself;
- required HCU platform primitives stay active until a safe fallback exists;
- checkpoint/config/model compatibility adapters are not operator choices.

Any new exception found in the Step5 audit must be documented in the Step5
integration guide with its reason and fallback analysis. An undocumented
exception is a defect.

## Architecture

### Central effective-policy resolvers

`vllm_hcu.platforms.envs.custom_ops_enabled()` remains the single dynamic
source for the master value, and
`optional_custom_op_enabled(feature_enabled)` remains the shared precedence
helper. Step5 and HY4 add small dependency-light resolvers where topology or
child parsing is needed; they delegate to the shared helper rather than reading
the master independently.

Raw child getters remain raw so logs can distinguish “requested” from
“effective.” Runtime and configuration code consume the same effective
resolver. Boolean parsing uses the repository's accepted true/false vocabulary
consistently; individual call sites do not invent new parsers.

Module-level provider constants may remain import-time decisions because the
master is a startup contract, but their initializer must call an effective
resolver. Tests use cold subprocess imports so a previously materialized
module cannot hide incorrect precedence.

### Selection before layout mutation

The effective provider is selected before any of the following:

- weight allocation or load-time quantization;
- packed/fused parameter construction;
- custom-op registration or compilation splitting registration;
- IPC buffer allocation;
- CUDA/HCU graph capture.

Master-off behavior uses an existing baseline callable and portable layout. It
does not run an optimized provider, catch a failure, and retry after state has
already been mutated.

### Configuration and runtime agreement

Configuration validation and runtime construction must use identical effective
values. Features that alter layouts or graph structure remain part of the
compile/config fingerprint. The existing Step5 compile-prefix environment
capture is retained, and tests verify that master state changes the effective
selection even if a child flag is `1`.

HY4 PCP validation reports separate errors for an explicitly disabled child
and for a disabled master, while both errors state that the effective shard is
required for TP1/PCP8.

### Registration behavior

Correctness adapters continue to register unconditionally. Optional patch
callbacks may register a lightweight wrapper while the master is off only if
the wrapper delegates exclusively to the baseline and does not import or
initialize the optimized provider. Where registration itself changes global
behavior, it is skipped when the effective policy is off.

## Considered approaches

### Chosen: effective resolver at each provider boundary

This preserves raw child configuration, makes precedence testable, and keeps
layout-changing decisions next to the code that owns the layout. It also
matches the repository's existing custom-op policy.

### Rejected: rewrite all child environment variables at startup

Mutating `os.environ` loses the distinction between requested and effective
state, is order-dependent for already imported modules, and makes diagnostics
misleading.

### Rejected: gate only patch registration

Several Step5 paths select providers directly inside model constructors or at
module import. Registration-only gating would miss load-time quantization,
fused parameter layouts, and direct runtime imports.

## Error handling and observability

- Unsupported master-off combinations fail during config validation, before
  allocating model weights or distributed buffers.
- Error messages name both `VLLM_HCU_USE_CUSTOM_OPS` and the relevant child or
  topology so the operator can choose deliberately.
- Existing one-time provider logs are retained. Where no provider log exists,
  Step5 logs one concise effective-policy message per process.
- Master-off never reports an optimized provider as selected.

## Validation

### CPU and source-level tests

1. Add a master/child matrix for each new resolver: master absent/`0`/`1`
   crossed with child absent/`0`/`1`.
2. Cover HY4 TP1/PCP8 topology-aware defaulting, explicit child disable, and
   master-off rejection.
3. Use call spies to prove optimized constructors, imports, registration
   callbacks, and execution entry points are not reached with the master off.
4. Use cold-import subprocess tests for Step5 module-level constants and
   custom-op registration.
5. Verify baseline output/provider selection for projection, router, SWA,
   LM-head, token-quant, EP collective, DP all-gather, host-sequence, and SWA
   decode paths.
6. Verify DSA remains enabled under master-off and that only audited optional
   DSA subproviders are suppressed.
7. Run the existing Step5 runtime, optimization-scope, environment-contract,
   DSA schedule/geometry, projection-fusion, MTP compatibility, HY4 attention,
   and PCP config suites.
8. Run syntax checks, `git diff --check`, production-boundary checks, and patch
   coverage checks.

Tests assert behavior rather than only searching source text. Layout-changing
features receive construction/load tests in addition to resolver tests.

### Hardware validation

After CPU verification, validate the branch on the supported eight-card host:

- master enabled: HY4 TP1/PCP8 + EP8 + DeepEP high-throughput + MTP3 starts
  without explicitly setting the linear-gate child flag, records the sharded
  gate layout, serves a request, and runs the established HumanEval 0-15
  concurrent gate;
- master disabled: the same PCP8 command fails during configuration with the
  expected actionable error and allocates no model weights;
- Step5 master enabled: a representative supported launch exercises the
  optimized defaults;
- Step5 master disabled: a representative supported launch uses the audited
  baseline providers, keeps DSA semantics active, serves a request longer than
  4096 tokens, and runs the existing short correctness gate.

Record the exact commit, vLLM wheel, model, complete command/environment, GPU
memory after load, provider evidence, request results, and cleanup state. A
CPU/source-level pass is not presented as hardware proof.

## Documentation and delivery

Update `docs/step5_runtime_integration.md` with the master/child contract,
managed feature table, DSA semantic exception, and validation evidence. Update
the HY4 PCP validation note so launch commands no longer require the child flag
when the master is enabled.

The implementation remains on
`codex/fix-hy4-pcp-mtp-draft-config`, is pushed to the personal fork, and is
submitted as a pull request to `HYGON-AI/vllm-plugin-das:v0.28.1-dev` after
review and verification.
