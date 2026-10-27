# Step5 runtime integration

Step5 uses the existing vLLM-HCU exact-import patch coordinator. Installation
and runtime do not modify the installed vLLM package.

## Ownership

- `vllm_hcu/models/step5_text.py`, `step5_vl.py`, `step5_mtp.py`: model implementations.
- `vllm_hcu/models/step5_config.py`: checkpoint configuration classes.
- `vllm_hcu/model_executor/layers/step4_dsa.py` and `step4_dsa_kernels.py`: DSA backend and kernels.
- `vllm_hcu/models/step5_swa_fused.py` and `vllm_hcu/ops/step5`: SWA, router, cached Triton driver launcher and FP8 kernels.
- `vllm_hcu/v1/step5_*.py`: Step5 metadata, execution and multi-module MTP implementations.
- `vllm_hcu/distributed/eplb*_runtime.py`: mixed-precision expert movement.

`patch/module_exchange.py` lists eight exact canonical Step5 aliases. They are
armed atomically with the existing cold replacements before importing model
or custom-op modules. Canonical and plugin imports resolve to the same objects.

## Registration order

Platform callbacks first install the compatible DeepGEMM exports before cold
MoE imports bind those functions during configuration deserialization. Later
Worker registration reuses the same idempotent adapters. Platform callbacks
then register configuration exports, model types, MTP overrides, compile
splitting operations, compile prefix hashing and EP graph capture.
EP capture is registered before consumers copy the public distributed export.
The Worker dispatcher registers metadata, block-table mirrors, MTP loading and
sampling, MoE combine, token FP8, logits and EPLB callbacks.
DP dispatch keeps the upstream synchronization contract: reuse is only valid for
the same agreed batch; a rank must not silently start a new collective on its own.

MoE combine hook state is owned by the plugin and can be registered before
Worker dispatch. Step5 execution and EPLB helpers resolve the canonical or owned
implementation rather than retaining exports copied before callback dispatch.

Substantial implementations remain in owned modules. Small adapters validate
exact target names and signatures, verify reviewed source fingerprints where
shared methods are replaced, and record target identities for idempotency.
Incompatible or stale targets fail startup instead of silently losing behavior.
No source rewriting or runtime patch-file application is used.

Step5 runner and metadata paths are selected by model type. Other model families
retain their native execution and MTP paths. Existing HCU feature sidecars and
operator switches remain the configuration boundary; upstream config and
attention metadata dataclasses are not extended in place.

## Optimization boundaries

`VLLM_HCU_USE_CUSTOM_OPS` is the master for optional HCU acceleration. Its
effective-policy rule is strict: master `0` wins over a child explicitly set to
`1`; master `1` preserves each child's existing default and explicit disable.
The implementation resolves this policy before importing an optional provider,
changing a weight layout, registering an optional split, or allocating an
optional communicator buffer.

The managed Step5 children are:

| Area | Child settings / selected providers | Master-off baseline |
| --- | --- | --- |
| Model load and projections | `VLLM_HCU_STEP5_FUSE_QKVG`, `VLLM_HCU_STEP5_BF16_ROUTER`, `VLLM_HCU_STEP5_FUSED_ROUTER_SHARED`, `VLLM_HCU_STEP5_FP8_DENSE`, `VLLM_HCU_STEP5_FP8_BF16_MOE`, `VLLM_HCU_STEP5_SWA_FUSED_QK`, `VLLM_HCU_STEP5_FUSED_GATE` | Original modules and native weight layouts |
| Logits and quantization | `STEP5_SKINNY_LM_HEAD`, `VLLM_HCU_TRITON_TOKEN_FP8_QUANT` | Upstream logits/quantization implementations |
| MoE and communication | `VLLM_HCU_EP_FUSED_MOE`, `VLLM_HCU_EP_FUSED_COMBINE`, `VLLM_HCU_DP_CUSTOM_AG` | DeepGEMM/normal combine and process-group collectives; no optional communicator allocation |
| Runner and attention | `VLLM_HCU_HOST_SEQ_LENS`, `VLLM_HCU_SWA_DECODE` | Device-derived sequence lengths and baseline paged attention |
| DSA acceleration | per-request and FP8-dot speculative indexers, BoltOps/FlashAttention stages, optional FP8 attention math, fused speculative summaries, and merged projection | Full-batch BF16 indexer, vendored Triton stages, unfused summaries, and separate projections |

DSA itself is a required Step5 model semantic, not an optional custom-op
provider. Therefore master-off does **not** set `VLLM_STEP5_DISABLE_DSA`, remove
`vllm::step4_dsa_index` from compilation, or disable the DSA metadata, cache,
planner, selector, CSA compression, and vendored Triton baseline. Disable DSA
only with its explicit `VLLM_STEP5_DISABLE_DSA` setting. Diagnostic and tuning
values such as slot validation, plan dumps, query chunking, and graph-region
sizing are also not provider selections and retain their meanings.

Only Step5-created logits processors opt into the skinny LM-head kernel. The
existing `STEP5_SKINNY_LM_HEAD` switch can still disable it. Step5 EPLB uses
per-layer rank maxima; other models retain the upstream rank-total metric.
Shared DeepGEMM and mixed-format transfer adapters remain compatibility helpers.

QKV/gate fusion requires bias-free, matching BF16/FP16 projections in the NN
weight layout. Unsupported layouts use the original linear methods. DSA validates
indexer geometry at construction and RoPE tensor layout before kernel launch.

`VLLM_STEP5_DISABLE_DSA` uses the same parser in platform, config and model code:
`1` or `true` (case-insensitive, surrounding whitespace ignored) disables DSA.
The old `VLLM_STEP5_ENABLE_QKVG_PROJ`, `VLLM_STEP5_DSA_INDEX_TP_SIZE`,
`VLLM_STEP5_FUSE_INDEXER_NORM` and `VLLM_STEP5_O_PROJ_REDUCE_SCATTER`
variables had no implementation and are no longer registered. Explicit settings
produce a warning; the actual QKVG switch is `VLLM_HCU_STEP5_FUSE_QKVG`.

### Audit of the `75aa13701` runtime surface

The complete Python runtime surface introduced by `75aa13701` was classified
as follows. This grouping covers every path returned by `git diff-tree` for the
commit; a helper inherits the classification of its only caller.

| Files / subsystem | Classification | Reason |
| --- | --- | --- |
| `models/step5_text.py`, `models/step5_swa_fused.py`, `models/step5/kernels.py`, `layers/skinny_lm_head.py`, `ops/token_fp8_quant.py` and the corresponding provider modules under `ops/step5/` | Managed optional acceleration | These replace projections, routers, logits, quantization, or fused kernels and have native fallbacks. |
| `layers/fused_moe/{experts/deep_gemm_moe.py,moe_runner.py,ep_fused_moe*.py,combine_runtime.py}`, `patch_cuda_communicator.py`, `distributed/device_communicators/ipc_all_gather.py` | Managed optional acceleration plus required support helpers | Provider selection/allocation is master-gated; capability checks and cleanup are required once a selected provider owns resources. |
| `model_executor/layers/step4_dsa.py` alternate providers and fusion selectors | Managed optional acceleration | Master-off selects the full-batch/BF16/unfused/separate-projection vendored baseline. |
| `step4_dsa.py` planner/cache/metadata plus `step4_dsa_kernels.py` | Required model semantics | They implement Step5 sparse attention and remain active unless DSA is explicitly disabled. |
| `models/step5_config.py`, `models/{step5_mtp.py,step5_vl.py}`, `v1/step5_*.py`, `v1/hcu_model_runner_v2.py` non-host-mirror paths | Correctness and compatibility | Config identity, metadata ownership, MTP loading/sampling, and runner selection are needed for a valid Step5 execution path. |
| `distributed/eplb*_runtime.py` and Step5 EPLB adapters | Correctness for explicitly configured EPLB | They preserve mixed-format expert state and lifecycle; they are not an unconditional kernel substitution. |
| `patch/module_exchange.py`, patch registration/contract adapters, `models/__init__.py`, `models/step5/__init__.py`, `runtime_compat/step5*.py`, `platforms/step5_envs.py`, `platforms/hcu.py` | Compatibility and registration | These establish exact imports, fail-closed signatures, config exports, and shared environment parsing. |
| DSA slot checks, plan dump, query-chunk and graph-region controls | Diagnostic/tuning | They change diagnostics or bounded scheduling geometry, not operator-provider ownership. |

The audit intentionally does not claim that master-off removes all plugin code;
it guarantees optional accelerator fallback while keeping the model-correctness
and compatibility layers required to run Step5.

## MoE and communicator contracts

`VLLM_HCU_EP_FUSED_MOE` is a channel-FP8 optimization with dynamic per-token
activations and SWIGLUSTEP. Weight preparation and execution share the same
layout checks. Block-FP8 keeps its original DeepGEMM path; unsupported channel
layouts retain normal weight packing. AITER advertises SWIGLUSTEP only for its
implemented FP8/INT8 channel-weight and dynamic-token activation pairs.

Mixed attention backends retain a compatible upstream KV manager block and use
kernel virtual splitting. Incompatible hybrid or heterogeneous page constraints
fail explicitly instead of silently replacing their aligned manager block.

The optional IPC all-gather supports same-node DP groups of 2 to 8 ranks.
Inputs share logical row layouts and global sizes across ranks; local strided
views are packed on either communication path. Initialization errors are
coordinated during startup, without adding decode-time collectives. Communicator
destruction synchronizes local work, closes imported mappings through the
plugin's native `close_mem_handle` operation, confirms all importers have
closed, then frees owned raw buffers. A failed group or import-close operation
must terminate the worker instead of freeing buffers that peers may still use.
Garbage collection never starts a collective or frees exported IPC memory.
Rebuild the plugin extension when upgrading to this native operation.

Router top-k uses a padded power-of-two internal vector with masked writes.
Non-power-of-two values remain valid, while topk8 retains its original layout.

## Pending-region ownership

Decode, speculative verification and mixed prefill schedules resolve and touch
all resumed region heads before allocating any new pending rows. A retired row
may still hold tokens needed after speculative rejection; an earlier request in
the batch must not recycle a later request's head. Missing buffered tokens fail
instead of producing an incorrect summary.

Quantization tools, server launchers, performance clients and evaluation run
artifacts remain outside the source package. The package contains runtime
implementations, compatibility adapters, their tests and integration guidance.

## Validation

Use the supported vLLM 0.28.1 base and the plugin on `PYTHONPATH`:

```bash
python -m pytest -q tests/runtime_patch/test_step5_runtime.py \
  tests/runtime_patch/test_step5_dp_sync_safety.py \
  tests/runtime_patch/test_step5_optimization_scope.py \
  tests/runtime_patch/test_step5_env_contract.py \
  tests/model_executor/test_step4_dsa_schedule.py \
  tests/model_executor/test_step4_dsa_geometry.py \
  tests/model_executor/test_step5_projection_fusion.py
python tools/check_production_boundary.py
python tools/check_patch_test_coverage.py
```

The focused runtime tests cover adapter contracts and registration. Pending
buffer ownership regressions live separately in
`tests/model_executor/test_step4_dsa_schedule.py`.

For the full `tests/patch` suite, set `VLLM_V0251_SOURCE_ROOT` to the directory
containing the reviewed vLLM 0.28.1 package. This is the existing test harness's
legacy variable name; it does not require a separate vLLM 0.25.1 checkout.

The adapters accept the reviewed clean base and, where needed, the previous
source-patched API. Updating vLLM requires reviewing the compatibility contracts.
A clean-base import/configuration test verifies independence from historical
source patches; CPU tests do not establish GPU correctness or performance.
GPU startup and inference should be verified separately when resources allow.
