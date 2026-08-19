# HCU Model Runner V2 Design

## Goal

Run vLLM v0.25.1 with Model Runner V2 on the HCU platform while retaining the
existing Model Runner V1 path as a controlled fallback. The first acceptance
model is `/models/Qwen3-8B` on one HCU device.

## Current State

`HcuGPUWorker` already reads `VllmConfig.use_v2_model_runner`. When enabled it
constructs the upstream `vllm.v1.worker.gpu.model_runner.GPUModelRunner`
directly; when disabled it constructs the plugin-owned V1 runner in
`vllm_hcu.v1.hcu_model_runner`. This makes the upstream V2 runner reachable,
but gives HCU no explicit compatibility boundary for runner-specific behavior.

The plugin's import-time runtime patches, platform registration, attention
backends, model implementations, and custom operators remain shared by both
paths. They should continue to be the primary customization mechanism.

## Approach

Add a thin plugin-owned `HcuGPUModelRunnerV2` subclass around the upstream V2
runner and make `HcuGPUWorker` instantiate it when V2 is enabled. The subclass
is an integration boundary, not a fork: it initially inherits upstream
behavior unchanged and gains overrides only when an HCU-specific difference is
demonstrated by a focused test or the Qwen3-8B smoke run.

Do not copy the upstream V2 runner into the plugin and do not mechanically port
the monolithic V1 HCU runner. Existing module replacement and callback patches
remain responsible for operator/model adaptations that are independent of the
runner implementation.

## Configuration and Data Flow

1. The caller sets `VLLM_USE_V2_MODEL_RUNNER=1` to select MRV2.
2. `VllmConfig` validates upstream MRV2 feature constraints.
3. `HCUPlatform` selects `HcuGPUWorker` as today.
4. `HcuGPUWorker` applies worker patches before calling its parent constructor.
5. During device initialization, the worker constructs
   `HcuGPUModelRunnerV2` for V2 or the existing `GPUModelRunner` for V1.
6. The HCU V2 subclass delegates to upstream MRV2 and uses the already
   registered HCU attention backends, model modules, and custom operators.

`VLLM_USE_V2_MODEL_RUNNER=0` remains the explicit V1 rollback mechanism. This
change does not make MRV2 unconditional and does not remove the V1 runner.

## Compatibility Policy

The wrapper imports MRV2 from the vLLM v0.25.1 module path and is intentionally
version-coupled to the plugin's existing v0.25.1 compatibility contract.
Unsupported upstream MRV2 features must fail through vLLM's normal validation;
the plugin must not silently force V1 after the user explicitly selects V2.

Overrides are permitted only for HCU-specific behavior. Each override requires
a contract test that records the upstream signature or behavior relied upon,
so future vLLM upgrades fail clearly instead of drifting silently.

## Error Handling

- Import or constructor incompatibilities fail with their original traceback.
- HCU runtime patch validation continues after model loading on both runners.
- An MRV2-specific HCU incompatibility receives a focused error or a narrowly
  scoped override; it is not hidden by automatic fallback.
- Operators unavailable on the installed HCU stack remain normal startup or
  execution failures with enough context to identify the failing component.

## Testing and Acceptance

1. Add a unit contract proving the worker selects the plugin-owned V2 runner
   when `use_v2_model_runner` is true and preserves the V1 selection otherwise.
2. Run the relevant plugin unit/runtime-patch suite.
3. Run one-device offline inference with `/models/Qwen3-8B`, first with eager
   execution to isolate basic loading and forward execution.
4. Run the same minimal inference with the normal graph configuration.
5. Confirm logs identify Model Runner V2 and generation returns non-empty text.
6. If feasible on the available device, compare deterministic V1 and V2 token
   output for the same prompt and sampling parameters.

The implementation is accepted when MRV2 completes the Qwen3-8B smoke run,
the V1 fallback remains selectable, and focused tests pass. Hardware or driver
failures outside the code change are reported separately with captured logs.

## Out of Scope

- Removing Model Runner V1.
- Making MRV2 the unconditional default for all models and features.
- Porting every optimization from the V1 HCU runner without evidence it is
  required by MRV2.
- Broad performance tuning, multi-node validation, speculative decoding, LoRA,
  pooling, or disaggregated serving in this first migration.
