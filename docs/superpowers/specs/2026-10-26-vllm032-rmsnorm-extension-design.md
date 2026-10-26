# vLLM 0.32 DAS RMSNorm Extension Design

## Intent

Extend the independently installable `hcu-vllm-das-plugin` from the validated
SiluAndMul-only `0.1.0.dev0` slice to `0.2.0.dev0` with the model-independent
LightOp RMSNorm and fused-add RMSNorm paths from the pinned DAS reference.
`hcu-vllm-adapter` remains the only owner of HCU platform selection and runtime
lifecycle; the DAS wheel remains optimization-only.

The source reference remains `HYGON-AI/vllm-plugin-das` branch `v0.28.1-dev`
at `cd412a0d53b9b517fbc3dcdf96121dc3daa1096c`. The vLLM target remains official
`v0.32.0` at `5df63a7203cdb9328d824a27b682e77c7b9928be`. The prepared LightOp runtime is
`0.6.0+dtk2610.torch2110.2610222215`.

## Goals

- Add capability `lightop-rms-norm` to the existing adapter extension.
- Register vLLM 0.32 OOT replacement `HcuRMSNorm` and two uniquely named Torch
  custom operators for ordinary and fused-add RMSNorm.
- Preserve all new vLLM 0.32 semantics by routing unsupported cases to
  `RMSNorm.forward_cuda`.
- Keep the master and operator-specific environment rollback effective after
  layer construction.
- Prove ordinary and fused-add BF16/FP16 numerical behavior on `gfx938` from
  installed wheels.
- Preserve the exact SiluAndMul behavior and ownership boundary already
  validated in `0.1.0.dev0`.

## Non-goals

- Do not port `GemmaRMSNorm`, `RMSNormGated`, RMSNorm-plus-quantization, model
  patches, attention, MoE, or native kernels in this slice.
- Do not claim `gfx92a`, model-level accuracy, throughput, graph replay, or
  multi-device validation.
- Do not add a second vLLM platform/general entry point or package legacy
  `vllm_hcu` modules.
- Do not replace the prepared HCU PyTorch, DTK, vLLM, Adapter, or LightOp stack
  through pip dependency resolution.

## Compatibility findings

The old DAS RMSNorm assumes a weighted, contiguous reduction with the last
dimension as its variance domain. vLLM 0.32 `RMSNorm` also supports
`has_weight=False`, `var_hidden_size`, batch-invariant execution, and strided
inputs. Passing those cases to the old LightOp kernel would silently narrow
the vLLM contract or risk invalid memory/layout assumptions.

The optimized path is therefore eligible only when all of these are true:

- `VLLM_HCU_USE_CUSTOM_OPS` and `VLLM_HCU_USE_CUSTOM_RMS_NORM` are enabled;
- `VLLM_BATCH_INVARIANT` is false;
- the selected ordinary/fused path passes a real weight;
- `variance_size_override` is `None`;
- the input width is at least 64 and equals the layer hidden size;
- input and optional residual are contiguous, shape-compatible, and share
  dtype;
- the weight is contiguous and has the input dtype.

Every other case delegates to vLLM 0.32 `forward_cuda`, including narrow
Mamba projections. The checks run inside `forward_hip`, so either rollback
switch takes effect without reconstructing the model.

## Approaches

### Copy the v0.28.1 module unchanged

This is smallest but does not account for weightless, variance override,
batch-invariant, or strided-input behavior added or formalized in vLLM 0.32.
It is rejected as unsafe for production.

### Add a second adapter extension entry point

This isolates RMSNorm activation failures from SiluAndMul, but expands the
public discovery surface and changes the existing one-extension ownership
contract. It is deferred until the Adapter protocol can report capability-level
activation independently.

### Expand the existing extension with guarded RMSNorm (selected)

Preflight all SiluAndMul and RMSNorm symbols before importing either capability,
then register both under the same success latch. A missing dependency fails the
optimization extension before product modules are imported. The guarded OOT
implementation retains upstream behavior for every unported vLLM 0.32 case.
This keeps packaging and operational rollback simple at the cost that both
capabilities are an all-or-nothing activation unit.

## Activation and packaging

Discovery remains dependency-light. Activation validates release `0.32.0`,
then resolves:

- `lightop.activation.silu_and_mul_opt`;
- `lightop.norm.rmsnorm_forward_autograd`;
- `lightop.norm.fused_add_rms_norm`;
- vLLM `SiluAndMul.register_oot` and `RMSNorm.register_oot`;
- vLLM `direct_register_custom_op`.

Only after all checks pass does activation import `hcu_vllm_das.silu_and_mul`
and `hcu_vllm_das.rms_norm`. Success latches once under the existing lock.
The wheel continues to declare only `hcu_vllm_adapter.extensions:das_hcu`.

The plugin version becomes `0.2.0.dev0`; the previous `0.1.0.dev0` wheel and
its validation record remain immutable historical evidence.

## Verification

Portable fake-runtime tests prove registration, output/fake behavior, mutation
schema, eligibility and every fallback class. Frozen-source tests pin the exact
vLLM 0.32 RMSNorm fields and methods consumed by the plugin. Installed-wheel
attestation must find both OOT owners and both capabilities while rejecting
legacy platform/general ownership.

Live `gfx938` validation instantiates upstream `RMSNorm`, proves OOT replacement,
and compares ordinary and fused-add BF16/FP16 outputs with independent float32
references. It also exercises the master switch, child switch, narrow-width,
weightless, and variance-override fallbacks. `gfx92a` remains explicitly
unvalidated and should run with `VLLM_HCU_USE_CUSTOM_RMS_NORM=0` until it has its
own evidence record.

## Follow-on

After RMSNorm, validate a small model A/B using both migrated capabilities,
then port attention/operator adapters, native cache/collective kernels, and MoE
one evidence-backed slice at a time.
