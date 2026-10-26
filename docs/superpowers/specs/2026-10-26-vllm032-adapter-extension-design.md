# vLLM 0.32 DAS Adapter Extension Design

## Intent

Produce a real, independently installable DAS optimization plugin for the
official vLLM `v0.32.0` release while preserving `hcu-vllm-adapter` as the
only owner of HCU platform selection and baseline runtime lifecycle.

The source reference is `HYGON-AI/vllm-plugin-das` branch `v0.28.1-dev` at
`cd412a0d53b9b517fbc3dcdf96121dc3daa1096c`. The vLLM target is official
commit `5df63a7203cdb9328d824a27b682e77c7b9928be`. This first production slice
ports the model-independent LightOp `SiluAndMul` path and establishes the
packaging, activation, wheel, and GPU-validation boundary for later
capabilities.

## Goals

- Build a pure-Python `hcu-vllm-das-plugin` wheel targeting vLLM release
  `0.32.0`.
- Expose exactly one runtime entry point:
  `hcu_vllm_adapter.extensions: das_hcu =
  hcu_vllm_das.extension:extension_spec`.
- Import no Torch, vLLM, LightOp, legacy DAS runtime, or device code during
  extension discovery.
- Reject missing, malformed, epoch-shifted, or non-0.32 vLLM versions before
  runtime registration.
- Activate the vLLM 0.32 OOT `SiluAndMul` implementation only after LightOp
  and the exact vLLM interface are available.
- Keep activation deterministic and idempotent in one process.
- Prove wheel contents, installed entry points, vLLM 0.32 OOT registration,
  and HCU numeric correctness.

## Non-goals

- Do not register `vllm.platform_plugins` or `vllm.general_plugins`.
- Do not package the legacy `vllm_hcu` platform, Worker, model, patch, or
  native-extension modules in the new wheel.
- Do not claim that MoE, attention, EPLB, checkpoint-specific, multi-GPU, or
  model-level optimizations are ported by this slice.
- Do not modify vLLM installation files or depend on post-install patching.
- Do not promote the plugin to production from unit tests alone; installed
  wheel and live HCU evidence remain required.

## Constraints

- `hcu-vllm-adapter` API version 1 owns discovery and activation.
- Compatible vLLM versions have PEP 440 epoch zero and release tuple exactly
  `(0, 32, 0)`; pre/dev/post/local suffixes are allowed.
- The plugin must remain optional. An activation failure is reported by the
  adapter and must not change platform ownership.
- The global `VLLM_HCU_USE_CUSTOM_OPS=0` rollback and the child
  `VLLM_HCU_USE_CUSTOM_SILU_AND_MUL=0` rollback both select vLLM's native
  implementation.
- Source reference and license/notice history remain in the repository even
  though legacy runtime packages are excluded from this wheel.

## Approaches

### Relabel the complete 0.28.1 wheel as 0.32

This would retain conflicting platform/general entry points and execute old
platform, Worker, and model patches against changed vLLM interfaces. It is
rejected as unsafe.

### Keep the `vllm_hcu` wheel and selectively disable old entry points

This reduces packaging work, but the wheel would still contain a large dormant
runtime surface and importing the root namespace could execute legacy setup.
An accidental import would be difficult to distinguish from an authorized
port. It is rejected for the first production slice.

### New optimization-only namespace and wheel (selected)

Create `hcu_vllm_das` and package only migrated modules. The original
`vllm_hcu` tree stays in Git as the pinned migration source but is absent from
the wheel. Each capability enters the new namespace only with its own vLLM
0.32 contract and runtime evidence. This makes wheel inspection an enforceable
production boundary and avoids package-file collisions with historical DAS
installations.

## Package and activation boundary

`setup.py` becomes dependency-light and does not import Torch or compile the
legacy native extension. It discovers only `hcu_vllm_das` packages and defines
only the adapter extension entry point. Runtime environments supply the
audited vLLM, adapter, Torch/DTK, and LightOp wheels; installation uses
`--no-deps` so pip cannot replace that prepared stack.

`hcu_vllm_das.extension` is safe during metadata discovery. Its factory
returns adapter `ExtensionSpec(name="das-hcu", api_version=1, priority=100,
capabilities={"lightop-silu-and-mul"})`. Activation performs, in order:

1. validate `ExtensionContext.vllm_version` against release `0.32.0`;
2. resolve the required `lightop.activation.silu_and_mul_opt` symbol;
3. resolve the vLLM 0.32 `SiluAndMul` OOT and custom-op APIs;
4. import/register the migrated implementation once;
5. latch success only after registration completes.

A failed attempt is not reported as active. Repeated successful activation is
a no-op. The adapter's process-local lifecycle remains the outer once-only
boundary.

## First capability: LightOp SiluAndMul

The migrated implementation registers one Torch custom op and one vLLM OOT
class. The OOT class overrides `forward_hip`; when both HCU custom-op switches
are enabled it calls LightOp, otherwise it calls vLLM 0.32 `forward_cuda`.
The output shape, dtype, and device match vLLM's native SwiGLU contract.

This path is chosen first because it is model-independent, single-device,
does not require legacy platform or Worker patches, and the target vLLM still
exposes the same OOT abstraction. It provides a meaningful plugin-enabled A/B
path without pretending that the wider 0.28 runtime is compatible.

## Verification and production status

Portable tests verify version gating, discovery import isolation, entry-point
ownership, failure ordering, idempotence, rollback switches, and exact wheel
contents. An isolated environment then installs the official vLLM 0.32 wheel,
the adapter wheel, and the new plugin wheel and verifies the real entry point
and OOT registry.

Live HCU validation compares the LightOp result against a float32 PyTorch
reference for representative BF16 shapes with both rollback switches. A model
smoke/A-B run may follow on the same installed wheel tuple. This slice is
production-candidate only after those commands pass; it does not promote any
unported capability.

## Follow-on order

After this slice, port capabilities one at a time from the reviewed intake:
RMSNorm, attention/operator adapters, native cache/collective kernels, then MoE
and model-specific paths. Every addition must retain the optimization-only
entry point and add its own vLLM 0.32 contract plus disabled/enabled A/B
evidence.
