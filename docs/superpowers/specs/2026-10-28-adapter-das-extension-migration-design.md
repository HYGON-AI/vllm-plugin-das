# adapter-das Extension Migration Design

## Intent

Move the existing vLLM 0.32 DAS optimization wheel from the standalone
`hcu-vllm-adapter` protocol to the multi-host `adapter-das` protocol without
changing its validated LightOp operator implementations.

## Goals

- Consume Extension API v1 from `hcu_adapter.extensions`.
- Register exactly one entry point in group `hcu_adapter.extensions`.
- Depend on the adapter-das distribution `sgl-kernel>=0.1.0`.
- Preserve the vLLM 0.32 compatibility gate, capability names, priority,
  idempotent activation, master switch, operator switches, and upstream
  fallbacks.
- Update source, wheel, verifier, portable, and live-HCU contracts together.

## Non-goals and anti-goals

- Do not add `vllm.platform_plugins` or `vllm.general_plugins` ownership.
- Do not package legacy `vllm_hcu` or either adapter implementation.
- Do not alter RMSNorm or SiluAndMul math in this migration.
- Do not inherit model-support claims from the former adapter pairing.

## Approaches

1. **Mechanical protocol migration — selected.** Change only the dependency,
   extension namespace, verifier ownership expectations, and related tests.
   This preserves already validated optimization code and makes regressions
   attributable to the integration boundary.
2. Support both old and new extension groups in one wheel. Rejected because
   installing both adapters could activate the same optimization twice.
3. Give the plugin its own vLLM general entry point. Rejected because it
   recreates competing runtime owners and defeats atomic adapter allow-listing.

## Interface and lifecycle

The wheel declares `hcu_adapter.extensions: das_hcu =
hcu_vllm_das.extension:extension_spec`. `extension_spec()` returns adapter
`ExtensionSpec(name="das-hcu", api_version=1, priority=100,
capabilities={"lightop-rms-norm", "lightop-silu-and-mul"}, ...)`.
Activation continues to validate official vLLM release `0.32.0`, LightOp
callables, and vLLM OOT registries before importing the operator modules.

## Testing and acceptance

- Portable contracts fail if the old adapter package/group remains anywhere
  in runtime source, build metadata, verifier, or v0.32 tests.
- Wheel inspection proves there are no vLLM platform/general entries and only
  the new adapter extension entry.
- The installed verifier requires adapter-owned vLLM callbacks at
  `hcu_adapter.bridges.vllm` and the extension protocol at
  `hcu_adapter.extensions` from distribution `sgl-kernel`.
- Existing RMSNorm/SiluAndMul portable tests remain green unchanged in meaning.
- Live-HCU operator tests and model/HumanEval gates must be rerun with the new
  adapter/plugin wheel pair before a production statement.
