# SlimQuant Marlin vLLM 0.28.1 Compatibility Design

## Goal

Make `-q slimquant_marlin` reach quantization configuration validation on the
DTK 26.10 vLLM 0.28.1 wheel instead of failing in activation-dtype discovery.

## Cause

The public SlimQuant facade calls activation-dtype discovery on the selected
configuration class.  In the target vLLM wheel,
`CompressedTensorsConfig.get_supported_act_dtypes` is not decorated as a class
method, although its parameter is named `cls`.  The inherited lookup therefore
raises a missing-argument `TypeError` when called through the facade.

## Design

Add an explicit class method on
`SlimQuantCompressedTensorsMarlinConfig` that delegates to the target vLLM
implementation while supplying the concrete class argument.  Keep the generic
facade unchanged because the W4A8 implementation already exposes a normal
class method and must retain its lazy-loading contract.

Add a regression test through the real `SlimQuantMarlinFacade` and concrete
configuration class.  It must assert the target-supported activation dtypes,
so removal of the compatibility override reproduces the original `TypeError`.

## Boundaries

- This change fixes configuration discovery only; it does not claim that a
  plain BF16 checkpoint can be converted into a SlimQuant checkpoint at load
  time.
- Kernel loading, model-weight compatibility, graph capture, and request
  accuracy are validated separately with the requested Qwen3.5 model.
- LightOp W16A16 eligibility/configuration remains a separate change because
  its full-model path currently reaches an HCU VMFault.
