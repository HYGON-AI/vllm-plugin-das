# LightOp AutoAWQ Backend Design

## Goal

Add an optional HCU AutoAWQ W4A16 backend that uses the public LightOp Marlin
operators for the small-token shapes they tune, while preserving the current
vLLM Triton/Marlin route for every unsupported configuration.

## User-visible contract

`VLLM_HCU_USE_LIGHTOP_AWQ=1` opts into the backend; it defaults to disabled and
is subordinate to `VLLM_HCU_USE_CUSTOM_OPS`.  Eligibility requires FP16
activations, four-bit zero-point AutoAWQ, group size 128, and a LightOp-tuned
K/N pair.  Unsupported dtypes, shapes, groups, disabled policy, and missing
LightOp delegate to the original v0.28.1 quant method without changing its
weight layout.

Production code uses only
`lightop.gemm_ops.awq_gemm_marlin_weight_repack` and
`lightop.gemm_ops.gemm_awq_w4a16_marlin`.  It does not import LMSlim or a
private LightOp module.

## Architecture

A hybrid linear method wraps the quant method selected by vLLM.  It lets that
method create standard AutoAWQ parameters, decides eligibility before any
layout mutation, and either delegates completely or performs a one-time pure
Torch conversion followed by the public LightOp repack.  At inference the
LightOp path applies the public GEMM and restores the original output shape;
the delegate path is unchanged.

The conversion unpacks AutoAWQ int4 nibbles in `[0,4,1,5,2,6,3,7]` order,
transposes to the kernel's logical orientation, encodes qzero plus 64 together
with FP16 scales, and then calls the public repack function.  No duplicate
Triton and LightOp packed weights are retained for an eligible layer.

## Validation

CPU/mock tests cover layout conversion against a small independent reference,
selection and every fallback boundary, bias/output-shape behavior, public API
ownership, and idempotent patching.  HCU tests compare output accuracy and
latency with the existing v0.28.1 AutoAWQ route for representative tuned
shapes and token counts.  If no local AutoAWQ model exists, the MR records
that service-level validation is unavailable rather than claiming it.
