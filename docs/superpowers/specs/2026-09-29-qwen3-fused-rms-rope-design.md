# Qwen3 Fused RMSNorm and Rotary Embedding Design

## Goal

Accelerate the Qwen3 dense-attention query/key preparation path on HCU by
replacing the separate Q/K RMSNorm and rotary-embedding calls with the public
LightOp `lightop.attention.rms_rotary_embedding_fuse` operator.

## User-visible contract

The optimization is enabled by default through
`VLLM_HCU_USE_FUSED_RMS_ROPE=1` and can be disabled with `0`.  The setting is
latched during worker patch installation.  It is also governed by the
`VLLM_HCU_USE_CUSTOM_OPS` master switch.

The fused route is used only for the Qwen3 text attention implementation when
positions are one-dimensional, the rotary dimension equals the attention head
dimension, the rotary cache is available, and dual-chunk attention is not
active.  Every other case delegates to the original vLLM forward method before
any input is mutated.

Production code imports only the categorized public LightOp API.  It must not
import LMSlim, `lightop.op`, `lightop.gemmopt`, or LightOp private modules.

## Architecture

`vllm_hcu.ops.rms_rope` owns one Torch custom-op wrapper whose real
implementation calls the public LightOp attention function and whose fake
implementation preserves the two output shapes.  The schema declares query
and key as mutable because the vendor kernel updates them in place.

`patch_qwen3_attention` installs an exact-signature replacement for
`Qwen3Attention.forward`.  The replacement mirrors the upstream QKV split,
uses the fused wrapper for Q/K normalization and RoPE, then invokes the
unchanged vLLM attention and output projection.  A saved original method is
used for all unsupported inputs and when the feature is disabled.

## Validation

CPU/mock tests cover environment precedence, exact target/signature,
idempotence, supported routing, disabled routing, two-dimensional positions,
dual-chunk attention, and public LightOp API ownership.  HCU validation covers
single-operator BF16 accuracy/performance and Qwen3-8B service startup,
`/health`, `/v1/models`, HumanEval prompts 0-7, and an A/B throughput run.
