# HYV4 INT8 Quantization Reuse Design

## Goal

Reduce the HYV4 Channel-INT4 W4A8 decode gap against the Channel-FP8 model
without changing checkpoint weights or quantization semantics. The optimized
model must lose no more than 2 percentage points on the 64-case HumanEval gate
and must show an end-to-end throughput or TPOT improvement at input length 1,
output length 1024, concurrency 128.

## Evidence and scope

The routed W4A8 expert GEMMs are 44%--47% faster than their FP8 equivalents.
The regression comes from the non-expert INT8 path: per-token INT8 activation
quantization costs about 0.026--0.028 ms versus 0.007--0.008 ms for FP8, and
HYV4 quantizes identical activations repeatedly.

This change is limited to the plugin repository. It does not modify the model,
LightOp, DeepGEMM configuration files, or the installed Python environment.

## Design

When the existing `VLLM_HCU_USE_FUSED_RMS_QUANT` and custom-op switches are
enabled and the consumer uses INT8 activation quantization:

1. Attention quantizes its normalized input once for both `q_a_proj` and
   `kv_a_proj_with_mqa` instead of letting each linear quantize it separately.
2. The normalized Q-LoRA activation is quantized once and reused by both
   `q_b_proj` and the sparse indexer's `wq_b` projection.
3. `HYV4MoEFused` quantizes its input once and passes the same pair to routed
   W4A8 experts and INT8 shared experts. Chunked MTP execution slices the
   quantized tensors with the same token boundaries.
4. The dense feed-forward path accepts a pre-quantized input and uses the
   existing fused SiLU-multiply-quant operator for its INT8 down projection.

An attempted decoder-level fused RMSNorm/quant implementation reduced some
kernel work but reduced end-to-end throughput. The delivered implementation
therefore keeps ordinary RMSNorm and creates quantized pairs inside attention
and MoE. The final reuse path retains the latency benefit, but captured graph
memory remains 3.63 GiB versus the 2.37 GiB baseline; this cost must be stated
alongside benchmark results.

All new arguments are optional. With the feature flag off, non-INT8 weights,
or an unsupported quantization method, HYV4 keeps its current path.

## Validation gates

- Unit tests prove each shared pair reaches all intended consumers, chunk
  slicing stays aligned, and disabled/non-INT8 paths preserve old behavior.
- The repository's HYV4/runtime/accuracy regression suites remain green.
- HumanEval cases 0--63 score at least 63/64 (no more than 2 percentage points
  below the 64/64 baseline).
- The controlled greedy performance run uses DP8, EP8, MTP3, DeepEP low
  latency, FP8 E4M3 KV cache, FULL_AND_PIECEWISE graphs, input 1, output 1024,
  concurrency 128, 16 warmups, and 128 measured requests. A separate PR is
  opened only if throughput increases or TPOT decreases.

## Delivery

The optimization is developed on `perf/hy4-w4a8-quant-reuse`, stacked on the
existing correctness branch `fix/hy4-w4a8-dp8-ep8-mtp3`. Its PR targets that
branch so the performance diff remains independent of the correctness PR.
