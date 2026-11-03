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

## Review follow-up: large offsets and internal overlap

The persistent masked activation kernel must promote the expert index to
64-bit before multiplying it by any tensor stride. This keeps input, output,
scale, and token-count address arithmetic correct when a valid expert
workspace crosses the signed 32-bit element boundary. A CPU semantic test
will exercise the boundary without allocating the corresponding multi-GiB
tensor, and the existing HCU accuracy test will cover kernel compilation and
normal-size execution.

For `MK_INTERNAL_OVERLAPPED`, the selected design adds a shared-only
pre-quantized-input channel through `MoERunner`, the SlimQuant W4A8 quant
method, and the HCU modular kernel. The routed DeepEP input remains BF16;
only the shared expert receives the reusable `(int8, scale)` pair during the
modular kernel's finalize phase. This avoids changing DeepEP dispatch while
removing the duplicate shared gate/up quantization.

Two alternatives were rejected: dropping the pair for internal overlap would
discard work already produced by fused RMS quantization, while extending the
upstream `RoutedExperts.forward_modular` public surface would broaden the
replacement contract unnecessarily. The new contract tests must prove that
internal overlap receives the pair exactly once, routed dispatch stays BF16,
NO_OVERLAP behavior is unchanged, and unsupported methods retain their
existing public-forward fallback.

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

## Final review-follow-up results

The final feature HEAD `35f35d3`, plus the review-hardening commit `c120cf1`,
passed the portable contract gate with 2,574 tests collected and passing (192
deselected), including 372 focused HYV4, DeepEP, and quantized-GEMM cases. The
real HCU persistent-kernel accuracy entry also passed. HumanEval evaluated 64
unique cases with no API errors and scored 64/64 (`mean_acc_pass@1 = 1.0`),
so the accuracy change is 0 percentage points.

The requested DP8/EP8/MTP3 service used DeepEP low latency, FP8 E4M3 KV
cache, and FULL_AND_PIECEWISE graphs. The two controlled 128-request runs
produced 1294.01 and 1266.70 output tok/s, for a 1280.36 tok/s mean. This is
2.13% below the 1308.22 tok/s W4A8 reference mean, narrowly outside the 2%
raw-throughput threshold, and 6.69% above the 1200.10 tok/s FP8 reference.

The required investigation found that full-run throughput is dominated by
one or two data-parallel tail requests: the prior W4A8 reference itself spans
4.41% between its two runs. A same-host before/after check showed that the
steady per-token cost did not regress: mean TPOT improved from 73.760 ms at
`01e694f` to 73.355--73.393 ms at `35f35d3` (0.50%--0.55%), and after
excluding the two makespan outliers the paired throughput was 1321.48 versus
1322.66 tok/s. Raw makespan should therefore be reported with this variance
qualification rather than treated as evidence of a kernel regression.

## Delivery

The optimization is developed on `perf/hy4-w4a8-quant-reuse`, stacked on the
existing correctness branch `fix/hy4-w4a8-dp8-ep8-mtp3`. Its PR targets that
branch so the performance diff remains independent of the correctness PR.
