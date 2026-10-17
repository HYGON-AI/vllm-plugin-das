# Qwen4Exp dense-prefill/QSA-decode implementation plan

**Goal:** Add an opt-in HCU path that runs dense varlen FlashAttention for
Qwen4Exp prefill-only batches. Context-free prefill consumes current BF16
Q/K/V directly; prefix/chunked prefill gathers the manager-paged KV cache and
dequantizes FP8 storage before attention when the fixed-stride workspace stays
within amplification and absolute byte limits. Oversized/ragged history,
decode, speculative decode, and mixed batches preserve the existing QSA path.

**Control:** `VLLM_HCU_QSA_DENSE_PREFILL=1`, additionally gated by
`VLLM_HCU_USE_CUSTOM_OPS`. The default is disabled.

## Tasks

1. Add failing environment-contract tests for the child switch, default, and
   custom-op master gate.
2. Add failing runtime-patch tests proving prefill-only batches route to dense
   varlen FlashAttention, decode and mixed batches stay on QSA, ragged history
   lengths remain distinct, unsafe history falls back to QSA, and QSA side
   caches are still updated.
3. Implement the environment resolver and a narrowly scoped runtime adapter
   around Qwen4Exp QSA attention.
4. Reuse HCU's public varlen FlashAttention entry point for context-free BF16
   Q/K/V. For history, gather the aligned 3200-token manager pages into stable
   fixed-stride BF16 segments, and pass both segment offsets and actual K
   lengths to the native varlen ABI so ragged padding is never attended. Keep
   QSA selection available for history and fall back when fixed-stride
   amplification exceeds 4x or the estimated workspace exceeds 1 GiB.
5. Run focused unit tests, the runtime patch suite, and formatting/lint checks.
6. Run A/B serving with identical DP4/EP4 settings, warm up both instances,
   then measure BS32 input 4096/output 1024 TTFT and TPOT. Run the requested
   accuracy evaluation and inspect an ON trace to prove dense prefill and QSA
   decode routing.
7. Commit and update the stacked MR only if accuracy passes and the target
   workload has a reproducible prefill/end-to-end improvement; report TPOT and
   speculative-acceptance changes separately because decode routing is
   unchanged.
