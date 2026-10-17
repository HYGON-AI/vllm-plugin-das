# Qwen4Exp dense-prefill/QSA-decode implementation plan

**Goal:** Add an opt-in HCU path that runs dense varlen FlashAttention from
the current BF16 Q/K/V tensors for context-free Qwen4Exp prefill batches,
while preserving the existing QSA path for prefix/chunked prefill, decode,
speculative decode, and mixed batches.

**Control:** `VLLM_HCU_QSA_DENSE_PREFILL=1`, additionally gated by
`VLLM_HCU_USE_CUSTOM_OPS`. The default is disabled.

## Tasks

1. Add failing environment-contract tests for the child switch, default, and
   custom-op master gate.
2. Add failing runtime-patch tests proving context-free prefill routes to
   dense varlen FlashAttention, prefix/chunked prefill, decode, and mixed
   batches stay on QSA, and QSA side caches are still updated.
3. Implement the environment resolver and a narrowly scoped runtime adapter
   around Qwen4Exp QSA attention.
4. Reuse HCU's varlen FlashAttention entry point with the current BF16 Q/K/V.
   Do not read the aligned physical KV cache for this path: hybrid Mamba cache
   alignment expands the page to 3200 tokens, and the vendor fallback would
   cast the query to the FP8 cache dtype while requiring a BF16 output.
5. Run focused unit tests, the runtime patch suite, and formatting/lint checks.
6. Run A/B serving with identical DP4/EP4 settings, warm up both instances,
   then measure BS32 input 4096/output 1024 TTFT and TPOT. Run the requested
   accuracy evaluation and inspect an ON trace to prove dense prefill and QSA
   decode routing.
7. Commit and open a stacked MR only if accuracy passes and performance has a
   reproducible improvement without a material TPOT regression.
