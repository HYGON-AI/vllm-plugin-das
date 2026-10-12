# DeepSeek-V4 sparse prefill integration validation

## Scope

`tests/accuracy/check_dsv4_flashmla_prefill_integration_gpu.py` directly calls
the installed vLLM `DeepseekV4ROCMAiterMLAAttention._forward_prefill` with
synthetic paged FP8 caches and metadata. It uses real cache quantization,
dequantization/gather, `combine_topk_swa_indices`, workspace allocation, original
AITER prefill, and the installed production FlashMLA padding adapter. No model
weights or substituted gather/combiner kernels are used.

Coverage:

- TP-local heads: 16, 32, 96, 64, 128.
- SWA, C4A, C128A; nonzero attention sinks and interior negative top-k indices.
- Three requests with query lengths 17/33/65 and sequence lengths 173/301/521.
- A preceding decode metadata row with two tokens, making prefill query offsets
  nonzero; only the prefill method is executed.
- Disjoint reversed physical cache pages, multiple pages per request, and
  compressed-cache versus SWA gather offsets.
- Chunk sizes 2 and 3: a partial final chunk and all requests in one chunk.
- Batched AITER versus independently executed requests; batched adapter versus
  batched AITER and independently executed requests. Independent execution resets
  request-local gathered KV offsets and provides an additional offset check.

This covers the prefill attention call chain, not metadata-builder correctness,
full-model logits, TP collectives, or TP4/TP8 end-to-end model accuracy.

## HCU run, 2026-10-12 UTC

Repository base: `5f4bfdb947d8155ba4934227bfae5a192bc77650`, plus the new
integration script. Installed vLLM version: `0.25.1`. Adapter import resolved to
`/home/zhoujie6/vllm-plugin-das-bw1000/vllm_hcu/patch/worker/core_fix/patch_deepseek_v4_rocm_flashmla_sparse.py`.

Command (run outside the device-isolated sandbox):

```bash
python tests/accuracy/check_dsv4_flashmla_prefill_integration_gpu.py
```

Result: exit code 0, `PREFILL INTEGRATION CHECKS PASSED`.
All 30 configurations (5 head layouts × 3 cache modes × 2 chunk sizes) and
90 comparison assertions passed. Every batched AITER/independent-request
comparison had maximum and mean error 0. The largest adapter/AITER absolute
error was 0.0078125, below the 0.015625 threshold. The same bound held against
independently executed requests. All outputs were finite.

The existing `check_dsv4_flashmla_prefill_padding_gpu.py` was also run on HCU at
the base commit in this session: all 60 cases passed, exit code 0 and
`PREFILL PADDING CHECKS PASSED`. Maximum absolute error was 0.015625; every
`valid_tail` case was at most 0.0078125. This separately tests legal indices
outside `topk_length` and internal negative-index masking.
