# Decode-only benchmarking with packed FP8 caches

The HCU connector factory now maps the existing `DecodeBenchConnector` name to
the plugin implementation. This ports the FP8 fill fix and opt-in startup fill
from [vLLM PR #58472](https://github.com/vllm-project/vllm/pull/58472), adapted to
vLLM 0.25.1, and adds native cache writes for two packed layouts:

- `fp8_ds_mla`: 512 FP8 NoPE values, four FP32 scales, and 64 BF16 RoPE values
  (656 bytes per token). This covers HY4 and other models using that same
  DeepSeek V3.2 MLA layout.
- `DEEPSEEK_V32_INDEXER`: 128 FP8 key values and one FP32 scale per token
  (132 bytes per token of page storage). The installed native writer handles
  HCU value shuffling, block-tail scales, and UE8M0 scale rounding.

Plain floating-point caches and plain FP8 values stored as bytes are also
supported. Unknown packed layouts remain zero-filled with a warning. DeepSeek
V4's different packed MLA geometry and FP4 indexer layouts are not supported
by the new packed writers.

Example:

```bash
--kv-transfer-config '{"kv_connector":"DecodeBenchConnector","kv_role":"kv_both","kv_connector_extra_config":{"fill_mean":0.015,"fill_std":0.1,"fill_seed":0}}'
```

`fill_std > 0` samples Gaussian values in FP32 and then converts or writes
them through the native quantization operators. Nonzero constant filling is
also supported with `fill_std: 0`. Default mean and standard deviation remain
0.015 and 0.0; supported packed caches now hold valid nonzero synthetic history
under that default, rather than bytes truncated to zero.

`fill_seed` defaults to zero. Each layer, DP rank, and DCP rank has a separate
deterministic random stream, independent of the model's global RNG. Independent
DCP streams avoid creating identical synthetic key sequences on both ranks.
Random keys can reduce top-k ties, but do not guarantee equal selections per
query or reproduce real prefill's key distribution.

`startup_fill: true` fills all registered history once before serving and
skips per-request filling for eligible groups. Groups whose blocks the engine
zeroes remain filled per request. Circular buffers remain zero-filled to protect
embedded metadata. Sleep mode that discards caches requires restarting/refilling
before reusing this benchmark setup.

Temporary sampling and native-write inputs are chunked.
`fill_chunk_tokens` (default 16384) bounds the token chunk; the implementation
also limits chunks to approximately four million cache elements, except when a
single block/state row already exceeds that size.

Per-request metadata uses actual KV-cache groups and DCP-aware block spans.
The connector fills whole physical blocks covering the synthetic history,
including a partially occupied boundary block. Real decode writes replace
current-token slots before attention. Use disabled prefix caching for synthetic
benchmarking; partially overlapping shared prefix blocks are not safe to fill.

These caches are synthetic. Generated text and accuracy scores are not meaningful.

Validation:

```bash
python3 -m pytest -m 'not hcu' tests/kv_connector/test_decode_bench_connector.py
# Only on a released HCU:
python3 -m pytest -m hcu tests/kv_connector/test_decode_bench_connector.py
```
