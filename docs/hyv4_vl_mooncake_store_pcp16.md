# HYV4-VL PCP16 with Mooncake Store

This HCU path adapts the native vLLM `MooncakeStoreConnector` for HYV4-VL
PCP16. The connector batches multi-buffer Store requests and can optionally
canonicalize PCP cache keys so one PCP replica writes the cache and the other
PCP ranks can read the same key.

## Supported configuration

- Architecture: `HYV4VLForConditionalGeneration`, compressed-tensors W4A8.
- Topology: PP1, TP1, PCP16, DP1, DCP1, expert parallelism enabled.
- MoE: DeepEP high-throughput all-to-all and DeepGEMM.
- KV connector: registered `MooncakeStoreConnector`, role `kv_both`, without a
  custom connector module path.
- KV cache: FP8 E4M3.
- Native checkpoint MTP is admitted for the direct Mooncake Store path when
  the HYV4 PCP MTP contract passes. Other PCP paths retain their existing
  validation.

## Single PCP replica

Set `pcp_single_replica: true` in
`kv_connector_extra_config` to use one Store copy across the PCP group. PCP
rank 0 remains the writer; other PCP ranks use the canonical PCP key namespace
to read the same stored keys. This option requires PCP size greater than 1.
Without it, the connector preserves the normal per-rank Store behavior.

The HCU adapter limits each native multi-buffer put/get call to 32 keys per
batch to avoid oversized native transport submissions.

## Validation record

On the single-node 16-card environment, PCP16 + direct Mooncake Store + native
MTP3 started successfully. A one-row cache probe used a 54,251-token prompt;
after clearing the GPU prefix cache, Mooncake memory returned 54,144 tokens
(99.80% external prefix hit rate), with 107 tokens recomputed and zero failed
Store keys. This was a single-request validation, not a full-dataset run.

The initial and replayed generation output hashes differed, so output
equivalence with MTP was not established by that probe.
