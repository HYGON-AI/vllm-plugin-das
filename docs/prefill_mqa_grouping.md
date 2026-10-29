# Experimental prefill MQA request grouping

The eager FP8 sparse-indexer path can partition consecutive requests before
running MQA and prefill TopK. This reduces the width and lifetime of temporary
logits matrices while retaining the existing LightOp kernels and request-relative
indices. CPU request lengths are planned once when chunk metadata is built.

The option is **disabled by default**:

```bash
export VLLM_HCU_ENABLE_PREFILL_MQA_GROUPING=1
export VLLM_HCU_PREFILL_MQA_GROUP_MIN_SAVED_CELLS=2000000
```

The second setting expresses the cost of an additional launch in equivalent
logit cells. A larger value discourages splitting. It is a heuristic, not a
latency guarantee: synthetic single-device tests found both improvements and
regressions depending on request sizes and query counts.

The optimized path is restricted to eager gfx938 FP8 prefill, LightOp dispatch,
uncompressed indexer metadata and DCP=1. Decode, graph execution/capture,
DCP>1 and unsupported or absent plans retain the existing implementation.
PCP16/DCP1 is distinct from DCP16: request-relative indices must be preserved
for the existing attention consumers, and the DCP score-exchange path must
retain its original logits matrix.

Model accuracy and actual serving TTFT/throughput must be checked for a target
configuration before enabling this experimental option. A reduction in the
standalone MQA/TopK time is not an end-to-end RPS claim.

The contiguous grouping cost model is adapted from
[sglang-das d50074e](https://github.com/maxiaobetter/sglang-das/commit/d50074e6a6942df5c91da0076acf273c977738cc),
licensed under Apache-2.0.
