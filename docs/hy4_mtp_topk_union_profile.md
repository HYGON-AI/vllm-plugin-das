# HY4 MTP top-k union profiling

HY4 MTP queries belonging to one request may select overlapping sparse-KV
rows. The diagnostic profiler reports the number of unique selected rows
(`U`) for every request before logical indices are converted to physical cache
slots. Within one request that conversion is one-to-one, so it does not change
`U`.

Run the target workload in eager mode and enable a bounded sample:

```bash
export VLLM_HCU_HYV4_MTP_TOPK_UNION_PROFILE_STEPS=1
export VLLM_HCU_HYV4_MTP_TOPK_UNION_PROFILE_SKIP_STEPS=0
vllm serve ... --enforce-eager
```

Each sparse attention implementation reports one line such as:

```text
HY V4 MTP top-k union profile: layer=model.layers.1.self_attn.attn, \
query_lengths=[4, 4], \
valid_counts=[8192, 8192], union_sizes=[3072, 2944], \
aggregate_reuse_factor=2.7234
```

`union_sizes` contains `U` for each request. The reuse factor is the number of
valid selections divided by the number of unique selections. For MTP3 with
top-k 2048 it ranges from 1 (no sharing) to 4 (identical selections).

Model warmup may execute matching MTP-shaped batches. Increase
`VLLM_HCU_HYV4_MTP_TOPK_UNION_PROFILE_SKIP_STEPS` to skip those calls. Both
settings apply per sparse attention implementation, are non-negative, and
default to zero. Profiling copies indices to the CPU and synchronizes the
device; never enable it for performance measurements or production serving.
