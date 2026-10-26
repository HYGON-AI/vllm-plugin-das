# Step5 prefill from installed wheels

This profile reproduces the fixed E23 DeepEP NORMAL MNVL path on four hosts
with four gfx938 GPUs each. The measured SM16 profile uses dispatch/combine
chunks of 8/8, two independent DBO communication buffers, fused dispatch
scatter, and unchanged vendor DeepGEMM WASP kernels. Its previous three-round
mean was 46,827.1 input tokens/s. The deferred-Gather prototype is excluded.

Install the matching plugin wheel and supplied `deep-ep` MNVL wheel normally
in a dedicated container. A virtual environment is optional. The MNVL-only
dependency excludes RDMA and low-latency modes. Do not replace a general
purpose environment with it. No source-directory `PYTHONPATH`, `LD_PRELOAD`,
or standalone extension override is required.

Machine tuning and expert placement files stay outside this repository.
The release assets contain the fixed E23 token-hash replica map, original
E22 placement, tokenizer, Dense tuning configuration, and their manifest.
The launcher checks asset hashes and the installed DeepEP extension.

On each host, use the same committed launcher with a fresh state directory:

```sh
python3 /opt/step5-release/source/benchmarks/step5/serve_deepep_prefill.py \
  --rank NODE_RANK --head-ip HEAD_IP --node-ip LOCAL_IP \
  --assets /opt/step5-release/assets \
  --state /var/lib/step5-release/run \
  --cache /var/cache/step5-release
```

Ranks are 0 through 3. The fixed workload is DP16/TP1/EP16, 64 concurrent
requests, 65,536 input tokens and one output token, batch-token budget 8192,
model limit 70000, FP8 KV cache, memory utilization 0.875, and disabled prefix
caching. Synthetic MTP3 uses `[0.838992, 0.598322, 0.362686]` exactly.

After the service is ready, copy all four fresh startup logs to the head
node and collect actual initialized KV capacities:

```sh
python3 /opt/step5-release/source/benchmarks/step5/collect_initialized_capacities.py \
  --profile prefill-e23 --logs node0.log node1.log node2.log node3.log \
  --output /var/lib/step5-release/run/startup-capacities.json
python3 /opt/step5-release/source/benchmarks/step5/measure_deepep_prefill.py \
  --client /opt/step5-release/client/prefill-e23 \
  --capacities /var/lib/step5-release/run/startup-capacities.json \
  --state /var/lib/step5-release/run --label fresh-release
```

The measurement script runs one excluded full warmup and three formal
rounds using the delivered benchmark unchanged. It requires all 64 requests
to complete without preemption. Profiling is disabled. Synthetic acceptance
is for throughput measurement; GSM8K quality results use standard MTP.
