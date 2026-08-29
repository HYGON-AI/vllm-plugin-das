# DeepSeek-V4-Flash-0731 DSpark Mooncake P/D validation

## Scope and compatibility

This profile adds single-node 4P+4D prefill/decode disaggregation for both:

- `/models/DeepSeek-V4-Flash-0731-Channel-FP8-w8a8`
- `/models/DeepSeek-V4-Flash-0731-Channel-INT8-w8a8`

It follows vLLM's merged DeepSeek-V4 MLA Mooncake support in
[vllm-project/vllm#46807](https://github.com/vllm-project/vllm/pull/46807).
The plugin already carried the group-aware Mooncake/MLA data path; the missing
piece was a conservative HCU configuration guard that rejected every DSpark
configuration with `kv_transfer_config`. The guard now permits only
`DeepseekV4ForCausalLM` + `MooncakeConnector`. PCP+P/D and non-Mooncake DSpark
connectors remain rejected.

The public MoE selection remains one `--all2all-backend deepep_auto` argument.
There are no public high-throughput, low-latency, or `--moe-backend` switches.
The HCU runtime selects contiguous DeepEP/DeepGEMM for high-throughput forwards
and masked DeepEP/DeepGEMM for low-latency forwards. In Mooncake P/D mode the
role fixes that internal choice: the producer retains only the contiguous
Marlin layout and the consumer retains only the masked layout. This makes DP4
fit without adding a role-specific public MoE option; ordinary single-service
`deepep_auto` still retains both layouts and selects per forward. Both services
use `--kv-cache-dtype fp8`, including the Channel-INT8 checkpoint.

## Choose the model

FP8:

```bash
export MODEL="${VLLM_HCU_DEEPSEEK_V4_FLASH_0731_MODEL:-/models/DeepSeek-V4-Flash-0731-Channel-FP8-w8a8}"
export SERVED_MODEL=DeepSeek-V4-Flash-0731-Channel-FP8-w8a8
```

INT8:

```bash
export MODEL="${VLLM_HCU_DEEPSEEK_V4_FLASH_0731_INT8_MODEL:-/models/DeepSeek-V4-Flash-0731-Channel-INT8-w8a8}"
export SERVED_MODEL=DeepSeek-V4-Flash-0731-Channel-INT8-w8a8
```

Run the following P, D, and proxy commands in separate shells after choosing
one model. The two DP services use distinct RPC, rendezvous, and Mooncake
bootstrap ports so they can coexist on one node.

## Prefill service: cards 0-3

```bash
export HIP_VISIBLE_DEVICES=0,1,2,3
export VLLM_MOONCAKE_BOOTSTRAP_PORT=18998
export VLLM_DP_MASTER_IP=127.0.0.1
export VLLM_DP_MASTER_PORT=29561
export VLLM_HCU_MOONCAKE_TTFT_TRACE=1
export VLLM_LOGGING_LEVEL=DEBUG
export VLLM_HCU_USE_FLASH_ATTN_UNIFIED=1

vllm serve "$MODEL" \
  --trust-remote-code \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --tokenizer-mode deepseek_v4 \
  --distributed-executor-backend mp \
  --max-model-len 4096 \
  --max-num-batched-tokens 512 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.9 \
  --tensor-parallel-size 1 \
  --data-parallel-size 4 \
  --enable-expert-parallel \
  --all2all-backend deepep_auto \
  --speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}' \
  --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_producer","kv_connector_extra_config":{"mooncake_protocol":"rdma"}}' \
  --served-model-name "$SERVED_MODEL" \
  --port 10141 \
  --data-parallel-rpc-port 29551
```

## Decode service: cards 4-7

```bash
export HIP_VISIBLE_DEVICES=4,5,6,7
export VLLM_MOONCAKE_BOOTSTRAP_PORT=18999
export VLLM_DP_MASTER_IP=127.0.0.1
export VLLM_DP_MASTER_PORT=29562
export VLLM_HCU_MOONCAKE_TTFT_TRACE=1
export VLLM_LOGGING_LEVEL=DEBUG
export VLLM_HCU_USE_FLASH_ATTN_UNIFIED=1

vllm serve "$MODEL" \
  --trust-remote-code \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --tokenizer-mode deepseek_v4 \
  --distributed-executor-backend mp \
  --max-model-len 4096 \
  --max-num-batched-tokens 64 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.9 \
  --tensor-parallel-size 1 \
  --data-parallel-size 4 \
  --enable-expert-parallel \
  --all2all-backend deepep_auto \
  --speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}' \
  --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_consumer","kv_connector_extra_config":{"mooncake_protocol":"rdma"}}' \
  --served-model-name "$SERVED_MODEL" \
  --port 10142 \
  --data-parallel-rpc-port 29552
```

The decode pool uses 64 batched tokens because DeepEP LL capacity is
`max_num_seqs * (1 + num_speculative_tokens)` (8 * 8). This also keeps the
vLLM startup profiling forward inside the same validated LL dispatch capacity;
it is a standard vLLM scheduler option, not a public HT/LL backend switch.

## Official Mooncake proxy

```bash
export VLLM_V0251_SOURCE_ROOT=/models/zb/vllm_025/vllm

python3 \
  "$VLLM_V0251_SOURCE_ROOT/examples/disaggregated/mooncake_connector/mooncake_connector_proxy.py" \
  --prefill http://127.0.0.1:10141 18998 \
  --decode http://127.0.0.1:10142 \
  --host 127.0.0.1 \
  --port 10140
```

## Client smoke request

```bash
curl --noproxy '*' http://127.0.0.1:10140/v1/completions \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"${SERVED_MODEL}\",\"prompt\":\"Write a Python hello-world program.\",\"temperature\":0,\"max_tokens\":128}"
```

## Automated HumanEval-32 acceptance

The runner owns only the P, D, and proxy process trees it creates. It starts P,
waits for health, starts D, waits for health, starts the proxy, retries a real
`/v1/completions` route while the proxy discovers P-side DP engines, then runs
EvalScope through port 10140. Cleanup is always proxy, D, then P.

FP8:

```bash
VLLM_V0251_SOURCE_ROOT=/models/zb/vllm_025/vllm \
PYTHONPATH=. \
pytest -q -s \
  tests/integration/server/test_evalscope_deepseek_v4_dspark_mooncake_pd.py::test_deepseek_v4_dspark_humaneval_fp8_mooncake_pd
```

INT8:

```bash
VLLM_V0251_SOURCE_ROOT=/models/zb/vllm_025/vllm \
PYTHONPATH=. \
pytest -q -s \
  tests/integration/server/test_evalscope_deepseek_v4_dspark_mooncake_pd.py::test_deepseek_v4_dspark_humaneval_int8_mooncake_pd
```

The acceptance gate requires exactly 32 predictions and reviews, normalized
HumanEval accuracy and pass@1 of 32/32, successful Mooncake send/receive TTFT
events, both contiguous HT and masked LL DeepEP/DeepGEMM markers, no known
Mooncake transfer failure, and positive DSpark draft and accepted token totals.
`VLLM_V0251_SOURCE_ROOT` locates only the official proxy script. Do not add that
source tree to `PYTHONPATH`: doing so hides the compiled extensions in the
installed vLLM 0.25.1 wheel.

## Observed results

Fresh FP8 and INT8 4P+4D results are recorded here after the live HCU runs.
