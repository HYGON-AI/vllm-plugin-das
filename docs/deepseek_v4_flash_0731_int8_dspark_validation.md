# DeepSeek-V4-Flash-0731 Channel-INT8 DSpark validation

## Scope

This profile targets
`/models/DeepSeek-V4-Flash-0731-Channel-INT8-w8a8` on one eight-card BW1100
node. It supports TP8+DSpark and single-service DP8+EP8+DSpark. PCP+DSpark and
prefill/decode disaggregation are outside this profile.

The public CLI follows vLLM's DeepSeek-V4 flow. It does not expose a separate
MoE backend, P/D role, throughput, or latency switch. DP+EP selects
`deepep_auto` once; the HCU backend uses contiguous DeepGEMM for HT forwards
and masked DeepGEMM for LL forwards. Channel-INT8 uses the public
`m_grouped_i8_gemm_nt_contiguous`/`marlin_i8_contiguous_weight` and
`m_grouped_i8_gemm_nt_masked`/`marlin_i8_masked_weight` pairs. LightOp performs
the `swiglu_limit=10.0` activation and dynamic per-token INT8 quantization.

## Server commands

TP8+DSpark:

```bash
vllm serve /models/DeepSeek-V4-Flash-0731-Channel-INT8-w8a8 \
  --host 0.0.0.0 \
  --port 10138 \
  --trust-remote-code \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --tokenizer-mode deepseek_v4 \
  --distributed-executor-backend mp \
  --max-model-len 4096 \
  --max-num-batched-tokens 512 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.9 \
  --tensor-parallel-size 8 \
  --served-model-name DeepSeek-V4-Flash-0731-Channel-INT8-w8a8 \
  --speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}'
```

DP8+EP8+DSpark:

```bash
vllm serve /models/DeepSeek-V4-Flash-0731-Channel-INT8-w8a8 \
  --host 0.0.0.0 \
  --port 10139 \
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
  --data-parallel-size 8 \
  --enable-expert-parallel \
  --all2all-backend deepep_auto \
  --served-model-name DeepSeek-V4-Flash-0731-Channel-INT8-w8a8 \
  --speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}'
```

Neither command needs `--enforce-eager` or `--moe-backend`. The DP profile
uses the same automatic graph fallback as the FP8 profile.

## Client commands

TP8 service:

```bash
curl --noproxy '*' http://127.0.0.1:10138/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "DeepSeek-V4-Flash-0731-Channel-INT8-w8a8",
    "messages": [{"role": "user", "content": "Write a Python hello-world program."}],
    "temperature": 0,
    "max_tokens": 128,
    "stream": false,
    "chat_template_kwargs": {"thinking": false}
  }'
```

Use the same request against port `10139` for the DP8+EP8 service:

```bash
curl --noproxy '*' http://127.0.0.1:10139/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "DeepSeek-V4-Flash-0731-Channel-INT8-w8a8",
    "messages": [{"role": "user", "content": "Write a Python hello-world program."}],
    "temperature": 0,
    "max_tokens": 128,
    "stream": false,
    "chat_template_kwargs": {"thinking": false}
  }'
```

## Reproducible gates

```bash
VLLM_HCU_DEEPSEEK_V4_FLASH_0731_INT8_MODEL=/models/DeepSeek-V4-Flash-0731-Channel-INT8-w8a8 \
python -m pytest -q -s \
  tests/integration/spec_decode/test_deepseek_v4_flash_dspark.py \
  -k deepseek_v4_flash_int8_dspark_tp8

VLLM_HCU_DEEPSEEK_V4_FLASH_0731_INT8_MODEL=/models/DeepSeek-V4-Flash-0731-Channel-INT8-w8a8 \
python -m pytest -q -s \
  tests/integration/spec_decode/test_deepseek_v4_flash_dspark.py \
  -k deepseek_v4_flash_int8_dspark_dp8_ep8

VLLM_HCU_DEEPSEEK_V4_FLASH_0731_INT8_MODEL=/models/DeepSeek-V4-Flash-0731-Channel-INT8-w8a8 \
python -m pytest -q -s \
  tests/integration/server/test_evalscope_deepseek_v4_dspark_humaneval.py \
  -k deepseek_v4_int8_dspark_humaneval_tp8

VLLM_HCU_DEEPSEEK_V4_FLASH_0731_INT8_MODEL=/models/DeepSeek-V4-Flash-0731-Channel-INT8-w8a8 \
python -m pytest -q -s \
  tests/integration/server/test_evalscope_deepseek_v4_dspark_humaneval.py \
  -k deepseek_v4_int8_dspark_humaneval_dp8_ep8
```

HumanEval acceptance requires exactly 32 predictions and 32 reviews. The
repository-owned normalization reruns all completions after removing one
complete or truncated Markdown fence; normalized `mean_acc` and
`mean_acc_pass@1` must both equal `1.0`.

## Current observed status (2026-08-29)

- TP8+DSpark smoke passed in 148.46 seconds. DP8+EP8+DSpark smoke passed in
  141.79 seconds on the local 48-shard, 286.61 GiB checkpoint.
- The DP log recorded construction and execution of both HT contiguous and LL
  masked experts under the single `deepep_auto` selector.
- The combined gfx938 operator suite passed 10/10 cases, including INT8
  contiguous/masked DeepGEMM and clamped LightOp INT8 quantization.
- TP8 HumanEval produced 32 predictions and 32 reviews: raw accuracy/pass@1
  was 29/32 (`0.9062`), and normalized accuracy/pass@1 was 32/32 (`1.0000`).
  The end-to-end pytest gate passed in 367.53 seconds; EvalScope reported
  2.428-second average latency and 49.21 output tokens/s.
- DP8+EP8 HumanEval produced 32 predictions and 32 reviews: raw
  accuracy/pass@1 was 29/32 (`0.9062`), and normalized accuracy/pass@1 was
  32/32 (`1.0000`). The end-to-end pytest gate passed in 545.06 seconds;
  EvalScope reported 4.061-second average latency and 29.98 output tokens/s.
