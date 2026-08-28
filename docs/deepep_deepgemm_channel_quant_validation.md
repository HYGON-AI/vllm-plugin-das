# DeepEP and DeepGEMM channel-quantized model validation

This document provides the server and client commands used to validate
channel-wise INT8 and FP8 MoE models on vLLM 0.25.1 with the HCU plugin.

## Environment

- Plugin checkout: the current repository root
- Installed vLLM: `0.25.1`
- INT8 model: `/models/GLM-5.2-Channel-INT8-w8a8`
- FP8 model: `/models/DeepSeek-V3.2-channel-fp8`
- Topology: TP1 / DP8 / EP8
- EvalScope: 1.10.0
- Eval dataset: ModelScope HumanEval, first 32 samples

Run the following setup from the plugin checkout or change `PLUGIN_ROOT` to
the worktree being validated:

```bash
export PLUGIN_ROOT="$(git rev-parse --show-toplevel)"
export PYTHONPATH="${PLUGIN_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
unset VLLM_USE_DEEP_GEMM
unset VLLM_PLUGINS
cd "${PLUGIN_ROOT}"
```

`--moe-backend deep_gemm` selects DeepGEMM directly, so
`VLLM_USE_DEEP_GEMM` is not required. Leave `VLLM_PLUGINS` unset so vLLM
loads the `hcu`, `hcu_model`, and `hcu_ops` entry points. Setting it to only
`hcu` excludes the HCU model and operator plugins.

Only start one of the following servers at a time. Each command uses all eight
visible HCUs.

## GLM-5.2 Channel-INT8 servers

High-throughput mode uses `m_grouped_i8_gemm_nt_contiguous` with
`marlin_i8_contiguous_weight`:

```bash
vllm serve /models/GLM-5.2-Channel-INT8-w8a8 \
  --port 10164 \
  --served-model-name glm52-int8-ht \
  --trust-remote-code \
  --max-model-len 4096 \
  --data-parallel-size 8 \
  --enable-expert-parallel \
  --all2all-backend deepep_high_throughput \
  --gpu-memory-utilization 0.9 \
  --max-num-batched-tokens 256 \
  --max-num-seqs 32 \
  --moe-backend deep_gemm \
  --default-chat-template-kwargs '{"enable_thinking":false}' \
  --enforce-eager \
  --disable-log-stats
```

Low-latency mode uses `m_grouped_i8_gemm_nt_masked` with
`marlin_i8_masked_weight`:

```bash
vllm serve /models/GLM-5.2-Channel-INT8-w8a8 \
  --port 10165 \
  --served-model-name glm52-int8-ll \
  --trust-remote-code \
  --max-model-len 4096 \
  --data-parallel-size 8 \
  --enable-expert-parallel \
  --all2all-backend deepep_low_latency \
  --gpu-memory-utilization 0.9 \
  --max-num-batched-tokens 256 \
  --max-num-seqs 32 \
  --moe-backend deep_gemm \
  --default-chat-template-kwargs '{"enable_thinking":false}' \
  --enforce-eager \
  --disable-log-stats
```

## DeepSeek-V3.2 Channel-FP8 servers

High-throughput mode uses `m_grouped_fp8_gemm_nt_contiguous` with
`marlin_fp8_contiguous_weight`:

```bash
vllm serve /models/DeepSeek-V3.2-channel-fp8 \
  --port 10174 \
  --served-model-name deepseek-v32-fp8-ht \
  --trust-remote-code \
  --max-model-len 4096 \
  --data-parallel-size 8 \
  --enable-expert-parallel \
  --all2all-backend deepep_high_throughput \
  --gpu-memory-utilization 0.9 \
  --max-num-batched-tokens 256 \
  --max-num-seqs 32 \
  --moe-backend deep_gemm \
  --default-chat-template-kwargs '{"enable_thinking":false}' \
  --enforce-eager \
  --disable-log-stats
```

Low-latency mode uses `m_grouped_fp8_gemm_nt_masked` with
`marlin_fp8_masked_weight`:

```bash
vllm serve /models/DeepSeek-V3.2-channel-fp8 \
  --port 10175 \
  --served-model-name deepseek-v32-fp8-ll \
  --trust-remote-code \
  --max-model-len 4096 \
  --data-parallel-size 8 \
  --enable-expert-parallel \
  --all2all-backend deepep_low_latency \
  --gpu-memory-utilization 0.9 \
  --max-num-batched-tokens 256 \
  --max-num-seqs 32 \
  --moe-backend deep_gemm \
  --default-chat-template-kwargs '{"enable_thinking":false}' \
  --enforce-eager \
  --disable-log-stats
```

The startup log should report `DeepEPHTPrepareAndFinalize` for HT or
`DeepEPLLPrepareAndFinalize` for LL. The explicit modes retain only their
selected packed weight layout. `deepep_auto` retains both layouts because it
can switch modes dynamically.

## Client health and generation requests

Set the port and served model name for the server under test. The following
example targets the FP8 low-latency server:

```bash
export SERVER_PORT=10175
export SERVED_MODEL=deepseek-v32-fp8-ll
```

Check server health and model registration:

```bash
curl --noproxy '*' -fsS "http://127.0.0.1:${SERVER_PORT}/health"
curl --noproxy '*' -fsS "http://127.0.0.1:${SERVER_PORT}/v1/models"
```

Send a deterministic OpenAI-compatible chat request:

```bash
curl --noproxy '*' -fsS \
  "http://127.0.0.1:${SERVER_PORT}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{
    \"model\": \"${SERVED_MODEL}\",
    \"messages\": [{
      \"role\": \"user\",
      \"content\": \"Return only the number: 2+3\"
    }],
    \"temperature\": 0,
    \"max_tokens\": 16,
    \"chat_template_kwargs\": {\"enable_thinking\": false}
  }"
```

The expected assistant content is `5`.

For the other servers, set the variables as follows:

| Mode | `SERVER_PORT` | `SERVED_MODEL` |
| --- | ---: | --- |
| GLM-5.2 INT8 HT | 10164 | `glm52-int8-ht` |
| GLM-5.2 INT8 LL | 10165 | `glm52-int8-ll` |
| DeepSeek-V3.2 FP8 HT | 10174 | `deepseek-v32-fp8-ht` |
| DeepSeek-V3.2 FP8 LL | 10175 | `deepseek-v32-fp8-ll` |

## HumanEval 32 client

The validated machine has EvalScope installed under
`/tmp/evalscope-target-v025`. Change `EVAL_WORK_DIR`, `SERVER_PORT`, and
`SERVED_MODEL` for each run. Keep `--eval-batch-size 1` when comparing with
the recorded accuracy results.

```bash
export SERVER_PORT=10175
export SERVED_MODEL=deepseek-v32-fp8-ll
export EVAL_WORK_DIR=/tmp/deepseek-v32-fp8-ll-humaneval32

env \
  -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  -u http_proxy -u https_proxy -u all_proxy \
  NO_PROXY=127.0.0.1,localhost \
  no_proxy=127.0.0.1,localhost \
  PYTHONPATH=/tmp/evalscope-target-v025 \
python -m evalscope.cli.cli eval \
  --model "${SERVED_MODEL}" \
  --model-id "${SERVED_MODEL}" \
  --api-url "http://127.0.0.1:${SERVER_PORT}/v1" \
  --api-key EMPTY \
  --eval-type openai_api \
  --datasets humaneval \
  --dataset-hub modelscope \
  --limit 32 \
  --eval-batch-size 1 \
  --generation-config '{"max_tokens":2048,"temperature":0,"do_sample":false,"extra_body":{"chat_template_kwargs":{"enable_thinking":false}}}' \
  --work-dir "${EVAL_WORK_DIR}" \
  --no-timestamp
```

The report is written to
`${EVAL_WORK_DIR}/reports/${SERVED_MODEL}/humaneval.json`. Verify that both
`mean_acc` and `mean_acc_pass@1` have `num: 32` before comparing scores.

## Recorded HumanEval results

| Model | Mode | Predictions/reviews | mean_acc | Pass@1 |
| --- | --- | ---: | ---: | ---: |
| GLM-5.2 Channel-INT8 | HT / contiguous | 32/32 | 1.0 | 1.0 |
| GLM-5.2 Channel-INT8 | LL / masked | 32/32 | 1.0 | 1.0 |
| DeepSeek-V3.2 Channel-FP8 | HT / contiguous | 32/32 | 1.0 | 1.0 |
| DeepSeek-V3.2 Channel-FP8 | LL / masked | 32/32 | 1.0 | 1.0 |

The DeepSeek-V3.2 FP8 HT run averaged 32.3817 seconds per request and 4.61
output tokens/s. The LL run averaged 25.5985 seconds per request and 5.91
output tokens/s.
