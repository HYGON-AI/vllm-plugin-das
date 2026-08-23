# AITER quantized MoE validation on vLLM 0.25.1

This document records the validation of explicit `--moe-backend aiter`
support for compressed-tensors INT8 and channel-wise FP8 MoE models.

## Environment

- vLLM source: `/models/zb/vllm_025/vllm`
- Plugin branch: `fix/hcu-v0251-aiter-int8-quant-alignment` (PR 19)
- Model runner: V2 (`VLLM_USE_V2_MODEL_RUNNER=1`)
- KV cache layout: HND
- Graph mode: `FULL_AND_PIECEWISE` (eager mode was not enabled)
- EvalScope: 1.10.0
- Eval dataset: HumanEval, first 32 samples, batch size 1
- Generation: temperature 0, `max_tokens=32768`, thinking disabled

`VLLM_ROCM_USE_AITER` and `VLLM_ROCM_USE_AITER_MOE` are not needed. The
backend is selected explicitly with `--moe-backend aiter`.

`VLLM_HCU_USE_FLASH_ATTN_UNIFIED=1` selects the HCU unified attention path;
it does not select the MoE implementation.

## INT8 AITER server

```bash
export PLUGIN_ROOT=/models/zb/vllm_025_hcu/vllm-plugin-das
export HIP_VISIBLE_DEVICES=1
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_KV_CACHE_LAYOUT=HND
export VLLM_HCU_USE_FLASH_ATTN_UNIFIED=1
export PYTHONPATH="${PLUGIN_ROOT}"

vllm serve /models/Qwen3.5-35B-A3B-W8A8 \
  --served-model-name qwen35-int8-aiter \
  --tensor-parallel-size 1 \
  --max-model-len 65536 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.90 \
  --trust-remote-code \
  --moe-backend aiter \
  --port 8011
```

The startup log must contain all of the following:

- `Using AITER Int8 MoE backend`
- `cudagraph_mode=<CUDAGraphMode.FULL_AND_PIECEWISE`
- successful AITER W8A8 stage-1 and stage-2 ASM module loads
- completed PIECEWISE and FULL graph capture

## slimquant_marlin INT8 baseline

```bash
export PLUGIN_ROOT=/models/zb/vllm_025_hcu/vllm-plugin-das
export HIP_VISIBLE_DEVICES=3
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_KV_CACHE_LAYOUT=HND
export VLLM_HCU_USE_FLASH_ATTN_UNIFIED=1
export VLLM_CACHE_ROOT=/tmp/vllm-cache-qwen35-slimquant
export PYTHONPATH="${PLUGIN_ROOT}"

vllm serve /models/Qwen3.5-35B-A3B-W8A8 \
  --served-model-name qwen35-int8-slimquant \
  --tensor-parallel-size 1 \
  --max-model-len 65536 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.90 \
  --trust-remote-code \
  --quantization slimquant_marlin \
  --port 8012
```

## Channel-FP8 AITER server

```bash
export PLUGIN_ROOT=/models/zb/vllm_025_hcu/vllm-plugin-das
export HIP_VISIBLE_DEVICES=2
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_KV_CACHE_LAYOUT=HND
export VLLM_HCU_USE_FLASH_ATTN_UNIFIED=1
export VLLM_CACHE_ROOT=/tmp/vllm-cache-qwen35-fp8-aiter
export PYTHONPATH="${PLUGIN_ROOT}"

vllm serve /models/Qwen3.5-35B-A3B-CHANNEL-FP8 \
  --served-model-name qwen35-fp8-aiter \
  --tensor-parallel-size 1 \
  --max-model-len 65536 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.90 \
  --trust-remote-code \
  --moe-backend aiter \
  --port 8013
```

The startup log must contain `Using AITER Fp8 MoE backend`, successful FP8
W8A8 stage-1 and stage-2 ASM module loads, and completed PIECEWISE and FULL
graph capture.

## Channel-FP8 Triton baseline

```bash
export PLUGIN_ROOT=/models/zb/vllm_025_hcu/vllm-plugin-das
export HIP_VISIBLE_DEVICES=3
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_KV_CACHE_LAYOUT=HND
export VLLM_HCU_USE_FLASH_ATTN_UNIFIED=1
export VLLM_CACHE_ROOT=/tmp/vllm-cache-qwen35-fp8-triton
export PYTHONPATH="${PLUGIN_ROOT}"

vllm serve /models/Qwen3.5-35B-A3B-CHANNEL-FP8 \
  --served-model-name qwen35-fp8-triton \
  --tensor-parallel-size 1 \
  --max-model-len 65536 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.90 \
  --trust-remote-code \
  --moe-backend triton \
  --port 8014
```

## Client sanity request

Change the port and model name for the backend under test.

```bash
curl -sS http://127.0.0.1:8013/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen35-fp8-aiter",
    "messages": [{
      "role": "user",
      "content": "Write a Python function add(a, b) that returns the sum. Output code only."
    }],
    "temperature": 0,
    "max_tokens": 256,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

The AITER FP8 response completed normally, returned `reasoning=null`, and
contained valid Python without garbled text.

## EvalScope command

The tested machine used an isolated EvalScope installation under
`/tmp/evalscope-target-v025`. Change `--model`, `--model-id`, `--api-url`, and
`--work-dir` for each backend.

```bash
PYTHONPATH=/tmp/evalscope-target-v025 \
python3 -m evalscope.cli.cli eval \
  --model qwen35-int8-aiter \
  --model-id qwen35-int8-aiter \
  --api-url http://127.0.0.1:8011/v1 \
  --api-key EMPTY \
  --eval-type openai_api \
  --datasets humaneval \
  --dataset-hub modelscope \
  --limit 32 \
  --eval-batch-size 1 \
  --generation-config '{"temperature":0,"max_tokens":32768,"timeout":1800,"extra_body":{"chat_template_kwargs":{"enable_thinking":false}}}' \
  --seed 0 \
  --no-collect-perf \
  --work-dir /tmp/evalscope-qwen35-int8-aiter-32 \
  --no-timestamp
```

The 1800-second request timeout is necessary because the INT8 AITER result for
sample index 17 reaches the 32768-token generation limit at about 44 tokens/s.
EvalScope's shorter default request timeout expires just before that request
finishes.

## Results

| Model | MoE path | HumanEval pass@1 | Correct | Max output tokens | Wall time |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen3.5-35B-A3B-W8A8 | AITER W8A8 ASM | 0.7188 | 23/32 | 32768 | 885.88 s |
| Qwen3.5-35B-A3B-W8A8 | slimquant_marlin | 0.8438 | 27/32 | 261 | 154.69 s |
| Qwen3.5-35B-A3B-CHANNEL-FP8 | AITER FP8 W8A8 ASM | 0.8125 | 26/32 | 213 | 167.46 s |
| Qwen3.5-35B-A3B-CHANNEL-FP8 | AITER FP8 W8A8 ASM, vLLM SiLU | 0.8750 | 28/32 | 253 | 163.76 s |
| Qwen3.5-35B-A3B-CHANNEL-FP8 | Triton FP8 W8A8 | 0.8750 | 28/32 | 247 | 171.27 s |

No Unicode replacement characters were found in the recorded 32-sample
outputs. The INT8 AITER failure mode is a changed numerical/code-generation
trajectory, not text encoding corruption.

## Operator-level comparison with real checkpoint weights

The model's layer-0 expert weights and scales were loaded directly from the
checkpoint and AITER ASM was compared with AITER's Triton solution at M=1, 16,
and 128, with E=256, K=2048, N=512, and top-k=8.

| Quantization | M | max abs diff | mean abs diff | mean relative diff |
| --- | ---: | ---: | ---: | ---: |
| INT8 W8A8 | 1 | 0.001343 | 0.000294 | 4.47% |
| INT8 W8A8 | 16 | 0.002441 | 0.000335 | 5.32% |
| INT8 W8A8 | 128 | 0.035156 | 0.000532 | 3.32% |
| FP8 W8A8 | 1 | 0.000122 | 0.0000058 | 0.09% |
| FP8 W8A8 | 16 | 0.000732 | 0.0000137 | 0.21% |
| FP8 W8A8 | 128 | 0.017578 | 0.0000319 | 0.20% |

The INT8 weights, channel scales, and dynamic-token quantization metadata are
loaded correctly by the plugin. The remaining INT8 accuracy gap is consistent
with the larger numerical difference in the installed AITER INT8 ASM path and
is reproducible independently of vLLM graph capture and EvalScope.

## PR 19 FP8 precision revalidation

The graph-enabled V2 runner was revalidated on GPU 5 with the server and
EvalScope commands above. A fresh official Triton run scored 29/32 (0.9062),
while the unmodified AITER ASM path reproduced 26/32 (0.8125). The AITER
failures were `HumanEval/0`, `/1`, `/4`, `/18`, `/19`, and `/20`.
The retained implementation is code commit `555269b` on
`fix/hcu-v0251-aiter-int8-quant-alignment`.

| Fresh run | Score | Correct | Wall time |
| --- | ---: | ---: | ---: |
| AITER ASM before alignment | 0.8125 | 26/32 | 156.23 s |
| Official vLLM Triton | 0.9062 | 29/32 | 188.11 s |
| AITER ASM with scoped vLLM SiLU | 0.8750 | 28/32 | 163.76 s |

Operator diagnostics ruled out dynamic per-token FP8 quantization as the main
source of the gap. vLLM, AITER, and BoltOps produced equivalent scales (the
largest observed vLLM-to-AITER scale difference was approximately `1.86e-9`).
Replacing AITER's FP8 quantizer therefore did not improve the end-to-end score
and was not retained.

The retained change uses vLLM's canonical `_C.silu_and_mul` only while an
explicit channel-FP8 AITER ASM call is active. INT8, AITER MOE_C, and the
official Triton MoE path remain unchanged. The aligned AITER run scored 28/32
(0.8750), failed only `HumanEval/1`, `/4`, `/19`, and `/20`, and completed in
163.76 seconds. Its longest output was 253 tokens; no Unicode replacement
characters or NUL bytes were present.

The startup log confirmed `Using AITER Fp8 MoE backend`, loaded both FP8 W8A8
ASM stages, and completed PIECEWISE and FULL graph capture. Static regression
verification completed with:

```text
168 passed, 14 warnings in 64.14s
```
