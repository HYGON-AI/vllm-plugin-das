# AITER quantized MoE validation on vLLM 0.25.1

This document records the validation of explicit `--moe-backend aiter`
support for compressed-tensors INT8 and channel-wise FP8 MoE models.

## Environment

- vLLM source: `/models/zb/vllm_025/vllm`
- AITER: `0.1.5+das185.dtk2604.torch2110.2608180853.g40a705`
- Audited AITER activation ABI: `_apply_activation(activation, is_gated,
  activated_out, ffn1_out_2d, gemm1_alpha, gemm1_limit)`
- Plugin branch: `fix/hcu-v025-aiter-w8a8-quantized`
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

The 1800-second request timeout is required only to reproduce the original
AITER-SiLU baseline: its sample index 17 reaches the 32768-token generation
limit at about 44 tokens/s. The aligned run's longest output is 528 tokens.

## Results

| Model | MoE path | HumanEval pass@1 | Correct | Max output tokens | Wall time |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen3.5-35B-A3B-W8A8 | AITER W8A8 ASM, original AITER SiLU | 0.7188 | 23/32 | 32768 | 885.88 s |
| Qwen3.5-35B-A3B-W8A8 | AITER W8A8 ASM, vLLM-compatible SiLU | 0.9062 | 29/32 | 528 | 175.44 s |
| Qwen3.5-35B-A3B-W8A8 | slimquant_marlin | 0.8438 | 27/32 | 261 | 154.69 s |
| Qwen3.5-35B-A3B-CHANNEL-FP8 | AITER FP8 W8A8 ASM | 0.8125 | 26/32 | 213 | 167.46 s |
| Qwen3.5-35B-A3B-CHANNEL-FP8 | Triton FP8 W8A8 | 0.8750 | 28/32 | 247 | 171.27 s |

No Unicode replacement characters or NUL bytes were found in any of the
32-sample outputs. The aligned INT8 run returned no request errors and failed
only HumanEval indices 4, 19, and 20. The previous 32768-token runaway output
was eliminated.

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
loaded correctly by the plugin. Controlled stage capture showed that AITER ASM
and vLLM's true-W8A8 Triton operator agree at GEMM1, while the installed
AITER Triton SiLU implementation introduces the first material difference.

The plugin therefore replaces the intermediate activation only while an
explicit INT8-W8A8 AITER ASM call is active. It uses vLLM's canonical
`_C.silu_and_mul` operator under a context-local guard. FP8, W16A16, other
AITER solution types, and the official vLLM Triton MoE path retain their
original behavior.

The same real layer-0 weights were then compared against the unmodified vLLM
Triton operator forced to true W8A8 at operator level. NMAE is normalized by
the reference output mean absolute value.

| M | Original AITER vs vLLM NMAE | Aligned AITER vs vLLM NMAE |
| ---: | ---: | ---: |
| 1 | 1.12132% | 0.00000% |
| 16 | 0.88458% | 0.38930% |
| 128 | 0.88683% | 0.26516% |

The remaining M=16/128 difference comes after the activation boundary from
GEMM2 and expert-combine numerical ordering. The graph-enabled service
completed both PIECEWISE and FULL capture, served a normal request with
`reasoning=null`, and completed the aligned 32-sample evaluation in 175.44
seconds.

Static verification after this alignment completed with `159 passed` and the
same 14 pre-existing deprecation warnings:

```bash
python3 -m compileall -q vllm_hcu tests/runtime_patch
VLLM_V0251_SOURCE_ROOT=/models/zb/vllm_025/vllm \
python3 -m pytest -q \
  tests/runtime_patch/test_quant_gemm_aiter.py \
  tests/runtime_patch/test_moe_deepep.py \
  tests/runtime_patch/test_platform_hcu_config.py
```
