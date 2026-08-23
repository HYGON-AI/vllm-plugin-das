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
| Qwen3.5-35B-A3B-CHANNEL-FP8 | AITER FP8 W8A8 ASM, native | 0.8125 | 26/32 | 213 | 167.46 s |
| Qwen3.5-35B-A3B-CHANNEL-FP8 | AITER FP8 W8A8 ASM, aligned | 0.8750 | 28/32 | 240 | 158.13 s |
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

| Fresh run | Score | Correct | Wall time |
| --- | ---: | ---: | ---: |
| AITER ASM | 0.8125 | 26/32 | 156.23 s |
| Official vLLM Triton | 0.9062 | 29/32 | 188.11 s |

The isolated SiLU test does not establish an AITER correctness defect. A fresh
run with uniformly distributed BF16 inputs measured about 0.141% NMAE for
AITER versus FP32 and about 0.193% for vLLM versus FP32. The two implementations
are numerically different, but AITER was not less accurate against that
reference.

A second diagnostic loaded the model's real layer-0 expert weights and captured
input quantization, GEMM1, gated activation, bridge quantization, GEMM2, and
expert reduction. A complete two-variable ablation compared the activation and
second per-token FP8 quantizer independently:

| AITER activation | Second FP8 quantizer | M=1 output NMAE | M=16 | M=128 |
| --- | --- | ---: | ---: | ---: |
| native | native | 1.80010% | 1.17462% | 1.24115% |
| native | vLLM | 1.80010% | 1.16848% | 1.22752% |
| vLLM | native | 0.09349% | 0.08673% | 0.10758% |
| vLLM | vLLM | 0% | 0% | 0.02732% |

This proves only numerical alignment with the official Triton path, not that
AITER's SiLU is incorrect. The retained production policy therefore preserves
AITER's native gated-SiLU implementation. Only the immediately following FP8
bridge quantization uses vLLM's `scaled_fp8_quant`; the first input
quantization remains AITER-native. The quantizer substitution is scoped to the
exact combination `FP8 W8A8 + ASM + gated SiLU`. INT8, MOE_C, GELU, non-gated
activation, and unsupported alpha/limit variants retain their existing
implementations. The compatibility wrappers validate the installed AITER call
signatures and fail closed if the ABI changes.

The staged diagnostic can be reproduced with:

```bash
HIP_VISIBLE_DEVICES=5 \
PYTHONPATH=/models/zb/vllm_025_hcu/vllm-plugin-das \
python3 tests/accuracy/diagnose_aiter_fp8_moe_stages.py \
  --model /models/Qwen3.5-35B-A3B-CHANNEL-FP8 \
  --tokens 1 16 128 \
  --aiter-activation native \
  --aiter-quant2 vllm \
  --output /tmp/pr19-fp8-stage-native-silu-vllm-quant2.json
```

The diagnostic's manual quant2 option isolates the same quantizer used by the
production scoped wrapper while leaving AITER activation native.

For historical context, the experiment that replaced both SiLU and quant2 was
repeated twice on one graph-enabled V2 server. Both runs produced
byte-identical generated code, scored 28/32 (0.875), and failed the same cases:
`HumanEval/4`, `/19`, `/20`, and `/27`. This result motivated the staged
diagnostic but is not evidence that AITER's native SiLU is incorrect, and that
SiLU replacement is no longer retained. No Unicode replacement characters or
NUL bytes were present.

The startup log confirmed `Using AITER Fp8 MoE backend`, loaded both FP8 W8A8
ASM stages, and completed PIECEWISE and FULL graph capture. Static regression
verification completed with:

```text
174 passed, 14 warnings in 63.40s
```

### Native-SiLU retained-policy validation on latest v0.25.1

The retained policy was also applied without conflicts on top of plugin
`v0.25.1@47d3fb884fdb9e4f03d1ddc993af5d31f18cc865` and tested against vLLM
`7b108ad1a51b217e9abec0ddc047978405481bae`. The service used Model Runner V2,
FULL and PIECEWISE graph capture, NHD KV-cache layout, parameterized
`FLASH_ATTN_VARLEN`, and parameterized AITER MoE. No legacy AITER or unified-FA
environment gate was set.

```bash
export HIP_VISIBLE_DEVICES=5
export PYTHONPATH=/models/zb/vllm_025_hcu/vllm-plugin-das
export VLLM_CACHE_ROOT=/tmp/vllm-cache-pr19-native-silu-varlen-aiter
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_KV_CACHE_LAYOUT=NHD
unset VLLM_HCU_USE_FLASH_ATTN_UNIFIED
unset VLLM_HCU_USE_FLASH_ATTN_VARLEN
unset VLLM_ROCM_USE_AITER
unset VLLM_ROCM_USE_AITER_MOE

python3 -m vllm.entrypoints.openai.api_server \
  --model /models/Qwen3.5-35B-A3B-CHANNEL-FP8 \
  --served-model-name qwen35-fp8-varlen-aiter-native-silu \
  --port 8016 \
  --trust-remote-code \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.9 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 8 \
  --attention-backend FLASH_ATTN_VARLEN \
  --moe-backend aiter
```

```bash
env -u ALL_PROXY -u HTTP_PROXY -u HTTPS_PROXY \
    -u all_proxy -u http_proxy -u https_proxy \
  PYTHONPATH=/tmp/evalscope-target-v025 \
  python3 -m evalscope.cli.cli eval \
    --model qwen35-fp8-varlen-aiter-native-silu \
    --model-id qwen35-fp8-varlen-aiter-native-silu \
    --api-url http://127.0.0.1:8016/v1 \
    --api-key EMPTY \
    --eval-type openai_api \
    --datasets humaneval \
    --dataset-hub modelscope \
    --limit 32 \
    --eval-batch-size 1 \
    --generation-config '{"temperature":0,"max_tokens":32768,"timeout":1800,"extra_body":{"chat_template_kwargs":{"enable_thinking":false}}}' \
    --seed 0 \
    --no-collect-perf \
    --work-dir /tmp/evalscope-pr19-native-silu-varlen-aiter-32 \
    --no-timestamp
```

The run scored 32/32 (`mean_acc=1.0`, `pass@1=1.0`) with no request errors,
Unicode replacement characters, or NUL bytes. Mean recorded request latency
was 1.736 s (minimum 0.372 s, maximum 3.368 s), and the 32 responses contained
3,595 output tokens. The output-content SHA-256 was
`636736abda369ce7a41f9ddbb74aa199b066591a4f97134f4d166dd48f1bd495`.
The startup log confirmed the `FLASH_ATTN_VARLEN` argument, selected the AITER
FP8 MoE backend, loaded both AITER FP8 W8A8 ASM stages, and completed graph
capture. The scoped regression suite passed with:

```text
138 passed, 18 warnings in 36.38s
```
