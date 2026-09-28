# LightOp AutoAWQ validation

## Scope

This backend is opt-in and uses only the public categorized LightOp APIs
`lightop.gemm_ops.awq_gemm_marlin_weight_repack` and
`lightop.gemm_ops.gemm_awq_w4a16_marlin`. It is selected only for FP16,
4-bit zero-point AutoAWQ layers with group size 128 and a tuned `(K, N)`
shape. Every other layer retains the original vLLM 0.28.1 method and weight
layout.

Enable it with:

```bash
export VLLM_HCU_USE_LIGHTOP_AWQ=1
```

`VLLM_HCU_USE_CUSTOM_OPS=0` remains a master opt-out. The LightOp AWQ flag is
disabled by default until model-level coverage is available.

## HCU operator check

The following command was run on 2026-09-29:

```bash
HIP_VISIBLE_DEVICES=1 pytest -q -s \
  tests/accuracy/test_lightop_autoawq_accuracy.py --disable-warnings
```

The test uses FP16, `K=4096`, `N=8192`, group size 128, and compares the
converted public-LightOp path with both the current v0.28.1 Triton AWQ GEMM
and dequantized FP16 matmul.

| M | v0.28.1 AWQ (ms) | LightOp (ms) | Speedup | Max error vs v0.28.1 | Max error vs dequant |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.211448 | 0.027154 | 7.787x | 0.00018311 | 0.00006104 |
| 2 | 0.210513 | 0.027513 | 7.652x | 0.00018311 | 0.00003052 |
| 8 | 0.211141 | 0.031693 | 6.662x | 0.00024414 | 0.00006104 |
| 16 | 0.215892 | 0.034624 | 6.235x | 0.00024414 | 0.00012207 |
| 32 | 0.296156 | 0.069454 | 4.264x | 0.00024414 | 0.00012207 |
| 64 | 0.646547 | 0.137539 | 4.701x | 0.00024414 | 0.00012207 |
| 128 | 1.158311 | 0.197223 | 5.873x | 0.00024414 | 0.00000000 |

Result: `1 passed`. Timings are single-process kernel measurements on the
available HCU and are intended to verify this exact software and shape
combination rather than guarantee service throughput.

## Model inventory and service command

No model whose `config.json` declares an AWQ quantization method was found
under `/model`, `/models`, or `/data/models` to depth four. Consequently, no
model-level accuracy result is claimed for this change.

Use the following command for a compatible AutoAWQ checkpoint. Keep the
model dtype at FP16; BF16 deliberately uses the existing v0.28.1 backend.

```bash
export VLLM_HCU_USE_LIGHTOP_AWQ=1

HIP_VISIBLE_DEVICES=0 vllm serve /path/to/autoawq-model \
  -tp 1 \
  --dtype float16 \
  --trust-remote-code \
  --max-model-len 32768 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.9
```

After startup, bypass configured proxies for local probes:

```bash
curl --noproxy '*' http://127.0.0.1:8000/health
curl --noproxy '*' http://127.0.0.1:8000/v1/models
```
