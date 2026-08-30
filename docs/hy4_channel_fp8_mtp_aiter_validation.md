# Hy4 channel-FP8 MTP and AITER validation

This document records the TP8 graph-mode validation of
`/models/Hy4-preview-Channel-FP8-w8a8-v2` on plugin commit `1ee1b06` and
vLLM `0.25.1`. The four configurations differ only in MoE backend and whether
three Hy4 MTP draft tokens are enabled.

## Environment and server command

- Model runner: V2 (`VLLM_USE_V2_MODEL_RUNNER=1`)
- KV cache layout: NHD
- Tensor parallelism: 8; expert parallelism disabled
- Graph mode: `FULL_AND_PIECEWISE`, capture sizes 1 and 16
- EvalScope: 1.11.0
- Dataset: all 164 OpenAI HumanEval tasks
- Request concurrency: 16
- Generation: `temperature=0`, `top_p=1`, `seed=42`,
  `max_tokens=1024`, `reasoning_effort=no_think`

Use `MOE_BACKEND=triton` or `MOE_BACKEND=aiter`. Leave `SPEC_ARGS` empty for
the target-only baseline, or set it to the shown MTP configuration.

```bash
export PLUGIN_ROOT=/models/zb/hy4/.worktrees/hy-v4-mtp-blockwise-v0251-merge
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_KV_CACHE_LAYOUT=NHD
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=300
export VLLM_ENGINE_ITERATION_TIMEOUT_S=300
export PYTHONPATH="${PLUGIN_ROOT}"

MOE_BACKEND=triton
SPEC_ARGS=()
# For MTP3 instead:
# SPEC_ARGS=(--speculative-config '{"method":"mtp","num_speculative_tokens":3}')

vllm serve /models/Hy4-preview-Channel-FP8-w8a8-v2 \
  --served-model-name hy4-channel \
  --tensor-parallel-size 8 \
  --no-enable-expert-parallel \
  --moe-backend "${MOE_BACKEND}" \
  --gpu-memory-utilization 0.95 \
  --max-model-len 4096 \
  --max-num-seqs 16 \
  --max-num-batched-tokens 4096 \
  --default-chat-template-kwargs '{"reasoning_effort":"no_think"}' \
  --reasoning-parser hy_v4 \
  --enable-auto-tool-choice \
  --tool-call-parser hy_v4 \
  --compilation-config '{"cudagraph_capture_sizes":[1,16]}' \
  "${SPEC_ARGS[@]}" \
  --port 8000
```

Do not force `--attention-backend FLASH_ATTN_VARLEN` for this model. The
successful service automatically selects `FLASH_ATTN MLA prefill backend`,
which supplies the sparse MLA capability required by Hy4.

## EvalScope command

Change `--model-id` and `--work-dir` for each configuration. Do not pass
`--limit`.

```bash
env -u ALL_PROXY -u HTTP_PROXY -u HTTPS_PROXY \
    -u all_proxy -u http_proxy -u https_proxy \
python -m evalscope.cli.cli eval \
  --model hy4-channel \
  --model-id Hy4-preview-Channel-FP8-w8a8-v2-triton-mtpoff-no_think-164 \
  --api-url http://127.0.0.1:8000/v1 \
  --api-key EMPTY \
  --eval-type openai_api \
  --datasets humaneval \
  --dataset-hub modelscope \
  --eval-batch-size 16 \
  --generation-config '{"batch_size":16,"max_tokens":1024,"temperature":0.0,"top_p":1.0,"seed":42,"timeout":1800.0,"extra_body":{"chat_template_kwargs":{"reasoning_effort":"no_think"}}}' \
  --seed 42 \
  --work-dir /models/evalscope_hy4_channel_triton_mtpoff_humaneval164_20260830 \
  --no-timestamp
```

Each run produced exactly 164 prediction rows and 164 review rows.

## Full HumanEval results

| MoE backend | MTP draft tokens | Correct | pass@1 | Avg latency | Avg throughput | Avg output | Wall time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Triton | off | 134/164 | 81.7% | 75.178 s | 4.45 tok/s | 335 | 880.05 s |
| Triton | 3 | 132/164 | 80.5% | 37.276 s | 8.79 tok/s | 328 | 468.12 s |
| AITER MOE_C | off | 135/164 | 82.3% | 62.289 s | 4.99 tok/s | 311 | 718.40 s |
| AITER MOE_C | 3 | 130/164 | 79.3% | 36.096 s | 8.77 tok/s | 317 | 431.70 s |

Within Triton, MTP3 changes 9 tasks from pass to fail and 7 from fail to pass,
for a net change of -2/164 (-1.22 percentage points). Within AITER, it changes
7 tasks from pass to fail and 2 from fail to pass, for a net change of -5/164
(-3.05 percentage points). MTP therefore still lowers pass@1 in this run,
although the changes are bidirectional rather than a fixed set of corrupt
outputs.

### Triton: MTP off to MTP3

- Pass to fail: `HumanEval/1`, `/21`, `/54`, `/59`, `/70`, `/73`, `/93`,
  `/132`, `/156`
- Fail to pass: `HumanEval/67`, `/83`, `/99`, `/103`, `/125`, `/134`, `/144`
- Both pass: 125; both fail: 23; byte-identical predictions: 81/164

### AITER: MTP off to MTP3

- Pass to fail: `HumanEval/59`, `/83`, `/93`, `/108`, `/114`, `/116`, `/140`
- Fail to pass: `HumanEval/32`, `/91`
- Both pass: 128; both fail: 27; byte-identical predictions: 82/164

### Backend-only comparison with MTP disabled

Triton and AITER have 127 common passes and 22 common failures. Seven Triton
passes fail under AITER, while eight Triton failures pass under AITER, so the
AITER baseline is one task higher overall. Only 81/164 complete predictions
are byte-identical. This is a numerical trajectory change, not a backend
crash, NaN, or text corruption.

## MTP acceptance

| MoE backend | Accepted | Drafted | Overall | Position 0 | Position 1 | Position 2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Triton | 37,397 | 49,521 | 75.52% | 88.19% | 75.23% | 63.12% |
| AITER MOE_C | 36,099 | 48,000 | 75.21% | 87.53% | 74.89% | 63.20% |

The nearly identical acceptance profiles do not explain the extra three-task
net drop in the AITER+MTP3 run. MTP3 nevertheless cuts average request latency
by about 50% for Triton and 42% for AITER in this workload.

## Operator-level diagnosis

The real layer-1 channel-FP8 expert weights were loaded from the checkpoint,
sliced exactly as TP8 rank 0, and evaluated at `M=1,2,4,8,16,32`, `E=256`,
`K=6144`, `top_k=8`, BF16 compute, routed scaling 2.827, and Hy4's clamped
SwiGLU limit 10. AITER always selected deterministic `MOE_C`; repeated M=1
and M=16 calls were bitwise equal.

| Tokens | AITER vs Triton max abs | Mean abs | p99 abs | Fraction > 0.03 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.125000 | 0.025891 | 0.093750 | 50.37% |
| 2 | 0.062500 | 0.0000736 | 0 | 0.081% |
| 4 | 0.062500 | 0.003274 | 0.031250 | 5.58% |
| 8 | 0.125000 | 0.003200 | 0.062500 | 5.97% |
| 16 | 0.515625 | 0.007945 | 0.195312 | 6.49% |
| 32 | 0.515625 | 0.006757 | 0.117188 | 9.37% |

An independent check against AITER's own channel-FP8 golden implementation,
with unclamped SiLU so the reference contracts match, produced exact equality
at M=1. At M=16 the AITER-versus-golden maximum was 0.0078125 and mean was
`1.19e-7`; Triton-versus-golden mean was 0.007596. Thus the plugin's AITER
layout, scales, and public API calls are correct, and the installed MOE_C
kernel agrees with its reference algorithm. The AITER/Triton difference comes
from different dynamic FP8 quantization, accumulation, and reduction paths.

## Review decision

No production source change is justified by this investigation:

- AITER starts normally with TP8, graph mode, MTP off, and MTP3.
- Channel-wise weights, scales, shuffle layout, routed weights, output dtype,
  and clamped SwiGLU arguments are wired correctly.
- AITER is deterministic and matches its own golden implementation.
- AITER-off scores one task above Triton-off on the full dataset; therefore the
  earlier 15/16 versus 16/16 result was a small-sample effect, not evidence of
  a general AITER regression.
- MTP3 improves throughput substantially but has a measurable pass@1 tradeoff
  in both backends. Accuracy-sensitive deployments should leave MTP disabled;
  throughput-sensitive deployments should validate on their own workload.

Changing plugin kernels to force Triton-like outputs would be an alignment
policy or fallback mode, not a correctness fix. It should not be introduced
without an explicit product requirement and a wider accuracy/performance
evaluation.
