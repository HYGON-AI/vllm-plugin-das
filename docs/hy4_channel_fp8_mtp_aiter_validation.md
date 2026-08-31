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

## Opt-in Channel-FP8 to INT8-W8A16 path (2026-08-31)

The HCU plugin can optionally requantize each already TP-local Channel-FP8
expert matrix to symmetric INT8 per output channel and execute it with BF16
activations through the AITER-owned BoltOps W8A16 path. This avoids the eager
AITER FP8 weight shuffle, keeps the standard expert layout, and leaves the
existing FP8 route unchanged unless explicitly enabled.

Add this environment variable to the server command above:

```bash
export VLLM_HCU_USE_CHANNEL_FP8_W8A16_MOE=1
MOE_BACKEND=aiter
```

The installed BoltOps build has no tuned JSON for the production TP8 shape
`E=256,N=256,arch=gfx938,dtype=int8_w8a16`; the runtime therefore uses the
BoltOps default Triton configuration. Both MTP-off and MTP3 services loaded,
captured FULL and PIECEWISE graphs for batch sizes 1 and 16, and returned a
normal HTTP 200 chat completion. Online weight reload is rejected explicitly
after requantization because replaying the original FP8 loaders into INT8
parameters would silently corrupt weights.

The following is a 16-task diagnostic sample, not a formal full HumanEval run.
All runs used TP8, graph mode, batch 16, `temperature=0`, seed 42, and
`reasoning_effort=no_think`.

| Expert execution | MTP | Correct | pass@1 | Avg latency | Avg throughput |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original Channel-FP8 AITER | off | 15/16 | 93.75% | 34.764 s | 4.75 tok/s |
| Original Channel-FP8 AITER | 3 | 14/16 | 87.5% | 31.347 s | 6.47 tok/s |
| Requantized INT8-W8A16 BoltOps | off | 14/16 | 87.5% | 47.016 s | 3.66 tok/s |
| Requantized INT8-W8A16 BoltOps | 3 | 14/16 | 87.5% | 23.374 s | 8.63 tok/s |

The W8A16 runs both failed `HumanEval/1` and `HumanEval/10`. The original FP8
MTP-off run failed only `/10`, while its MTP3 run failed `/1` and `/10`.
Therefore W8A16 introduced one pass-to-fail flip without MTP in this small
sample; enabling MTP did not introduce another pass/fail change on top of
W8A16. Output lengths differ, and the W8A16 kernels use an untuned default
configuration, so the throughput numbers are directional only.

Numerical diagnostics provide a less noisy view of the expert math. For a real
TP8 production-shaped layer (`E=256`, top-k 8), the BoltOps kernel differed from
an explicit INT8 reference by 0.657% relative L2 at batch 16 and from the
original Channel-FP8 weight reference by 1.655%. Against the downloaded layer
40 BF16 reference, explicit INT8-W8A16 measured 4.805% relative L2 versus
6.272% for the existing Channel-FP8 W8A8 route. A separate live HCU synthetic
kernel check measured 0.379% relative L2 and 0.999993 cosine versus explicit
BF16 expert math.

## Opt-in Channel-FP8 weight-only BF16 diagnostic (2026-08-31)

`VLLM_HCU_USE_CHANNEL_FP8_BF16_MOE=1` isolates the original Channel-FP8
checkpoint weights from both dynamic activation quantization and the FP8 MoE
kernel. The weights and FP32 channel scales remain resident in their canonical
checkpoint layout. For each current MoE layer, the diagnostic materializes
temporary BF16 weights and invokes the unquantized BoltOps Triton fused MoE.
The temporary storage is released for allocator/graph-pool reuse after that
layer. This path is intentionally accuracy-only and is not a production
performance backend. It is mutually exclusive with
`VLLM_HCU_USE_CHANNEL_FP8_W8A16_MOE`.

```bash
export PLUGIN_ROOT=/models/zb/hy4/.worktrees/hy-v4-mtp-blockwise-v0251-merge
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH="${PLUGIN_ROOT}"
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_KV_CACHE_LAYOUT=NHD
export VLLM_HCU_USE_CHANNEL_FP8_BF16_MOE=1
unset VLLM_HCU_USE_CHANNEL_FP8_W8A16_MOE

vllm serve /models/Hy4-preview-Channel-FP8-w8a8-v2 \
  --served-model-name hy4-channel \
  --tensor-parallel-size 8 \
  --no-enable-expert-parallel \
  --moe-backend aiter \
  --gpu-memory-utilization 0.95 \
  --max-model-len 4096 \
  --max-num-seqs 16 \
  --max-num-batched-tokens 4096 \
  --default-chat-template-kwargs '{"reasoning_effort":"no_think"}' \
  --reasoning-parser hy_v4 \
  --enable-auto-tool-choice \
  --tool-call-parser hy_v4 \
  --compilation-config '{"cudagraph_capture_sizes":[1,16]}' \
  --port 8000
```

The TP8 service captured both PIECEWISE and FULL graphs for sizes 1 and 16;
graph capture consumed 2.51 GiB per rank, confirming that BF16 layer
temporaries were reused rather than retained for all 77 layers. A short chat
completion returned HTTP 200. On a live BW1100, a synthetic BF16 diagnostic
call was bitwise identical to the same BoltOps call with explicitly
dequantized weights (`max_abs=0`, relative L2 `0`).

The first 16 HumanEval tasks were then run with MTP disabled, batch 16,
`temperature=0`, seed 42, and `max_tokens=1024`. The diagnostic is slow (about
0.49 token/s for a single long request), so the API timeout must exceed the
original 1800 seconds:

```bash
env -u ALL_PROXY -u HTTP_PROXY -u HTTPS_PROXY \
    -u all_proxy -u http_proxy -u https_proxy \
python -m evalscope.cli.cli eval \
  --model hy4-channel \
  --model-id Hy4-preview-Channel-FP8-w8a8-v2-bf16-mtpoff-no_think \
  --api-url http://127.0.0.1:8000/v1 \
  --api-key EMPTY \
  --eval-type openai_api \
  --datasets humaneval \
  --dataset-hub modelscope \
  --limit 16 \
  --eval-batch-size 16 \
  --generation-config '{"batch_size":16,"max_tokens":1024,"temperature":0.0,"top_p":1.0,"seed":42,"timeout":3600.0,"extra_body":{"chat_template_kwargs":{"reasoning_effort":"no_think"}}}' \
  --seed 42 \
  --work-dir /models/evalscope_hy4_channel_bf16_mtpoff_humaneval16_20260831 \
  --no-timestamp
```

| Expert execution | MTP | Correct | pass@1 | Failed tasks |
| --- | ---: | ---: | ---: | --- |
| Original Channel-FP8 AITER W8A8 | off | 15/16 | 93.75% | `/10` |
| Requantized INT8-W8A16 BoltOps | off | 14/16 | 87.5% | `/1`, `/10` |
| Channel-FP8 weight-only BF16 diagnostic | off | 15/16 | 93.75% | `/10` |

The weight-only diagnostic recovered `HumanEval/1`, which the extra INT8
requantization had changed from pass to fail. It did not recover `/10`; the
BF16 and original AITER outputs both contain the same palindrome-suffix index
error. In this 16-task isolation, the extra loss is therefore attributable to
Channel-FP8-to-INT8 weight requantization, not dynamic activation quantization,
FlashMLA, or the original AITER FP8 MoE route.

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

### Max-token-capped failure rerun

Fourteen of the 29 AITER/MTP-off failures consumed exactly the configured
1,024 output tokens and ended with incomplete Python. They were rerun as one
14-request batch with the same server, prompt, graph mode, sampling parameters,
and `reasoning_effort=no_think`; only `max_tokens` changed from 1,024 to 3,072.

- Recovered and passed: `HumanEval/91`, `/97`, `/103`, `/109`, `/119`
- Still reached 3,072 tokens: `HumanEval/32`, `/73`, `/76`, `/77`, `/127`,
  `/129`, `/130`, `/132`, `/134`
- The five recovered requests stopped normally at 727, 292, 901, 1,086 and
  1,264 tokens respectively.
- The remaining nine predictions repeatedly reasoned inside Python comments
  and never completed the function before the larger limit. There were no API,
  kernel, NaN, or text-decoding errors.

Substituting the five recovered results into the original run gives a paired
diagnostic score of 140/164 (85.4%). This is not a new formal full-dataset
pass@1 run because the failed subset was generated again with a different
batch composition. It demonstrates that five of the original failures were
caused by the 1,024-token evaluation limit, while nine are persistent
long-output degeneration rather than ordinary short code failures.

### Upstream PR 54160 audit

The merged upstream implementation in
[vllm-project/vllm#54160](https://github.com/vllm-project/vllm/pull/54160)
was compared at head `184421e`. Its accuracy-relevant changes are aligned in
this plugin: FP32 lm-head accumulation, FP32 router logits and weights, generic
indexer FP8 scale shapes, routed scaling 2.827, normalized sigmoid top-k,
routed-expert SwiGLU limit 10, iHC formulas, sparse top-k buffer ownership, and
the HY V4 MTP hidden-width rule.

Upstream PR 54160 implements NVIDIA MXFP8/TRTLLM experts; it does not contain a
channel-wise AITER expert path. Consequently it is a framework/model reference,
but not an AITER channel-FP8 operator oracle. A direct comparison of BoltOps
`ihc_pre`, `ihc_post`, and `ihc_head` against the upstream PyTorch formulas at
the real `hc=4`, `d=6144` shape found zero p99 BF16 error for T=1 and T=16. At
T=16, the maximum errors were 0.0009766 for pre/head and 0.0039063 for post.
Together with the AITER golden comparison above, this does not support a
framework wiring or iHC/MOE_C correctness defect as the cause of the low
HumanEval score.

## Review decision

No production source change is justified by this investigation:

- AITER starts normally with TP8, graph mode, MTP off, and MTP3.
- Channel-wise weights, scales, shuffle layout, routed weights, output dtype,
  and clamped SwiGLU arguments are wired correctly.
- AITER is deterministic and matches its own golden implementation.
- AITER-off scores one task above Triton-off on the full dataset; therefore the
  earlier 15/16 versus 16/16 result was a small-sample effect, not evidence of
  a general AITER regression.
- Raising only the output limit recovers 5/14 capped AITER failures. Nine hard
  cases continue generating repetitive reasoning comments through 3,072
  tokens, so evaluation length and model output behavior materially depress
  the formal 1,024-token score.
- MTP3 improves throughput substantially but has a measurable pass@1 tradeoff
  in both backends. Accuracy-sensitive deployments should leave MTP disabled;
  throughput-sensitive deployments should validate on their own workload.

Changing plugin kernels to force Triton-like outputs would be an alignment
policy or fallback mode, not a correctness fix. It should not be introduced
without an explicit product requirement and a wider accuracy/performance
evaluation.

## TP4 + PCP2 validation (2026-08-31)

Hy4 sparse MLA now accepts prefill context parallelism with a fail-closed
contract. The validated topology is `TP4 * PCP2 = 8` worker ranks. Expert
parallelism is enabled, so
each rank owns 32 of the 256 experts. Eager mode is required for this initial
PCP support; MTP values 1 and 2 are accepted, while MTP3 is rejected during
configuration validation.

The following command starts the validated Triton service. Omit `SPEC_ARGS`
for target-only generation, or select one of the two MTP variants shown below.

```bash
export PLUGIN_ROOT=/models/zb/hy4/.worktrees/hy-v4-mtp-blockwise-v0251-merge
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_KV_CACHE_LAYOUT=NHD
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=300
export VLLM_ENGINE_ITERATION_TIMEOUT_S=300
export PYTHONPATH="${PLUGIN_ROOT}"
unset VLLM_HCU_USE_CHANNEL_FP8_BF16_MOE
unset VLLM_HCU_USE_CHANNEL_FP8_W8A16_MOE

SPEC_ARGS=()
# MTP1:
# SPEC_ARGS=(--speculative-config '{"method":"mtp","num_speculative_tokens":1}')
# MTP2:
# SPEC_ARGS=(--speculative-config '{"method":"mtp","num_speculative_tokens":2}')

vllm serve /models/Hy4-preview-Channel-FP8-w8a8-v2 \
  --served-model-name hy4-channel-pcp \
  --tensor-parallel-size 4 \
  --prefill-context-parallel-size 2 \
  --enable-expert-parallel \
  --enforce-eager \
  --moe-backend triton \
  --gpu-memory-utilization 0.95 \
  --max-model-len 4096 \
  --max-num-seqs 16 \
  --max-num-batched-tokens 4096 \
  --default-chat-template-kwargs '{"reasoning_effort":"no_think"}' \
  --reasoning-parser hy_v4 \
  --enable-auto-tool-choice \
  --tool-call-parser hy_v4 \
  "${SPEC_ARGS[@]}" \
  --port 8000
```

All three variants created ranks `PCP0/1_TP0..3_EP0..7`, selected the
`FLASH_ATTN MLA prefill backend` and native vLLM `TRITON Fp8 MoE` backend,
completed model loading, and returned HTTP 200 with assistant content `OK`:

```bash
env -u ALL_PROXY -u HTTP_PROXY -u HTTPS_PROXY \
    -u all_proxy -u http_proxy -u https_proxy \
curl --fail-with-body --max-time 600 -sS \
  http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"hy4-channel-pcp","messages":[{"role":"user","content":"Reply with exactly: OK"}],"temperature":0,"max_tokens":16,"chat_template_kwargs":{"reasoning_effort":"no_think"}}'
```

The target-only service also passed this long-prefill smoke, which exercises
PCP with a meaningful context rather than only a short prompt:

```bash
python - <<'PY' | \
env -u ALL_PROXY -u HTTP_PROXY -u HTTPS_PROXY \
    -u all_proxy -u http_proxy -u https_proxy \
curl --fail-with-body --max-time 600 -sS \
  http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data-binary @-
import json

content = (
    "Read the following filler context, then reply with exactly: OK\n\n"
    + ("alpha " * 3000)
    + "\n\nEnd of filler context. Reply with exactly: OK"
)
print(json.dumps({
    "model": "hy4-channel-pcp",
    "messages": [{"role": "user", "content": content}],
    "temperature": 0,
    "max_tokens": 16,
    "chat_template_kwargs": {"reasoning_effort": "no_think"},
}))
PY
```

The response was HTTP 200 with assistant content `OK`, 3,049 prompt tokens,
and two completion tokens. The service log showed the PCP indexer-cache gather
kernel executing and no request or kernel error. The service then shut down
without an orphan worker.

The first 16 HumanEval tasks were run at request batch size 16:

```bash
env -u ALL_PROXY -u HTTP_PROXY -u HTTPS_PROXY \
    -u all_proxy -u http_proxy -u https_proxy \
python -m evalscope.cli.cli eval \
  --model hy4-channel-pcp \
  --model-id Hy4-preview-Channel-FP8-w8a8-v2-triton-pcp2-mtpoff-no_think \
  --api-url http://127.0.0.1:8000/v1 \
  --api-key EMPTY \
  --eval-type openai_api \
  --datasets humaneval \
  --dataset-hub modelscope \
  --limit 16 \
  --eval-batch-size 16 \
  --generation-config '{"batch_size":16,"max_tokens":1024,"temperature":0.0,"top_p":1.0,"seed":42,"timeout":1800.0,"extra_body":{"chat_template_kwargs":{"reasoning_effort":"no_think"}}}' \
  --seed 42 \
  --work-dir /models/evalscope_hy4_channel_pcp2_triton_mtpoff_humaneval16_20260831 \
  --no-timestamp
```

Change `mtpoff` in the model ID and work directory to `mtp1` or `mtp2` as
appropriate when running the speculative variants.

The three reports are retained under:

```text
/models/evalscope_hy4_channel_pcp2_triton_mtpoff_humaneval16_20260831
/models/evalscope_hy4_channel_pcp2_triton_mtp1_humaneval16_20260831
/models/evalscope_hy4_channel_pcp2_triton_mtp2_humaneval16_20260831
```

| MoE backend | PCP | MTP | Correct | pass@1 | Failed tasks | Avg latency | Avg throughput | Avg output |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| Triton | 2 | off | 15/16 | 93.8% | `/10` | 75.078 s | 2.60 tok/s | 196 |
| Triton | 2 | 1 | 14/16 | 87.5% | `/1`, `/10` | 49.911 s | 4.67 tok/s | 233 |
| Triton | 2 | 2 | 15/16 | 93.8% | `/10` | 38.955 s | 4.70 tok/s | 183 |

Target-only and MTP2 have the same 15/16 score and the same `/10` failure,
whose generated code contains the same palindrome-suffix index error. MTP2
therefore had no pass/fail flips and did not reduce pass@1 in this sample.
MTP1 introduced one pass-to-fail flip, `/1`, and no fail-to-pass flips; that
output followed a longer, different generation trajectory. No API or runtime
error was logged, but this small sample alone does not isolate whether the
trajectory change comes from speculation, PCP-sensitive numerics, or ordinary
generation variance. Each service was stopped after its run; no vLLM worker
remained and all eight cards returned to zero utilization.

### AITER PCP limitation

The same TP4+PCP2+EP topology was also attempted with `--moe-backend aiter`.
The attempted target-only command is the service command above with
`--moe-backend triton` changed to `--moe-backend aiter`; MTP1 and MTP2 add the
corresponding `SPEC_ARGS` value. It fails during the startup profile before an
API request can be served.
It intentionally remains fail-closed in this plugin because the installed
AITER package has no valid Channel-FP8 expert-parallel solution for the local
shape `M=8192,E=32,top_k=8,dtype=bf16,quant_type=fp8_w8a8`:

```text
HcuCompressedTensorsMoeError: AITER quantized MoE found no backend config for
M=8192, E=32, top_k=8, dtype=torch.bfloat16, quant_type=fp8_w8a8
```

AITER's MOE_C table contains a configuration for the global `E=256` shape,
but its kernel performs another lookup using the local EP shape and returns an
incomplete default without `MODE`. Forcing that global configuration would
also bypass part of the expert-map contract. The public AITER Triton fallback
was tested separately and caused an HCU VM fault for this production EP shape;
it is not the native vLLM modular Triton backend validated above. No unsafe
fallback was retained. This is an installed AITER/BoltOps capability gap, not
a PCP configuration or Hy4 model-registration failure.

The fail-closed configuration and sparse-backend capability changes are in
commits `5682574` and `d1a47b6`, respectively.
