# DeepSeek-V4-Flash-0731 Channel-INT8 DSpark validation

## Scope

This profile targets
`/models/DeepSeek-V4-Flash-0731-Channel-INT8-w8a8` on one eight-card BW1100
node. It supports TP8+DSpark with either Triton or the HCU AITER extension, and
single-service DP8+EP8+DSpark. PCP+DSpark and prefill/decode disaggregation are
outside this profile.

The public CLI follows vLLM's DeepSeek-V4 flow. Pure TP8 selects the public
`--moe-backend triton` or `--moe-backend aiter` value. INT8 AITER is an HCU
plugin extension; upstream vLLM 0.25.1 provides the Triton path. DP+EP does not
set a MoE backend and selects `deepep_auto` once; the HCU backend uses
contiguous DeepGEMM for HT forwards and masked DeepGEMM for LL forwards.
Channel-INT8 uses the public
`m_grouped_i8_gemm_nt_contiguous`/`marlin_i8_contiguous_weight` and
`m_grouped_i8_gemm_nt_masked`/`marlin_i8_masked_weight` pairs. LightOp performs
the `swiglu_limit=10.0` activation and dynamic per-token INT8 quantization.

## Server commands

TP8+Triton+DSpark:

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
  --moe-backend triton \
  --served-model-name DeepSeek-V4-Flash-0731-Channel-INT8-w8a8 \
  --speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}'
```

TP8+AITER+DSpark:

```bash
vllm serve /models/DeepSeek-V4-Flash-0731-Channel-INT8-w8a8 \
  --host 0.0.0.0 \
  --port 10141 \
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
  --moe-backend aiter \
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

None of the commands needs `--enforce-eager`. The DP profile does not set
`--moe-backend` and uses the same automatic graph fallback as the FP8 profile.
For explicit AITER, do not set `VLLM_ROCM_USE_AITER=1` or
`VLLM_ROCM_USE_AITER_MOE=1`; the plugin checks the requested MoE capability
without enabling unrelated AITER communication or model paths.

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

Use port `10141` for AITER TP8, or port `10139` for DP8+EP8.

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
  tests/integration/server/test_evalscope_deepseek_v4_dspark_humaneval.py::test_deepseek_v4_int8_dspark_humaneval_tp8

VLLM_HCU_DEEPSEEK_V4_FLASH_0731_INT8_MODEL=/models/DeepSeek-V4-Flash-0731-Channel-INT8-w8a8 \
python -m pytest -q -s \
  tests/integration/server/test_evalscope_deepseek_v4_dspark_humaneval.py::test_deepseek_v4_int8_dspark_humaneval_tp8_aiter

VLLM_HCU_DEEPSEEK_V4_FLASH_0731_INT8_MODEL=/models/DeepSeek-V4-Flash-0731-Channel-INT8-w8a8 \
python -m pytest -q -s \
  tests/integration/server/test_evalscope_deepseek_v4_dspark_humaneval.py \
  -k deepseek_v4_int8_dspark_humaneval_dp8_ep8
```

HumanEval acceptance requires exactly 32 predictions and 32 reviews. The
repository-owned normalization reruns all completions after removing one
complete or truncated Markdown fence; normalized `mean_acc` and
`mean_acc_pass@1` must both equal `1.0`.

## Current observed status (2026-08-30)

- TP8+DSpark smoke passed in 148.46 seconds. DP8+EP8+DSpark smoke passed in
  141.79 seconds on the local 48-shard, 286.61 GiB checkpoint.
- The DP log recorded construction and execution of both HT contiguous and LL
  masked experts under the single `deepep_auto` selector.
- The combined gfx938 operator suite passed 10/10 cases, including INT8
  contiguous/masked DeepGEMM and clamped LightOp INT8 quantization.
- Triton TP8 HumanEval produced 32 predictions and 32 reviews: raw
  accuracy/pass@1 was 29/32 (`0.9062`), normalized accuracy/pass@1 was 32/32
  (`1.0000`), and EvalScope reported 2.427-second average latency and 49.23
  output tokens/s.
- AITER TP8 HumanEval also produced 32 predictions and 32 reviews: raw
  accuracy/pass@1 was 29/32 (`0.9062`), normalized accuracy/pass@1 was 32/32
  (`1.0000`), and EvalScope reported 2.121-second average latency and 57.0
  output tokens/s. The log selected `AITER Int8 MoE`, loaded gfx938 W8A8 ASM
  stage-1/stage-2 kernels, and recorded roughly 80--83% DSpark acceptance.
- DP8+EP8 HumanEval produced 32 predictions and 32 reviews: raw
  accuracy/pass@1 was 29/32 (`0.9062`), and normalized accuracy/pass@1 was
  32/32 (`1.0000`). The end-to-end pytest gate passed in 545.06 seconds;
  EvalScope reported 4.061-second average latency and 29.98 output tokens/s.

## gfx936 v0.25.1 operator parity (2026-10-05)

The current v0.28.1 HCU path was compared on one live gfx936 device with the
operator sources extracted from
`vllm==0.25.1+das185.dtk2604.torch2110.2608171710.g7b108a`. The same seeded
inputs produced byte-identical `fp8_ds_mla` cache writes, element-identical
gather/dequant output, identical active-token top-k mapping, and identical
combined sparse indices. The current top-k helper intentionally writes `-1`
for padding rows whose length is zero; v0.25.1 left ignored values there.

The v0.25.1 optimized Triton path and the current HCU reference path differed
by at most `0.0078125` for sparse prefill, `0.00390625` for SWA-only sparse
decode, and `0.0078125` for combined compressed-KV-plus-SWA sparse decode,
within the recorded BF16 gate (`rtol=0.02`, `atol=0.03`). The current
categorized LightOp fused QNorm/RoPE/KVNorm/cache insert generated cache bytes
identical to the v0.25.1 cache contract on gfx936; its fused query output had a
maximum absolute difference of `0.015567779541015625` from the unfused FP32
reference and passed `rtol=0.01`, `atol=0.01`.

Reproduce the cross-version comparison with:

```bash
HIP_VISIBLE_DEVICES=0 \
PYTHONPATH=/data/models/vllm-plugin-das/.worktrees/fix-remaining-language-model-validation \
python /data/models/model-language-validation/dsv4-flash-0731-w4a8/compare_v0251_operator_parity.py
```

The live gfx936 regression gate is:

```bash
HIP_VISIBLE_DEVICES=0 \
PYTHONPATH=/data/models/vllm-plugin-das/.worktrees/fix-remaining-language-model-validation \
pytest -q -s \
  tests/accuracy/deepseek_v4_dspark_ops_cases.py::test_dspark_non_pcp_lightop_context_insert_writes_fp8_cache \
  tests/accuracy/deepseek_v4_dspark_ops_cases.py::test_dspark_non_pcp_lightop_context_insert_matches_vllm_reference \
  tests/accuracy/test_hcu_kernel_accuracy.py::test_lightop_deepseek_v4_fused_insert_updates_q_and_cache \
  tests/accuracy/test_hcu_kernel_accuracy.py::test_lightop_sparse_mqa_matches_fp32_reference \
  tests/accuracy/test_hcu_kernel_accuracy.py::test_lightop_paged_sparse_mqa_matches_packed_cache_reference
```

Observed result: `5 passed`. These cache and sparse-MQA cases are valid on
gfx936; the Channel-FP8/INT8 DeepGEMM cases in the shared accuracy module remain
gfx938-only.
