# DeepSeek-V4-Flash-0731 Channel-FP8 DSpark validation

## Scope

This adaptation targets
`/models/DeepSeek-V4-Flash-0731-Channel-FP8-w8a8` on one eight-card BW1100 node.
It supports pure TP8+DSpark with either the Triton or AITER MoE backend, and
single-service DP+EP+DSpark. The public configuration follows the standard
vLLM `EngineArgs`/CLI path and the
[official DeepSeek-V4-Flash recipe](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Flash):
DSpark uses seven speculative tokens with probabilistic draft sampling.

PCP+DSpark and prefill/decode disaggregation are excluded. There are no public
high-throughput or low-latency flags. DP+EP requests `deepep_auto` once; the HCU
runtime selects contiguous high-throughput or masked low-latency experts for
each forward and snapshots that selection across prepare, experts, and
finalize.

## Supported commands

TP8+Triton+DSpark:

```bash
vllm serve /models/DeepSeek-V4-Flash-0731-Channel-FP8-w8a8 \
  --host 0.0.0.0 \
  --port 10136 \
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
  --served-model-name DeepSeek-V4-Flash-0731-Channel-FP8-w8a8 \
  --speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}'
```

TP8+AITER+DSpark changes only the public MoE backend and port:

```bash
vllm serve /models/DeepSeek-V4-Flash-0731-Channel-FP8-w8a8 \
  --host 0.0.0.0 \
  --port 10140 \
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
  --served-model-name DeepSeek-V4-Flash-0731-Channel-FP8-w8a8 \
  --speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}'
```

DP8+EP8+DSpark, one service and one automatic DeepEP setting:

```bash
vllm serve /models/DeepSeek-V4-Flash-0731-Channel-FP8-w8a8 \
  --host 0.0.0.0 \
  --port 10137 \
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
  --served-model-name DeepSeek-V4-Flash-0731-Channel-FP8-w8a8 \
  --speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}'
```

Do not set `VLLM_ROCM_USE_AITER=1` or `VLLM_ROCM_USE_AITER_MOE=1` for the
explicit AITER profile. Those are global feature gates and enable unrelated
AITER communication and model paths. The plugin's explicit capability check
allows `--moe-backend aiter` to select only the MoE implementation.

DP8+EP8 is the supported automatic dual-layout topology. DP4+EP4 does not
have enough per-card memory to retain both Marlin weight layouts for this
checkpoint. No command needs `--enforce-eager`: TP8 retains vLLM graph
capture, while DP `deepep_auto` internally sets the CUDA Graph mode to `NONE`
because its high-throughput DeepEP dispatch is graph-incompatible. This follows
vLLM's automatic fallback for the official DeepEP high-throughput backend.

All services use the same OpenAI-compatible client request. Select port
`10136` for Triton TP8, `10140` for AITER TP8, or `10137` for DP8+EP8:

```bash
curl --noproxy '*' http://127.0.0.1:10136/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "DeepSeek-V4-Flash-0731-Channel-FP8-w8a8",
    "messages": [{"role": "user", "content": "Write a Python hello-world program."}],
    "temperature": 0,
    "max_tokens": 128,
    "chat_template_kwargs": {"thinking": false}
  }'
```

## Operator evidence

The live gfx938 operator suite passed all ten FP8/INT8 cases:

```bash
HIP_VISIBLE_DEVICES=0 VLLM_PLUGINS=__disabled__ \
python -m pytest -q tests/accuracy/test_deepseek_v4_dspark_ops.py -m hcu -s
```

Observed result: `10 passed`. FP8 coverage includes:

- `marlin_fp8_contiguous_weight` with
  `m_grouped_fp8_gemm_nt_contiguous`;
- `marlin_fp8_masked_weight` with `m_grouped_fp8_gemm_nt_masked` at the
  DeepSeek-V4 dimensions `(K=7168, N=4096)` and `(K=2048, N=7168)`;
- LightOp HT and LL FP8 SiLU quantization;
- non-PCP `fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert` cache insertion,
  including parity against vLLM's `fp8_ds_mla` reference cache contract.

The same suite also covers the Channel-INT8 contiguous and masked DeepGEMM
pairs and LightOp's clamped HT/LL dynamic INT8 quantization path.

The installed DeepGEMM wheel requires E4M3FN inputs for the contiguous path;
E4M3FNUZ is rejected. The test uses E4M3FN with its 448 quantization bound.

## Runtime and HumanEval gates

Run the model-runtime gates in order:

```bash
VLLM_HCU_DEEPSEEK_V4_FLASH_0731_MODEL=/models/DeepSeek-V4-Flash-0731-Channel-FP8-w8a8 \
python tools/run_patch_tests.py --suite model -- -k deepseek_v4_flash_dspark_tp8

VLLM_HCU_DEEPSEEK_V4_FLASH_0731_MODEL=/models/DeepSeek-V4-Flash-0731-Channel-FP8-w8a8 \
python tools/run_patch_tests.py --suite model -- -k deepseek_v4_flash_dspark_dp8_ep8
```

Run HumanEval-32 after TP8 passes, then run the unified DP8 profile:

```bash
VLLM_HCU_DEEPSEEK_V4_FLASH_0731_MODEL=/models/DeepSeek-V4-Flash-0731-Channel-FP8-w8a8 \
python -m pytest -q \
  tests/integration/server/test_evalscope_deepseek_v4_dspark_humaneval.py::test_deepseek_v4_dspark_humaneval_tp8 \
  -s

VLLM_HCU_DEEPSEEK_V4_FLASH_0731_MODEL=/models/DeepSeek-V4-Flash-0731-Channel-FP8-w8a8 \
python -m pytest -q \
  tests/integration/server/test_evalscope_deepseek_v4_dspark_humaneval.py::test_deepseek_v4_dspark_humaneval_tp8_aiter \
  -s

VLLM_HCU_DEEPSEEK_V4_FLASH_0731_MODEL=/models/DeepSeek-V4-Flash-0731-Channel-FP8-w8a8 \
python -m pytest -q \
  tests/integration/server/test_evalscope_deepseek_v4_dspark_humaneval.py \
  -k humaneval_dp8_ep8 -s
```

Acceptance requires exactly 32 prediction JSONL records and 32 review JSONL
records. EvalScope 1.11 leaves an opening Markdown fence in some otherwise
valid completions, so the repository-owned gate removes one complete or
truncated Python fence and re-executes all 32 samples. Both normalized
`mean_acc` and normalized `mean_acc_pass@1` must be exactly `1.0`; the raw
EvalScope metrics remain in the log for comparison.

## Current observed status (2026-08-30)

The corrected checkpoint passed both runtime topology gates:

- TP8+DSpark produced non-empty deterministic tokens with PCP world size 1;
- DP8+EP8+DSpark produced non-empty tokens from one `deepep_auto` service;
- the DP8 log recorded both contiguous high-throughput and masked low-latency
  expert forwards;
- public `--kv-cache-dtype fp8` selected vLLM's internal `fp8_ds_mla` cache.

Both TP8 backends produced 32 predictions and 32 reviews. Triton's raw
EvalScope score was 28/32 (`0.8750`) and AITER's was 30/32 (`0.9375`). The
source-controlled code-fence normalization passed 32/32 (`1.0`) for both.
The AITER log selected `AITER Fp8 MoE`, loaded gfx938 FP8-W8A8 ASM stage-1 and
stage-2 kernels, retained `CUSTOM/PYNCCL` TP communication, and recorded
DSpark acceptance rates around 80--86%. The single-service DP path synchronizes
decode-phase evidence so active and empty ranks select the same HT/LL
collective, while short prefill and mixed batches remain on HT. Artifacts are
stored under:

- `/tmp/vllm-hcu-evalscope/deepseek_v4_flash_0731_dspark_tp8`
- `/tmp/vllm-hcu-evalscope/deepseek_v4_flash_0731_dspark_tp8_aiter`
- `/tmp/vllm-hcu-evalscope/deepseek_v4_flash_0731_dspark_dp8_ep8`

The original `/models/DeepSeek-V4-Flash-0731-FP8-Channel` directory was
incomplete; it was not used for these claims.

Observed environment:

- 8 × BW1100 (gfx938), 147440 MiB VRAM per card;
- vLLM `0.25.1+das185.dtk2604.torch2110.2608171710.g7b108a`;
- torch `2.11.0+das.opt1.dtk2604.202604021232.g1175f0`;
- deep-ep `1.1.0+das185.dtk2604.torch2110.2608181058.gb5b9ab`;
- deepgemm `2.1.0+das185.dtk2604.torch2110.2608171132.g493d80`;
- lightop `0.6.0+das.dtk2604.torch2110.2608171227.g8c835c`;
- EvalScope `1.11.0`.
