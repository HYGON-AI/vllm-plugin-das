# BF16 DeepEP auto with DeepGEMM validation

Qwen3.5-122B-A10B is unquantized BF16. The original unquantized MoE oracle
rejects `--moe-backend deep_gemm`; the existing HCU auto experts accept only
channel FP8/INT8. This implementation adds an explicit BF16 selection for
`--all2all-backend deepep_auto --moe-backend deep_gemm`.

The HT path calls `deepgemm.m_grouped_bf16_gemm_nt_contiguous` with canonical
weights. The LL path calls `deepgemm.m_grouped_bf16_gemm_nt_masked` with HIPC
packed weights and MODE 2000. GEMMs operate on BF16 weights/activations.
BF16 scatter uses the generic Triton copy kernel because LightOp's fused
quantized scatter accepts only byte activations. Completely padded HT tiles
use a valid expert index because this BF16 scheduler does not skip -1 tiles.
The inverse permutation excludes these padded results from the output.

Supported configurations currently require BF16, SiLU gate/up, dimensions
that are multiples of 256, DeepEP auto, no bias/LoRA, and output router
weighting. Unsupported explicit requests fail during backend selection.

LL needs a separate packed weight copy, created during weight loading and
included in memory profiling. For Qwen3.5-122B-A10B with DP4/EP4, the extra
48-layer MoE weight storage is approximately 54 GiB per rank. This memory
cost and the original 180000 context configuration require model validation.
Weight updates refresh the packed copy while retaining its storage address.

## Validation on 2026-11-05

- 70 runtime tests passed (69 existing combined checks plus a deferred-quantization regression) across `test_deepep_bf16_moe.py`,
  `test_lightop_w16a16_moe.py`, and `test_deepep_exact_ht_layout.py`.
- Real BF16 HT and LL GEMM probes (E=2, N=512, K=512) matched the BF16 PyTorch
  reference exactly on valid rows.
- Four full expert pipeline cases passed (E=4, hidden/intermediate=512/256
  and 3072/1024, tokens=1/17, top-k=2), exercising HT -> LL -> HT,
  invalid routes, an empty expert, router weighting, and output/workspace aliasing.
  An independent FP32 reference avoids BF16 reference rounding at every stage.
  Observed relative L2 error: 0.34%-0.45%; maximum absolute error below 0.01.
  Tests require relative L2 below 2% and per-element rtol=4%/atol=0.02.
  The earlier BF16 intermediate reference had one outlier (absolute difference
  0.00531) at hidden=3072, tokens=17, motivating the FP32 reference check.
- DP4/EP4 full model startup passed on physical GPUs 4-7, with BF16 DeepGEMM,
  DeepEP auto, FP8 KV cache, MTP=3, eager mode, 8192 batched tokens, 64 sequences,
  and max-model-len=180000. Worker logs confirmed both HT and LL selection.
- Default gpu-memory-utilization=0.90 failed KV capacity (0.16 GiB available).
  Setting 0.95 succeeded: model loading 129.92 GiB/rank, available KV cache
  4.48 GiB, reported cache capacity 265263 tokens per rank.
- BF16 dispatch must leave expects_unquantized_inputs=False: quant_dtype=None
  already preserves BF16; True requests deferred quantization, rejected by LL.
- BF16 DeepGEMM HumanEval32 completed: pass@1=32/32, temperature=0,
  max_tokens=32768, thinking disabled, seed=0, eval-batch-size=1.
  EvalScope benchmark elapsed 393.0 seconds; API request latency sum
  314.455 seconds; 4501 output tokens (14.31 output tokens/request-second,
  including prompt processing). This is a serial end-to-end sample measurement.
- Earlier Triton HT baseline: pass@1=32/32, latency sum 211.488 seconds,
  4601 output tokens (21.75 output tokens/request-second). This baseline ran
  the previous day; outputs and hardware activity differ, so this is not a
  controlled kernel throughput comparison.
- AITER comparison uses allgather_reducescatter, not DeepEP auto, and 0.90
  GPU utilization (the same eager/MTP/KV/context/generation settings otherwise).
  Startup selects the AITER backend, but no ASM solution exists for the
  BF16 E=256,N1=2048,N2=3072,K=3072,top-k=8,M=1 loading probe. The plugin
  installs the native layout and routes execution to vLLM Triton. Therefore
  this measures the AITER configuration with fallback, not pure AITER kernels.
- AITER configuration (Triton fallback) HumanEval32: pass@1=32/32,
  benchmark elapsed 236.2 seconds, request latency sum 163.539 seconds,
  4654 output tokens (28.46 output tokens/request-second).
- Graphs remain unverified. All model evaluations used eager mode.

| Configuration | HumanEval pass@1 | Sum API latency (s) | Output tokens | Output tokens / API second |
| --- | --- | ---: | ---: | ---: |
| DeepGEMM + DeepEP auto | 32/32 | 314.455 | 4501 | 14.31 |
| AITER config, native Triton fallback + allgather/reducescatter | 32/32 | 163.539 | 4654 | 28.46 |
| Earlier Triton + DeepEP HT | 32/32 | 211.488 | 4601 | 21.76 |

These serial API measurements include prefill and decode. Communication
backends, memory budgets, model outputs, run dates and other-device activity
differ; they are not isolated kernel benchmarks. This BF16 DeepGEMM
implementation demonstrates correctness for the sampled tasks, not a speedup.

Host artifacts:
- `/home/evalscope-qwen35-das-deepgemm-32/reports/report.html`
- `/home/evalscope-qwen35-das-aiter-32/reports/report.html`
- `/home/bf16-deepep-validation/comparison.json`
- `/home/bf16-deepep-validation/server-fixed.log`
- `/home/bf16-deepep-validation/server-aiter.log`

The main host script `/home/vllm_1102_zb.sh` now uses the validated DeepGEMM
configuration. The original eager Triton baseline script is preserved at
`/home/vllm_1102_zb.triton_baseline.sh`. Test services were stopped after evaluation.

A previous, separate eager Triton + DeepEP high-throughput run achieved
HumanEval pass@1 32/32. It does **not** validate this DeepGEMM implementation
or the original AITER + auto configuration.

Reproduce the numerical cases on a free HCU:

```bash
HIP_VISIBLE_DEVICES=4 PYTHONPATH="$PWD" pytest -q tests/accuracy/test_deepep_bf16_moe_accuracy.py
```

Use the repository source and matching built extension when testing a server.
The validation should retain explicit backend logging and verify both HT and LL
selection before collecting model evaluation results.

Server invocation used the original script's environment with:

```bash
PYTHONPATH=/home/vllm-plugin-das HIP_VISIBLE_DEVICES=4,5,6,7 \
vllm serve /data/model/Qwen3.5-122B-A10B \
  --trust-remote-code --served-model-name qwen35-das-deepgemm \
  --all2all-backend deepep_auto --data-parallel-size 4 --enable-expert-parallel \
  --attention-backend FLASH_ATTN --moe-backend deep_gemm \
  --kv-cache-dtype fp8_e4m3 --enable-prefix-caching \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
  --gpu-memory-utilization 0.95 --max-model-len 180000 --enable-chunked-prefill \
  --enforce-eager --max-num-batched-tokens 8192 --max-num-seqs 64
```
