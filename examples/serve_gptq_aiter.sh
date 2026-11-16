#!/usr/bin/env bash
set -euo pipefail

# Text inference profile; install the matching HCU plugin wheel first.
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to the GPTQ checkpoint directory}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-4}"
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_KV_CACHE_LAYOUT=HND

exec vllm serve "${MODEL_PATH}" \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 16384 \
  --max-num-seqs 64 \
  --max-num-batched-tokens 8192 \
  --attention-backend FLASH_ATTN \
  --block-size 64 \
  --moe-backend aiter \
  --enable-prefix-caching \
  --mamba-cache-mode align \
  --prefix-match-unit 64 \
  --enable-mamba-fine-grained-prefix-cache \
  --kv-cache-dtype fp8_e5m2 \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 3 \
  --skip-mm-profiling \
  --port "${PORT:-8004}" "$@"
