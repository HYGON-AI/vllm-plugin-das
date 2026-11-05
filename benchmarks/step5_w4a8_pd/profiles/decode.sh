#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
export VLLM_HCU_STEP5_W4A8_PROFILE=decode
export STEP5_EXP_ROLE=decode
export HCU_DP_MOE_PIPE_MIN_TOKENS=0
export HCU_DBO_MIN_TOKENS=0
export VLLM_STEP5_EXPERT_PLACEMENT=""
exec python3 -m vllm.entrypoints.cli.main serve "$MODEL_PATH" "${STEP5_COMMON_ARGS[@]}" \
  --tensor-parallel-size 4 \
  --data-parallel-size 2 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 32 \
  --max-cudagraph-capture-size 256 \
  --gpu-memory-utilization 0.50
