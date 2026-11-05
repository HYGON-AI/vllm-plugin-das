#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
export VLLM_HCU_STEP5_W4A8_PROFILE=prefill
export STEP5_EXP_ROLE=prefill
export HCU_DP_MOE_PIPE_MIN_TOKENS=16384
export HCU_MTP_PREFILL_TAIL=1536
export HCU_DBO_MIN_TOKENS=8192
export HCU_SHARED_EXPERTS_STREAM_MIN_TOKENS=4096
export NCCL_MAX_NCHANNELS=8
: "${PLACEMENT_PATH:?Set the validated EP8 static placement}"
[[ -r "$PLACEMENT_PATH" ]] || exit 2
export VLLM_STEP5_EXPERT_PLACEMENT="$PLACEMENT_PATH"
export HCU_EP_DIRECT_SUM=0
exec python3 -m vllm.entrypoints.cli.main serve "$MODEL_PATH" "${STEP5_COMMON_ARGS[@]}" \
  --tensor-parallel-size 1 \
  --data-parallel-size 8 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 64 \
  --max-cudagraph-capture-size 64 \
  --gpu-memory-utilization 0.56
