#!/usr/bin/env bash
# Sourced only by the explicit Step5 performance profiles.
set -euo pipefail
[[ $# == 0 ]] || { echo "No CLI overrides; edit the profile explicitly" >&2; exit 2; }
RECIPE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
: "${MODEL_PATH:?Set the packed W4A8 checkpoint}"
: "${TOKENIZER_PATH:?Set the benchmark tokenizer}"
: "${STEP5_EXP_RUN:?Set a NEW output directory}"
[[ -n "${STEP5_W4_BUDGET_GUARD:-}" && -f "$STEP5_EXP_RUN/guard-identity.json" ]] || { echo "Launch through budget_guard.py --execute; this script starts nothing without its guard" >&2; exit 2; }

unset HCU_STEP5_DG_FULLMODEL_EXPERIMENT HCU_STEP5_DG_DENSE_FP8_EXPERIMENT
unset HCU_MTP_PREFILL_TAIL HCU_EP_DIRECT_SUM HCU_SHARED_EXPERTS_STREAM_MIN_TOKENS NCCL_MAX_NCHANNELS
unset ROCBLAS_TENSILE_LIBPATH HIPBLASLT_TUNING_OVERRIDE_FILE
if [[ -n "${STEP5_TOPO_FILE:-}" ]]; then export NCCL_TOPO_FILE="$STEP5_TOPO_FILE"; else unset NCCL_TOPO_FILE; fi
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_SERVER_DEV_MODE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VLLM_HCU_USE_CUSTOM_OPS=1
export VLLM_USE_NN=1
export VLLM_HCU_STEP5_FP8_DENSE=1
export VLLM_HCU_STEP5_FUSE_QKVG=1
export VLLM_HCU_STEP5_FP8_BF16_MOE=0
export VLLM_HCU_EP_FUSED_MOE=1
export VLLM_HCU_EP_FUSED_COMBINE=1
export VLLM_HCU_DP_CUSTOM_AG=1
export VLLM_HCU_STEP5_SWA_FUSED_QK=1
export VLLM_HCU_STEP5_FUSED_GATE=1
export VLLM_HCU_STEP5_QKV_INDEXER=1
export VLLM_HCU_TRITON_TOKEN_FP8_QUANT=1
export VLLM_HCU_USE_FLASH_ATTN=1
export VLLM_HCU_USE_FLASH_ATTN_UNIFIED=1
export VLLM_USE_FLASH_ATTN_FP8=1
export VLLM_USE_PD_SPLIT=1
export VLLM_HCU_USE_LIGHTOP_EP_SCATTER=1
export VLLM_HCU_USE_LIGHTOP_MOE_ALIGN=1
export VLLM_HCU_USE_CUSTOM_QUANTIZATION_GEMM=1
export VLLM_HCU_USE_FUSE_MOE_GATE=1
export VLLM_HCU_USE_CUSTOM_SILU_AND_MUL=1
export VLLM_HCU_USE_CUSTOM_GEMMA_RMS_NORM=1
export VLLM_ROCM_USE_AITER_MOE=1
export VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD=0
export VLLM_HCU_PIN_MEMORY=1
export HIP_FORCE_DEV_KERNARG=1
export GPU_MAX_HW_QUEUES=4
export FA_DECODE_NUM_SPLITS=2
export HSA_FORCE_FINE_GRAIN_PCIE=1
export NCCL_IB_DISABLE=1
export VLLM_STEP4_DSA_GRAPH_REGIONS=8704
export HCU_STEP5_DENSE_TUNING=0
export VLLM_HCU_ALLOCATOR_LIMIT_GIB=86
unset EP_FUSED_MOE_TABLE

STEP5_COMMON_ARGS=(
  --tokenizer
  "$TOKENIZER_PATH"
  --served-model-name
  step5
  --reasoning-parser
  step3p5
  --host
  "${STEP5_HOST:-127.0.0.1}"
  --port
  "${STEP5_PORT:-8000}"
  --api-server-count
  1
  --enable-expert-parallel
  --moe-backend
  triton
  --all2all-backend
  allgather_reducescatter
  --max-model-len
  70000
  --no-enable-prefix-caching
  --speculative-config
  '{"method":"mtp","num_speculative_tokens":3,"rejection_sample_method":"synthetic","synthetic_acceptance_rates":[0.838992,0.598322,0.362686],"use_local_argmax_reduction":true}'
  --cudagraph-metrics
  --compilation-config
  '{"cudagraph_mode":"FULL_AND_PIECEWISE","compile_sizes":[16,64],"custom_ops":["+swiglustep_and_mul"]}'
  --kv-cache-dtype
  fp8_e4m3
  --limit-mm-per-prompt
  '{"image":0}'
)
