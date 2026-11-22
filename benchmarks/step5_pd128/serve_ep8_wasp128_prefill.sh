#!/usr/bin/env bash
set -euo pipefail

case "${1:-}" in
    --help)
        cat <<'EOF'
Start the single-node Step5 DeepEP / WASP128 prefill service.

Required environment:
  STEP5_MODEL_DIR        Step5 Channel-FP8 model
  STEP5_TOKENIZER_DIR    ByteLevel tokenizer used by the benchmark
  STEP5_EXPERT_MAP       EP8 physical-to-logical expert placement
  STEP5_DENSE_TUNING     gfx938 dense tuning JSON
  STEP5_HIPBLASLT_DIR    Qualified hipBLASLt and library_gfx938 directory
  STEP5_RUN_DIR          Output and compilation-cache directory on local disk

Optional: STEP5_PYTHON (python3), STEP5_PORT (28302).
Use --dry-run to print the arguments without starting the server.
EOF
        exit
        ;;
    ''|--dry-run) ;;
    *) printf 'Unexpected argument: %s\n' "$1" >&2; exit 2 ;;
esac

model=$(realpath -e "${STEP5_MODEL_DIR:?Set STEP5_MODEL_DIR}")
tokenizer=$(realpath -e "${STEP5_TOKENIZER_DIR:?Set STEP5_TOKENIZER_DIR}")
VLLM_STEP5_EXPERT_PLACEMENT=$(realpath -e "${STEP5_EXPERT_MAP:?Set STEP5_EXPERT_MAP}")
HCU_STEP5_TUNING_CONFIG=$(realpath -e "${STEP5_DENSE_TUNING:?Set STEP5_DENSE_TUNING}")
export VLLM_STEP5_EXPERT_PLACEMENT HCU_STEP5_TUNING_CONFIG
hipblaslt=$(realpath -e "${STEP5_HIPBLASLT_DIR:?Set STEP5_HIPBLASLT_DIR}")
run_dir=$(realpath -m "${STEP5_RUN_DIR:?Set STEP5_RUN_DIR}")
python="${STEP5_PYTHON:-python3}"
port="${STEP5_PORT:-28302}"

unset VLLM_PLUGINS NCCL_TOPO_FILE NCCL_MIN_NCHANNELS
unset HIPBLASLT_TUNING_OVERRIDE_FILE ROCBLAS_TENSILE_LIBPATH
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_USE_V2_MODEL_RUNNER=1 VLLM_ENGINE_READY_TIMEOUT_S=4200
export VLLM_HCU_USE_CUSTOM_OPS=1 VLLM_USE_DEEP_GEMM=1 VLLM_MOE_USE_DEEP_GEMM=1
export VLLM_HCU_STEP5_FP8_DENSE=1 VLLM_HCU_STEP5_FP8_BF16_MOE=1
export VLLM_HCU_EP_FUSED_MOE=0 VLLM_HCU_EP_FUSED_COMBINE=1
export VLLM_HCU_USE_CUSTOM_QUANTIZATION_GEMM=1 VLLM_HCU_USE_FUSE_MOE_GATE=1
export VLLM_HCU_USE_CUSTOM_SILU_AND_MUL=1 VLLM_HCU_USE_CUSTOM_GEMMA_RMS_NORM=1
export VLLM_HCU_USE_FLASH_ATTN=1 VLLM_HCU_USE_FLASH_ATTN_UNIFIED=1
export VLLM_HCU_USE_FLASH_ATTN_VARLEN=0 VLLM_STEP4_DSA_FLASH_ATTN=prefill,decode
export VLLM_HCU_STEP5_SWA_FUSED_QK=1 VLLM_HCU_STEP5_FUSED_GATE=1
export VLLM_HCU_TRITON_TOKEN_FP8_QUANT=1 VLLM_HCU_STEP5_QKV_INDEXER=1
export VLLM_HCU_PIN_MEMORY=1 VLLM_USE_FLASH_ATTN_FP8=1 VLLM_USE_PD_SPLIT=1
export VLLM_HCU_USE_LIGHTOP_EP_SCATTER=1 VLLM_HCU_USE_LIGHTOP_MOE_ALIGN=1
export VLLM_HCU_DP_CUSTOM_AG=0 VLLM_HCU_DP_PACKED_AG=0 VLLM_ROCM_USE_AITER_MOE=1
export VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD=0 FA_DECODE_NUM_SPLITS=2
export HCU_DP_MOE_PIPE_MIN_TOKENS=0 HCU_EP_DIRECT_SUM=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HSA_FORCE_FINE_GRAIN_PCIE=1 HIP_FORCE_DEV_KERNARG=1 GPU_MAX_HW_QUEUES=4
export NCCL_IB_DISABLE=1 NCCL_MAX_NCHANNELS=32 VLLM_HOST_IP=127.0.0.1
export DEEP_EP_NORMAL_MNVL=0 VLLM_DEEPEP_HIGH_THROUGHPUT_FORCE_INTRA_NODE=1
export VLLM_DEEPEP_BUFFER_SIZE_MB=1536 HCU_DBO_MIN_TOKENS=4096
export HCU_STEP5_MEGAMOE_PREFILL=0 HCU_STEP5_DSA_TOPK_EXACT_EMIT=1
export HCU_STEP5_DEEPEP_FUSED_SCATTER=1 HCU_STEP5_DEEPEP_COMPACT_ROWS=1
export HCU_STEP5_DEEPEP_PROFILE_DBO_OUTPUT=1 HCU_STEP5_DEEPEP_DBO_BUFFER_PAIR=1
export HCU_STEP5_DENSE_TUNING=1 HCU_STEP5_EP8_SPARSE_GROUP_MAJOR=1
export HCU_MTP_PREFILL_TAIL=1536 HCU_SHARED_EXPERTS_STREAM_MIN_TOKENS=4096
export VLLM_STEP4_DSA_GRAPH_REGIONS=17408 HCU_STEP5_CONTEXT_GRAPH_REGIONS=0

export LD_LIBRARY_PATH="$hipblaslt${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HIPBLASLT_TENSILE_LIBPATH="$hipblaslt/library_gfx938"
export HIPBLASLT_EXT_OP_LIBRARY_PATH="$HIPBLASLT_TENSILE_LIBPATH/hipblasltExtOpLibrary.dat"
export VLLM_CACHE_ROOT="$run_dir/cache/vllm"
export TORCHINDUCTOR_CACHE_DIR="$run_dir/cache/inductor"
export TRITON_CACHE_DIR="$run_dir/cache/triton"
export HCU_STEP5_DEEPEP_RUNTIME_CONFIG="$run_dir/communication.json"

comm='{"version":1,"num_sms":16,"dispatch_send":8,"combine_send":8,"recv_tokens":512,"async_layout":true,"overlap":true}'
compilation='{"mode":3,"cudagraph_mode":"NONE","compile_sizes":[16,64],"custom_ops":["+swiglustep_and_mul"],"inductor_compile_config":{"combo_kernels":false,"benchmark_combo_kernel":false,"coordinate_descent_tuning":false,"max_autotune":false,"max_autotune_pointwise":false,"max_autotune_gemm":false}}'
command=("$python" -m vllm.entrypoints.cli.main serve "$model"
    --tokenizer "$tokenizer" --served-model-name step5 --reasoning-parser step3p5
    --host 127.0.0.1 --port "$port" --tensor-parallel-size 1 --data-parallel-size 8
    --enable-expert-parallel --all2all-backend deepep_high_throughput --moe-backend auto
    --max-num-batched-tokens 8192 --max-model-len 140000 --max-num-seqs 16
    --no-enable-prefix-caching --kv-cache-dtype fp8_e4m3 --limit-mm-per-prompt '{"image":0}'
    --gpu-memory-utilization 0.90 --max-cudagraph-capture-size 64
    --compilation-config "$compilation"
    --speculative-config '{"method":"mtp","num_speculative_tokens":3,"use_local_argmax_reduction":true}'
    --additional-config '{"hcu":{"hcu_flash_attn_mode":"cutlass"}}' --api-server-count 1)

if [[ "${1:-}" == --dry-run ]]; then
    printf '%s\n' "$comm"
    printf '%q ' "${command[@]}"
    printf '\n'
    exit
fi

[[ -r "$HIPBLASLT_EXT_OP_LIBRARY_PATH" ]] || {
    printf 'Missing hipBLASLt extension library: %s\n' "$HIPBLASLT_EXT_OP_LIBRARY_PATH" >&2
    exit 1
}
mkdir -p "$run_dir"
(set -o noclobber; printf '%s\n' "$comm" > "$HCU_STEP5_DEEPEP_RUNTIME_CONFIG")
exec "${command[@]}"
