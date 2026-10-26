#!/usr/bin/env bash
# Step5 performance profile; synthetic MTP3 is for throughput measurement.
set -euo pipefail

# Four nodes run the same script. Set NODE_RANK=0,1,2,3 respectively.
: "${NODE_RANK:?Set NODE_RANK=0,1,2,3 on p5c1,p5c2,p5c3,p5c4 respectively}"
case "$NODE_RANK" in 0|1|2|3) ;; *) printf 'Invalid NODE_RANK\n' >&2; exit 2 ;; esac
STEP5_NODE_IPS=(172.20.2.22 172.20.2.23 172.20.2.24 172.20.2.25)
HEAD_IP="${HEAD_IP:-172.20.2.22}"
NODE_IP="${NODE_IP:-${STEP5_NODE_IPS[$NODE_RANK]}}"
STEP5_IFACE="${STEP5_IFACE:-eno2}"
STEP5_MODEL_DIR="${STEP5_MODEL_DIR:-/data/model/Step5-Channel-FP8-w8a8}"
STEP5_RELEASE_DIR="${STEP5_RELEASE_DIR:-/opt/step5-release}"
STEP5_CACHE_DIR="${STEP5_CACHE_DIR:-/var/cache/step5-release}"

# Clear the other service profile when switching Prefill/Decode.
for step5_name in $(compgen -v); do
  case "$step5_name" in
    HCU_STEP5_*|VLLM_STEP5_*|VLLM_STEP4_*|HCU_DP_MOE_*|HCU_DBO_*|HCU_STATIC_EPLB_*) unset "$step5_name" ;;
  esac
done
unset PYTHONPATH LD_PRELOAD LD_AUDIT HIPBLASLT_TUNING_OVERRIDE_FILE ROCBLAS_TENSILE_LIBPATH
unset PYTORCH_ALLOC_CONF PYTORCH_CUDA_ALLOC_CONF VLLM_PLUGINS EP_FUSED_MOE_TABLE
unset NCCL_TOPO_FILE NCCL_IB_DISABLE NCCL_MIN_NCHANNELS NCCL_MAX_NCHANNELS
unset VLLM_HCU_DP_PACKED_AG VLLM_USE_DEEP_GEMM VLLM_MOE_USE_DEEP_GEMM DEEP_EP_NORMAL_MNVL
export VLLM_HOST_IP="$NODE_IP"
export NCCL_SOCKET_IFNAME="$STEP5_IFACE" GLOO_SOCKET_IFNAME="$STEP5_IFACE"

# Validated operator and communication switches.
export HSA_FORCE_FINE_GRAIN_PCIE=1
export OMP_NUM_THREADS=1
export GPU_MAX_HW_QUEUES=4
export VLLM_HCU_USE_CUSTOM_QUANTIZATION_GEMM=1
export VLLM_HCU_USE_FUSE_MOE_GATE=1
export VLLM_HCU_USE_FLASH_ATTN=1
export VLLM_HCU_USE_FLASH_ATTN_UNIFIED=1
export VLLM_USE_PD_SPLIT=1
export VLLM_HCU_USE_CUSTOM_OPS=1
export VLLM_HCU_USE_LIGHTOP_EP_SCATTER=1
export VLLM_HCU_USE_CUSTOM_SILU_AND_MUL=1
export VLLM_HCU_USE_CUSTOM_GEMMA_RMS_NORM=1
export VLLM_ROCM_USE_AITER_MOE=1
export VLLM_USE_FLASH_ATTN_FP8=1
export VLLM_HCU_USE_LIGHTOP_MOE_ALIGN=1
export FA_DECODE_NUM_SPLITS=2
export VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD=0
export VLLM_HCU_DP_CUSTOM_AG=1
export VLLM_HCU_STEP5_FP8_DENSE=1
export VLLM_HCU_STEP5_FP8_BF16_MOE=1
export VLLM_HCU_STEP5_SWA_FUSED_QK=1
export VLLM_HCU_STEP5_FUSED_GATE=1
export VLLM_HCU_TRITON_TOKEN_FP8_QUANT=1
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export VLLM_HCU_PIN_MEMORY=1
export HIP_FORCE_DEV_KERNARG=1
export VLLM_SERVER_DEV_MODE=1
export VLLM_USE_V2_MODEL_RUNNER=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NCCL_DEBUG=INFO
export VLLM_HCU_STEP5_QKV_INDEXER=1
export VLLM_STEP4_DSA_FLASH_ATTN=prefill,decode
export HCU_MTP_PREFILL_TAIL=1536
export HCU_SHARED_EXPERTS_STREAM_MIN_TOKENS=4096
export HCU_DP_MOE_PIPE_MIN_TOKENS=0
export HCU_DBO_MIN_TOKENS=4096
export HCU_EP_DIRECT_SUM=0
export HCU_STEP5_EXPERT_GRID=0
export HCU_STEP5_FUSED_ROUTER_STREAM=0
export HCU_STEP5_BOUNDED_EP_WORKSPACE=0
export EP_FUSED_MOE_CHUNK=32768
export VLLM_STEP4_DSA_GRAPH_REGIONS=8704
export VLLM_HCU_EP_FUSED_MOE=0
export VLLM_HCU_EP_FUSED_COMBINE=1
export HCU_STEP5_DENSE_TUNING=1
export HCU_STEP5_PREFILL_ASM_G1=0
export HCU_STEP5_PREFILL_G2=0
export HCU_STEP5_SPARSE_RS=0
export VLLM_HCU_SHARED_EXPERTS_EARLY_LAUNCH=0
export VLLM_DEEPEP_BUFFER_SIZE_MB=1536
export HCU_STEP5_DEEPEP_FUSED_SCATTER=1
export HCU_STEP5_DEEPEP_COMPACT_ROWS=1
export HCU_STEP5_DEEPEP_PROFILE_DBO_OUTPUT=1
export HCU_STEP5_DEEPEP_DBO_BUFFER_PAIR=1
export DEEP_EP_NORMAL_MNVL=1
export VLLM_USE_DEEP_GEMM=1
export VLLM_MOE_USE_DEEP_GEMM=1
export NCCL_MAX_NCHANNELS=32

# Prefill: DP16/TP1/EP16, fixed E23, DeepEP MNVL + DeepGEMM/WASP.
STEP5_STATE_DIR="${STEP5_STATE_DIR:-/var/lib/step5-release/run}"
STEP5_RUN_HOME="$STEP5_STATE_DIR/home"
STEP5_PREFILL_ASSETS="${STEP5_PREFILL_ASSETS:-$STEP5_RELEASE_DIR/assets}"
STEP5_TOKENIZER="${STEP5_TOKENIZER:-$STEP5_PREFILL_ASSETS/tokenizer-bytelevel}"
export HCU_STEP5_TUNING_CONFIG="$STEP5_PREFILL_ASSETS/tuning/step5-gfx938-image1021.json"
export VLLM_STEP5_EXPERT_PLACEMENT="$STEP5_PREFILL_ASSETS/expert_placement_prefill_8192_ep16_20261020.pt"
STEP5_EXPERT_STATIC_MAP="$STEP5_PREFILL_ASSETS/expert_placement_prefill_e23_static_tokenhash_20261023.json"
for required_file in "$STEP5_MODEL_DIR/config.json" "$HCU_STEP5_TUNING_CONFIG" "$VLLM_STEP5_EXPERT_PLACEMENT" "$STEP5_EXPERT_STATIC_MAP"; do
  [[ -r "$required_file" ]] || { printf 'Missing dependency: %s\n' "$required_file" >&2; exit 2; }
done

# Keep compilation caches on the supplied writable cache volume.
export TMPDIR="$STEP5_CACHE_DIR/tmp"
export XDG_CACHE_HOME="$STEP5_CACHE_DIR" HF_HOME="$STEP5_CACHE_DIR/huggingface"
export VLLM_CACHE_ROOT="$STEP5_CACHE_DIR/vllm"
export TORCHINDUCTOR_CACHE_DIR="$STEP5_CACHE_DIR/torchinductor"
export TRITON_CACHE_DIR="$STEP5_CACHE_DIR/triton" CUDA_CACHE_PATH="$STEP5_CACHE_DIR/cuda"
export NUMBA_CACHE_DIR="$STEP5_CACHE_DIR/numba"
export MIOPEN_USER_DB_PATH="$STEP5_CACHE_DIR/miopen" MIOPEN_CUSTOM_CACHE_DIR="$STEP5_CACHE_DIR/miopen"
export PYTHONPYCACHEPREFIX="$STEP5_CACHE_DIR/pycache"
mkdir -p "$STEP5_STATE_DIR" "$STEP5_RUN_HOME" "$TMPDIR" "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$MIOPEN_USER_DB_PATH"

# Fixed runtime policy, generated automatically; no dynamic weight migration.
export HCU_STEP5_DEEPEP_RUNTIME_CONFIG="$STEP5_STATE_DIR/communication.json"
cat > "$HCU_STEP5_DEEPEP_RUNTIME_CONFIG" <<'JSON'
{"version":1,"num_sms":16,"dispatch_send":8,"combine_send":8,"recv_tokens":512,"async_layout":true,"overlap":true}
JSON
STEP5_EPLB_CONFIG="{\"num_redundant_experts\":16,\"use_async\":false,\"disable_rearrange\":true,\"expert_map_path\":\"$STEP5_EXPERT_STATIC_MAP\",\"static_dispatch_policy\":\"token_hash\"}"
step5_role_args=(--api-server-count 1)
if [[ "$NODE_RANK" != 0 ]]; then
  step5_role_args=(--headless --data-parallel-start-rank "$((NODE_RANK * 4))")
fi

exec env HOME="$STEP5_RUN_HOME" vllm serve "$STEP5_MODEL_DIR" \
  --tokenizer "$STEP5_TOKENIZER" --served-model-name step5 \
  --host 127.0.0.1 --port 8018 \
  --tensor-parallel-size 1 --data-parallel-size 16 --data-parallel-size-local 4 \
  --data-parallel-address "$HEAD_IP" --data-parallel-rpc-port 18192 \
  --distributed-executor-backend mp --enable-expert-parallel \
  --all2all-backend deepep_high_throughput \
  --moe-backend auto \
  --max-num-batched-tokens 8192 --max-model-len 70000 --max-num-seqs 64 \
  --long-prefill-token-threshold 0 \
  --no-enable-prefix-caching \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3,"rejection_sample_method":"synthetic","synthetic_acceptance_rates":[0.838992,0.598322,0.362686],"use_local_argmax_reduction":true}' \
  --max-cudagraph-capture-size 64 \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["+swiglustep_and_mul"],"compile_sizes":[16,64]}' \
  --kv-cache-dtype fp8_e4m3 --limit-mm-per-prompt '{"image":0}' \
  --gpu-memory-utilization 0.875 \
  --enable-eplb --eplb-config "$STEP5_EPLB_CONFIG" \
  "${step5_role_args[@]}" "$@"
