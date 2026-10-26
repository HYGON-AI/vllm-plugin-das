#!/usr/bin/env bash
# Four hosts x four BW1300_LC cards; benchmark-only synthetic MTP sampling.
set -euo pipefail

phase="${1:?Usage: launch_multinode.sh prefill|decode NODE_RANK}"
rank="${2:?NODE_RANK must be 0, 1, 2 or 3}"
case "$phase" in prefill|decode) ;; *) exit 2 ;; esac
case "$rank" in 0|1|2|3) ;; *) exit 2 ;; esac
: "${STEP5_HEAD_IP:?Set the head node IP reachable by all four nodes}"
: "${VLLM_HOST_IP:?Set the local node communication IP}"
: "${STEP5_TOKENIZER:?Set the tokenizer directory used by the benchmark}"
: "${STEP5_STATE_DIR:?Set a directory on the data-backed container disk}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source_dir="${STEP5_SOURCE_DIR:-$repo_root}"
model_dir="${STEP5_MODEL_DIR:-/data/model/Step5-Channel-FP8-w8a8}"
export PYTHONPATH="$source_dir${PYTHONPATH:+:$PYTHONPATH}"
export EP_FUSED_MOE_TABLE="${STEP5_EP_TABLE:?Set an external tuned MoE table; machine tuning data is not shipped in the source}"
export HOME="$STEP5_STATE_DIR/$phase/home"
# ZMQ appends a UUID to TMPDIR for Unix sockets (107-byte path limit).
export TMPDIR="${STEP5_TMP_DIR:-/var/tmp/xuwq_step5}"
export XDG_CACHE_HOME="${STEP5_CACHE_DIR:-$STEP5_STATE_DIR/cache}"
export HF_HOME="$XDG_CACHE_HOME/huggingface"
export VLLM_CACHE_ROOT="$XDG_CACHE_HOME/vllm"
export TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/torchinductor"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda"
export NUMBA_CACHE_DIR="$XDG_CACHE_HOME/numba"
export MIOPEN_USER_DB_PATH="$XDG_CACHE_HOME/miopen"
export MIOPEN_CUSTOM_CACHE_DIR="$XDG_CACHE_HOME/miopen"
export PYTHONPYCACHEPREFIX="$XDG_CACHE_HOME/pycache"
export VLLM_HOST_IP
export HSA_FORCE_FINE_GRAIN_PCIE=1 OMP_NUM_THREADS=1
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eno2}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-$NCCL_SOCKET_IFNAME}"
unset NCCL_TOPO_FILE NCCL_IB_DISABLE NCCL_MAX_NCHANNELS NCCL_MIN_NCHANNELS
unset VLLM_STEP5_EXPERT_PLACEMENT
unset VLLM_HCU_DP_PACKED_AG HCU_STEP5_ROUTER_SHARED_MAX_M
unset VLLM_PLUGINS ROCBLAS_TENSILE_LIBPATH HIPBLASLT_TUNING_OVERRIDE_FILE
export GPU_MAX_HW_QUEUES=4
export VLLM_HCU_USE_CUSTOM_QUANTIZATION_GEMM=1
export VLLM_HCU_USE_FUSE_MOE_GATE=1
export VLLM_HCU_USE_FLASH_ATTN=1 VLLM_HCU_USE_FLASH_ATTN_UNIFIED=1
export VLLM_USE_PD_SPLIT=1 VLLM_HCU_USE_CUSTOM_OPS=1
export VLLM_HCU_USE_LIGHTOP_EP_SCATTER=1
export VLLM_HCU_USE_CUSTOM_SILU_AND_MUL=1
export VLLM_HCU_USE_CUSTOM_GEMMA_RMS_NORM=1
export VLLM_ROCM_USE_AITER_MOE=1 VLLM_USE_FLASH_ATTN_FP8=1
export VLLM_HCU_USE_LIGHTOP_MOE_ALIGN=1 FA_DECODE_NUM_SPLITS=2
export VLLM_HCU_EP_FUSED_MOE="${STEP5_PREFILL_EP_FUSED_MOE:-1}" VLLM_HCU_EP_FUSED_COMBINE=1
case "$VLLM_HCU_EP_FUSED_MOE" in 0|1) ;; *) printf 'Unsupported EP fused-MoE switch\n' >&2; exit 2 ;; esac
export VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD=0
export VLLM_HCU_DP_CUSTOM_AG=1
export VLLM_HCU_STEP5_FP8_DENSE=1 VLLM_HCU_STEP5_FP8_BF16_MOE=1
export VLLM_HCU_STEP5_SWA_FUSED_QK=1 VLLM_HCU_STEP5_FUSED_GATE=1
export VLLM_HCU_TRITON_TOKEN_FP8_QUANT=1
export VLLM_ENGINE_READY_TIMEOUT_S=3600 VLLM_HCU_PIN_MEMORY=1
export HIP_FORCE_DEV_KERNARG=1 VLLM_SERVER_DEV_MODE=1
export VLLM_USE_V2_MODEL_RUNNER=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NCCL_DEBUG=INFO
export VLLM_HCU_STEP5_QKV_INDEXER=1
export VLLM_STEP4_DSA_FLASH_ATTN=prefill,decode
export HCU_MTP_PREFILL_TAIL=1536 HCU_SHARED_EXPERTS_STREAM_MIN_TOKENS=4096
export HCU_DP_MOE_PIPE_MIN_TOKENS=0 HCU_DBO_MIN_TOKENS=0 HCU_EP_DIRECT_SUM=0
export HCU_STEP5_EXPERT_GRID=0
export HCU_STEP5_FUSED_ROUTER_STREAM=0
export HCU_STEP5_BOUNDED_EP_WORKSPACE=0
export EP_FUSED_MOE_CHUNK=32768
export VLLM_STEP4_DSA_GRAPH_REGIONS=8704
scheduler_extra=()
moe_backend_extra=()
eplb_extra=()
profiler_extra=()
table_extra=()
physical_experts=352

if [[ "$phase" == prefill ]]; then
    unset HCU_STEP5_DP_FABRIC HCU_STEP5_CONTEXT_GRAPH_REGIONS
    tp=1; dp="${STEP5_PREFILL_DP_SIZE:-16}"; local_dp=4; tokens="${STEP5_PREFILL_BATCHED_TOKENS:-8192}"; max_len="${STEP5_PREFILL_MAX_MODEL_LEN:-70000}"; gpu_util="${STEP5_PREFILL_GPU_MEMORY_UTILIZATION:-0.875}"
    case "$dp" in 8|16) ;; *) printf 'Unsupported Prefill DP size\n' >&2; exit 2 ;; esac
    ((rank * local_dp < dp)) || { printf 'Node rank exceeds Prefill DP size\n' >&2; exit 2; }
    api_host="${STEP5_API_HOST:-127.0.0.1}"; api_port="${STEP5_PORT:-8018}"
    case "$gpu_util" in 0.875|0.91) ;; *) printf 'Unsupported Prefill GPU memory utilization\n' >&2; exit 2 ;; esac
    case "$max_len" in 70000|1048576) ;; *) printf 'Unsupported Prefill context limit\n' >&2; exit 2 ;; esac
    export HCU_STEP5_BOUNDED_EP_WORKSPACE="${STEP5_PREFILL_BOUNDED_WORKSPACE:-0}"
    case "$HCU_STEP5_BOUNDED_EP_WORKSPACE" in 0|1) ;; *) printf 'Unsupported Prefill workspace switch\n' >&2; exit 2 ;; esac
    case "$tokens" in 4096|8192|16384|32768) ;; *) printf 'Unsupported Prefill batched-token limit\n' >&2; exit 2 ;; esac
    export EP_FUSED_MOE_CHUNK="${STEP5_PREFILL_MOE_CHUNK:-32768}"
    case "$EP_FUSED_MOE_CHUNK" in 16384|32768|65536|131072|262144) ;; *) printf 'Unsupported Prefill MoE chunk size\n' >&2; exit 2 ;; esac
    export HCU_DP_MOE_PIPE_MIN_TOKENS="${STEP5_PREFILL_PIPE_MIN_TOKENS:-0}"
    export HCU_DBO_MIN_TOKENS="${STEP5_PREFILL_DBO_MIN_TOKENS:-$HCU_DBO_MIN_TOKENS}"
    export HCU_EP_DIRECT_SUM="${STEP5_PREFILL_DIRECT_SUM_MIN_TOKENS:-$HCU_EP_DIRECT_SUM}"
    for threshold in "$HCU_DBO_MIN_TOKENS" "$HCU_EP_DIRECT_SUM"; do
        [[ "$threshold" =~ ^[0-9]+$ ]] || { printf 'Invalid Prefill overlap threshold\n' >&2; exit 2; }
    done
    case "$HCU_DP_MOE_PIPE_MIN_TOKENS" in 0|16384|32768|65536|131072|262144) ;; *) printf 'Unsupported Prefill pipeline threshold\n' >&2; exit 2 ;; esac
    export HCU_SHARED_EXPERTS_STREAM_MIN_TOKENS="${STEP5_PREFILL_SHARED_STREAM_MIN_TOKENS:-4096}"
    case "$HCU_SHARED_EXPERTS_STREAM_MIN_TOKENS" in 0|4096|8192|16384) ;; *) printf 'Unsupported Prefill shared-expert stream threshold\n' >&2; exit 2 ;; esac
    case "${STEP5_PREFILL_MOE_BACKEND:-auto}" in
        auto) ;;
        aiter) moe_backend_extra=(--moe-backend aiter) ;;
        *) printf 'Unsupported Prefill MoE backend\n' >&2; exit 2 ;;
    esac
    # Explicit reproduction profiles; historical defaults remain unchanged.
    case "${STEP5_PREFILL_PROFILE:-baseline}" in
        baseline) ;;
        dense_1021|g2_1021|asm_g1_1021|e23_asm_1021)
            [[ "$dp" == 16 && "$tp" == 1 && "$tokens" == 8192 && "$max_len" == 70000 && "$gpu_util" == 0.875 ]] || {
                printf '1021 Prefill profiles require DP16/TP1, tokens8192, length70000 and memory0.875\n' >&2; exit 2;
            }
            [[ "$VLLM_HCU_EP_FUSED_MOE" == 1 && "${STEP5_PREFILL_MOE_BACKEND:-auto}" == auto ]] || {
                printf '1021 Prefill profiles require the existing EP fused MoE path\n' >&2; exit 2;
            }
            : "${STEP5_PREFILL_PLACEMENT:?Set the external measured E22 Prefill placement file}"
            : "${HCU_STEP5_TUNING_CONFIG:?Set the external Dense/ASM tuning profile}"
            export HCU_DP_MOE_PIPE_MIN_TOKENS=65536 HCU_DBO_MIN_TOKENS=4096 HCU_EP_DIRECT_SUM=65536
            export EP_FUSED_MOE_CHUNK=65536 HCU_STEP5_BOUNDED_EP_WORKSPACE=1
            export HCU_SHARED_EXPERTS_STREAM_MIN_TOKENS=4096 HCU_DP_MOE_PIPE_UNIFORM=1
            export HCU_STEP5_DENSE_TUNING=1 HCU_STEP5_PREFILL_G2=0 HCU_STEP5_PREFILL_ASM_G1=0
            case "$STEP5_PREFILL_PROFILE" in
                g2_1021) export HCU_STEP5_PREFILL_G2=1 ;;
                asm_g1_1021)
                    [[ "${STEP5_PREFILL_REDUNDANT_EXPERTS:-0}" == 0 && -z "${STEP5_PREFILL_STATIC_MAP:-}" ]] || {
                        printf 'E22 ASM profile does not use redundant experts\n' >&2; exit 2;
                    }
                    export HCU_STEP5_PREFILL_ASM_G1=1 HCU_STEP5_PREFILL_G2=1
                    ;;
                e23_asm_1021)
                    [[ "${STEP5_PREFILL_REDUNDANT_EXPERTS:-0}" == 16 && -r "${STEP5_PREFILL_STATIC_MAP:-}" ]] || {
                        printf 'E23 ASM requires 16 fixed replicas and a readable static map\n' >&2; exit 2;
                    }
                    export HCU_STEP5_PREFILL_ASM_G1=1 HCU_STEP5_PREFILL_G2=1 GPU_MAX_HW_QUEUES=8
                    ;;
            esac
            scheduler_extra=(--long-prefill-token-threshold 0)
            ;;
        *) printf 'Unsupported Prefill reproduction profile\n' >&2; exit 2 ;;
    esac
    export NCCL_MAX_NCHANNELS=32
    if [[ -n "${STEP5_PREFILL_PLACEMENT:-}" ]]; then
        export VLLM_STEP5_EXPERT_PLACEMENT="$STEP5_PREFILL_PLACEMENT"
    fi
    redundant="${STEP5_PREFILL_REDUNDANT_EXPERTS:-0}"
    static_map="${STEP5_PREFILL_STATIC_MAP:-}"
    case "$redundant" in
        0) [[ -z "$static_map" ]] || { printf 'Static Prefill map requires 16 redundant experts\n' >&2; exit 2; } ;;
        16)
            [[ "$dp" == 16 && -r "$static_map" ]] || { printf 'E23 requires DP16 and a readable static map\n' >&2; exit 2; }
            eplb_json="$(python3 - "$static_map" <<'PY'
import json,sys
print(json.dumps(dict(num_redundant_experts=16,use_async=False,disable_rearrange=True,
                     expert_map_path=sys.argv[1],static_dispatch_policy='token_hash')))
PY
)"
            eplb_extra=(--enable-eplb --eplb-config "$eplb_json")
            physical_experts=368
            table_extra=(--require-m 65536)
            ;;
        *) printf 'Unsupported static Prefill redundant-expert count\n' >&2; exit 2 ;;
    esac
    if [[ -n "${STEP5_PREFILL_PROFILER_CONFIG:-}" ]]; then
        profiler_extra=(--profiler-config "$STEP5_PREFILL_PROFILER_CONFIG")
    fi
else
    export HCU_STEP5_DENSE_TUNING=0 HCU_STEP5_PREFILL_G2=0 HCU_STEP5_PREFILL_ASM_G1=0 HCU_DP_MOE_PIPE_UNIFORM=0
    tokens=32768; max_len=1048576
    case "${STEP5_DECODE_PARALLEL:-dp4tp4}" in
        dp4tp4) tp=4; dp=4; local_dp=1; gpu_util=0.85 ;;
        dp16tp1)
            tp=1; dp=16; local_dp=4; gpu_util=0.91
            # Size persistent DSA scores from the advertised context limit.
            # The legacy narrow graph rejects longer decode histories.
            export VLLM_STEP4_DSA_GRAPH_REGIONS=0
            export HCU_STEP5_BOUNDED_EP_WORKSPACE=1
            export EP_FUSED_MOE_CHUNK="${STEP5_DECODE_MOE_CHUNK:-65536}"
            case "$EP_FUSED_MOE_CHUNK" in 32768|65536) ;; *) printf 'Unsupported Decode MoE chunk\n' >&2; exit 2 ;; esac
            # Bound each prompt's score matrix without lowering the scheduler's
            # total token budget or its 1M context limit. Live validation is needed.
            threshold="${STEP5_DECODE_LONG_PREFILL_TOKEN_THRESHOLD:-2048}"
            case "$threshold" in 0|1024|2048|4096|8192) ;; *) printf 'Unsupported Decode long-prefill threshold\n' >&2; exit 2 ;; esac
            if [[ "$threshold" != 0 ]]; then
                scheduler_extra=(--long-prefill-token-threshold "$threshold")
            fi
            ;;
        *) printf 'Unsupported Decode parallel configuration\n' >&2; exit 2 ;;
    esac
    gpu_util="${STEP5_DECODE_GPU_MEMORY_UTILIZATION:-$gpu_util}"
    case "$gpu_util" in 0.85|0.875|0.91) ;; *) printf 'Unsupported Decode GPU memory utilization\n' >&2; exit 2 ;; esac
    api_host="${STEP5_API_HOST:-0.0.0.0}"; api_port="${STEP5_PORT:-18118}"
    export HCU_STEP5_EXPERT_GRID="${STEP5_DECODE_EXPERT_GRID:-0}"
    case "$HCU_STEP5_EXPERT_GRID" in 0|1) ;; *) printf 'Unsupported Decode expert-grid switch\n' >&2; exit 2 ;; esac
    export HCU_STEP5_FUSED_ROUTER_STREAM="${STEP5_DECODE_FUSED_ROUTER_STREAM:-0}"
    case "$HCU_STEP5_FUSED_ROUTER_STREAM" in 0|1) ;; *) printf 'Unsupported Decode fused-router stream switch\n' >&2; exit 2 ;; esac
    if [[ -n "${STEP5_DECODE_NCCL_CHANNELS:-}" ]]; then
        case "$STEP5_DECODE_NCCL_CHANNELS" in 1|2|4|8|16|32|64) ;; *) printf 'Unsupported Decode NCCL channel count\n' >&2; exit 2 ;; esac
        export NCCL_MIN_NCHANNELS="$STEP5_DECODE_NCCL_CHANNELS"
        export NCCL_MAX_NCHANNELS="$STEP5_DECODE_NCCL_CHANNELS"
    fi
    if [[ "${STEP5_DECODE_PROFILE:-}" == itl6479 ]]; then
        [[ "$dp" == 4 && "$tp" == 4 ]] || { printf 'itl6479 requires DP4/TP4\n' >&2; exit 2; }
        export VLLM_HCU_DP_PACKED_AG=1
        export HCU_STEP5_ROUTER_SHARED_MAX_M="${STEP5_ROUTER_SHARED_MAX_M:-32}"
        export HCU_SHARED_EXPERTS_STREAM_MIN_TOKENS="${STEP5_SHARED_STREAM_MIN_TOKENS:-64}"
        export VLLM_STEP5_EXPERT_PLACEMENT="${STEP5_DECODE_PLACEMENT:?Set the measured Decode expert placement file}"
    elif [[ "${STEP5_DECODE_PROFILE:-}" == dp16_native ]]; then
        [[ "$dp" == 16 && "$tp" == 1 ]] || { printf 'dp16_native requires DP16/TP1\n' >&2; exit 2; }
        export HCU_STEP5_ROUTER_SHARED_MAX_M="${STEP5_ROUTER_SHARED_MAX_M:-32}"
        export HCU_SHARED_EXPERTS_STREAM_MIN_TOKENS="${STEP5_SHARED_STREAM_MIN_TOKENS:-64}"
    fi
fi
if [[ "$phase" == decode ]]; then
    case "${HCU_STEP5_DP_FABRIC:-0}" in
        0) ;;
        1)
            [[ "$dp" == 4 && "$tp" == 4 && "${STEP5_DECODE_PROFILE:-}" == itl6479 ]] || {
                printf 'Measured Decode Fabric requires the DP4/TP4 itl6479 profile\n' >&2; exit 2;
            }
            export VLLM_HCU_DP_CUSTOM_AG=0
            ;;
        *) printf 'Unsupported Decode Fabric switch\n' >&2; exit 2 ;;
    esac
    case "${HCU_STEP5_CONTEXT_GRAPH_REGIONS:-0}" in
        0) ;;
        8736)
            [[ "$dp" == 4 && "$tp" == 4 && "${STEP5_DECODE_PROFILE:-}" == itl6479 ]] || {
                printf 'Measured hot-context graph requires DP4/TP4 itl6479\n' >&2; exit 2;
            }
            export VLLM_STEP4_DSA_GRAPH_REGIONS=8736
            ;;
        *) printf 'Unsupported measured hot-context graph width\n' >&2; exit 2 ;;
    esac
fi
extra=(--api-server-count 1)
if [[ "$rank" != 0 ]]; then
    extra=(--headless --data-parallel-start-rank "$((rank * local_dp))")
fi
command=(python3 -u -m vllm.entrypoints.cli.main serve "$model_dir"
    --tokenizer "$STEP5_TOKENIZER" --served-model-name step5
    --host "$api_host" --port "$api_port"
    --tensor-parallel-size "$tp" --data-parallel-size "$dp"
    --data-parallel-size-local "$local_dp"
    --data-parallel-address "$STEP5_HEAD_IP"
    --data-parallel-rpc-port "${STEP5_RPC_PORT:-18192}"
    --distributed-executor-backend mp --enable-expert-parallel
    --all2all-backend allgather_reducescatter
    --max-model-len "$max_len" --max-num-seqs 64 --max-num-batched-tokens "$tokens"
    --kv-cache-dtype fp8_e4m3 --gpu-memory-utilization "$gpu_util"
    --no-enable-prefix-caching --max-cudagraph-capture-size 64
    --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["+swiglustep_and_mul"],"compile_sizes":[16,64]}'
    --speculative-config '{"method":"mtp","num_speculative_tokens":3,"rejection_sample_method":"synthetic","synthetic_acceptance_rates":[0.838992,0.598322,0.362686],"use_local_argmax_reduction":true}'
    --limit-mm-per-prompt '{"image":0}' "${moe_backend_extra[@]}" "${scheduler_extra[@]}" "${eplb_extra[@]}" "${profiler_extra[@]}" "${extra[@]}")
if [[ "${STEP5_DRY_RUN:-0}" == 1 ]]; then
    printf 'EP_FUSED_MOE_TABLE=%q\n' "$EP_FUSED_MOE_TABLE"
    printf 'VLLM_HCU_DP_PACKED_AG=%q HCU_STEP5_ROUTER_SHARED_MAX_M=%q HCU_SHARED_EXPERTS_STREAM_MIN_TOKENS=%q\n' "${VLLM_HCU_DP_PACKED_AG:-0}" "${HCU_STEP5_ROUTER_SHARED_MAX_M:-64}" "$HCU_SHARED_EXPERTS_STREAM_MIN_TOKENS"
    printf 'VLLM_STEP5_EXPERT_PLACEMENT=%q\n' "${VLLM_STEP5_EXPERT_PLACEMENT:-}"
    printf 'NCCL_MIN_NCHANNELS=%q NCCL_MAX_NCHANNELS=%q\n' "${NCCL_MIN_NCHANNELS:-auto}" "${NCCL_MAX_NCHANNELS:-auto}"
    printf 'HCU_DP_MOE_PIPE_MIN_TOKENS=%q\n' "$HCU_DP_MOE_PIPE_MIN_TOKENS"
    printf 'HCU_DBO_MIN_TOKENS=%q HCU_EP_DIRECT_SUM=%q HCU_DP_MOE_PIPE_UNIFORM=%q\n' "$HCU_DBO_MIN_TOKENS" "$HCU_EP_DIRECT_SUM" "${HCU_DP_MOE_PIPE_UNIFORM:-0}"
    printf 'HCU_STEP5_DENSE_TUNING=%q HCU_STEP5_PREFILL_G2=%q\n' "${HCU_STEP5_DENSE_TUNING:-0}" "${HCU_STEP5_PREFILL_G2:-0}"
    printf 'HCU_STEP5_PREFILL_ASM_G1=%q GPU_MAX_HW_QUEUES=%q\n' "${HCU_STEP5_PREFILL_ASM_G1:-0}" "$GPU_MAX_HW_QUEUES"
    printf 'VLLM_HCU_DP_CUSTOM_AG=%q\n' "$VLLM_HCU_DP_CUSTOM_AG"
    printf 'HCU_STEP5_DP_FABRIC=%q HCU_STEP5_CONTEXT_GRAPH_REGIONS=%q\n' "${HCU_STEP5_DP_FABRIC:-0}" "${HCU_STEP5_CONTEXT_GRAPH_REGIONS:-0}"
    printf 'HCU_STEP5_EXPERT_GRID=%q\n' "$HCU_STEP5_EXPERT_GRID"
    printf 'HCU_STEP5_FUSED_ROUTER_STREAM=%q\n' "$HCU_STEP5_FUSED_ROUTER_STREAM"
    printf 'EP_FUSED_MOE_CHUNK=%q\n' "$EP_FUSED_MOE_CHUNK"
    printf 'HCU_STEP5_BOUNDED_EP_WORKSPACE=%q\n' "$HCU_STEP5_BOUNDED_EP_WORKSPACE"
    printf 'VLLM_HCU_EP_FUSED_MOE=%q\n' "$VLLM_HCU_EP_FUSED_MOE"
    printf 'VLLM_STEP4_DSA_GRAPH_REGIONS=%q\n' "$VLLM_STEP4_DSA_GRAPH_REGIONS"
    printf '%q ' "${command[@]}"
    printf '\n'
    exit 0
fi
if [[ "${STEP5_DECODE_PROFILE:-}" == itl6479 && "$phase" == decode ]]; then
    [[ -f "$VLLM_STEP5_EXPERT_PLACEMENT" ]] || { printf 'Missing placement: %s\n' "$VLLM_STEP5_EXPERT_PLACEMENT" >&2; exit 2; }
fi
if [[ "$phase" == prefill && -n "${VLLM_STEP5_EXPERT_PLACEMENT:-}" ]]; then
    [[ -f "$VLLM_STEP5_EXPERT_PLACEMENT" ]] || { printf 'Missing Prefill placement: %s\n' "$VLLM_STEP5_EXPERT_PLACEMENT" >&2; exit 2; }
fi
python3 "$repo_root/benchmarks/step5/check_ep_table.py" "$EP_FUSED_MOE_TABLE" --ep-size "$((tp * dp))" --physical-experts "$physical_experts" "${table_extra[@]}"
mkdir -p "$TMPDIR" "$HOME" "$XDG_CACHE_HOME" "$VLLM_CACHE_ROOT" \
    "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$MIOPEN_USER_DB_PATH"
exec "${command[@]}"
