#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mode="${1:-}"
case "$mode" in prefill|prefill-wasp128|decode|decode-perf) ;; *) echo 'Usage: serve.sh prefill|prefill-wasp128|decode|decode-perf [--dry-run]' >&2; exit 2;; esac
dry_run=0
if [[ "${2:-}" == --dry-run ]]; then dry_run=1; fi
[[ $# == 1 || ( $# == 2 && "$2" == --dry-run ) ]] || exit 2
source "${STEP5_CONFIG_FILE:-$script_dir/config.env}"
: "${NODE_RANK:?Set NODE_RANK=0 on the head node or 1 on its peer}"
[[ "$NODE_RANK" == 0 || "$NODE_RANK" == 1 ]] || exit 2
[[ ${#NODE_IPS[@]} == 2 ]] || exit 2
cache_base="$CACHE_ROOT/$mode/node$NODE_RANK"
CACHE_DIR="$cache_base"
if [[ "$mode" == prefill || "$mode" == prefill-wasp128 ]]; then
  unset VLLM_HCU_STEP5_DECODE_PRODUCERS VLLM_HCU_STEP5_DSA_FUSED_QK
  source "$script_dir/prefill.env"
  role=(--api-server-count 1)
  if [[ "$NODE_RANK" == 1 ]]; then role=(--headless --data-parallel-start-rank 8); fi
  eplb="{\"num_redundant_experts\":16,\"use_async\":false,\"disable_rearrange\":true,\"expert_map_path\":\"$STEP5_EXPERT_STATIC_MAP\",\"static_dispatch_policy\":\"token_hash\"}"
  phase=(--port 18128 --tensor-parallel-size 1 --data-parallel-size 16 --data-parallel-size-local 8
    --data-parallel-rpc-port 18392 --all2all-backend allgather_reducescatter
    --enable-eplb --eplb-config "$eplb" --max-num-batched-tokens 8192
    --long-prefill-token-threshold 0 --kv-cache-memory-bytes 5100273664
    --gpu-memory-utilization .590 --no-async-scheduling)
  if [[ "$mode" == prefill-wasp128 ]]; then
    source "$script_dir/prefill-wasp128.env"
    # Keep the same TP1/DP16 layout and physical expert map; use cross-node HT.
    for ((i=0; i<${#phase[@]}; i++)); do
      if [[ "${phase[$i]}" == --all2all-backend ]]; then phase[$((i+1))]=deepep_high_throughput; fi
    done
    phase+=(--moe-backend auto)
  fi
else
  if [[ "$mode" == decode-perf ]]; then
    export STEP5_SPECULATIVE_CONFIG='{"method":"mtp","num_speculative_tokens":3,"rejection_sample_method":"synthetic","synthetic_acceptance_rates":[0.838992,0.598322,0.362686],"use_local_argmax_reduction":true}'
  else
    export STEP5_SPECULATIVE_CONFIG='{"method":"mtp","num_speculative_tokens":3,"rejection_sample_method":"standard","use_local_argmax_reduction":true}'
  fi
  source "$script_dir/decode.env"
  role=(--api-server-count 1)
  if [[ "$NODE_RANK" == 1 ]]; then role=(--headless --data-parallel-start-rank 2); fi
  export VLLM_STEP5_EXPERT_PLACEMENT="$ASSETS_DIR/expert_placement_decode.pt"
  phase=(--port 18118 --tensor-parallel-size 4 --data-parallel-size 4 --data-parallel-size-local 2
    --data-parallel-rpc-port 18292 --all2all-backend deepep_low_latency --moe-backend deep_gemm
    --expert-placement-strategy linear --max-num-batched-tokens 512
    --long-prefill-token-threshold 32 --block-size 64 --gpu-memory-utilization .5
    --speculative-config "$STEP5_SPECULATIVE_CONFIG" --safetensors-load-strategy prefetch)
fi
export STEP5_GPU_ORDER_S7="$GPU_ORDER_HEAD" STEP5_GPU_ORDER_S9="$GPU_ORDER_PEER"
export HIP_VISIBLE_DEVICES="$GPU_ORDER_HEAD"
if [[ "$NODE_RANK" == 1 ]]; then export HIP_VISIBLE_DEVICES="$GPU_ORDER_PEER"; fi
export VLLM_HOST_IP="${NODE_IPS[$NODE_RANK]}"
export NCCL_SOCKET_IFNAME="$IFACE" GLOO_SOCKET_IFNAME="$IFACE" NCCL_IB_HCA="$IB_HCA"
export ROCSHMEM_BOOTSTRAP_SOCKET_IFNAME="$IFACE"
export TRITON_CACHE_DIR="$cache_base/triton" STEP5_TRITON_KERNEL_CACHE="$cache_base/triton"
export VLLM_CACHE_ROOT="$cache_base/vllm" TORCHINDUCTOR_CACHE_DIR="$cache_base/inductor"
export VLLM_USE_AOT_COMPILE=0 VLLM_DISABLE_COMPILE_CACHE=1
export TORCHINDUCTOR_FX_GRAPH_CACHE=0 TORCHINDUCTOR_AUTOGRAD_CACHE=0
export VLLM_SERVER_DEV_MODE=1
command=(vllm serve "$MODEL_DIR" --tokenizer "$TOKENIZER_DIR" --served-model-name step5
  --host 0.0.0.0 --data-parallel-address "${NODE_IPS[0]}" --distributed-executor-backend mp
  --enable-expert-parallel --max-model-len 70000 --max-num-seqs 64
  --max-cudagraph-capture-size 64 --no-enable-prefix-caching --kv-cache-dtype fp8_e4m3
  --compilation-config "$STEP5_COMPILATION_CONFIG" --limit-mm-per-prompt '{"image":0}'
  --worker-extension-cls vllm_hcu.v1.step5_source_identity.Step5SourceIdentityExtension
  "${phase[@]}" "${role[@]}")
if [[ "$dry_run" == 1 ]]; then printf '%q ' "${command[@]}"; printf '\n'; exit 0; fi
[[ -r "$MODEL_DIR/config.json" && -r "$TOKENIZER_DIR/tokenizer.json" ]] || { echo 'Model or tokenizer missing' >&2; exit 2; }
mkdir -p "$cache_base" "$TRITON_CACHE_DIR"
exec 9>"$cache_base/service.lock"
flock -n 9 || { echo 'This profile already has an active service' >&2; exit 2; }
ulimit -l unlimited
exec "${command[@]}"
