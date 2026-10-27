#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
rank="${1:-0}"
[[ "$rank" == 0 || "$rank" == 1 ]] || { echo 'rank must be 0 or 1' >&2; exit 2; }
: "${MODEL:?Set MODEL to the Kimi-K3 checkpoint}"
: "${RUN_DIR:?Set RUN_DIR to an absolute, shared container path}"
mkdir -p "$RUN_DIR/logs" "$RUN_DIR/trace"
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_KIMI_KDA_PREFILL_BACKEND="${VLLM_KIMI_KDA_PREFILL_BACKEND:-boltops}"
export VLLM_KIMI_KDA_CACHE_STRATEGY="${VLLM_KIMI_KDA_CACHE_STRATEGY:-extra_buffer}"
export VLLM_KIMI_KDA_STATE_BACKEND="${VLLM_KIMI_KDA_STATE_BACKEND:-boltops}"
export VLLM_KIMI_KDA_DEBUG="${VLLM_KIMI_KDA_DEBUG:-0}"
export VLLM_HCU_USE_CUSTOM_OPS=1
export VLLM_KIMI_FLASHKDA_BACKEND=1
export VLLM_KIMI_CONV1D_BACKEND=1
export VLLM_USE_FLASHMLA=1
export VLLM_ROCM_USE_AITER_CUSTOM_AR=1
export AITER_AR_MAX_SIZE_MB=256
export NCCL_MAX_NCHANNELS=1
export GPU_MAX_HW_QUEUES=4
export PYTHONUNBUFFERED=1
export VLLM_USE_NN=0
export VLLM_ENGINE_READY_TIMEOUT_S=7200
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-ens61f1np1}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-ens61f1np1}"
export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1,mlx5_8:1,mlx5_9:1}"
if [[ "$rank" == 0 ]]; then
  export VLLM_HOST_IP="${VLLM_HOST_IP:-${MASTER_ADDR:-13.13.4.22}}"
else
  export VLLM_HOST_IP="${VLLM_HOST_IP:-${NODE1_HOST:-13.13.4.20}}"
fi
unset PYTHONPATH VLLM_SKIP_WEIGHT
cmd=(vllm serve "$MODEL" --trust-remote-code --language-model-only
  --quantization slimquant_w4a8 --load-format fastsafetensors --dtype bfloat16
  --tensor-parallel-size "${TP_SIZE:-8}" --pipeline-parallel-size "${PP_SIZE:-2}"
  --nnodes "${NNODES:-2}" --node-rank "$rank" --distributed-executor-backend mp
  --master-addr "${MASTER_ADDR:-13.13.4.22}" --master-port "${MASTER_PORT:-25403}"
  --distributed-timeout-seconds 7200 --max-model-len "${MAX_MODEL_LEN:-16384}"
  --gpu-memory-utilization 0.85 --kv-cache-dtype fp8_e4m3 --block-size 64
  --mamba-ssm-cache-dtype bfloat16 --enable-prefix-caching --mamba-cache-mode align
  --max-num-seqs 128 --max-num-batched-tokens "${TOKEN_BUDGET:-8192}"
  --enforce-eager --reasoning-parser kimi_k3 --enable-auto-tool-choice --tool-call-parser kimi_k3)
if [[ "${KIMI_LAYERS:-0}" != 0 ]]; then
  cmd+=(--hf-overrides "{\"text_config\":{\"num_hidden_layers\":${KIMI_LAYERS}}}")
fi
if [[ "${ENABLE_PROFILER:-0}" == 1 ]]; then
  profile_config="$(python3 - "$RUN_DIR/trace" <<'PY'
import json, sys
print(json.dumps(dict(profiler="torch", torch_profiler_dir=sys.argv[1],
    torch_profiler_with_stack=True, torch_profiler_record_shapes=True,
    torch_profiler_with_memory=False, torch_profiler_use_gzip=True,
    ignore_frontend=True, max_iterations=0)))
PY
)"
  cmd+=(--profiler-config "$profile_config")
fi
if [[ "$rank" == 0 ]]; then
  cmd+=(--host 0.0.0.0 --port "${API_PORT:-19401}")
else
  cmd+=(--headless)
fi
printf '%q ' "${cmd[@]}" > "$RUN_DIR/logs/command_node${rank}.txt"
printf '\n' >> "$RUN_DIR/logs/command_node${rank}.txt"
python3 - "$RUN_DIR/logs/env_node${rank}.json" <<'PY'
import json, os, sys
names = ["VLLM_KIMI_KDA_PREFILL_BACKEND", "VLLM_KIMI_KDA_CACHE_STRATEGY",
         "VLLM_KIMI_KDA_STATE_BACKEND",
         "VLLM_KIMI_KDA_DEBUG",
         "VLLM_HCU_USE_CUSTOM_OPS",
         "VLLM_USE_V2_MODEL_RUNNER", "VLLM_KIMI_FLASHKDA_BACKEND", "VLLM_KIMI_CONV1D_BACKEND",
         "VLLM_ROCM_USE_AITER_CUSTOM_AR", "AITER_AR_MAX_SIZE_MB", "VLLM_HOST_IP",
         "HIP_VISIBLE_DEVICES", "NCCL_SOCKET_IFNAME", "NCCL_IB_HCA", "NCCL_MAX_NCHANNELS"]
with open(sys.argv[1], "w") as stream:
    json.dump({name: os.environ.get(name) for name in names}, stream, indent=2)
PY
exec "${cmd[@]}" 2>&1 | tee "$RUN_DIR/logs/node${rank}_$(hostname -s).log"
