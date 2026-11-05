#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Two-node TP8/PP2 Kimi-K3 DSpark validation with ordinary checkpoint loading.
set -euo pipefail
rank="${1:?usage: run_kimi_k3_pp_dspark_server.sh NODE_RANK}"
[[ "$rank" == 0 || "$rank" == 1 ]] || { echo 'NODE_RANK must be 0 or 1' >&2; exit 2; }
: "${MODEL:?Set MODEL to the full Kimi-K3 checkpoint}"
: "${DRAFT_MODEL:?Set DRAFT_MODEL to the full DSpark checkpoint}"
: "${MASTER_ADDR:?Set MASTER_ADDR to node 0 IP}"
: "${VLLM_HOST_IP:?Set VLLM_HOST_IP to this node IP}"
: "${RUN_DIR:?Set RUN_DIR to a writable absolute path}"
mkdir -p "$RUN_DIR"

export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_USE_NN=0
export VLLM_HCU_USE_CUSTOM_OPS=1
export VLLM_KIMI_CONV1D_BACKEND=1
export VLLM_KIMI_FLASHKDA_BACKEND=1
export VLLM_KIMI_KDA_PREFILL_BACKEND=legacy
export VLLM_KIMI_KDA_CACHE_STRATEGY=align
export HSA_FORCE_FINE_GRAIN_PCIE=1
export GPU_MAX_HW_QUEUES=4
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export VLLM_ENGINE_READY_TIMEOUT_S=7200

spec=$(python3 - "$DRAFT_MODEL" <<'PY'
import json, sys
print(json.dumps({"model": sys.argv[1], "method": "dspark",
                  "num_speculative_tokens": 7,
                  "draft_sample_method": "probabilistic",
                  "rejection_sample_method": "block"}))
PY
)

# The draft runs on the last PP stage only. Use safetensors for both models:
# fastsafetensors uses WORLD collectives that the earlier stage cannot join
# during draft loading.
cmd=(python3 -m vllm.entrypoints.cli.main serve "$MODEL"
  --trust-remote-code --language-model-only --quantization slimquant_w4a8
  --load-format safetensors --dtype bfloat16 --speculative-config "$spec"
  --tensor-parallel-size 8 --pipeline-parallel-size 2 --nnodes 2
  --node-rank "$rank" --distributed-executor-backend mp
  --master-addr "$MASTER_ADDR" --master-port "${MASTER_PORT:-29562}"
  --distributed-timeout-seconds 7200 --max-model-len "${MAX_MODEL_LEN:-8192}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}"
  --kv-cache-dtype fp8_e4m3 --block-size 64 --disable-custom-all-reduce
  --mamba-ssm-cache-dtype bfloat16 --enable-prefix-caching --mamba-cache-mode align
  --max-num-seqs 1 --max-num-batched-tokens 1024 --enforce-eager
  --reasoning-parser kimi_k3 --no-async-scheduling)
if [[ "$rank" == 0 ]]; then
  cmd+=(--host 0.0.0.0 --port "${API_PORT:-19442}")
else
  cmd+=(--headless)
fi
printf '%q ' "${cmd[@]}" > "$RUN_DIR/node${rank}.command"
printf '\n' >> "$RUN_DIR/node${rank}.command"
python3 - "$RUN_DIR/node${rank}.environment.json" <<'PY'
import json, os, sys, vllm, vllm_hcu
names = ("PYTHONPATH", "VLLM_HOST_IP", "NCCL_SOCKET_IFNAME", "GLOO_SOCKET_IFNAME",
         "NCCL_IB_HCA", "VLLM_USE_NN", "HSA_FORCE_FINE_GRAIN_PCIE",
         "VLLM_KIMI_KDA_PREFILL_BACKEND", "VLLM_KIMI_KDA_CACHE_STRATEGY")
with open(sys.argv[1], "w") as stream:
    json.dump({"plugin_import": vllm_hcu.__file__, "vllm_version": vllm.__version__,
               "environment": {name: os.environ.get(name) for name in names}}, stream, indent=2)
PY
exec "${cmd[@]}"
