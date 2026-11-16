#!/usr/bin/env bash
set -euo pipefail
node=${1:?node name}
case "$node" in nmz20|nmz22) ;; *) exit 2;; esac
TASK=/home/xiesl/inference/verify_torch213_image/task_dir/support_kimi31_pro
export PYTHONPATH="$TASK/quant_w4a8_tp16/experimental_entry:/home/xiesl/inference/verify_torch213_image/code_dir/plugin/adapter_extension/src"
export HCU_ADAPTER_VLLM_REQUIRED_EXTENSIONS=kimi31_slimquant_exp
export K31_PROBE_OUTPUT="$TASK/quant_w4a8_tp16/probe_extension_spawn_${node}.json"
exec bash "$TASK/full93/gpu_rdma_candidate/run_full93_python.sh" "$TASK/quant_w4a8_tp16/probe_extension_spawn.py"
