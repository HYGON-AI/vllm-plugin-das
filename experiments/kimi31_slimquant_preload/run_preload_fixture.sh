#!/usr/bin/env bash
set -euo pipefail
if [[ ${1:-} == --help ]]; then
  echo 'Usage: run_preload_fixture.sh NODE_RANK MASTER_ADDR MODEL_DIR ADAPTER_ROOT PYTHON SOCKET_DIR'
  echo 'Provide the matching Kimi preload Adapter/Python and RDMA NIC settings externally.'
  exit 0
fi
rank=${1:?provide node rank 0 or 1}
master_addr=${2:?provide rendezvous address}
model=${3:?provide diagnostic fixture directory}
adapter_root=${4:?provide companion Kimi preload Adapter checkout}
python_exe=${5:?provide matching Python executable}
socket_dir=${6:?provide fresh socket directory}
case "$rank" in 0|1) ;; *) echo 'node rank must be 0 or 1' >&2; exit 2;; esac
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
plugin_root=$(cd -- "$script_dir/../.." && pwd)
export PYTHONPATH="$plugin_root/adapter_extension/src:$plugin_root:$adapter_root${PYTHONPATH:+:$PYTHONPATH}"
export HCU_ADAPTER_HOST=vllm HCU_VLLM_KIMI31_PRELOAD_IMPL=nvidia-load-only
export HCU_ADAPTER_VLLM_REQUIRED_EXTENSIONS=kimi31_slimquant_exp VLLM_PLUGINS=hcu_adapter
export HCU_VLLM_FASTSAFETENSORS_STAGING=gpu_rdma VLLM_KIMI_K3_GEMM_AR=0 VLLM_FASTSAFETENSORS_QUEUE_SIZE=-1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
exec "$python_exe" -m vllm.entrypoints.cli.main preload \
 --model "$model" --trust-remote-code --load-format fastsafetensors \
 --model-loader-extra-config '{"enable_weights_track":true}' --tensor-parallel-size 16 --data-parallel-size 1 --pipeline-parallel-size 1 \
 --distributed-executor-backend mp --nnodes 2 --node-rank "$rank" --master-addr "$master_addr" --master-port 29571 --weight-cache-master-port 29671 \
 --weight-cache-socket-dir "$socket_dir" --max-model-len 4096 --max-num-seqs 2 --enforce-eager --disable-custom-all-reduce \
 --moe-backend emulation --kv-cache-dtype bfloat16 --no-enable-flashinfer-autotune
