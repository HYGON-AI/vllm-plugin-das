#!/usr/bin/env bash
set -euo pipefail
node=${1:?nmz22 or nmz20}
case "$node" in nmz22) rank=0;; nmz20) rank=1;; *) exit 2;; esac
root=/home/xiesl/inference/verify_torch213_image/task_dir/support_kimi31_pro
base="$root/quant_w4a8_tp16"
export PYTHONPATH="$base/experimental_entry:/home/xiesl/inference/verify_torch213_image/code_dir/plugin/adapter_extension/src:/home/xiesl/inference/verify_torch213_image/code_dir/adapter-kimi31-preload-opt:${PYTHONPATH:-}"
export HCU_ADAPTER_HOST=vllm HCU_VLLM_KIMI31_PRELOAD_IMPL=nvidia-load-only
export HCU_ADAPTER_VLLM_REQUIRED_EXTENSIONS=kimi31_slimquant_exp VLLM_PLUGINS=hcu_adapter
export HCU_VLLM_FASTSAFETENSORS_STAGING=gpu_rdma VLLM_KIMI_K3_GEMM_AR=0 VLLM_FASTSAFETENSORS_QUEUE_SIZE=-1
export OMP_NUM_THREADS=1 GLOO_SOCKET_IFNAME=ens61f1np1 NCCL_SOCKET_IFNAME=ens61f1np1 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=0 NCCL_NET=IB
export NCCL_IB_HCA=mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1,mlx5_8:1,mlx5_9:1
export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH PYTHONUNBUFFERED=1 VLLM_ENGINE_READY_TIMEOUT_S=14400
export LD_LIBRARY_PATH=/tmp/xiesl_kimi31_preload_venv/lib/lib:/opt/mpi/lib:/opt/ucx/lib:/opt/hwloc/lib:/opt/dtk/dcc/gcvm/lib:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/hyhal/lib:/opt/hyhal/lib64
exec /tmp/xiesl_kimi31_preload_venv/bin/python -m vllm.entrypoints.cli.main preload \
 --model "$base/fixture_4layer_2expert_v1" --trust-remote-code --load-format fastsafetensors \
 --model-loader-extra-config '{"enable_weights_track":true}' --tensor-parallel-size 16 --data-parallel-size 1 --pipeline-parallel-size 1 \
 --distributed-executor-backend mp --nnodes 2 --node-rank "$rank" --master-addr 13.13.4.22 --master-port 29571 --weight-cache-master-port 29671 \
 --weight-cache-socket-dir /tmp/xiesl_k31w4fixture --max-model-len 4096 --max-num-seqs 2 --enforce-eager --disable-custom-all-reduce \
 --moe-backend emulation --kv-cache-dtype bfloat16 --no-enable-flashinfer-autotune
