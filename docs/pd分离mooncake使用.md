# 基于 MooncakeConnector 的 PD 分离使用指南

## 目录

- [关于 Mooncake](#关于-mooncake)
- [前置条件](#前置条件)
  - [下载和安装](#下载和安装)
- [使用方法](#使用方法)
  - [环境变量](#环境变量)
  - [TTFT 分段追踪](#ttft-分段追踪)
  - [P、D 单实例单节点](#pd-单实例单节点)
  - [P：TP2PP4  D：TP4PP2 (1P1D)](#ptp2pp4-dtp4pp2-1p1d)
  - [P：TP8  D：DP8EP8 (1P1D)](#ptp8-ddp8ep8-1p1d)
  - [P：PP16  D：TP8 (2P1D)](#ppp16-dtp8-2p1d)
  - [P：SP8  D：DP16EP16 (1P2D)](#psp8-ddp16ep16-1p2d)

## 关于 Mooncake

Mooncake 旨在提升大语言模型（LLM）的推理效率，尤其是在对象存储速度较慢的环境中。它通过在高速互连的 DRAM/SSD 资源上构建多级缓存池来实现这一目标。与传统的缓存系统相比，Mooncake 利用（GPUDirect）RDMA 技术以零拷贝方式直接传输数据，同时最大化利用单机上的多 NIC 资源。

## 前置条件

### 下载和安装

mooncake 代码仓库：http://42.228.13.241:10068/dcutoolkit/deeplearing/mooncake

mooncake whl 包（ubuntu2204）路径：http://pypi.sourcefind.cn:666/das_nightly/dtk2604-rc4-mooncake/+f/79c/add379d74452d/mooncake_transfer_engine-0.3.10.post1+das.opt1.dtk2604.2605131137.gd34f6f-cp310-cp310-manylinux_2_35_x86_64.whl 

通过 pip 安装 mooncake：`pip install mooncake_transfer_engine*.whl`。

## 使用方法

### 环境变量

```bash
# prefill 端和 decode 端都需要设置
export VLLM_HOST_IP=${HOST_IP}           # 本机 ip 地址
export MC_ENABLE_DEST_DEVICE_AFFINITY=1  # 优先选择和本地网卡同名的远端网卡进行通信
```

### TTFT 分段追踪

开启后，Prefill / Decode 进程会在关键路径上输出结构化 **TTFT_EVENT** 日志，用于 PD 分离场景下的 TTFT 六段分解分析。

#### 启用方式

Prefill 与 Decode **均需**设置（与 `--disable-log-stats` 无关）。该变量由 `vllm_hcu.platforms.envs` 维护：

```bash
export VLLM_HCU_MOONCAKE_TTFT_TRACE=1
export VLLM_LOGGING_LEVEL=DEBUG   # TTFT_EVENT 以 DEBUG 级别输出，需同步开启
```

#### 日志格式

每条事件一行，包含事件名、墙钟时间戳（秒，浮点）、`transfer_id` 与 `req_id`：

```text
Mooncake TTFT_EVENT event=p_alloc ts=1751510400.123456 transfer_id=xfer-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx req_id=cmpl-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
```

从 Prefill / Decode 日志中过滤事件：

```bash
grep 'Mooncake TTFT_EVENT' prefiller.log decoder.log
```

#### 八事件与六段模型

| 侧 | 事件 | 含义 |
|----|------|------|
| P（kv_producer） | `p_alloc` | 请求分配 KV |
| P | `p_ready` | Prefill 完成 |
| P | `p_send_kv_start` | 开始发送 KV |
| P | `p_send_kv_done` | KV 发送完成 |
| D（kv_consumer） | `d_alloc` | Decode 侧分配 KV |
| D | `d_kv_ready` | KV 拉取完成 |
| D | `d_kv_sched_ready` | Scheduler 就绪，可开始 decode |
| D | `d_first_token` | 首个 token 输出 |

六段时延（相邻事件时间差，单位 ms）：

| 段 | 起止事件 | 说明 |
|----|----------|------|
| P prefill | `p_alloc` → `p_ready` | Prefill 计算 |
| P ready→send | `p_ready` → `p_send_kv_start` | Prefill 完成到开始发 KV |
| P send | `p_send_kv_start` → `p_send_kv_done` | KV 发送 |
| D wait KV | `d_alloc` → `d_kv_ready` | Decode 等待 KV 传输 |
| D KV sched | `d_kv_ready` → `d_kv_sched_ready` | KV 就绪到可调度 decode |
| D decode | `d_kv_sched_ready` → `d_first_token` | 首 token 前 decode |

### P、D 单实例单节点

#### Prefill 节点（10.63.60.113）

```bash
vllm serve Qwen3/Qwen3-8B \
  --port 8010 \
  --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_producer"}' 
```

#### Decode 节点（10.63.60.113）

```bash
HIP_VISIBLE_DEVICES=1 vllm serve Qwen3/Qwen3-8B \
  --port 8020 \
  --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_consumer"}'
```

#### 代理服务器

```bash
python3 examples/online_serving/disaggregated_serving/mooncake_connector/mooncake_connector_proxy.py \
  --prefill "http://0.0.0.0:8010" "8998" \
  --decode "http://0.0.0.0:8020" \
  --port 8001
```

### P：TP2PP4  D：TP4PP2 (1P1D)

#### Prefill 节点（10.63.60.113）

```bash
vllm serve /mnt/deepseek-v2/DeepSeek-V2-Lite-Chat \
  --enforce-eager \
  --port 8010 \
  -tp 2 \
  -pp 4 \
  --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_producer"}' 
```

#### Decode 节点（10.63.60.114）

```bash
vllm serve /mnt/deepseek-v2/DeepSeek-V2-Lite-Chat \
  --enforce-eager \
  --port 8020 \
  -tp 4 \
  -pp 2 \
  --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_consumer"}' 
```

#### 代理服务器

```bash
python3 examples/online_serving/disaggregated_serving/mooncake_connector/mooncake_connector_proxy.py \
  --prefill "http://10.63.60.113:8010" "8998" \
  --decode "http://10.63.60.114:8020" \
  --port 8000
```

### P：TP8  D：DP8EP8 (1P1D)

#### Prefill 节点（10.16.1.15）

```bash
vllm serve /models/vllm-w8a8-models/GLM-5-W8A8 \
  -q slimquant_marlin \
  --trust-remote-code \
  --dtype bfloat16 \
  --max-model-len 65536 \
  --max-num-batched-tokens 8192 \
  --enforce-eager \
  -tp 8 \
  --port 9348 \
  --gpu-memory-utilization 0.92 \
  --max-num-seqs 64 \
  --enable-prefix-caching \
  --block-size 64 \
  --kv-cache-dtype fp8_ds_mla \
  --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_producer"}'
```

#### Decode 节点（10.16.1.16）

```bash
export NCCL_NET_GDR_LEVEL=7
export NCCL_SDMA_COPY_ENABLE=0
export NCCL_IB_HCA=mlx5_0:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1,mlx5_8:1,mlx5_9:1
export ROCSHMEM_HEAP_SIZE=4000000000
export ROCSHMEM_TOPO_FILE_FORCE=/workspace/topo.config
export USE_SPE_MQP=1
export ROCSHMEM_SQ_SIZE=1024
export ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX=256
export VLLM_MOE_DP_CHUNK_SIZE=128
export VLLM_HCU_ALL2ALL_BACKEND=deepep_low_latency
export VLLM_HCU_USE_FLASHMLA=1
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
vllm serve /models/vllm-w8a8-models/GLM-5-W8A8  \
  -q slimquant_marlin \
  --trust-remote-code \
  --dtype bfloat16 \
  --max-model-len 65536 \
  --max-num-batched-tokens 128 \
  -dp 8 \
  --port 9349 \
  --max-num-seqs 64 \
  --gpu-memory-utilization 0.92 \
  --block-size 64 \
  --kv-cache-dtype fp8_ds_mla \
  --enable-expert-parallel \
  --all2all-backend deepep_low_latency \
  --disable-custom-all-reduce \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  -cc '{"inductor_compile_config":{"combo_kernels": false}}' \
  --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_consumer"}'
```

#### 代理服务器

```bash
python3 /workspace/vllm/examples/online_serving/disaggregated_serving/mooncake_connector/mooncake_connector_proxy.py \
  --prefill "http://10.16.1.15:9348" "8998" \
  --decode "http://10.16.1.16:9349" \
  --port 8000
```

### P：PP16  D：TP8 (2P1D)

#### Prefill 节点（10.16.1.15, 10.16.1.16）

```bash
10.16.1.15:
export VLLM_HCU_USE_FLASHMLA=1
export LMSLIM_USE_GLOBAL_MOE_CACHE=1
export VLLM_DP_MASTER_IP=10.16.1.15
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
ray start --head --node-ip-address=10.16.1.15  --port=1255 --num-gpus=8 --num-cpus=32
10.16.1.16:
export VLLM_HCU_USE_FLASHMLA=1
export LMSLIM_USE_GLOBAL_MOE_CACHE=1
export VLLM_DP_MASTER_IP=10.16.1.15
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
ray start --address=10.16.1.15:1255  --num-gpus=8 --num-cpus=32
10.16.1.15:
vllm serve /models/v2_6/GLM-w4a8-V2_6_test \
  --trust-remote-code \
  -pp 16 \
  --dtype bfloat16 \
  --max-model-len 65536 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 64 \
  --kv-cache-dtype fp8_ds_mla \
  --gpu-memory-utilization 0.9 \
  --distributed-executor-backend ray \
  --enforce-eager \
  -cc '{"inductor_compile_config":{"combo_kernels": false}}' \
  --port 9348 \
  --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_producer"}'
```

#### Decode 节点（10.16.1.18）

```bash
export VLLM_HCU_USE_FLASHMLA=1
export LMSLIM_USE_GLOBAL_MOE_CACHE=1
export VLLM_DP_MASTER_IP=10.16.1.15
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
vllm serve /models/v2_6/GLM-w4a8-V2_6_test \
  --trust-remote-code \
  -tp 8 \
  --dtype bfloat16 \
  --max-model-len 65536 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 64 \
  --kv-cache-dtype fp8_ds_mla \
  --gpu-memory-utilization 0.9 \
  -cc '{"inductor_compile_config":{"combo_kernels": false}}' \
  --port 9349 \
  --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_consumer"}'
```

#### 代理服务器

```bash
python3 /workspace/vllm/examples/online_serving/disaggregated_serving/mooncake_connector/mooncake_connector_proxy.py \
  --prefill "http://10.16.1.15:9348" "8998" \
  --decode "http://10.16.1.18:9349" \
  --port 8000
```

### P：SP8  D：DP16EP16 (1P2D)

#### Prefill 节点（10.16.1.15）

```bash
export VLLM_HCU_USE_CUSTOM_FLASH_ATTN=1
export GPU_MAX_HW_QUEUES=4
export VLLM_HCU_USE_LIGHTOP_MOE_ALIGN=1
export LMSLIM_USE_LIGHTOP=1
export HIPBLASLT_TUNING_OVERRIDE_FILE=/workspace/rocblas/hipblaslt.config
export ROCBLAS_TENSILE_LIBPATH=/workspace/rocblas/rocblas_hy3_fp8_zmy
export MC_ENABLE_DEST_DEVICE_AFFINITY=1

LMSLIM_USE_FUSED_RMS_QUANT=1 \
VLLM_HCU_USE_FUSED_QKV_SPLIT_RMS_ROPE_KVSTORE=0 \
vllm serve /models/Hy3-CHANNEL-FP8-w8a8-sero-ignore-from-script3 \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 2 \
  --max-model-len 65536 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 128 \
  --dtype bfloat16 \
  --tensor-parallel-size 8 \
  --no-enable-prefix-caching \
  --tool-call-parser hy_v3 \
  --reasoning-parser hy_v3 \
  --enable-auto-tool-choice \
  --enable-custom-sp \
  --enforce-eager \
  --kv_cache_dtype fp8_e4m3 \
  --port 8010 \
  --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_producer"}'
```

#### Decode 节点（10.16.1.16, 10.16.1.18）

```bash
10.16.1.16:
export VLLM_HOST_IP=10.16.1.16
export NCCL_SOCKET_IFNAME=ens14f0
export GLOO_SOCKET_IFNAME=ens14f0
export VLLM_TORCH_PROFILER_DIR=/data/vllm_profile
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export NCCL_MIN_NCHANNELS=16
export NCCL_MAX_NCHANNELS=16

export Allgather_Base_STREAM_WITH_COMPUTE=1
export SENDRECV_STREAM_WITH_COMPUTE=1
export HIP_KERNEL_EVENT_SYSTENFENCE=1
export VLLM_RPC_TIMEOUT=1800000
export VLLM_USE_PIECEWISE=0
export VLLM_REJECT_SAMPLE_OPT=0

export USE_FUSED_RMS_QUANT=0
export USE_FUSED_SILU_MUL_QUANT=0

export VLLM_ROCM_USE_AITER=0
export VLLM_ROCM_USE_AITER_MOE=0
export VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=0

export VLLM_USE_GLOBAL_CACHE13=1
export VLLM_FUSED_MOE_CHUNK_SIZE=16384
export VLLM_CUSTOM_CACHE=0
export VLLM_USE_OPT_CAT=1
export VLLM_USE_FUSED_FILL_RMS_CAT=1
export VLLM_USE_LIGHTOP_MOE_SUM_MUL_ADD=0
export VLLM_USE_LIGHTOP_RMS_ROPE_CONCAT=0
export VLLM_USE_V32_ENCODE=1
export VLLM_HCU_USE_FLASHMLA=1
export VLLM_HCU_DISABLE_DSA=0
export USE_LIGHTOP_TOPK=1
export USE_LIGHTOP_PER_TOKEN_GROUP_QUANT_FP8=1
export USE_LIGHTOP_CONVERT_REQ_INDEX_TO_GLOBAL_INDEX=1

export NCCL_NET_GDR_LEVEL=7
export NCCL_SDMA_COPY_ENABLE=0
export NCCL_IB_HCA=mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1,mlx5_8:1,mlx5_9:1

export ROCSHMEM_HEAP_SIZE=4000000000
#郑州节点需要设置
export ROCSHMEM_TOPO_FILE_FORCE=/workspace/topo.config
export ROCSHMEM_ALLOWED_IBV_DEVICES=mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9
export USE_SPE_MQP=1
export ROCSHMEM_SQ_SIZE=1024

export VLLM_MOE_DP_CHUNK_SIZE=128
export ROCSHMEM_IB_GID_INDEX=0

export VLLM_USE_LIGHTOP=1
export VLLM_HCU_USE_CUSTOM_FLASH_ATTN=1
export GPU_MAX_HW_QUEUES=4
export MC_ENABLE_DEST_DEVICE_AFFINITY=1

vllm serve /models/Hy3-CHANNEL-FP8-w8a8-sero-ignore-from-script3 \
  --trust-remote-code \
  -dp 16 \
  -tp 1 \
  --enable-expert-parallel \
  --disable-custom-all-reduce \
  --dtype bfloat16 \
  --enable-chunked-prefill \
  --max-model-len 53000 \
  --enable-prefix-caching \
  --block-size 64 \
  --gpu-memory-utilization 0.89 \
  --data-parallel-size-local 8 \
  --data-parallel-address 10.16.1.16 \
  --data-parallel-rpc-port 1127 \
  --data-parallel-start-rank 0 \
  --kv-cache-dtype fp8_e4m3 \
  -q slimquant_marlin \
  --max-num-seqs 256 \
  --all2all_backend=deepep_low_latency \
  --speculative_config '{"method":"mtp","num_speculative_tokens":2, "quantization": "slimquant_marlin"}' \
  --port 8020 \
  --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_consumer"}'

10.16.1.18:
export VLLM_HOST_IP=10.16.1.18
export NCCL_SOCKET_IFNAME=ens14f0
export GLOO_SOCKET_IFNAME=ens14f0
export VLLM_TORCH_PROFILER_DIR=/data/vllm_profile
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export NCCL_MIN_NCHANNELS=16
export NCCL_MAX_NCHANNELS=16

export Allgather_Base_STREAM_WITH_COMPUTE=1
export SENDRECV_STREAM_WITH_COMPUTE=1
export HIP_KERNEL_EVENT_SYSTENFENCE=1
export VLLM_RPC_TIMEOUT=1800000
export VLLM_USE_PIECEWISE=0
export VLLM_REJECT_SAMPLE_OPT=0

export USE_FUSED_RMS_QUANT=0
export USE_FUSED_SILU_MUL_QUANT=0
export VLLM_ROCM_USE_AITER=0
export VLLM_ROCM_USE_AITER_MOE=0
export VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=0
export VLLM_USE_GLOBAL_CACHE13=1
export VLLM_FUSED_MOE_CHUNK_SIZE=16384
export VLLM_CUSTOM_CACHE=0
export VLLM_USE_OPT_CAT=1
export VLLM_USE_FUSED_FILL_RMS_CAT=1
export VLLM_USE_LIGHTOP_MOE_SUM_MUL_ADD=0
export VLLM_USE_LIGHTOP_RMS_ROPE_CONCAT=0
export VLLM_USE_V32_ENCODE=1
export VLLM_HCU_USE_FLASHMLA=1
export VLLM_HCU_DISABLE_DSA=0
export USE_LIGHTOP_TOPK=1
export USE_LIGHTOP_PER_TOKEN_GROUP_QUANT_FP8=1
export USE_LIGHTOP_CONVERT_REQ_INDEX_TO_GLOBAL_INDEX=1

export NCCL_NET_GDR_LEVEL=7
export NCCL_SDMA_COPY_ENABLE=0
export NCCL_IB_HCA=mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1,mlx5_8:1,mlx5_9:1

export ROCSHMEM_HEAP_SIZE=4000000000
#郑州节点需要设置
export ROCSHMEM_TOPO_FILE_FORCE=/workspace/topo.config
export ROCSHMEM_ALLOWED_IBV_DEVICES=mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9
export USE_SPE_MQP=1
export ROCSHMEM_SQ_SIZE=1024

export VLLM_MOE_DP_CHUNK_SIZE=128
export ROCSHMEM_IB_GID_INDEX=0

export VLLM_USE_LIGHTOP=1
export VLLM_HCU_USE_CUSTOM_FLASH_ATTN=1
export GPU_MAX_HW_QUEUES=4
export MC_ENABLE_DEST_DEVICE_AFFINITY=1

vllm serve /models/Hy3-CHANNEL-FP8-w8a8-sero-ignore-from-script3 \
  --trust-remote-code \
  -dp 16 \
  -tp 1 \
  --enable-expert-parallel \
  --disable-custom-all-reduce \
  --dtype bfloat16 \
  --enable-chunked-prefill \
  --max-model-len 53000 \
  --enable-prefix-caching \
  --block-size 64 \
  --gpu-memory-utilization 0.89 \
  --data-parallel-size-local 8 \
  --data-parallel-address 10.16.1.16 \
  --data-parallel-rpc-port 1127 \
  --data-parallel-start-rank 8 \
  --kv-cache-dtype fp8_e4m3 \
  -q slimquant_marlin \
  --max-num-seqs 256 \
  --headless \
  --all2all_backend=deepep_low_latency \
  --speculative_config '{"method":"mtp","num_speculative_tokens":2, "quantization": "slimquant_marlin"}' \
  --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_consumer"}'
```

#### 代理服务器

```bash
python3 /workspace/vllm/examples/online_serving/disaggregated_serving/mooncake_connector/mooncake_connector_proxy.py \
  --prefill "http://10.16.1.15:8010" "8998" \
  --decode "http://10.16.1.16:8020" \
  --port 8000
```
