# 基于 MooncakeConnector 的 PD 分离使用指南

## 目录

- [关于 Mooncake](#关于-mooncake)
- [前置条件](#前置条件)
  - [下载和安装](#下载和安装)
- [使用方法](#使用方法)
  - [环境变量](#环境变量)
  - [P、D 单实例单节点](#pd-单实例单节点)
  - [P：TP2PP4  D：TP4PP2 (1P1D)](#ptp2pp4-dtp4pp2-1p1d)
  - [P：TP8  D：DP8EP8 (1P1D)](#ptp8-ddp8ep8-1p1d)
  - [P：PP16  D：TP8 (2P1D)](#ppp16-dtp8-2p1d)

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
