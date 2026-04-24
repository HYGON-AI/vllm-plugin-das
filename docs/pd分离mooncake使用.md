# 基于 MooncakeConnector 的 PD 分离使用指南

## 关于 Mooncake

Mooncake 旨在提升大语言模型（LLM）的推理效率，尤其是在对象存储速度较慢的环境中。它通过在高速互连的 DRAM/SSD 资源上构建多级缓存池来实现这一目标。与传统的缓存系统相比，Mooncake 利用（GPUDirect）RDMA 技术以零拷贝方式直接传输数据，同时最大化利用单机上的多 NIC 资源。

已支持的特性：（1）DP 并行；（2）TP 并行；（3）PD 对称切分。

尚未支持的特性：（1）PP 并行；（2）PD 非对称切分。

## 前置条件

### 下载和安装

mooncake 代码仓库：http://42.228.13.241:10068/dcutoolkit/deeplearing/mooncake

mooncake whl 包（ubuntu2204）路径：http://42.228.13.241:18000/customized/mooncake/dtk-26.04-ubuntu2204/

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

#### 测试 

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-8B",
    "messages": [
      {"role": "user", "content": "Tell me a long story about artificial intelligence."}
    ]
  }'
```

### P：DP2TP4  D：DP2TP4

#### Prefill 节点（10.63.60.113）

```bash
vllm serve Qwen3/Qwen3-8B \
  --port 8010 \
  --data-parallel-size 2 --tensor-parallel-size 4 \
  --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_producer"}' 
```

#### Decode 节点（10.63.60.114）

```bash
vllm serve Qwen3/Qwen3-8B \
  --port 8020 \
  --data-parallel-size 2 --tensor-parallel-size 4 \
  --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_consumer"}'
```

#### 代理服务器

```bash
python3 examples/online_serving/disaggregated_serving/mooncake_connector/mooncake_connector_proxy.py \
  --prefill "http://10.63.60.113:8010" "8998" \
  --decode "http://10.63.60.114:8020" \
  --port 8001
```

#### 测试

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-8B",
    "messages": [
      {"role": "user", "content": "Tell me a long story about artificial intelligence."}
    ]
  }'
```