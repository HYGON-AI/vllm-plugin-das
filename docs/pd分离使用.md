## PD 分离
#### 注释：enable_multiple_machines:true：是否是跨机的这里P和D的服务都要设置，只要有一个跨机，就要设置true；enable_asymmetric_p2p：是否是非对称切分；remote_tp_size：D的tpsize；remote_pp_size：D的ppsize （这里的非对成切分支持mla的模型）
### 环境变量

```bash
export NCCL_NCHANNELS_PER_PEER=2
export IP_CONFIG_FILE=/data/ip_config.txt ## 第一个ip为D的第一个节点，第二个ip为D的第二个节点
export NCCL_IB_HCA=,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1,mlx5_8:1,mlx5_9:1
export VLLM_HOST_IP=10.16.1.76  #ip地址 不同的节点这个需要对应修改
export NCCL_SOCKET_IFNAME=enp33s0f3u1
export GLOO_SOCKET_IFNAME=enp33s0f3u1
export NCCL_MIN_NCHANNELS=16
export NCCL_MAX_NCHANNELS=16
export NCCL_NET_GDR_READ=1
```
## P、D单实例单机的任意切分方式（满足D的tp>=P的tp)使用。
### 代理
```bash
在P的节点，例子里是75节点：
vllm-hcu源码：cd vllm-hcu/examples/disaggregated_serving_du_swift_xpyd
python3 disagg_proxy_du_swift_xpyd.py
特别注意，这里如果服务重启，代理也需要重启
```
### P的运行指令：
```bash
 vllm serve /module/DeepSeek-R1-W4A8-V2/   --port 20011 --trust-remote-code  --dtype bfloat16 --max-model-len 49152  --max-num-batched-tokens 8192  -tp 1 -pp 8  --gpu-memory-utilization 0.9 --max-num-seqs 256  --disable-log-requests  --block-size 64 --enforce-eager -q slimquant_w4a8_marlin --speculative_config '{"method": "deepseek_mtp", "num_speculative_tokens": 2}'      --kv-transfer-config '{"kv_connector":"P2pNcclConnector","kv_role":"kv_producer","kv_buffer_size":"1e1","kv_port":"23001","kv_connector_extra_config":{"enable_asymmetric_p2p":true,"remote_tp_size":2,"remote_pp_size":4,"proxy_ip":"10.16.1.75","proxy_port":"30001","http_port":"20011","send_type":"PUT_ASYNC"}}'  --kv-cache-dtype fp8_e5m2
```
### D的运行指令：
```bash
vllm serve /module/DeepSeek-R1-W4A8-V2/ --host 0.0.0.0   --port 20009 --trust-remote-code --dtype bfloat16 -q slimquant_w4a8_marlin --max-model-len 16484 -tp 2 -pp 4  --gpu-memory-utilization 0.90 --max-num-seqs 100 --block-size 64 --disable-log-requests  --max-num-batched-tokens 16484  --speculative_config '{"method": "deepseek_mtp", "num_speculative_tokens": 2}'  --kv-cache-dtype fp8_e5m2     --kv-transfer-config '{"kv_connector":"P2pNcclConnector","kv_role":"kv_consumer","kv_buffer_size":"1e8","kv_port":"22001","kv_connector_extra_config":{"proxy_ip":"10.16.1.75","proxy_port":"30001","http_port":"20009","send_type":"PUT_ASYNC","mem_pool_size_gb":128}}'
```
## P：PP2 TP8 D：TP8
### 代理
```bash
在P的节点（例子里是75和76节点）：
vllm-hcu源码：cd vllm-hcu/examples/disaggregated_serving_du_swift_xpyd
python3 disagg_proxy_du_swift_xpyd_mult_mac.py
```
### P的运行指令：
```bash
在75节点运行：ray start --head --node-ip-address=10.16.1.75 --port=8244 --num-gpus=8 --num-cpus=32
在76节点运行：ray start --address='10.16.1.75:8244' --num-gpus=8 --num-cpus=32
在75节点启动服务：vllm serve  /module/DeepSeek-R1-W4A8-V2/ --host 0.0.0.0   --port 20005  --trust-remote-code --distributed-executor-backend ray --dtype bfloat16 --max-model-len 32768  -tp 8 -pp 2  --gpu-memory-utilization 0.90 --max-num-seqs 256 --speculative_config '{"method": "deepseek_mtp", "num_speculative_tokens": 2}' --disable-log-requests --block-size 64 --enable-chunked-prefill --max-num-batched-tokens 6144 --no-enable-prefix-caching  --enforce-eager --kv-cache-dtype fp8_e5m2 -q slimquant_marlin --kv-transfer-config '{"kv_connector":"P2pNcclConnector","kv_role":"kv_producer","kv_buffer_size":"1e1","kv_port":"21001","kv_connector_extra_config":{"enable_multiple_machines":true,"enable_asymmetric_p2p":false,"remote_tp_size":8,"remote_pp_size":1,"proxy_ip":"10.16.1.75","proxy_port":"30001","http_port":"20005","send_type":"PUT_ASYNC","mem_pool_size_gb":64}}'

```
### D的运行指令：
```bash
vllm serve /module/DeepSeek-R1-W4A8-V2/ --host 0.0.0.0   --port 20009 --trust-remote-code --dtype bfloat16 -q slimquant_w4a8_marlin --max-model-len 16484 -tp 8 --gpu-memory-utilization 0.90 --max-num-seqs 100 --block-size 64 --disable-log-requests  --max-num-batched-tokens 16484  --speculative_config '{"method": "deepseek_mtp", "num_speculative_tokens": 2}'  --kv-cache-dtype fp8_e5m2 --kv-transfer-config '{"kv_connector":"P2pNcclConnector","kv_role":"kv_consumer","kv_buffer_size":"1e8","kv_port":"22001","kv_connector_extra_config":{"enable_multiple_machines":true，"proxy_ip":"10.16.1.75","proxy_port":"30001","http_port":"20009","send_type":"PUT_ASYNC","mem_pool_size_gb":128}}'

```
## P：PP2 TP8 D：PP2 TP8
### 代理
```bash
在P的节点（例子里是75和76节点）：
vllm-hcu源码：cd vllm-hcu/examples/disaggregated_serving_du_swift_xpyd
python3 disagg_proxy_du_swift_xpyd_mult_mac.py
```
### P的运行指令：
```bash
在75节点运行：ray start --head --node-ip-address=10.16.1.75 --port=8244 --num-gpus=8 --num-cpus=32
在76节点运行：ray start --address='10.16.1.75:8244' --num-gpus=8 --num-cpus=32
在75节点启动服务：vllm serve  /module/DeepSeek-R1-W4A8-V2/ --host 0.0.0.0   --port 20005  --trust-remote-code --distributed-executor-backend ray --dtype bfloat16 --max-model-len 32768  -tp 8 -pp 2  --gpu-memory-utilization 0.90 --max-num-seqs 256 --speculative_config '{"method": "deepseek_mtp", "num_speculative_tokens": 2}' --disable-log-requests --block-size 64 --enable-chunked-prefill --max-num-batched-tokens 6144 --no-enable-prefix-caching  --enforce-eager --kv-cache-dtype fp8_e5m2 -q slimquant_marlin --kv-transfer-config '{"kv_connector":"P2pNcclConnector","kv_role":"kv_producer","kv_buffer_size":"1e1","kv_port":"21001","kv_connector_extra_config":{"enable_multiple_machines":true,"enable_asymmetric_p2p":false,"remote_tp_size":8,"remote_pp_size":1,"proxy_ip":"10.16.1.75","proxy_port":"30001","http_port":"20005","send_type":"PUT_ASYNC","mem_pool_size_gb":64}}'
```
### D的运行指令：
```bash
在77节点运行：ray start --head --node-ip-address=10.16.1.77 --port=9244 --num-gpus=8 --num-cpus=32
在26节点运行：ray start --address='10.16.1.77:9244' --num-gpus=8 --num-cpus=32
在77节点启动服务：vllm serve /module/DeepSeek-R1-W4A8-V2/ --host 0.0.0.0   --port 20009 --trust-remote-code --dtype bfloat16 -q slimquant_w4a8_marlin --max-model-len 16484 -tp 8 -pp 2 --gpu-memory-utilization 0.90 --max-num-seqs 100 --block-size 64 --disable-log-requests  --max-num-batched-tokens 16484  --speculative_config '{"method": "deepseek_mtp", "num_speculative_tokens": 2}'  --kv-cache-dtype fp8_e5m2 --kv-transfer-config '{"kv_connector":"P2pNcclConnector","kv_role":"kv_consumer","kv_buffer_size":"1e8","kv_port":"22001","kv_connector_extra_config":{"enable_multiple_machines":true，"proxy_ip":"10.16.1.75","proxy_port":"30001","http_port":"20009","send_type":"PUT_ASYNC","mem_pool_size_gb":128}}'
```