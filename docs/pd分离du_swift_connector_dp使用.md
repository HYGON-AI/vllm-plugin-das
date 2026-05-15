## PD分离dp并行
### 环境变量
```
export VLLM_HCU_USE_DP_CONNECTOR=1
export VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=1
export NCCL_MIN_NCHANNELS=16
export NCCL_MAX_NCHANNELS=16
```
### 代理(10.16.1.49):
```
python3 path-to-vllm-hcu/examples/disaggregated_serving_du_swift_xpyd/disagg_proxy_du_swift_xpyd_dp.py
```

### P的运行指令(10.16.1.49)：
```
vllm serve /llm-models/qwen3/Qwen3-8B \
--port 20012 \
--trust-remote-code \
--dtype bfloat16 \
-tp 1 \
--block-size 64 \
--gpu-memory-utilization 0.9 \
--enable-chunked-prefill \
--enable-prefix-caching \
--enforce-eager \
--kv-transfer-config '{"kv_connector":"DuSwiftConnectorDp","kv_role":"kv_producer","kv_buffer_size":"1e4","kv_port":"21002","kv_connector_extra_config":{"proxy_ip":"10.16.1.49","proxy_port":"30001","http_port":"20012","send_type":"PUT_ASYNC","instance_ip":"10.16.1.49"}}'
```
### D的运行指令(10.16.1.49)：
```
vllm serve /llm-models/qwen3/Qwen3-8B \
--port 21003 \
--trust-remote-code \
--dtype bfloat16 \
-dp 2 \
-tp 1 \
--block-size 64 \
--gpu-memory-utilization 0.85 \
--enable-chunked-prefill \
--max-num-batched-tokens 128 \
--enable-prefix-caching \
--kv-transfer-config '{"kv_connector":"DuSwiftConnectorDp","kv_role":"kv_consumer","kv_buffer_size":"1e9","kv_port":"25123","kv_connector_extra_config":{"proxy_ip":"10.16.1.49","proxy_port":"30001","http_port":"21003","send_type":"PUT_ASYNC","instance_ip":"10.16.1.49","dp_size":2}}' \
--data-parallel-size-local 2 \
--data-parallel-address 10.16.1.49 \
--data-parallel-rpc-port 1127 \
--disable-custom-all-reduce
```
### 测试(10.16.1.49):
```
curl http://localhost:10001/v1/completions     -H "Content-Type: application/json"     -d '{                                                      "model": "/llm-models/qwen3/Qwen3-8B",
        "prompt": "I believe the meaning of life is",
        "max_tokens": 100,
        "temperature": 0
    }'
```
