# HY4 Flash W4A8 on gfx936 with AITER Custom AllReduce

This configuration uses the gfx936 support at `04310dcae5e1`.
HY4 sparse `fp8_ds_mla` KV cache stays in E4M3.
The test used the container's stock FlashMLA build and an AITER build with
the Custom AllReduce module. The target and DFlash draft keep the same cache
format. W4A8 expert routing uses `--moe-backend aiter`. [Plugin #537](https://github.com/HYGON-AI/vllm-plugin-das/pull/537)

## Serve

Activate the environment containing the gfx936-enabled Plugin, Adapter,
AITER and FlashMLA before launching. Set `MODEL_DIR` to the local
`Hy4-flash-Channel-INT4-w4a8` checkpoint. The topology XML in the command
is available in the qualified container.

```bash
export MODEL_DIR=/data/models/Hy4-flash-Channel-INT4-w4a8
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export OMP_NUM_THREADS=4
export HCU_AITER_AR=1
export VLLM_ROCM_USE_AITER=0
export VLLM_ROCM_USE_AITER_CUSTOM_AR=1
export AITER_AR_TRANSPORT=ipc
export AITER_AR_MAX_SIZE_MB=64
export AITER_AR_ENABLE_REG_CAPTURE=0
export AITER_AR_SMALL_MESSAGE=off
export NCCL_TOPO_FILE=/usr/local/built-in-508-topo-input-tj-default.xml
export NCCL_SOCKET_IFNAME=lo
export NCCL_P2P_DISABLE=1

vllm serve "${MODEL_DIR}" \
  --host 127.0.0.1 \
  --port 48823 \
  --served-model-name hy4-flash-int4-w4a8 \
  --tensor-parallel-size 8 \
  --dtype bfloat16 \
  --max-model-len -1 \
  --max-num-seqs 64 \
  --max-num-batched-tokens 2048 \
  --gpu-memory-utilization 0.85 \
  --kv-cache-dtype fp8_ds_mla \
  --generation-config vllm \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --speculative-config '{"method":"dfly","num_speculative_tokens":5,"enforce_eager":false,"kv_cache_dtype":"fp8_ds_mla"}' \
  --attention-backend FLASHMLA_SPARSE \
  --no-enable-prefix-caching \
  --reasoning-parser hy_v4 \
  --moe-backend aiter \
  --kv-cache-memory-bytes 17179869184 \
  --compilation-config '{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE","inductor_compile_config":{"benchmark_combo_kernel":false}}'
```

The server log should report `Custom allreduce initialized with transport=ipc`
for each of the eight ranks.

## Validation

The comparison used one gfx936 TP8 node, the same installed model stack
and checkpoint, FULL_AND_PIECEWISE, DFlash with five draft tokens, and
64 concurrent requests of 4,096 input tokens and 256 output tokens.
Both runs completed 64 requests. The request seed and tokenized inputs matched.

| Configuration | Output tokens/s | Mean TTFT | Mean TPOT |
| --- | ---: | ---: | ---: |
| E4M3, regular AllReduce | 119.76 | 69.29 s | 154.14 ms |
| E4M3, AITER Custom AllReduce (IPC) | 155.76 | 54.47 s | 113.56 ms |

Output throughput increased by 30.06% in this comparison.
DFlash accepted 64.32% of draft tokens in the throughput
run. The E4M3 reference accepted 64.86% in the same workload.
EvalScope HumanEval at concurrency 64 scored **159/164** against the
recorded Eager reference of **161/164**, with zero request, infrastructure
or length-truncation failures. DFlash accepted
38.32% of draft tokens during HumanEval.
