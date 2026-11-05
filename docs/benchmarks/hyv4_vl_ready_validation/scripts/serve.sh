#!/usr/bin/env bash
set -euo pipefail
cd /tmp/vllm-plugin-das-pr
export PYTHONPATH=/root/hy4-perf-stage4:/tmp/vllm-plugin-das-pr
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_USE_BREAKABLE_CUDAGRAPH=1
export VLLM_SERVER_DEV_MODE=1
exec vllm serve /module/Hy4-testing-Channel-FP8-w8a8 \
  --served-model-name hy4-vl --tensor-parallel-size 8 --moe-backend triton \
  --gpu-memory-utilization 0.90 --max-model-len 4096 --max-num-seqs 2 \
  --max-num-batched-tokens 1024 \
  --compilation-config '{"mode":0,"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[1,2,128,256,512,768,1024]}' \
  --worker-extension-cls stage4_worker.Stage4Worker \
  --mm-processor-kwargs '{"max_pixels":1048576}' \
  --limit-mm-per-prompt '{"image":2}' --mm-processor-cache-gb 0 \
  --no-enable-prefix-caching \
  --default-chat-template-kwargs '{"reasoning_effort":"no_think"}' \
  --host 127.0.0.1 --port 8010
