#!/usr/bin/env bash
set -euo pipefail
plugin_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
export PYTHONPATH="$plugin_root${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_PLUGINS=hcu,hcu_model,hcu_ops
exec vllm serve /data/models/hy4-preview-channel-int4a8-gptq-911-2 \
  --served-model-name hy4-gptq911 \
  --tensor-parallel-size 8 --moe-backend aiter \
  --quantization slimquant_w4a8 \
  --hf-overrides '{"quantization_config":{"quant_method":"slimquant_w4a8","checkpoint_format":"hy4_w4a8_v1","checkpoint_index":"/data/models/hy4-preview-channel-int4a8-gptq-911-2/hy4-checkpoint.index.json"}}' \
  --gpu-memory-utilization 0.95 --max-model-len 4096 \
  --max-num-seqs 1 --max-num-batched-tokens 128 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --default-chat-template-kwargs '{"reasoning_effort":"no_think"}' \
  --host 127.0.0.1 --port 8000 "$@"
