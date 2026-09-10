#!/usr/bin/env bash
set -euo pipefail
plugin_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
export PYTHONPATH="$plugin_root${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_USE_V2_MODEL_RUNNER=1
exec vllm serve /data/models/hy4_w4a8_unverified \
  --served-model-name hy4-w4a8 \
  --tensor-parallel-size 8 --moe-backend aiter \
  --quantization slimquant_w4a8 \
  --hf-overrides '{"quantization_config":{"quant_method":"slimquant_w4a8","checkpoint_format":"hy4-w4a8-custom-v1","conversion_manifest":"/data/models/hy4_w4a8_unverified/conversion.json"}}' \
  --gpu-memory-utilization 0.95 --max-model-len 1024 \
  --max-num-seqs 1 --max-num-batched-tokens 128 \
  --host 127.0.0.1 --port 8000 "$@"
