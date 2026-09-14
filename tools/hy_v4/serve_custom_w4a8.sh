#!/usr/bin/env bash
# Hardware-unvalidated checkpoint adapter. Supply the converted checkpoint path.
set -euo pipefail
checkpoint_root=${1:?Usage: serve_custom_w4a8.sh CHECKPOINT_DIR [vllm serve options]}
shift
overrides=$(python3 -c 'import json, sys; print(json.dumps({"quantization_config": {"quant_method": "slimquant_w4a8", "checkpoint_format": "hy4-w4a8-custom-v1", "conversion_manifest": sys.argv[1] + "/conversion.json"}}))' "$checkpoint_root")
exec vllm serve "$checkpoint_root" \
  --served-model-name hy4-w4a8 \
  --tensor-parallel-size 8 --moe-backend aiter \
  --quantization slimquant_w4a8 --hf-overrides "$overrides" \
  --gpu-memory-utilization 0.95 --max-model-len 1024 \
  --max-num-seqs 1 --max-num-batched-tokens 128 \
  --host 127.0.0.1 --port 8000 "$@"
