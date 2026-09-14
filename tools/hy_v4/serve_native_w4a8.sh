#!/usr/bin/env bash
# Hardware-unvalidated checkpoint adapter. Supply the native checkpoint path.
set -euo pipefail
checkpoint_root=${1:?Usage: serve_native_w4a8.sh CHECKPOINT_DIR [vllm serve options]}
shift
overrides=$(python3 -c 'import json, sys; print(json.dumps({"quantization_config": {"quant_method": "slimquant_w4a8", "checkpoint_format": "hy4_w4a8_v1", "checkpoint_index": sys.argv[1] + "/hy4-checkpoint.index.json"}}))' "$checkpoint_root")
exec vllm serve "$checkpoint_root" \
  --served-model-name hy4-native-w4a8 \
  --tensor-parallel-size 8 --moe-backend aiter \
  --quantization slimquant_w4a8 --hf-overrides "$overrides" \
  --gpu-memory-utilization 0.95 --max-model-len 4096 \
  --max-num-seqs 1 --max-num-batched-tokens 128 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --default-chat-template-kwargs '{"reasoning_effort":"no_think"}' \
  --host 127.0.0.1 --port 8000 "$@"
