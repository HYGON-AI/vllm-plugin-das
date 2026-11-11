#!/usr/bin/env bash
# CPU source replay only; hardware admission, quantization and GEMM are doubles.
# Usage: VLLM_TEST_PYTHON=/absolute/python3.12 bash this-script OUTPUT_DIRECTORY
set -euo pipefail
loading_repo=$(realpath "$(dirname "${BASH_SOURCE[0]}")/../..")
loading_core=$(realpath "${VLLM_TARGET_ROOT:-${loading_repo}/../vllm}")
loading_adapter=$(realpath "${HCU_ADAPTER_SOURCE:-${loading_repo}/../adapter-das}")
loading_output=${1:?Provide a dedicated output directory}
: "${VLLM_TEST_PYTHON:?Set the existing Python 3.12 interpreter}"
mkdir -p "${loading_output}"
loading_output=$(realpath "${loading_output}")
export CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES=
export HCU_ADAPTER_HOST=none VLLM_PLUGINS=__disabled__ VLLM_TARGET_DEVICE=cpu
export VLLM_NO_USAGE_STATS=1 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_TARGET_ROOT="${loading_core}"
export PYTHONPATH="${loading_core}:${loading_adapter}:${loading_repo}/adapter_extension/src"
"${VLLM_TEST_PYTHON}" -B "${loading_repo}/tests/vllm033/replay_dsv41_channel_fp8_loading.py" \
  "${loading_output}/report.json" 2>&1 | tee "${loading_output}/replay.log"
