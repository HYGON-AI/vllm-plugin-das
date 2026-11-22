#!/usr/bin/env bash
# CPU source integration only; no model forward, device use or installation.
# Usage: VLLM_TEST_PYTHON=/absolute/python3.12 bash this-script OUTPUT_DIRECTORY
set -euo pipefail
draft_repo=$(realpath "$(dirname "${BASH_SOURCE[0]}")/../..")
draft_core=$(realpath "${VLLM_TARGET_ROOT:-${draft_repo}/../vllm}")
draft_adapter=$(realpath "${HCU_ADAPTER_SOURCE:-${draft_repo}/../adapter-das}")
draft_output=${1:?Provide a dedicated output directory}
: "${VLLM_TEST_PYTHON:?Set the existing Python 3.12 interpreter}"
mkdir -p "${draft_output}"
draft_output=$(realpath "${draft_output}")
export CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES=
export HCU_ADAPTER_HOST=none VLLM_PLUGINS=__disabled__ VLLM_TARGET_DEVICE=cpu
export VLLM_NO_USAGE_STATS=1 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_TARGET_ROOT="${draft_core}"
export PYTHONPATH="${draft_core}:${draft_adapter}:${draft_repo}/adapter_extension/src"
"${VLLM_TEST_PYTHON}" -B "${draft_repo}/tests/vllm033/replay_dsv41_dspark_scale_aliases.py" \
  "${draft_output}/report.json" 2>&1 | tee "${draft_output}/replay.log"
