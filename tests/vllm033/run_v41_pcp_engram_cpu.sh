#!/usr/bin/env bash
set -euo pipefail
engram_repo=$(realpath "$(dirname "${BASH_SOURCE[0]}")/../..")
engram_core=$(realpath "${VLLM_TARGET_ROOT:-${engram_repo}/../vllm}")
engram_adapter=$(realpath "${HCU_ADAPTER_SOURCE:-${engram_repo}/../adapter-das}")
engram_output=${1:?Provide output directory}
: "${VLLM_TEST_PYTHON:?Set existing Python interpreter}"
mkdir -p "${engram_output}"
engram_output=$(realpath "${engram_output}")
export CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES=
export HCU_ADAPTER_HOST=none VLLM_PLUGINS=__disabled__ VLLM_TARGET_DEVICE=cpu
export VLLM_NO_USAGE_STATS=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_TARGET_ROOT="${engram_core}"
export PYTHONPATH="${engram_core}:${engram_adapter}:${engram_repo}/adapter_extension/src"
"${VLLM_TEST_PYTHON}" -B "${engram_repo}/tests/vllm033/replay_v41_pcp_engram.py" "${engram_output}/report.json" 2>&1 | tee "${engram_output}/replay.log"
