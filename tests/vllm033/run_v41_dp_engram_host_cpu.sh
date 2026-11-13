#!/usr/bin/env bash
set -euo pipefail
host_repo=$(realpath "$(dirname "${BASH_SOURCE[0]}")/../..")
host_core=$(realpath "${VLLM_TARGET_ROOT:-${host_repo}/../vllm}")
host_adapter=$(realpath "${HCU_ADAPTER_SOURCE:-${host_repo}/../adapter-das}")
host_output=${1:?Provide output directory}
: "${VLLM_TEST_PYTHON:?Set existing Python interpreter}"
mkdir -p "${host_output}"
host_output=$(realpath "${host_output}")
export CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES=
export HCU_ADAPTER_HOST=none VLLM_PLUGINS=__disabled__ VLLM_TARGET_DEVICE=cpu
export VLLM_NO_USAGE_STATS=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_TARGET_ROOT="${host_core}"
export PYTHONPATH="${host_core}:${host_adapter}:${host_repo}/adapter_extension/src"
"${VLLM_TEST_PYTHON}" -B "${host_repo}/tests/vllm033/replay_v41_dp_engram_host.py" "${host_output}/report.json" 2>&1 | tee "${host_output}/replay.log"
