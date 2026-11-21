#!/usr/bin/env bash
# CPU source integration; device operations/providers are explicitly doubled.
set -euo pipefail
w4_repo=$(realpath "$(dirname "${BASH_SOURCE[0]}")/../..")
w4_core=$(realpath "${VLLM_TARGET_ROOT:-${w4_repo}/../vllm}")
w4_adapter=$(realpath "${HCU_ADAPTER_SOURCE:-${w4_repo}/../adapter-das}")
w4_output=${1:?Provide an output directory}
w4_mode=${2:-weights}
case "${w4_mode}" in weights|execution) ;; *) exit 2 ;; esac
: "${VLLM_TEST_PYTHON:?Set the existing Python 3.12 interpreter}"
mkdir -p "${w4_output}"
w4_output=$(realpath "${w4_output}")
export CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES=
export HCU_ADAPTER_HOST=none VLLM_PLUGINS=__disabled__ VLLM_TARGET_DEVICE=cpu
export VLLM_NO_USAGE_STATS=1 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_TARGET_ROOT="${w4_core}"
export PYTHONPATH="${w4_core}:${w4_adapter}:${w4_repo}/adapter_extension/src"
"${VLLM_TEST_PYTHON}" -B "${w4_repo}/tests/vllm033/replay_ct_w4a8_${w4_mode}.py" \
  "${w4_output}/report.json" 2>&1 | tee "${w4_output}/replay.log"
