#!/usr/bin/env bash
# Fixed CPU replay for the adapter-owned CT W8A8 INT8 method: only its dedicated
# ignored evidence directory is written.
set -euo pipefail
int8_repo=$(realpath "$(dirname "${BASH_SOURCE[0]}")/../..")
int8_core=$(realpath "${VLLM_TARGET_ROOT:-${int8_repo}/../vllm}")
int8_adapter=$(realpath "${HCU_ADAPTER_SOURCE:-${int8_repo}/../adapter-das}")
int8_output=${1:?Provide dedicated output directory}
: "${VLLM_TEST_PYTHON:?Set existing Python 3.12 interpreter}"
int8_output=$(realpath -m "${int8_output}")
case "${int8_output}" in
  "${int8_repo}/tests/vllm033/.cache/dsv41-int8-linear/"*) ;;
  *) echo 'Output must be inside the dedicated ignored INT8 linear evidence directory' >&2; exit 2 ;;
esac
mkdir -p "${int8_output}"
export CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES=
export HCU_ADAPTER_HOST=none VLLM_PLUGINS=__disabled__ VLLM_TARGET_DEVICE=cpu
export VLLM_NO_USAGE_STATS=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_TARGET_ROOT="${int8_core}"
export PYTHONPATH="${int8_core}:${int8_adapter}:${int8_repo}/adapter_extension/src"
"${VLLM_TEST_PYTHON}" -B "${int8_repo}/tests/vllm033/replay_dsv41_int8_linear.py" "${int8_output}/report.json" 2>&1 | tee "${int8_output}/replay.log"
