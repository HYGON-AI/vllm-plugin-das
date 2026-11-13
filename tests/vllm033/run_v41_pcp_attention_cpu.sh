#!/usr/bin/env bash
# CPU source replay only; transports and GPU kernel boundaries are doubled.
set -euo pipefail
pcp_repo=$(realpath "$(dirname "${BASH_SOURCE[0]}")/../..")
pcp_core=$(realpath "${VLLM_TARGET_ROOT:-${pcp_repo}/../vllm}")
pcp_adapter=$(realpath "${HCU_ADAPTER_SOURCE:-${pcp_repo}/../adapter-das}")
pcp_output=${1:?Provide an output directory}
: "${VLLM_TEST_PYTHON:?Set existing Python 3.12 interpreter}"
mkdir -p "${pcp_output}"
pcp_output=$(realpath "${pcp_output}")
export CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES=
export HCU_ADAPTER_HOST=none VLLM_PLUGINS=__disabled__ VLLM_TARGET_DEVICE=cpu
export VLLM_NO_USAGE_STATS=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_TARGET_ROOT="${pcp_core}"
export PYTHONPATH="${pcp_core}:${pcp_adapter}:${pcp_repo}/adapter_extension/src"
"${VLLM_TEST_PYTHON}" -B "${pcp_repo}/tests/vllm033/replay_v41_pcp_attention.py" "${pcp_output}/report.json" 2>&1 | tee "${pcp_output}/replay.log"
