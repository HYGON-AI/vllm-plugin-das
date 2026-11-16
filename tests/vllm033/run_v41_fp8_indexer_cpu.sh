#!/usr/bin/env bash
# CPU source integration; no device use, package install or model forward.
set -euo pipefail
index_repo=$(realpath "$(dirname "${BASH_SOURCE[0]}")/../..")
index_core=$(realpath "${VLLM_TARGET_ROOT:-${index_repo}/../vllm}")
index_adapter=$(realpath "${HCU_ADAPTER_SOURCE:-${index_repo}/../adapter-das}")
index_output=${1:?Provide an output directory}
: "${VLLM_TEST_PYTHON:?Set existing Python 3.12 interpreter}"
mkdir -p "${index_output}"
index_output=$(realpath "${index_output}")
export CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES=
export HCU_ADAPTER_HOST=none VLLM_PLUGINS=__disabled__ VLLM_TARGET_DEVICE=cpu
export VLLM_NO_USAGE_STATS=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_TARGET_ROOT="${index_core}"
export PYTHONPATH="${index_core}:${index_adapter}:${index_repo}/adapter_extension/src"
"${VLLM_TEST_PYTHON}" -B "${index_repo}/tests/vllm033/replay_v41_fp8_indexer.py" \
  "${index_output}/report.json" 2>&1 | tee "${index_output}/replay.log"
