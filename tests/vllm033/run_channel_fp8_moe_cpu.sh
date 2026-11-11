#!/usr/bin/env bash
# CPU source replay only; HCU providers, quantization, DeepEP and GEMM are doubles.
# The owned MoE method captures get_current_vllm_config() at construction, which
# production always has. The harness supplies that context; the replay itself is
# run unchanged.
set -euo pipefail
channel_repo=$(realpath "$(dirname "${BASH_SOURCE[0]}")/../..")
channel_core=$(realpath "${VLLM_TARGET_ROOT:-${channel_repo}/../vllm}")
channel_adapter=$(realpath "${HCU_ADAPTER_SOURCE:-${channel_repo}/../adapter-das}")
channel_output=${1:?Provide a dedicated output directory}
: "${VLLM_TEST_PYTHON:?Set the existing Python 3.12 interpreter}"
channel_output=$(realpath -m "${channel_output}")
case "${channel_output}" in
  "${channel_repo}/tests/vllm033/.cache/channel-fp8-moe/"*) ;;
  *) echo 'Output must be inside the dedicated ignored Channel-FP8 evidence directory' >&2; exit 2 ;;
esac
mkdir -p "${channel_output}"
export CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES=
export HCU_ADAPTER_HOST=none VLLM_PLUGINS=__disabled__ VLLM_TARGET_DEVICE=cpu
export VLLM_NO_USAGE_STATS=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_TARGET_ROOT="${channel_core}"
export PYTHONPATH="${channel_core}:${channel_adapter}:${channel_repo}/adapter_extension/src"
"${VLLM_TEST_PYTHON}" -B "${channel_repo}/tests/vllm033/replay_channel_fp8_moe_context.py" \
  "${channel_output}/report.json" 2>&1 | tee "${channel_output}/replay.log"
