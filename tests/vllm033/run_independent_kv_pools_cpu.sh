#!/usr/bin/env bash
# Installed Core wheel; do not add the Core source checkout to PYTHONPATH.
set -euo pipefail
plugin_root=$(realpath "$(dirname "${BASH_SOURCE[0]}")/../..")
project_root=$(realpath "${plugin_root}/..")
: "${VLLM_TEST_PYTHON:?Set the installed vLLM 0.33 Python interpreter}"
output=${1:?Provide a result directory}
mkdir -p "${output}"
output=$(realpath "${output}")
export CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES=
export HCU_ADAPTER_HOST=none VLLM_PLUGINS=__disabled__ VLLM_TARGET_DEVICE=cpu
export PYTHONSAFEPATH=1 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2
export PYTHONPATH="${project_root}/adapter-das:${plugin_root}/adapter_extension/src:${plugin_root}/tests/vllm033"
"${VLLM_TEST_PYTHON}" -m pytest --noconftest -q \
  "${plugin_root}/tests/vllm033/test_dsa_cache_pools.py" \
  "${plugin_root}/tests/vllm033/test_swa_cache_layer_split.py" \
  "${plugin_root}/tests/vllm033/test_dsa_cache_layout.py" \
  "${plugin_root}/tests/vllm033/test_mooncake_layer_split.py" \
  "${project_root}/adapter-das/tests/adapter/test_vllm_kv_cache.py" \
  2>&1 | tee "${output}/pytest.log"
"${VLLM_TEST_PYTHON}" -B "${plugin_root}/tests/vllm033/replay_v41_pcp_attention.py" \
  "${output}/pcp_replay.json" 2>&1 | tee "${output}/pcp_replay.log"
"${VLLM_TEST_PYTHON}" -B "${plugin_root}/tests/vllm033/report_dsa_cache_pools.py" \
  > "${output}/capacity.json"
