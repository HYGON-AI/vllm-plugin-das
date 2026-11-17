#!/usr/bin/env bash
# CPU-only source/configuration replay; no install, model, service, or GPU work.
# Usage: VLLM_TEST_PYTHON=/absolute/python3.12 bash this-script OUTPUT_DIRECTORY
set -euo pipefail

routing_repo=$(realpath "$(dirname "${BASH_SOURCE[0]}")/../..")
routing_core=$(realpath "${VLLM_TARGET_ROOT:-${routing_repo}/../vllm}")
routing_adapter=$(realpath "${HCU_ADAPTER_SOURCE:-${routing_repo}/../adapter-das}")
routing_output=${1:?Provide a dedicated output directory}
: "${VLLM_TEST_PYTHON:?Set the existing Python 3.12 interpreter}"
mkdir -p "${routing_output}"
routing_output=$(realpath "${routing_output}")

export CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES=
export HCU_ADAPTER_HOST=none VLLM_PLUGINS=__disabled__ VLLM_TARGET_DEVICE=cpu
export VLLM_NO_USAGE_STATS=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_INDEX=1
export PIP_DISABLE_PIP_VERSION_CHECK=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_TARGET_ROOT="${routing_core}"
export PYTHONPATH="${routing_core}:${routing_adapter}:${routing_repo}/adapter_extension/src"
# Propagate exact ownership exceptions to packaging subprocesses, never to the
# user's global Git configuration.
export GIT_CONFIG_COUNT=3
export GIT_CONFIG_KEY_0=safe.directory GIT_CONFIG_VALUE_0="${routing_repo}"
export GIT_CONFIG_KEY_1=safe.directory GIT_CONFIG_VALUE_1="${routing_core}"
export GIT_CONFIG_KEY_2=safe.directory GIT_CONFIG_VALUE_2="${routing_adapter}"

{
  git -c "safe.directory=${routing_repo}" -C "${routing_repo}" rev-parse HEAD
  git -c "safe.directory=${routing_core}" -C "${routing_core}" rev-parse HEAD
  git -c "safe.directory=${routing_adapter}" -C "${routing_adapter}" rev-parse HEAD
  "${VLLM_TEST_PYTHON}" -B - <<'PY'
import importlib.metadata as metadata
import inspect
import json
import sys
import torch
from vllm.config import ModelConfig
print(json.dumps({
    "python": sys.version,
    "installed_vllm_metadata": metadata.version("vllm"),
    "configuration_source": inspect.getsourcefile(ModelConfig),
    "cuda_initialized": torch.cuda.is_initialized(),
    "validation_level": "CPU configuration/source replay, not installed-core or GPU acceptance",
}, indent=2))
assert not torch.cuda.is_initialized()
PY
} 2>&1 | tee "${routing_output}/environment.log"

"${VLLM_TEST_PYTHON}" -B -m pytest -q -p no:cacheprovider \
  --junitxml="${routing_output}/pytest.xml" \
  "${routing_repo}/tests/vllm033/test_dsv41_channel_fp8.py" \
  "${routing_repo}/tests/vllm033/test_package_boundary.py" \
  "${routing_repo}/tests/vllm033/test_extension.py" \
  "${routing_repo}/tests/vllm033/test_vllm_033_contract.py" \
  2>&1 | tee "${routing_output}/pytest.log"
