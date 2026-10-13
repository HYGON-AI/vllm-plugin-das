#!/usr/bin/env bash

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
requirements_path=${1:-"${script_dir}/../../../docker/requirements-vllm-runtime.txt"}
requirements_path=$(realpath -- "${requirements_path}")
python=${PYTHON:-python3}

exec "${python}" -m pip install --no-cache-dir --requirement "${requirements_path}"
