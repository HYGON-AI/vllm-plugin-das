#!/usr/bin/env bash
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV LD_PRELOAD
export HCU_ADAPTER_HOST=none VLLM_PLUGINS=__disabled__
export HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES=
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
exec /usr/local/bin/python3 -B "$here/run.py" "$@"
