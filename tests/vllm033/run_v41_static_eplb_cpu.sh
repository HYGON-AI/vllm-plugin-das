#!/usr/bin/env bash
# CPU source replay only; EP membership and Core routing kernels are doubled.
set -euo pipefail
eplb_repo=$(realpath "$(dirname "${BASH_SOURCE[0]}")/../..")
eplb_adapter=$(realpath "${HCU_ADAPTER_SOURCE:-${eplb_repo}/../adapter-das}")
eplb_output=${1:?Provide an output directory}
: "${VLLM_TEST_PYTHON:?Set existing Python 3.12 interpreter}"
mkdir -p "${eplb_output}"
eplb_output=$(realpath "${eplb_output}")
export CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES=
export HCU_ADAPTER_HOST=none VLLM_PLUGINS=__disabled__ VLLM_TARGET_DEVICE=cpu
export VLLM_NO_USAGE_STATS=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# Source packages precede installed Core/dependencies. -I -S prevents an older
# sitecustomize or .pth owner from activating before those paths bind.
"${VLLM_TEST_PYTHON}" -I -S -B - \
  "${eplb_adapter}" "${eplb_repo}" "${VLLM_TARGET_ROOT:-}" \
  "${eplb_output}/report.json" <<'PY' 2>&1 | tee "${eplb_output}/replay.log"
import runpy
import sys
import sysconfig
from pathlib import Path

adapter, plugin, core, report = sys.argv[1:]
sys.path[:0] = [adapter, str(Path(plugin) / 'adapter_extension/src')]
if core:
    sys.path.insert(2, core)
for key in ('purelib', 'platlib'):
    path = sysconfig.get_path(key)
    if path not in sys.path:
        sys.path.append(path)
script = str(Path(plugin) / 'tests/vllm033/replay_v41_static_eplb.py')
sys.argv = [script, report]
runpy.run_path(script, run_name='__main__')
PY
