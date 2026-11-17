#!/usr/bin/env bash
set -euo pipefail
if [[ ${1:-} == --help ]]; then
  echo 'Usage: run_extension_probe_node.sh OUTPUT_JSON [PYTHON]'
  exit 0
fi
output=${1:?Usage: run_extension_probe_node.sh OUTPUT_JSON [PYTHON]}
python_exe=${2:-python}
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
plugin_root=$(cd -- "$script_dir/../.." && pwd)
export PYTHONPATH="$plugin_root/adapter_extension/src:$plugin_root${PYTHONPATH:+:$PYTHONPATH}"
export HCU_ADAPTER_VLLM_REQUIRED_EXTENSIONS=kimi31_slimquant_exp
export K31_PROBE_OUTPUT="$output"
exec "$python_exe" "$script_dir/probe_extension_spawn.py"
