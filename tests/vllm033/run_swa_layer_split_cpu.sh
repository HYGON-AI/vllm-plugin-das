#!/usr/bin/env bash
# CPU only: installed Core wheel + current Adapter and Plugin sources.
set -euo pipefail
plugin_root=$(realpath "$(dirname "${BASH_SOURCE[0]}")/../..")
project_root=$(realpath "${plugin_root}/..")
: "${VLLM_TEST_PYTHON:?Set the existing vLLM 0.33 Python interpreter}"
output=${1:?Provide a new result directory}
mkdir -p "${output}"
output=$(realpath "${output}")
export CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES=
export HCU_ADAPTER_HOST=none VLLM_PLUGINS=__disabled__ VLLM_TARGET_DEVICE=cpu
export PYTHONSAFEPATH=1 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2
export PYTHONPATH="${project_root}/adapter-das:${plugin_root}/adapter_extension/src:${plugin_root}/tests/vllm033"
"${VLLM_TEST_PYTHON}" -m pytest --noconftest -q \
  "${plugin_root}/tests/vllm033/test_swa_cache_layer_split.py" \
  "${plugin_root}/tests/vllm033/test_dsa_cache_layout.py" \
  "${plugin_root}/tests/vllm033/test_mooncake_layer_split.py" \
  "${project_root}/adapter-das/tests/adapter/test_vllm_kv_cache.py" \
  2>&1 | tee "${output}/pytest.log"
"${VLLM_TEST_PYTHON}" -B "${plugin_root}/tests/vllm033/replay_v41_pcp_attention.py" \
  "${output}/pcp_replay.json" 2>&1 | tee "${output}/pcp_replay.log"
"${VLLM_TEST_PYTHON}" - <<'PY' | tee "${output}/capacity.json"
import json
import torch
from test_swa_cache_layer_split import real_groups, config
from hcu_vllm_das import dsa_cache_layout as placement
from vllm.v1.core.kv_cache_utils import get_kv_cache_capacity
from vllm.v1.core.kv_cache_planning import _pool_bytes_per_block_from_builder

cfg, groups = config(), real_groups()
cfg.additional_config = {'hcu_layer_split': {'enabled': True}}
cfg.model_config.max_model_len = 16384
cfg.model_config.original_max_model_len = 65536
cfg.max_in_flight_tokens = 16384
cfg.num_prefill_lookahead_tokens = 0
builder = placement.DsaLayerSplitCacheBuilder()
cost = _pool_bytes_per_block_from_builder(builder, cfg, groups)
budget = 10 * 1024**3
rows = {}
for mode, stride in [('off', 116928), ('main_only_historical', 114304), ('main_and_swa', cost)]:
    plan = builder.get_kv_cache_config_from_groups(cfg, groups, budget // stride)
    tokens, concurrency = get_kv_cache_capacity(cfg, plan)
    rows[mode] = dict(bytes_per_physical_block=stride, blocks=plan.num_blocks,
                      capacity_tokens=tokens, concurrency=concurrency)
assert not torch.cuda.is_initialized()
print(json.dumps(dict(kind='CPU planning only; not a GPU measurement',
                     budget_bytes=budget, modes=rows, gpu_initialized=False), indent=2))
PY
