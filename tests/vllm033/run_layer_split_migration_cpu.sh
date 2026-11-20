#!/usr/bin/env bash
# Official 0.33 installed Core; no GPU services and no Core source PYTHONPATH.
set -euo pipefail
plugin_root=$(realpath "$(dirname "${BASH_SOURCE[0]}")/../..")
project_root=$(realpath "${plugin_root}/..")
: "${VLLM_TEST_PYTHON:?Set the installed vLLM 0.33 Python interpreter}"
output=${1:?Provide a result directory}
mkdir -p "$output"
output=$(realpath "$output")
export CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES=
export HCU_ADAPTER_HOST=none VLLM_PLUGINS=__disabled__ VLLM_TARGET_DEVICE=cpu
export PYTHONSAFEPATH=1 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2
export PYTHONPATH="${project_root}/adapter-das:${plugin_root}/adapter_extension/src:${plugin_root}/tests/vllm033"
export VLLM_HCU_DSV41_PCP_ATTENTION=0
export VLLM_HCU_DSV41_ENGRAM_HOST=0 HCU_VLLM_PCP_IMPL=stock
export HCU_VLLM_FP4_INDEXER_IMPL=stock HCU_DISABLE_OPS=
"$VLLM_TEST_PYTHON" -c 'import importlib.metadata as m, json, torch, vllm, hcu_adapter, hcu_vllm_das; print(json.dumps(dict(versions={n:m.version(n) for n in ("vllm","torch","triton")}, sources={n.__name__:n.__file__ for n in (vllm,hcu_adapter,hcu_vllm_das)},cuda_initialized=torch.cuda.is_initialized()), indent=2))' > "$output/environment.json"
git -C "$project_root/adapter-das" rev-parse HEAD > "$output/adapter-head.txt"
git -C "$plugin_root" rev-parse HEAD > "$output/plugin-head.txt"
"$VLLM_TEST_PYTHON" -m pytest --noconftest -q --junitxml="$output/pytest.xml" \
  "$plugin_root/tests/vllm033/test_dsa_cache_pools.py" \
  "$plugin_root/tests/vllm033/test_swa_cache_layer_split.py" \
  "$plugin_root/tests/vllm033/test_dsa_cache_layout.py" \
  "$plugin_root/tests/vllm033/test_mooncake_layer_split.py" \
  "$plugin_root/tests/vllm033/test_layer_split_migration.py" \
  "$plugin_root/tests/vllm033/test_fp4_layer_split.py" \
  "$plugin_root/tests/vllm033/test_pcp_merge_contracts.py" \
  "$plugin_root/tests/vllm033/test_dsv41_pcp_attention.py" \
  "$project_root/adapter-das/tests/adapter/test_vllm_kv_cache.py" \
  "$project_root/adapter-das/tests/adapter/test_pcp_mode_admission.py" \
  "$project_root/adapter-das/tests/adapter/test_pcp_transfer_contract.py" \
  2>&1 | tee "$output/pytest.log"
"$VLLM_TEST_PYTHON" -B "$plugin_root/tests/vllm033/replay_v41_pcp_attention.py" \
  "$output/pcp_replay.json" 2>&1 | tee "$output/pcp_replay.log"
"$VLLM_TEST_PYTHON" -B "$plugin_root/tests/vllm033/report_dsa_cache_pools.py" > "$output/capacity_theory.json"
