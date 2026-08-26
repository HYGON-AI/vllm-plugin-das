#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

set -euo pipefail

container_name="${HCU_CI_CONTAINER_NAME:-}"
workdir=/vllm-plugin-das
while [[ $# -gt 0 ]]; do
  case "$1" in
    --container-name) container_name="$2"; shift 2 ;;
    -w|--workdir) workdir="$2"; shift 2 ;;
    --) shift; break ;;
    *) echo "unknown argument before --: $1" >&2; exit 2 ;;
  esac
done
if [[ -z "$container_name" || $# -eq 0 ]]; then
  echo "container name and command after -- are required" >&2
  exit 2
fi

exec_args=(exec --workdir "$workdir")
for name in \
  HCU_CI_JOB_ID HCU_CI_JOB_ROOT HCU_CI_REGISTRY_JOB HCU_CI_ARCH HCU_CI_CARDS \
  HCU_CI_SUITE HCU_CI_PARTITION_ID HCU_CI_PARTITION_SIZE \
  HCU_CI_PYTEST_ARGS_JSON HCU_CI_REQUIREMENTS_JSON \
  VLLM_HCU_USE_FLASH_ATTN_UNIFIED VLLM_WORKER_MULTIPROC_METHOD \
  VLLM_HCU_TEST_STRICT_RESOURCES HF_HUB_OFFLINE HF_DATASETS_OFFLINE; do
  if [[ -n "${!name:-}" ]]; then
    exec_args+=(--env "$name=${!name}")
  fi
done

docker "${exec_args[@]}" "$container_name" \
  bash -lc 'source /opt/dtk/env.sh && exec "$@"' -- "$@"
