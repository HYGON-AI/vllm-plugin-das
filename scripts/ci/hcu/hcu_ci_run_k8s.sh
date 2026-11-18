#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

set -euo pipefail

if [[ "${HCU_CI_IN_K8S:-0}" != "1" ]]; then
  echo "native HCU execution requires the ARC job environment" >&2
  exit 2
fi
if [[ "${HCU_CI_ISOLATE_GENERATED_CODE:-0}" != "0" ]]; then
  echo "generated-code jobs require the existing hardened legacy sandbox" >&2
  exit 2
fi
if [[ $# -eq 0 ]]; then
  echo "command is required" >&2
  exit 2
fi

# DTK initialization must not replace scheduler-provided budgets or visibility.
declare -A platform_environment=()
for name in OMP_NUM_THREADS MAX_JOBS CMAKE_BUILD_PARALLEL_LEVEL HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES; do
  if [[ -v "$name" ]]; then platform_environment[$name]="${!name}"; fi
done
source /opt/dtk/env.sh
for name in "${!platform_environment[@]}"; do
  export "$name=${platform_environment[$name]}"
done
cd "${GITHUB_WORKSPACE:?GITHUB_WORKSPACE is required}"
export CI=1 VLLM_HCU_IS_IN_CI=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$GITHUB_WORKSPACE${PYTHONPATH:+:$PYTHONPATH}"
export HCU_CI_ENVIRONMENT_LOCK="$GITHUB_WORKSPACE/.github/workflows/configs/hcu-runner-environment.json"
# The platform owns this NFS mount. Never silently recreate it on local disk.
if [[ ! -d /ci_public || ! -w /ci_public ]]; then
  echo "ARC requires the platform-provided writable /ci_public mount" >&2
  exit 2
fi
# Do not consume a mutable cache published by another PR or a previous attempt.
# Namespace both architectures and all partitions by their original job ID.
if [[ ! "${GITHUB_REPOSITORY:-}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$ ||
      ! "${GITHUB_RUN_ID:-}" =~ ^[0-9]+$ ||
      ! "${GITHUB_RUN_ATTEMPT:-}" =~ ^[0-9]+$ ||
      ! "${HCU_CI_JOB_ID:-}" =~ ^[A-Za-z0-9_-]+$ ]]; then
  echo "invalid ARC scratch namespace" >&2
  exit 2
fi
scratch="/ci_public/hcu-ci/$GITHUB_REPOSITORY/$GITHUB_RUN_ID/$GITHUB_RUN_ATTEMPT/$HCU_CI_JOB_ID"
# Only installers opt into this temporary directory. Runtime TMPDIR stays local:
# vLLM/Python use it for Unix sockets, whose path length is limited on Linux.
export HCU_CI_NFS_TMPDIR="$scratch/tmp"
export PIP_CACHE_DIR="$scratch/pip-cache"
export TORCHINDUCTOR_CACHE_DIR="$scratch/torchinductor"
export XDG_CACHE_HOME="$scratch/cache"
mkdir -p "$HCU_CI_NFS_TMPDIR" "$PIP_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$XDG_CACHE_HOME" "${HCU_CI_JOB_ROOT:?HCU_CI_JOB_ROOT is required}"
echo "ARC scratch: $scratch; reports: $HCU_CI_JOB_ROOT" >&2
# Only trust this checkout for package metadata; never trust all directories.
git config --global --add safe.directory "$GITHUB_WORKSPACE"

resource_roots() {
  local configured="$1" suffix="$2" primary_env="$3" roots_env="$4"
  local root joined
  local roots=()
  if [[ -n "$configured" ]]; then
    if [[ "$suffix" == llm-models && -d "$configured/$suffix" ]]; then configured="$configured/$suffix"; fi
    if [[ ! -d "$configured" || "$configured" == *:* || "$configured" == *$'\n'* ]]; then
      echo "invalid configured resource root: $configured" >&2
      return 2
    fi
    roots+=("$configured")
  fi
  # Use only platform-provided read-only mounts. Do not recreate host mounts
  # or old /models and /datasets aliases inside the job container.
  for root in "/public/opendas/DL_DATA/$suffix" "/public4/opendas/DL_DATA/$suffix"; do
    if [[ -d "$root" ]]; then roots+=("$root"); fi
  done
  if [[ ${#roots[@]} -gt 0 ]]; then
    joined="$(IFS=:; echo "${roots[*]}")"
    export "$primary_env=${roots[0]}" "$roots_env=$joined"
  fi
}
resource_roots "${HCU_CI_MODEL_ROOT:-}" llm-models VLLM_HCU_TEST_MODEL_ROOT VLLM_HCU_TEST_MODEL_ROOTS
resource_roots "${HCU_CI_DATASET_ROOT:-}" ci_datasets VLLM_HCU_TEST_DATASET_ROOT VLLM_HCU_TEST_DATASET_ROOTS
# Keep platform-injected thread budgets and device visibility unchanged.
exec "$@"
