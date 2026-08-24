#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

set -euo pipefail

image="${HCU_CI_IMAGE:-}"
container_name="${HCU_CI_CONTAINER_NAME:-}"
workspace="${GITHUB_WORKSPACE:-$(pwd)}"
artifact_root="${HCU_CI_HOST_JOB_ROOT:-}"
model_root="${HCU_CI_MODEL_ROOT:-${VLLM_HCU_TEST_MODEL_ROOT:-}}"
dataset_root="${HCU_CI_DATASET_ROOT:-${VLLM_HCU_TEST_DATASET_ROOT:-}}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --image) image="$2"; shift 2 ;;
    --container-name) container_name="$2"; shift 2 ;;
    --workspace) workspace="$2"; shift 2 ;;
    --artifact-root) artifact_root="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$image" || -z "$container_name" || -z "$artifact_root" ]]; then
  echo "HCU CI image, container name, and host artifact root are required" >&2
  exit 2
fi
if [[ ! "$container_name" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]]; then
  echo "invalid HCU CI container name: $container_name" >&2
  exit 2
fi
if [[ "${HCU_CI_ALLOW_MUTABLE_IMAGE:-0}" != "1" && ! "$image" =~ @sha256:[0-9a-fA-F]{64}$ ]]; then
  echo "HCU_CI_IMAGE must be pinned by sha256 digest; set HCU_CI_ALLOW_MUTABLE_IMAGE=1 only for local debugging" >&2
  exit 2
fi

workspace="$(realpath "$workspace")"
mkdir -p "$artifact_root"
artifact_root="$(realpath "$artifact_root")"

if [[ "${HCU_CI_SKIP_PULL:-0}" != "1" ]]; then
  docker pull "$image"
fi
docker rm -f "$container_name" >/dev/null 2>&1 || true

docker_args=(
  run --detach
  --name "$container_name"
  --privileged
  --ipc host
  --shm-size "${HCU_CI_SHM_SIZE:-32g}"
  --cap-add SYS_PTRACE
  --security-opt seccomp=unconfined
  --workdir /vllm-plugin-das
  --volume "$workspace:/vllm-plugin-das"
  --volume "$artifact_root:/hcu-ci-artifacts"
  --env VLLM_HCU_IS_IN_CI=1
  --env PYTHONPATH=/vllm-plugin-das
)

missing_devices=()
for device in /dev/kfd /dev/dri; do
  if [[ -e "$device" ]]; then
    docker_args+=(--device "$device:$device")
  else
    missing_devices+=("$device")
  fi
done
if [[ "${#missing_devices[@]}" -gt 0 ]]; then
  echo "HCU device nodes are unavailable on runner: ${missing_devices[*]}" >&2
  echo "This self-hosted runner cannot execute HCU hardware tests." >&2
  exit 2
fi
if [[ -d /opt/hyhal ]]; then
  docker_args+=(--volume /opt/hyhal:/opt/hyhal:ro)
fi

if [[ -z "$model_root" ]]; then
  for candidate in \
    /models/llm-models \
    /public/opendas/DL_DATA/llm-models \
    /public/opendas/DL_DATA; do
    if [[ -d "$candidate/qwen3.5" || -d "$candidate/vllm-optest-models" ]]; then
      model_root="$candidate"
      break
    fi
  done
fi
if [[ -n "$model_root" ]]; then
  model_root="$(realpath "$model_root")"
  if [[ ! -d "$model_root" ]]; then
    echo "HCU model root does not exist: $model_root" >&2
    exit 2
  fi
  docker_args+=(
    --volume "$model_root:/models/llm-models:ro"
    --env VLLM_HCU_TEST_MODEL_ROOT=/models/llm-models
  )
fi

if [[ -n "$dataset_root" ]]; then
  dataset_root="$(realpath "$dataset_root")"
  if [[ ! -d "$dataset_root" ]]; then
    echo "HCU dataset root does not exist: $dataset_root" >&2
    exit 2
  fi
  docker_args+=(
    --volume "$dataset_root:/datasets:ro"
    --env VLLM_HCU_TEST_DATASET_ROOT=/datasets
  )
fi

visible_devices="${HCU_CI_VISIBLE_DEVICES:-}"
if [[ -z "$visible_devices" && "${HCU_CI_CARDS:-}" =~ ^[1-9][0-9]*$ ]]; then
  visible_devices="$(seq -s, 0 $((HCU_CI_CARDS - 1)))"
fi
if [[ -n "$visible_devices" ]]; then
  docker_args+=(
    --env "HIP_VISIBLE_DEVICES=$visible_devices"
    --env "CUDA_VISIBLE_DEVICES=$visible_devices"
  )
fi

for name in \
  HCU_CI_JOB_ID HCU_CI_JOB_ROOT HCU_CI_REGISTRY_JOB HCU_CI_ARCH HCU_CI_CARDS \
  HCU_CI_SUITE HCU_CI_PARTITION_ID HCU_CI_PARTITION_SIZE \
  HCU_CI_PYTEST_ARGS_JSON HCU_CI_REQUIREMENTS_JSON \
  VLLM_HCU_USE_FLASH_ATTN_UNIFIED VLLM_WORKER_MULTIPROC_METHOD \
  VLLM_HCU_TEST_STRICT_RESOURCES HF_HUB_OFFLINE HF_DATASETS_OFFLINE; do
  if [[ -n "${!name:-}" ]]; then
    docker_args+=(--env "$name=${!name}")
  fi
done

docker "${docker_args[@]}" "$image" bash -lc 'while true; do sleep 3600; done'
docker exec "$container_name" git config --global --add safe.directory /vllm-plugin-das
echo "started HCU CI container $container_name from $image"
