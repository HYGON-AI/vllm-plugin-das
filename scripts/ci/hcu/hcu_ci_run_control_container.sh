#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

set -euo pipefail

workdir="${GITHUB_WORKSPACE:-$(pwd)}"
if [[ "${1:-}" == "-w" || "${1:-}" == "--workdir" ]]; then
  workdir="$2"
  shift 2
fi
if [[ "${1:-}" == "--" ]]; then
  shift
fi
if [[ $# -eq 0 ]]; then
  echo "command is required" >&2
  exit 2
fi

# ARC K8s pools (HCU_CI_IN_K8S=1): the job already runs inside the HCU CI
# image without a docker daemon, so run the command in place with the same
# environment the control container would get.
if [[ "${HCU_CI_IN_K8S:-0}" == "1" ]]; then
  workspace="$(realpath "${GITHUB_WORKSPACE:-$workdir}")"
  workdir="$(realpath "$workdir")"
  export CI=1 HOME=/tmp/hcu-ci-home PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$workspace" \
    TORCHINDUCTOR_CACHE_DIR=/tmp/hcu-ci-torchinductor XDG_CACHE_HOME=/tmp/hcu-ci-cache \
    LOGNAME="$(id -un)" USER="$(id -un)"
  echo "using HCU CI control environment: current K8s job container" >&2
  mkdir -p "$HOME" "$TORCHINDUCTOR_CACHE_DIR" "$XDG_CACHE_HOME"
  cd "$workdir"
  exec bash -lc 'set -e; exec "$@"' -- "$@"
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
image="$("$script_dir/hcu_ci_resolve_image.sh")"
echo "using HCU CI control image: $image" >&2
workspace="$(realpath "${GITHUB_WORKSPACE:-$workdir}")"
workdir="$(realpath "$workdir")"
runner_user="$(id -un)"

docker_args=(
  run --rm
  --user "$(id -u):$(id -g)"
  --network host
  --ipc host
  --workdir "$workdir"
  --volume "$workspace:$workspace"
  --volume /etc/passwd:/etc/passwd:ro
  --volume /etc/group:/etc/group:ro
  --env CI=1
  --env HOME=/tmp/hcu-ci-home
  --env "LOGNAME=$runner_user"
  --env PYTHONDONTWRITEBYTECODE=1
  --env PYTHONPATH="$workspace"
  --env TORCHINDUCTOR_CACHE_DIR=/tmp/hcu-ci-torchinductor
  --env "USER=$runner_user"
  --env XDG_CACHE_HOME=/tmp/hcu-ci-cache
)

if [[ -n "${RUNNER_TEMP:-}" && -d "${RUNNER_TEMP:-}" ]]; then
  runner_temp="$(realpath "$RUNNER_TEMP")"
  docker_args+=(--volume "$runner_temp:$runner_temp")
fi
if [[ -d /opt/hyhal ]]; then
  docker_args+=(--volume /opt/hyhal:/opt/hyhal:ro)
fi

for name in \
  GITHUB_ACTION GITHUB_ACTIONS GITHUB_ACTOR GITHUB_API_URL GITHUB_BASE_REF \
  GITHUB_ENV GITHUB_EVENT_NAME GITHUB_EVENT_PATH GITHUB_GRAPHQL_URL \
  GITHUB_HEAD_REF GITHUB_OUTPUT GITHUB_REF GITHUB_REPOSITORY GITHUB_RUN_ATTEMPT \
  GITHUB_RUN_ID GITHUB_SERVER_URL GITHUB_SHA GITHUB_STEP_SUMMARY \
  GITHUB_TOKEN GITHUB_WORKSPACE RUNNER_NAME RUNNER_OS RUNNER_TEMP \
  PR_NUMBER BASE_REF BASE_SHA HEAD_SHA FULL_HCU ACCURACY_HCU; do
  if [[ -n "${!name:-}" ]]; then
    docker_args+=(--env "$name=${!name}")
  fi
done

docker "${docker_args[@]}" "$image" bash -lc '
  set -e
  mkdir -p "$HOME" "$TORCHINDUCTOR_CACHE_DIR" "$XDG_CACHE_HOME"
  exec "$@"
' -- "$@"
