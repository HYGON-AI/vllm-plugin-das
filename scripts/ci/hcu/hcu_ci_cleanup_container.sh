#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

set -euo pipefail

container_name="${HCU_CI_CONTAINER_NAME:-}"
if [[ -z "$container_name" ]]; then
  echo "HCU_CI_CONTAINER_NAME is required" >&2
  exit 2
fi
if [[ ! "$container_name" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]]; then
  echo "invalid HCU CI container name: $container_name" >&2
  exit 2
fi
uid="$(id -u)"
gid="$(id -g)"

if docker inspect "$container_name" >/dev/null 2>&1; then
  docker exec "$container_name" \
    chown -R "$uid:$gid" /vllm-plugin-das /hcu-ci-artifacts \
    >/dev/null 2>&1 || true
fi
docker rm -f "$container_name" >/dev/null 2>&1 || true

repair_path() {
  local path="$1"
  [[ -e "$path" ]] || return 0
  local blocked
  blocked="$(find "$path" \( ! -uid "$uid" -o \( -type d ! -writable \) \) -print -quit 2>/dev/null || true)"
  [[ -n "$blocked" ]] || return 0

  echo "repairing stale ownership under $path for $uid:$gid"
  if command -v sudo >/dev/null 2>&1 && sudo -n true >/dev/null 2>&1; then
    sudo chown -R "$uid:$gid" "$path"
  elif command -v docker >/dev/null 2>&1; then
    local repair_id repair_image
    local -a repair_images=()
    if [[ -n "${HCU_CI_IMAGE:-}" ]]; then
      repair_id="$(docker image inspect "$HCU_CI_IMAGE" --format '{{.Id}}' 2>/dev/null || true)"
      [[ -z "$repair_id" ]] || repair_images+=("$repair_id")
    fi
    while IFS= read -r repair_id; do
      [[ -z "$repair_id" ]] || repair_images+=("$repair_id")
    done < <(docker image ls --format '{{.ID}}' | awk '!seen[$0]++')
    local repaired=false
    for repair_image in "${repair_images[@]}"; do
      if docker run --rm \
          --user 0:0 \
          --entrypoint /bin/sh \
          --volume "$path:/hcu-ci-repair" \
          "$repair_image" \
          -c 'chown -R "$1:$2" /hcu-ci-repair' -- "$uid" "$gid"; then
        repaired=true
        break
      fi
    done
    if [[ "$repaired" != "true" ]]; then
      echo "no local Docker image could repair: $blocked" >&2
      return 1
    fi
  else
    echo "cannot repair non-writable path: $blocked" >&2
    return 1
  fi

  blocked="$(find "$path" \( ! -uid "$uid" -o \( -type d ! -writable \) \) -print -quit 2>/dev/null || true)"
  if [[ -n "$blocked" ]]; then
    echo "path still has invalid ownership or directory permissions: $blocked" >&2
    return 1
  fi
}

repair_path "${GITHUB_WORKSPACE:-}"
repair_path "${HCU_CI_HOST_JOB_ROOT:-}"
