#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

set -euo pipefail

inspect_ref() {
  local ref="$1"
  local digest image_id

  digest="$(
    docker image inspect "$ref" \
      --format '{{range .RepoDigests}}{{println .}}{{end}}' 2>/dev/null \
      | sed '/^$/d' \
      | head -n1
  )" || true
  if [[ -n "$digest" ]]; then
    printf '%s\n' "$digest"
    return 0
  fi

  image_id="$(docker image inspect "$ref" --format '{{.Id}}' 2>/dev/null || true)"
  if [[ "$image_id" =~ ^sha256:[0-9a-fA-F]{64}$ ]]; then
    printf '%s\n' "$image_id"
    return 0
  fi
  return 1
}

has_control_python() {
  local ref="$1"

  docker run --rm --entrypoint /bin/bash "$ref" \
    -lc 'test -x /usr/local/bin/python3.10' >/dev/null 2>&1
}

refresh_ref() {
  local ref="$1"

  if [[ "$ref" =~ ^sha256:[0-9a-fA-F]{64}$ ]]; then
    return 1
  fi
  docker pull "$ref" >/dev/null 2>&1
}

emit_usable_ref() {
  local ref="$1"

  if ! has_control_python "$ref"; then
    refresh_ref "$ref" || true
  fi
  if ! has_control_python "$ref"; then
    echo "skipping HCU CI image candidate without /usr/local/bin/python3.10: $ref" >&2
    return 1
  fi
  inspect_ref "$ref"
}

if [[ -n "${HCU_CI_IMAGE:-}" ]]; then
  printf '%s\n' "$HCU_CI_IMAGE"
  exit 0
fi

if [[ -r "${HCU_CI_IMAGE_FILE:-/etc/vllm-hcu-ci/image}" ]]; then
  image="$(sed -n 's/^[[:space:]]*//;s/[[:space:]]*$//;/^$/!p' "${HCU_CI_IMAGE_FILE:-/etc/vllm-hcu-ci/image}" | head -n1)"
  if [[ -n "$image" ]]; then
    printf '%s\n' "$image"
    exit 0
  fi
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is unavailable and HCU_CI_IMAGE is unset" >&2
  exit 2
fi

if emit_usable_ref vllm-hcu-ci:current; then
  exit 0
fi

while IFS= read -r ref; do
  [[ -n "$ref" ]] || continue
  if emit_usable_ref "$ref"; then
    exit 0
  fi
done < <(
  docker image ls \
    --format '{{.Repository}}:{{.Tag}}' \
    | awk '$1 !~ /<none>/ && $1 ~ /(^|\/)vllm:/ && $1 ~ /latest$/ {print}'
)

cat >&2 <<'EOF'
Unable to resolve the HCU CI image.
Set HCU_CI_IMAGE, create /etc/vllm-hcu-ci/image, tag the local image as
vllm-hcu-ci:current, or keep a local */vllm:*latest image on this runner.
EOF
exit 2
