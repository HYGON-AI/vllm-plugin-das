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
docker rm -f "$container_name" >/dev/null 2>&1 || true
