#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

set -euo pipefail

container_name="${HCU_CI_CONTAINER_NAME:-}"
if [[ -z "$container_name" ]]; then
  echo "HCU_CI_CONTAINER_NAME is required" >&2
  exit 2
fi

if [[ "${HCU_CI_SKIP_PROJECT_INSTALL:-0}" != "1" ]]; then
  docker exec --workdir /vllm-plugin-das "$container_name" \
    /usr/local/bin/python3.10 -m pip install --no-deps --no-build-isolation -e .
fi

docker exec -i --workdir /vllm-plugin-das "$container_name" /usr/local/bin/python3.10 - <<'PY'
import importlib.metadata
import sys

required = ("pytest", "torch", "vllm")
missing = []
for name in required:
    try:
        version = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        missing.append(name)
    else:
        print(f"{name}={version}")
if missing:
    raise SystemExit(f"HCU CI image is missing required distributions: {missing}")
print(f"python={sys.version}")
PY
