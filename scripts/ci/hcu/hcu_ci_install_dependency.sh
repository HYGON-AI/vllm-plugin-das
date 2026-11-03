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
    bash -lc \
    'source /opt/dtk/env.sh && exec /usr/local/bin/hcu-ci-python -m pip install --no-deps --no-build-isolation -e .'
fi

# Install CI requirements in this container; the base image stays unchanged.
docker exec --workdir /vllm-plugin-das "$container_name" \
  bash -lc \
  'source /opt/dtk/env.sh && exec /usr/local/bin/hcu-ci-python .github/scripts/hcu_ci/ensure_ci_dependencies.py'

docker exec -i --workdir /vllm-plugin-das "$container_name" \
  bash -lc 'source /opt/dtk/env.sh && exec /usr/local/bin/hcu-ci-python -' <<'PY'
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

# Evaluation jobs may bootstrap their declared evaluator in the ephemeral container.
docker exec --workdir /vllm-plugin-das "$container_name" \
  bash -lc \
  'source /opt/dtk/env.sh && exec /usr/local/bin/hcu-ci-python .github/scripts/hcu_ci/ensure_evalscope.py'
