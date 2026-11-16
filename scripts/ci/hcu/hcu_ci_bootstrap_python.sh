#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

set -euo pipefail

wrapper="${1:-/usr/local/bin/hcu-ci-python}"
legacy_python="${2:-/usr/local/bin/python3.10}"
creator="${3:-/tmp/hcu_ci_create_python_wrapper.sh}"

if [[ -x "$wrapper" ]]; then
  exit 0
fi

python_bin="$legacy_python"
if [[ ! -x "$python_bin" ]]; then
  python_bin="$(command -v python3 || true)"
fi
if [[ -z "$python_bin" || ! -x "$python_bin" ]]; then
  echo "no compatible Python entry exists in HCU CI image" >&2
  exit 2
fi
if [[ ! -r "$creator" ]]; then
  echo "Python wrapper creator is unavailable: $creator" >&2
  exit 2
fi

bash "$creator" "$python_bin" "$wrapper"
