#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

set -euo pipefail

python_bin="${1:-}"
wrapper="${2:-}"
if [[ -z "$python_bin" || -z "$wrapper" ]]; then
  echo "usage: $0 PYTHON_BIN WRAPPER" >&2
  exit 2
fi
if [[ ! -x "$python_bin" ]]; then
  echo "Python interpreter is not executable: $python_bin" >&2
  exit 2
fi

mkdir -p "$(dirname -- "$wrapper")"
printf '#!/usr/bin/env bash\nexec %q "$@"\n' "$python_bin" > "$wrapper"
chmod 0755 "$wrapper"
