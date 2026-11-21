#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

# 组件编译入口（v0.33 adapter extension），本地与 CI 共用。
# 产物：dist/hcu_vllm_das_plugin-<version>[+hcu.<YYMMDDHHMM>.g<commit6>[.dirty]]-*.whl
# 纯 Python wheel（py3-none-any），不依赖 DTK/torch/aicc，无需 manylinux 重打。
set -euo pipefail

usage() {
    echo "usage: build_hcu_wheel.sh [--no-stamp]"
    echo "  --no-stamp  不追加 +hcu.<时间>.g<commit> 本地版本戳，构建原始版本号"
    echo "  DAS_BUILD_STAMP=<时间>.g<commit>  显式指定版本戳（优先于自动计算）"
    echo "  自动戳在 adapter_extension 有未提交改动时追加 .dirty 后缀"
}

USE_STAMP=1
for arg in "$@"; do
    case "$arg" in
        --no-stamp) USE_STAMP=0 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "build_hcu_wheel.sh: unknown argument: $arg" >&2; usage >&2; exit 2 ;;
    esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROJECT="$ROOT/adapter_extension"

# ---- 前置检查：python/pip（本组件不依赖 DTK/torch/aicc）----
PY="${PYTHON:-}"
if [ -z "$PY" ]; then
    for cand in python3 python; do
        if command -v "$cand" >/dev/null 2>&1 \
            && "$cand" -c "import sys" >/dev/null 2>&1; then
            PY="$cand"
            break
        fi
    done
fi
if [ -z "$PY" ]; then
    echo "build_hcu_wheel.sh: python3/python not found on PATH" >&2
    exit 1
fi
if ! "$PY" -c 'import sys; raise SystemExit(not ((3, 11) <= sys.version_info[:2] < (3, 15)))'; then
    echo "build_hcu_wheel.sh: Python >=3.11,<3.15 is required" \
         "(adapter_extension/pyproject.toml), got $("$PY" --version 2>&1);" \
         "set PYTHON=/path/to/python3.12 to choose another interpreter" >&2
    exit 1
fi
if ! "$PY" -m pip --version >/dev/null 2>&1; then
    echo "build_hcu_wheel.sh: pip is required for $PY" >&2
    exit 1
fi

if [ ! -f "$PROJECT/pyproject.toml" ]; then
    echo "build_hcu_wheel.sh: adapter_extension/pyproject.toml not found (run from the repository)" >&2
    exit 1
fi

VERSION="$(sed -n 's/^version = "\(.*\)"$/\1/p' "$PROJECT/pyproject.toml" | head -1)"
if [ -z "$VERSION" ]; then
    echo "build_hcu_wheel.sh: cannot read version from adapter_extension/pyproject.toml" >&2
    exit 1
fi

if [ "$USE_STAMP" = 1 ] && [ -z "${DAS_BUILD_STAMP:-}" ]; then
    if ! { command -v git >/dev/null 2>&1 \
        && git -C "$ROOT" rev-parse --short=6 HEAD >/dev/null 2>&1; }; then
        echo "build_hcu_wheel.sh: git commit unavailable for the build stamp;" \
             "pass --no-stamp or set DAS_BUILD_STAMP explicitly" >&2
        exit 1
    fi
fi

# ---- 其余构建依赖由脚本自行安装 ----
# setuptools>=77 才支持 pyproject 里的 PEP 639 license 字段；
# pip wheel --no-build-isolation 要求当前环境已具备构建后端
if ! "$PY" - <<'EOF' >/dev/null 2>&1
import sys
try:
    from importlib.metadata import version
    major, minor = (int(part) for part in version("setuptools").split(".")[:2])
except Exception:
    raise SystemExit(1)
raise SystemExit(0 if (major, minor) >= (77, 0) else 1)
EOF
then
    echo "== installing build deps (setuptools>=77, wheel) =="
    "$PY" -m pip install --no-cache-dir "setuptools>=77" wheel
fi

# ---- 构建：版本单一来源是 adapter_extension/pyproject.toml ----
# 带戳构建使用临时副本改写版本，不修改被跟踪的 pyproject.toml
mkdir -p "$ROOT/dist"
BUILD="$PROJECT"
if [ "$USE_STAMP" = 1 ]; then
    if [ -z "${DAS_BUILD_STAMP:-}" ]; then
        DAS_BUILD_STAMP="$(date +%y%m%d%H%M).g$(git -C "$ROOT" rev-parse --short=6 HEAD)"
        # 拷贝带的是工作区当前内容, 有未提交改动时 HEAD 无法还原代码, 追加 .dirty 区分
        if [ -n "$(git -C "$PROJECT" status --porcelain -- .)" ]; then
            DAS_BUILD_STAMP="${DAS_BUILD_STAMP}.dirty"
        fi
    fi
    BUILD_VERSION="${VERSION%%+*}+hcu.${DAS_BUILD_STAMP}"
    BUILD="$(mktemp -d)"
    trap 'rm -rf "$BUILD"' EXIT
    tar -C "$PROJECT" \
        --exclude=build --exclude='*.egg-info' \
        --exclude=__pycache__ --exclude='*.pyc' \
        -cf - . | tar -C "$BUILD" -xf -
    sed -i "s/^version = \".*\"$/version = \"${BUILD_VERSION}\"/" "$BUILD/pyproject.toml"
    # 运行时 __version__ 与 distribution 元数据保持一致, 否则装好的包仍报告原始版本
    sed -i "s/^__version__ = \".*\"$/__version__ = \"${BUILD_VERSION}\"/" "$BUILD/src/hcu_vllm_das/version.py"
    if ! grep -qxF "version = \"${BUILD_VERSION}\"" "$BUILD/pyproject.toml" \
        || ! grep -qxF "__version__ = \"${BUILD_VERSION}\"" "$BUILD/src/hcu_vllm_das/version.py"; then
        echo "build_hcu_wheel.sh: failed to stamp version in the build copy" >&2
        exit 1
    fi
else
    BUILD_VERSION="$VERSION"
fi

echo "== building hcu-vllm-das-plugin ${BUILD_VERSION} ($BUILD) =="
"$PY" -m pip wheel --no-deps --no-build-isolation \
    --wheel-dir "$ROOT/dist" "$BUILD"

WHEEL="$(ls -t "$ROOT"/dist/hcu_vllm_das_plugin-"${BUILD_VERSION}"-*.whl | head -1)"
sha256sum "$WHEEL"
