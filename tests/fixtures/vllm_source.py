# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

"""Resolve the target vLLM tree without importing vLLM in the parent process."""

from __future__ import annotations

import importlib.util
import os
import sysconfig
from pathlib import Path


def resolve_target_vllm_root() -> Path:
    configured_root = os.environ.get("VLLM_SOURCE_ROOT")
    if configured_root is not None:
        candidates: tuple[Path | None, ...] = (Path(configured_root),)
    else:
        vllm_spec = importlib.util.find_spec("vllm")
        discovered_root = (
            Path(vllm_spec.origin).resolve().parents[1]
            if vllm_spec is not None and vllm_spec.origin is not None
            else None
        )
        installed_roots = tuple(
            Path(path)
            for key in ("platlib", "purelib")
            if (path := sysconfig.get_path(key))
        )
        candidates = (discovered_root, *installed_roots)

    for candidate in candidates:
        if candidate is None:
            continue
        resolved = candidate.resolve()
        if (resolved / "vllm" / "__init__.py").is_file():
            return resolved

    rendered = ", ".join(str(path) for path in candidates if path is not None)
    if configured_root is not None:
        raise RuntimeError(
            "VLLM_SOURCE_ROOT does not contain the target vllm package: "
            + rendered
        )
    raise RuntimeError(
        "no current vLLM source/install was found; checked: " + rendered
    )
