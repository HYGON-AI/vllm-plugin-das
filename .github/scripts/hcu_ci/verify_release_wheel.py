# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Verify that an isolated interpreter imports and loads the built wheel."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
from pathlib import Path
from typing import Sequence


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--kernel", action="store_true")
    args = parser.parse_args(argv)

    repository = args.repository.resolve()
    import vllm_hcu

    module_path = Path(vllm_hcu.__file__).resolve()
    if _is_relative_to(module_path, repository):
        print(
            "release validation imported checkout source instead of wheel: "
            f"{module_path}",
            file=sys.stderr,
        )
        return 2

    wheel_version = importlib.metadata.version("vllm-hcu")
    short_sha = args.expected_sha[:7]
    if f".{short_sha}" not in wheel_version:
        print(
            f"wheel version {wheel_version!r} does not contain checkout SHA "
            f"{short_sha}",
            file=sys.stderr,
        )
        return 2
    report: dict[str, object] = {
        "python": sys.version,
        "python_prefix": sys.prefix,
        "module_path": str(module_path),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("vllm-hcu", "vllm", "torch")
        },
        "expected_git_sha": args.expected_sha,
    }
    if args.kernel:
        import torch
        import vllm_hcu.hcu_ops  # noqa: F401

        namespace = getattr(torch.ops, "hcu_ops", None)
        if namespace is None or not hasattr(namespace, "meta_size"):
            print("wheel did not register torch.ops.hcu_ops.meta_size", file=sys.stderr)
            return 2
        meta_size = int(namespace.meta_size())
        if meta_size <= 0:
            print(f"invalid hcu_ops meta_size: {meta_size}", file=sys.stderr)
            return 2
        report["kernel"] = {
            "extension": str(Path(vllm_hcu.hcu_ops.__file__).resolve()),
            "meta_size": meta_size,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
