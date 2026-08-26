# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Syntax-check CI-critical Python and Python files changed by a PR."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tokenize
from pathlib import Path
from typing import Iterable, Sequence


REPOSITORY = Path(__file__).resolve().parents[3]
CRITICAL_PATHS = (
    REPOSITORY / ".github" / "scripts" / "hcu_ci",
    REPOSITORY / "tools" / "run_patch_tests.py",
    REPOSITORY / "tests" / "hcu_ci_registry.py",
)


class CompileError(RuntimeError):
    """Raised when changed Python sources cannot be syntax-checked."""


def _relative(path: Path) -> str:
    return path.relative_to(REPOSITORY).as_posix()


def _git_changed_paths(base: str, head: str) -> list[Path]:
    result = subprocess.run(
        ["git", "diff", "--name-only", "--diff-filter=ACMRTUXB", base, head],
        cwd=REPOSITORY,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode:
        raise CompileError(
            f"git diff failed for {base}..{head}: {result.stderr.strip()}"
        )
    paths: list[Path] = []
    for line in result.stdout.splitlines():
        if not line.endswith(".py"):
            continue
        path = (REPOSITORY / line).resolve()
        try:
            path.relative_to(REPOSITORY)
        except ValueError as exc:
            raise CompileError(f"changed path escapes repository: {line}") from exc
        if path.is_file():
            paths.append(path)
    return paths


def _critical_python_files(paths: Iterable[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(path.rglob("*.py")))
        elif path.is_file() and path.suffix == ".py":
            files.append(path)
    return files


def _compile(path: Path) -> None:
    with tokenize.open(path) as stream:
        source = stream.read()
    compile(source, str(path), "exec", dont_inherit=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", help="base commit used to find changed files")
    parser.add_argument("--head", help="head commit used to find changed files")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        files = _critical_python_files(CRITICAL_PATHS)
        if bool(args.base) != bool(args.head):
            raise CompileError("provide both --base and --head, or neither")
        if args.base and args.head:
            files.extend(_git_changed_paths(args.base, args.head))
        unique = sorted({path.resolve() for path in files}, key=_relative)
        for path in unique:
            _compile(path)
    except (OSError, SyntaxError, CompileError) as exc:
        print(f"HCU CI Python syntax check failed: {exc}", file=sys.stderr)
        return 2
    print(f"HCU CI Python syntax check passed: {len(unique)} file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
