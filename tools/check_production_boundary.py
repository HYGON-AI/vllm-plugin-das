#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fail-closed checks for the vLLM-HCU production package boundary."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence


_FORBIDDEN_CONTENT = (
    (
        "legacy_segment_id",
        re.compile(r"SP-V\d{3}-\d{4}"),
    ),
    (
        "legacy_audit_field",
        re.compile(
            r"\b(?:SEGMENT_TEST_IDS|INACTIVE_SEGMENTS|"
            r"INACTIVE_SEGMENT_TERMINALS|CORRECTED_SEGMENTS)\b"
        ),
    ),
    (
        "versioned_runtime_marker",
        re.compile(r"_hcu_v\d{3}_"),
    ),
)
_VERSIONED_MODULE = re.compile(r"_v\d{3}\.py$")


@dataclass(frozen=True, slots=True)
class Violation:
    kind: str
    path: str
    line: int | None
    value: str


def scan_production_package(package_root: Path) -> tuple[int, list[Violation]]:
    """Return the number of Python files scanned and all boundary violations."""

    package_root = package_root.resolve()
    violations: list[Violation] = []
    paths = sorted(package_root.rglob("*.py"))
    for path in paths:
        relative = path.relative_to(package_root).as_posix()
        if _VERSIONED_MODULE.search(path.name):
            violations.append(
                Violation("versioned_runtime_module", relative, None, path.name)
            )
        text = path.read_text(encoding="utf-8-sig")
        for line_number, line in enumerate(text.splitlines(), 1):
            for kind, pattern in _FORBIDDEN_CONTENT:
                match = pattern.search(line)
                if match is not None:
                    violations.append(
                        Violation(kind, relative, line_number, match.group(0))
                    )
    return len(paths), violations


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "vllm_hcu",
    )
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    scanned, violations = scan_production_package(args.package_root)
    result = {
        "ok": not violations,
        "package_root": str(args.package_root.resolve()),
        "python_files_scanned": scanned,
        "violations": [asdict(item) for item in violations],
    }
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    elif violations:
        for item in violations:
            location = item.path if item.line is None else f"{item.path}:{item.line}"
            print(f"{item.kind}: {location}: {item.value}")
    else:
        print(f"production boundary clean: {scanned} Python files")
    return 0 if not violations else 1


if __name__ == "__main__":
    raise SystemExit(main())
