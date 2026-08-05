# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Build full, nightly, or due-quarantine HCU matrices from versioned config."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path
from typing import Any, Sequence

from select_hcu_tests import (
    DEFAULT_CONFIG,
    _load_config,
    expand_job_partitions,
    validate_config,
)
from hcu_ci_register import matrix_github_outputs


REPOSITORY = Path(__file__).resolve().parents[3]
DEFAULT_QUARANTINE = (
    REPOSITORY
    / ".github"
    / "workflows"
    / "configs"
    / "hcu-quarantine.json"
)
QUARANTINE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")


class MatrixError(ValueError):
    """Raised when a versioned CI profile cannot safely drive hardware."""


def _parse_date(value: Any, field: str, entry_id: str) -> dt.date:
    if not isinstance(value, str):
        raise MatrixError(f"quarantine {entry_id!r} must declare {field}")
    try:
        return dt.date.fromisoformat(value)
    except ValueError as exc:
        raise MatrixError(
            f"quarantine {entry_id!r} has invalid {field}: {value!r}"
        ) from exc


def _profile_entry(
    raw: Any,
    jobs: dict[str, dict[str, Any]],
    *,
    id_prefix: str,
) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise MatrixError("matrix profile entries must be mappings")
    job_id = raw.get("job")
    if not isinstance(job_id, str) or job_id not in jobs:
        raise MatrixError(f"matrix profile references unknown job: {job_id!r}")
    allowed = {"job", "id", "timeout_minutes", "pytest_args", "partitions"}
    unknown = sorted(set(raw).difference(allowed))
    if unknown:
        raise MatrixError(
            f"matrix profile entry {job_id!r} has unsupported overrides: {unknown}"
        )
    selected = dict(jobs[job_id])
    selected["id"] = raw.get("id", f"{id_prefix}{job_id}")
    selected["registry_job"] = job_id
    if not isinstance(selected["id"], str) or not QUARANTINE_ID_RE.fullmatch(
        selected["id"]
    ):
        raise MatrixError(f"matrix profile has invalid id: {selected['id']!r}")
    if "timeout_minutes" in raw:
        timeout = raw["timeout_minutes"]
        if (
            not isinstance(timeout, int)
            or isinstance(timeout, bool)
            or not 1 <= timeout <= 360
        ):
            raise MatrixError(f"matrix profile {selected['id']!r} has invalid timeout")
        selected["timeout_minutes"] = timeout
    if "pytest_args" in raw:
        pytest_args = raw["pytest_args"]
        if not isinstance(pytest_args, list) or not all(
            isinstance(item, str) for item in pytest_args
        ):
            raise MatrixError(
                f"matrix profile {selected['id']!r} pytest_args must be strings"
            )
        selected["pytest_args"] = pytest_args
    if "partitions" in raw:
        partitions = raw["partitions"]
        if (
            not isinstance(partitions, int)
            or isinstance(partitions, bool)
            or not 1 <= partitions <= 32
        ):
            raise MatrixError(
                f"matrix profile {selected['id']!r} has invalid partitions"
            )
        selected["partitions"] = partitions
    return selected


def _load_quarantine(path: Path) -> list[dict[str, Any]]:
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MatrixError(f"cannot load quarantine config {path}: {exc}") from exc
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise MatrixError("quarantine config must be a schema_version=1 mapping")
    entries = config.get("entries")
    if not isinstance(entries, list):
        raise MatrixError("quarantine entries must be a list")
    return entries


def build_matrix(
    config: dict[str, Any],
    *,
    profile: str,
    quarantine_path: Path = DEFAULT_QUARANTINE,
    today: dt.date | None = None,
) -> list[dict[str, Any]]:
    jobs = validate_config(config)
    if profile == "full":
        return expand_job_partitions(list(jobs.values()))
    if profile == "nightly":
        entries = config.get("nightly_jobs")
        if not isinstance(entries, list) or not entries:
            raise MatrixError("nightly_jobs must be a non-empty list")
        return expand_job_partitions([
            _profile_entry(item, jobs, id_prefix="nightly-")
            for item in entries
        ])
    if profile != "quarantine":
        raise MatrixError(f"unsupported matrix profile: {profile}")

    current = today or dt.datetime.now(dt.timezone.utc).date()
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    quarantined_files: set[str] = set()
    for raw in _load_quarantine(quarantine_path):
        if not isinstance(raw, dict):
            raise MatrixError("quarantine entries must be mappings")
        entry_id = raw.get("id")
        if not isinstance(entry_id, str) or not QUARANTINE_ID_RE.fullmatch(entry_id):
            raise MatrixError(f"invalid quarantine id: {entry_id!r}")
        if entry_id in seen:
            raise MatrixError(f"duplicate quarantine id: {entry_id}")
        seen.add(entry_id)
        for field in ("owner", "reason", "issue"):
            if not isinstance(raw.get(field), str) or not raw[field].strip():
                raise MatrixError(
                    f"quarantine {entry_id!r} must declare a non-empty {field}"
                )
        nodeid = raw.get("nodeid")
        if not isinstance(nodeid, str) or "::" not in nodeid:
            raise MatrixError(
                f"quarantine {entry_id!r} must declare a pytest nodeid"
            )
        test_file = nodeid.split("::", 1)[0]
        if not test_file.startswith("tests/") or not (REPOSITORY / test_file).is_file():
            raise MatrixError(
                f"quarantine {entry_id!r} references missing test file {test_file!r}"
            )
        quarantined_files.add(test_file)
        pytest_args = raw.get("pytest_args")
        if not isinstance(pytest_args, list) or not pytest_args:
            raise MatrixError(
                f"quarantine {entry_id!r} must declare focused pytest_args"
            )
        retest_after = _parse_date(raw.get("retest_after"), "retest_after", entry_id)
        expires = _parse_date(raw.get("expires"), "expires", entry_id)
        if expires < retest_after:
            raise MatrixError(
                f"quarantine {entry_id!r} expires before its retest date"
            )
        if current > expires:
            raise MatrixError(
                f"quarantine {entry_id!r} expired on {expires}; remove it or renew it"
            )
        if retest_after <= current:
            profile_entry = {
                key: raw[key]
                for key in ("job", "timeout_minutes", "pytest_args")
                if key in raw
            }
            profile_entry["id"] = f"quarantine-{entry_id}"
            selected.append(
                _profile_entry(profile_entry, jobs, id_prefix="quarantine-")
            )
    xfail_files: set[str] = set()
    for path in (REPOSITORY / "tests").rglob("test_*.py"):
        source = path.read_text(encoding="utf-8")
        if "pytest.mark.xfail" in source or "pytest.xfail(" in source:
            xfail_files.add(path.relative_to(REPOSITORY).as_posix())
    untracked_xfails = sorted(xfail_files.difference(quarantined_files))
    if untracked_xfails:
        raise MatrixError(
            "xfail tests require hcu-quarantine.json entries: "
            f"{untracked_xfails}"
        )
    return expand_job_partitions(selected)


def _write_github_output(path: Path, matrix: list[dict[str, Any]]) -> None:
    values = matrix_github_outputs(matrix)
    with path.open("a", encoding="utf-8") as stream:
        for name, value in values.items():
            stream.write(f"{name}={value}\n")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        required=True,
        choices=("full", "nightly", "quarantine"),
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--quarantine", type=Path, default=DEFAULT_QUARANTINE)
    parser.add_argument("--today", type=dt.date.fromisoformat)
    parser.add_argument("--github-output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        matrix = build_matrix(
            _load_config(args.config),
            profile=args.profile,
            quarantine_path=args.quarantine,
            today=args.today,
        )
        print(json.dumps(matrix, indent=2, sort_keys=True))
        if args.github_output is not None:
            _write_github_output(args.github_output, matrix)
    except (MatrixError, OSError, ValueError) as exc:
        print(f"HCU CI matrix build failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
