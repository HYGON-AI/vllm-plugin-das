# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Select HCU CI jobs from changed repository paths.

The configuration file uses JSON syntax, which is also valid YAML. Keeping the
parser in the Python standard library makes selection runnable on a minimal
control-plane runner before the HCU test environment is initialized.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

from hcu_ci_register import (
    matrix_github_outputs,
    parse_registry,
    partition_registrations,
    registrations_for_job,
)


REPOSITORY = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPOSITORY / ".github/workflows/configs/hcu-test-map.yaml"
VALID_SUITES = {
    "accuracy-hcu",
    "contract-hcu",
    "integration-smoke",
    "model",
    "distributed-single-node",
    "distributed-multi-node",
    "distributed",
    "stress",
    "nightly",
    "full",
}
RUNNER_RE = re.compile(
    r"^(?:bw18|nmz36|hcu-ci-pr|hcu-linux-(?:cpu|gfx9[0-9]{2}-[1-9][0-9]*))$"
)
JOB_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")


class SelectionError(ValueError):
    """Raised when a selector configuration cannot safely drive CI."""


def _matches(path: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch.fnmatchcase(path, pattern) for pattern in patterns)


def _load_config(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as stream:
            config = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise SelectionError(f"cannot load selector config {path}: {exc}") from exc
    if not isinstance(config, dict) or config.get("version") != 1:
        raise SelectionError("selector config must be a version=1 mapping")
    for key in ("jobs", "groups", "fallback_jobs", "accuracy_jobs"):
        if key not in config:
            raise SelectionError(f"selector config is missing {key!r}")
    return config


def _validated_job(job_id: str, raw: Any) -> dict[str, Any]:
    if not JOB_ID_RE.fullmatch(job_id):
        raise SelectionError(f"invalid job id: {job_id!r}")
    if not isinstance(raw, dict):
        raise SelectionError(f"job {job_id!r} must be a mapping")
    required = {
        "runner",
        "arch",
        "cards",
        "suite",
        "pytest_args",
        "timeout_minutes",
        "requirements",
    }
    missing = sorted(required.difference(raw))
    if missing:
        raise SelectionError(f"job {job_id!r} is missing {missing}")
    runner = raw["runner"]
    arch = raw["arch"]
    cards = raw["cards"]
    suite = raw["suite"]
    timeout = raw["timeout_minutes"]
    pytest_args = raw["pytest_args"]
    requirements = raw["requirements"]
    partitions = raw.get("partitions", 1)
    if not isinstance(runner, str) or not RUNNER_RE.fullmatch(runner):
        raise SelectionError(f"job {job_id!r} has invalid runner {runner!r}")
    if arch not in {"gfx936", "gfx938"}:
        raise SelectionError(f"job {job_id!r} has invalid arch {arch!r}")
    if not isinstance(cards, int) or isinstance(cards, bool) or cards < 1:
        raise SelectionError(f"job {job_id!r} has invalid card count {cards!r}")
    if suite not in VALID_SUITES:
        raise SelectionError(f"job {job_id!r} has invalid suite {suite!r}")
    if (
        not isinstance(timeout, int)
        or isinstance(timeout, bool)
        or not 1 <= timeout <= 360
    ):
        raise SelectionError(f"job {job_id!r} has invalid timeout {timeout!r}")
    if not isinstance(pytest_args, list) or not all(
        isinstance(item, str) for item in pytest_args
    ):
        raise SelectionError(f"job {job_id!r} pytest_args must be strings")
    if not isinstance(requirements, list) or not all(
        isinstance(item, dict) for item in requirements
    ):
        raise SelectionError(f"job {job_id!r} requirements must be mappings")
    if (
        not isinstance(partitions, int)
        or isinstance(partitions, bool)
        or not 1 <= partitions <= 32
    ):
        raise SelectionError(f"job {job_id!r} has invalid partitions {partitions!r}")
    return {
        "id": job_id,
        "runner": runner,
        "arch": arch,
        "cards": cards,
        "suite": suite,
        "pytest_args": pytest_args,
        "timeout_minutes": timeout,
        "requirements": requirements,
        "partitions": partitions,
    }


def expand_job_partitions(
    selected_jobs: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Expand jobs into deterministic LPT partitions from static registration."""

    registrations = parse_registry()
    all_job_ids = {job.get("registry_job", job["id"]) for job in selected_jobs}
    # Full one-to-one validation is done by the static gate. Selection may be
    # operating on a subset, so validate only that selected jobs are registered.
    registered_ids = {item.job for item in registrations}
    missing = sorted(all_job_ids.difference(registered_ids))
    if missing:
        raise SelectionError(f"selected HCU jobs have no registration: {missing}")

    expanded: list[dict[str, Any]] = []
    for job in selected_jobs:
        display_id = job["id"]
        registry_job = job.get("registry_job", job["id"])
        enabled = registrations_for_job(registrations, registry_job)
        if not enabled:
            continue
        partition_size = job.get("partitions", 1)
        if partition_size > len(enabled):
            raise SelectionError(
                f"job {registry_job!r} requests {partition_size} partitions for "
                f"only {len(enabled)} enabled registration(s)"
            )
        for partition_id in range(partition_size):
            assigned = partition_registrations(
                enabled,
                partition_id,
                partition_size,
            )
            item = dict(job)
            item["registry_job"] = registry_job
            item["partition_id"] = partition_id
            item["partition_size"] = partition_size
            item["estimated_seconds"] = int(
                sum(registration.est_time for registration in assigned)
            )
            if partition_size > 1:
                item["id"] = (
                    f"{display_id}-p{partition_id + 1}of{partition_size}"
                )
            expanded.append(item)
    return expanded


def validate_config(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw_jobs = config["jobs"]
    raw_groups = config["groups"]
    if not isinstance(raw_jobs, dict) or not isinstance(raw_groups, dict):
        raise SelectionError("jobs and groups must be mappings")
    for key in ("fallback_jobs", "accuracy_jobs"):
        value = config[key]
        if not isinstance(value, list) or not all(
            isinstance(item, str) for item in value
        ):
            raise SelectionError(f"{key} must be a list of job ids")
    jobs = {
        job_id: _validated_job(job_id, raw)
        for job_id, raw in raw_jobs.items()
    }
    referenced: set[str] = set(config["fallback_jobs"])
    referenced.update(config["accuracy_jobs"])
    for group_name, group in raw_groups.items():
        if not isinstance(group, dict):
            raise SelectionError(f"group {group_name!r} must be a mapping")
        patterns = group.get("patterns")
        group_jobs = group.get("jobs")
        if not isinstance(patterns, list) or not all(
            isinstance(item, str) and item for item in patterns
        ):
            raise SelectionError(f"group {group_name!r} patterns are invalid")
        if not isinstance(group_jobs, list) or not all(
            isinstance(item, str) for item in group_jobs
        ):
            raise SelectionError(f"group {group_name!r} jobs are invalid")
        referenced.update(group_jobs)
    unknown = sorted(referenced.difference(jobs))
    if unknown:
        raise SelectionError(f"selector references unknown jobs: {unknown}")
    return jobs


def select_jobs(
    config: dict[str, Any],
    changed_paths: Sequence[str],
    *,
    full: bool = False,
    accuracy: bool = False,
) -> tuple[list[dict[str, Any]], list[str], bool]:
    jobs = validate_config(config)
    normalized = sorted(
        {
            path.strip().replace(os.sep, "/").removeprefix("./")
            for path in changed_paths
            if path.strip()
        }
    )
    if full:
        return expand_job_partitions(list(jobs.values())), ["full-hcu"], False
    if not normalized:
        return [], [], False

    non_hcu = config.get("non_hcu_patterns", [])
    docs_only = all(_matches(path, non_hcu) for path in normalized)
    if docs_only and not accuracy:
        return [], ["docs-only"], False

    selected_ids: set[str] = set()
    classified_paths: set[str] = set()
    selected_groups: list[str] = ["docs-only"] if docs_only else []
    for group_name, group in config["groups"].items():
        matched_paths = {
            path for path in normalized if _matches(path, group["patterns"])
        }
        if matched_paths:
            selected_groups.append(group_name)
            selected_ids.update(group["jobs"])
            classified_paths.update(matched_paths)

    production_patterns = config.get("production_patterns", [])
    test_patterns = config.get("test_patterns", [])
    fallback = any(
        path not in classified_paths
        and not _matches(path, non_hcu)
        and (
            _matches(path, production_patterns)
            or _matches(path, test_patterns)
        )
        for path in normalized
    )
    if fallback:
        selected_groups.append("conservative-fallback")
        selected_ids.update(config["fallback_jobs"])
    if accuracy:
        selected_groups.append("accuracy-hcu")
        selected_ids.update(config["accuracy_jobs"])

    ordered = expand_job_partitions(
        [job for job_id, job in jobs.items() if job_id in selected_ids]
    )
    return ordered, selected_groups, fallback


def _git_changed_paths(base: str, head: str) -> list[str]:
    result = subprocess.run(
        ["git", "diff", "--name-only", "--diff-filter=ACMRTUXB", base, head],
        cwd=REPOSITORY,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode:
        raise SelectionError(
            f"git diff failed for {base}..{head}: {result.stderr.strip()}"
        )
    return result.stdout.splitlines()


def _write_github_outputs(path: Path, payload: dict[str, Any]) -> None:
    values = matrix_github_outputs(payload["jobs"])
    values.update(
        {
            "groups": ",".join(payload["groups"]),
            "docs_only": str(payload["docs_only"]).lower(),
            "fallback": str(payload["fallback"]).lower(),
        }
    )
    with path.open("a", encoding="utf-8") as stream:
        for name, value in values.items():
            stream.write(f"{name}={value}\n")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--base")
    parser.add_argument("--head")
    parser.add_argument(
        "--paths-from",
        type=Path,
        help="read one changed path per line; use '-' for stdin",
    )
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--accuracy", action="store_true")
    parser.add_argument("--github-output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = _load_config(args.config)
        if args.paths_from is not None:
            if str(args.paths_from) == "-":
                changed_paths = sys.stdin.read().splitlines()
            else:
                changed_paths = args.paths_from.read_text(encoding="utf-8").splitlines()
        elif args.base and args.head:
            changed_paths = _git_changed_paths(args.base, args.head)
        else:
            raise SelectionError(
                "provide --base and --head, or provide --paths-from"
            )
        jobs, groups, fallback = select_jobs(
            config,
            changed_paths,
            full=args.full,
            accuracy=args.accuracy,
        )
        payload = {
            "jobs": jobs,
            "groups": groups,
            "docs_only": groups == ["docs-only"],
            "fallback": fallback,
            "changed_paths": sorted(changed_paths),
        }
        rendered = json.dumps(payload, indent=2, sort_keys=True)
        print(rendered)
        if args.github_output is not None:
            _write_github_outputs(args.github_output, payload)
    except (OSError, SelectionError) as exc:
        print(f"HCU CI selection failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
