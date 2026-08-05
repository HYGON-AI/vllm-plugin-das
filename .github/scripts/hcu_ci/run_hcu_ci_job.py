# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Run one selected HCU CI job with fail-closed report semantics."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

REPOSITORY = Path(__file__).resolve().parents[3]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from hcu_ci_preflight import (
    DEFAULT_ENVIRONMENT_LOCK,
    PreflightError,
    run_preflight,
)
from hcu_ci_register import (
    RegistrationError,
    parse_registry,
    partition_registrations,
    registrations_for_job,
)


def _required_environment(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise PreflightError(f"required environment variable is unset: {name}")
    return value


def _json_list(name: str) -> list[Any]:
    raw = os.environ.get(name, "[]")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise PreflightError(f"{name} is invalid JSON: {exc}") from exc
    if not isinstance(value, list):
        raise PreflightError(f"{name} must contain a JSON list")
    return value


def _junit_counts(path: Path) -> tuple[int, int, int]:
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as exc:
        raise PreflightError(f"cannot parse JUnit report {path}: {exc}") from exc
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    tests = sum(int(suite.attrib.get("tests", "0")) for suite in suites)
    skipped = sum(int(suite.attrib.get("skipped", "0")) for suite in suites)
    failures = sum(
        int(suite.attrib.get("failures", "0"))
        + int(suite.attrib.get("errors", "0"))
        for suite in suites
    )
    return tests, skipped, failures


def _tested_git_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode or not result.stdout.strip():
        raise PreflightError(
            f"cannot resolve tested git SHA: {result.stderr.strip()}"
        )
    return result.stdout.strip()


def main() -> int:
    try:
        job_id = _required_environment("HCU_CI_JOB_ID")
        arch = _required_environment("HCU_CI_ARCH")
        suite = _required_environment("HCU_CI_SUITE")
        registry_job = os.environ.get("HCU_CI_REGISTRY_JOB", job_id)
        partition_id = int(os.environ.get("HCU_CI_PARTITION_ID", "0"))
        partition_size = int(os.environ.get("HCU_CI_PARTITION_SIZE", "1"))
        cards = int(_required_environment("HCU_CI_CARDS"))
        job_root = Path(_required_environment("HCU_CI_JOB_ROOT")).resolve()
        pytest_args = _json_list("HCU_CI_PYTEST_ARGS_JSON")
        if not all(isinstance(item, str) for item in pytest_args):
            raise PreflightError("HCU_CI_PYTEST_ARGS_JSON must contain strings")
        requirements = _json_list("HCU_CI_REQUIREMENTS_JSON")
        if not all(isinstance(item, dict) for item in requirements):
            raise PreflightError(
                "HCU_CI_REQUIREMENTS_JSON must contain mappings"
            )
        registered = registrations_for_job(parse_registry(), registry_job)
        selected_registrations = partition_registrations(
            registered,
            partition_id,
            partition_size,
        )
        if not selected_registrations:
            raise PreflightError(
                f"registry job {registry_job!r} partition {partition_id}/"
                f"{partition_size} selected no tests"
            )
        registered_targets = [item.target for item in selected_registrations]
        environment_lock = Path(
            os.environ.get(
                "HCU_CI_ENVIRONMENT_LOCK",
                str(DEFAULT_ENVIRONMENT_LOCK),
            )
        ).resolve()
        tested_git_sha = _tested_git_sha()
        job_root.mkdir(parents=True, exist_ok=True)
        integration_dir = job_root / "integration"
        evalscope_dir = job_root / "evalscope"
        integration_dir.mkdir(parents=True, exist_ok=True)
        evalscope_dir.mkdir(parents=True, exist_ok=True)
        os.environ["VLLM_HCU_INTEGRATION_LOG_DIR"] = str(integration_dir)
        os.environ["VLLM_HCU_EVAL_WORK_DIR"] = str(evalscope_dir)

        request = {
            "job_id": job_id,
            "arch": arch,
            "cards": cards,
            "suite": suite,
            "pytest_args": pytest_args,
            "requirements": requirements,
            "registry_job": registry_job,
            "partition_id": partition_id,
            "partition_size": partition_size,
            "registered_targets": registered_targets,
            "git_sha": tested_git_sha,
            "runner_name": os.environ.get("RUNNER_NAME"),
        }
        (job_root / "request.json").write_text(
            json.dumps(request, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        try:
            preflight = run_preflight(
                expected_arch=arch,
                required_cards=cards,
                requirements=requirements,
                environment_lock=environment_lock,
            )
        except PreflightError as exc:
            (job_root / "preflight-error.txt").write_text(
                str(exc) + "\n",
                encoding="utf-8",
            )
            raise
        preflight.update(
            {
                "job_id": job_id,
                "suite": suite,
                "pytest_args": pytest_args,
                "git_sha": tested_git_sha,
                "runner_name": os.environ.get("RUNNER_NAME"),
            }
        )
        (job_root / "environment.json").write_text(
            json.dumps(preflight, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        junit_path = job_root / "pytest.xml"
        command = [
            sys.executable,
            "tools/run_patch_tests.py",
            "--suite",
            suite,
            *[
                argument
                for target in registered_targets
                for argument in ("--target", target)
            ],
            "--",
            *pytest_args,
            "-rsxX",
            f"--junitxml={junit_path}",
        ]
        command_text = " ".join(command)
        (job_root / "command.txt").write_text(
            command_text + "\n",
            encoding="utf-8",
        )
        print(f"HCU CI job={job_id}")
        print(f"artifacts={job_root}")
        print(f"command: {command_text}", flush=True)
        with (job_root / "pytest.log").open("wb") as log:
            process = subprocess.Popen(
                command,
                cwd=REPOSITORY,
                env=os.environ.copy(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            assert process.stdout is not None
            for chunk in iter(process.stdout.readline, b""):
                log.write(chunk)
                log.flush()
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
            returncode = process.wait()

        if not junit_path.is_file():
            raise PreflightError(
                f"pytest did not produce the required JUnit report: {junit_path}"
            )
        tests, skipped, failures = _junit_counts(junit_path)
        print(
            f"JUnit summary: tests={tests}, skipped={skipped}, "
            f"failures_or_errors={failures}"
        )
        if returncode:
            return returncode
        if tests < 1:
            raise PreflightError("selected HCU job collected no tests")
        if skipped:
            raise PreflightError(
                f"selected HCU job contains {skipped} skipped/xfail test(s)"
            )
        if failures:
            raise PreflightError(
                f"JUnit reports {failures} failure/error test(s)"
            )
    except (OSError, ValueError, PreflightError, RegistrationError) as exc:
        print(f"HCU CI job failed closed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
