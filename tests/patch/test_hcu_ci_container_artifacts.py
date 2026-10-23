# SPDX-License-Identifier: Apache-2.0
"""Exercise report export and cleanup without an accelerator or Docker daemon."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[2]
CLEANUP = REPOSITORY / "scripts/ci/hcu/hcu_ci_cleanup_container.sh"

FAKE_DOCKER = r"""#!/usr/bin/env python3
import json
import os
import shutil
import sys
from pathlib import Path

args = sys.argv[1:]
with Path(os.environ["FAKE_DOCKER_LOG"]).open("a") as stream:
    stream.write(json.dumps(args) + "\n")
if args[0] == "inspect":
    sys.exit(0 if os.environ.get("FAKE_PRESENT", "1") == "1" else 1)
if args[0] in ("exec", "stop"):
    sys.exit(0 if os.environ.get("FAKE_RUNNING", "1") == "1" else 1)
if args[0] == "cp":
    if os.environ.get("FAKE_COPY_FAIL") == "1":
        sys.exit(42)
    assert args[1] == "hcu_ci_artifact_test:/hcu-ci-artifacts/."
    shutil.copytree(
        os.environ["FAKE_ARTIFACTS"], args[2], dirs_exist_ok=True
    )
    sys.exit(0)
if args[0] == "rm":
    Path(os.environ["FAKE_REMOVED"]).touch()
    sys.exit(0)
raise SystemExit("unexpected Docker call: " + repr(args))
"""


@pytest.fixture
def artifact_runtime(tmp_path: Path):
    binaries = tmp_path / "bin"
    binaries.mkdir()
    docker = binaries / "docker"
    docker.write_text(FAKE_DOCKER)
    docker.chmod(0o755)
    reports = tmp_path / "container-reports"
    (reports / "nested logs").mkdir(parents=True)
    (reports / "pytest.xml").write_text("<testsuite tests='1'/>")
    (reports / "nested logs" / "server.log").write_text("complete\n")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    destination = tmp_path / "upload staging"
    calls = tmp_path / "docker.jsonl"
    removed = tmp_path / "removed"
    env = dict(
        os.environ,
        PATH=str(binaries) + os.pathsep + os.environ["PATH"],
        HCU_CI_CONTAINER_NAME="hcu_ci_artifact_test",
        HCU_CI_UPLOAD_ROOT=str(destination),
        GITHUB_WORKSPACE=str(workspace),
        FAKE_DOCKER_LOG=str(calls),
        FAKE_ARTIFACTS=str(reports),
        FAKE_REMOVED=str(removed),
    )
    return env, reports, destination, calls, removed


def _run_cleanup(env):
    return subprocess.run(
        ["bash", str(CLEANUP)],
        cwd=REPOSITORY,
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
    )


def _commands(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.mark.parametrize("running", ["0", "1"])
def test_cleanup_exports_reports_before_removing_container(artifact_runtime, running):
    env, _, destination, calls, removed = artifact_runtime
    env["FAKE_RUNNING"] = running

    result = _run_cleanup(env)

    assert result.returncode == 0, result.stdout + result.stderr
    assert (destination / "pytest.xml").read_text() == "<testsuite tests='1'/>"
    assert (destination / "nested logs" / "server.log").read_text() == "complete\n"
    commands = [args[0] for args in _commands(calls)]
    assert commands.index("stop") < commands.index("cp") < commands.index("rm")
    assert removed.exists()


def test_cleanup_preserves_stopped_container_when_export_fails(artifact_runtime):
    env, reports, _, calls, removed = artifact_runtime
    env["FAKE_COPY_FAIL"] = "1"

    result = _run_cleanup(env)

    assert result.returncode != 0
    assert "retaining stopped container hcu_ci_artifact_test" in result.stderr
    assert (reports / "pytest.xml").is_file()
    assert "stop" in [args[0] for args in _commands(calls)]
    assert not removed.exists()


def test_cleanup_without_artifact_destination_still_removes_container(artifact_runtime):
    env, _, destination, calls, removed = artifact_runtime
    env.pop("HCU_CI_UPLOAD_ROOT")

    result = _run_cleanup(env)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "cp" not in [args[0] for args in _commands(calls)]
    assert not destination.exists()
    assert removed.exists()


def test_cleanup_does_not_export_when_start_never_created_container(artifact_runtime):
    env, _, destination, calls, _ = artifact_runtime
    env["FAKE_PRESENT"] = "0"

    result = _run_cleanup(env)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "cp" not in [args[0] for args in _commands(calls)]
    assert not destination.exists()

def _artifact_step(name: str):
    import yaml

    workflow = yaml.safe_load(
        (REPOSITORY / ".github/workflows/_selected-hcu-tests.yml").read_text()
    )
    return next(step for step in workflow["jobs"]["hcu"]["steps"] if step["name"] == name)


def test_upload_staging_is_temporary_and_removed_after_upload(tmp_path: Path):
    temporary = tmp_path / "runner temporary"
    temporary.mkdir()
    env_file, outputs = tmp_path / "env", tmp_path / "outputs"
    env = dict(os.environ, RUNNER_TEMP=str(temporary), GITHUB_ENV=str(env_file), GITHUB_OUTPUT=str(outputs))
    prepare = _artifact_step("Prepare HCU artifact upload")
    subprocess.run(["bash", "-c", prepare["run"]], env=env, check=True)
    values = dict(line.split("=", 1) for line in env_file.read_text().splitlines())
    staging = Path(values["HCU_CI_UPLOAD_ROOT"])
    assert staging.parent == temporary
    assert dict(line.split("=", 1) for line in outputs.read_text().splitlines())["path"] == str(staging)
    (staging / "pytest.xml").write_text("report")
    neighbor = temporary / "other-job"
    neighbor.mkdir()
    (neighbor / "keep").write_text("keep")

    remove = _artifact_step("Remove uploaded HCU temporary reports")
    subprocess.run(["bash", "-c", remove["run"]], env=dict(env, **values), check=True)

    assert not staging.exists()
    assert (neighbor / "keep").read_text() == "keep"


def test_upload_cleanup_rejects_an_unrelated_directory(tmp_path: Path):
    unrelated = tmp_path / "other-job"
    unrelated.mkdir()
    (unrelated / "keep").write_text("keep")
    env = dict(os.environ, RUNNER_TEMP=str(tmp_path), HCU_CI_UPLOAD_ROOT=str(unrelated))

    remove = _artifact_step("Remove uploaded HCU temporary reports")
    result = subprocess.run(["bash", "-c", remove["run"]], env=env, capture_output=True, text=True)

    assert result.returncode != 0
    assert "Refusing to remove" in result.stderr
    assert (unrelated / "keep").read_text() == "keep"
