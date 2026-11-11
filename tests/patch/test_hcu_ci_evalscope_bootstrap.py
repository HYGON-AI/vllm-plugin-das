# SPDX-License-Identifier: Apache-2.0
"""Exercise job-scoped EvalScope setup without package downloads."""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "ensure_evalscope", REPOSITORY / ".github/scripts/hcu_ci/ensure_evalscope.py"
)
bootstrap = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bootstrap)
REQUIREMENT = {"kind": "distribution", "name": "evalscope", "match": "exact", "version": "1.9.1"}


@pytest.fixture
def installed(monkeypatch):
    versions = {"torch": "2.11.0+vendor", "vllm": "0.28.1", "numpy": "2.1.0"}

    def version(name):
        if name not in versions:
            raise bootstrap.metadata.PackageNotFoundError(name)
        return versions[name]

    monkeypatch.setattr(bootstrap.metadata, "version", version)
    return versions


def test_non_evaluation_job_does_not_install(monkeypatch, installed):
    monkeypatch.setattr(bootstrap.subprocess, "run", lambda *a, **k: pytest.fail("unexpected pip"))
    assert bootstrap.ensure_evalscope([{"kind": "model", "relative": "example"}]) is False


def test_matching_evalscope_reuses_existing_install(monkeypatch, installed):
    installed["evalscope"] = "1.9.1"
    monkeypatch.setattr(bootstrap.subprocess, "run", lambda *a, **k: pytest.fail("unexpected pip"))
    assert bootstrap.ensure_evalscope([REQUIREMENT]) is False


@pytest.mark.parametrize("existing", [None, "1.8.0"])
def test_missing_or_wrong_evalscope_is_installed_with_runtime_constraints(monkeypatch, installed, existing):
    if existing is not None:
        installed["evalscope"] = existing
    calls = []

    def install(command, **kwargs):
        assert command[:3] == [bootstrap.sys.executable, "-m", "pip"]
        assert command[-1] == "evalscope==1.9.1"
        constraints = Path(command[command.index("--constraint") + 1]).read_text()
        assert "torch==2.11.0+vendor\n" in constraints
        assert "vllm==0.28.1\n" in constraints
        assert "numpy==2.1.0\n" in constraints
        assert kwargs["check"] is True
        calls.append(command)
        installed["evalscope"] = "1.9.1"

    monkeypatch.setattr(bootstrap.subprocess, "run", install)
    assert bootstrap.ensure_evalscope([REQUIREMENT]) is True
    assert len(calls) == 1


def test_pip_failure_is_not_hidden(monkeypatch, installed):
    def fail(command, **kwargs):
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(bootstrap.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        bootstrap.ensure_evalscope([REQUIREMENT])


def test_runtime_drift_is_rejected(monkeypatch, installed):
    def install(command, **kwargs):
        installed["evalscope"] = "1.9.1"
        installed["torch"] = "2.12.0"

    monkeypatch.setattr(bootstrap.subprocess, "run", install)
    with pytest.raises(RuntimeError, match="protected runtime versions"):
        bootstrap.ensure_evalscope([REQUIREMENT])


def test_install_that_does_not_provide_requested_distribution_is_rejected(monkeypatch, installed):
    monkeypatch.setattr(bootstrap.subprocess, "run", lambda *a, **k: None)
    with pytest.raises(RuntimeError, match="did not install"):
        bootstrap.ensure_evalscope([REQUIREMENT])
