# SPDX-License-Identifier: Apache-2.0
"""Check ephemeral CI installation and preservation of the framework runtime."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / ".github/scripts/hcu_ci"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "ensure_ci_dependencies", SCRIPTS / "ensure_ci_dependencies.py"
)
bootstrap = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bootstrap)


@pytest.fixture
def installed(monkeypatch):
    versions = {
        "torch": "2.11.0+vendor", "vllm": "0.28.1", "numpy": "2.0.2",
        "pandas": "3.0.6", "pytest": "8.0",
    }
    monkeypatch.setattr(bootstrap, "_installed", versions.get)
    return versions


def _provide_dependencies(versions):
    versions.update({
        "ray": "2.58.0", "pandas": "2.3.3", "pytest": "9.1.1",
        "lm-format-enforcer": "0.11.3", "outlines_core": "0.2.14",
    })


@pytest.mark.parametrize("user", [False, True])
def test_install_ci_requirements_without_evalscope_and_preserve_runtime(
    monkeypatch, installed, user,
):
    monkeypatch.setattr(sys, "path", sys.path.copy())
    calls = []

    def install(command, **kwargs):
        assert command[:3] == [sys.executable, "-m", "pip"]
        assert ("--user" in command) is user
        assert "ray[data,serve,train,tune]==2.58.0" in command
        assert "pytest==9.1.1" in command
        assert "lm-format-enforcer==0.11.3" in command
        assert not any("evalscope" in argument for argument in command)
        constraints = Path(command[command.index("--constraint") + 1]).read_text()
        assert "torch==2.11.0+vendor\n" in constraints
        assert "numpy==2.0.2\n" in constraints
        assert "pandas==" not in constraints
        assert "pytest==" not in constraints
        assert kwargs["check"] is True
        _provide_dependencies(installed)
        calls.append(command)

    monkeypatch.setattr(bootstrap.subprocess, "run", install)
    bootstrap.ensure_ci_dependencies(user=user)
    assert len(calls) == 1


def test_ci_dependency_install_rejects_runtime_drift(monkeypatch, installed):
    def install(*args, **kwargs):
        _provide_dependencies(installed)
        installed["torch"] = "2.12.0"

    monkeypatch.setattr(bootstrap.subprocess, "run", install)
    with pytest.raises(RuntimeError, match="protected runtime drift"):
        bootstrap.ensure_ci_dependencies()


def test_ci_dependency_install_rejects_missing_requirements(monkeypatch, installed):
    monkeypatch.setattr(bootstrap.subprocess, "run", lambda *a, **k: None)
    with pytest.raises(RuntimeError, match="missing="):
        bootstrap.ensure_ci_dependencies()


def test_ci_dependency_install_propagates_pip_error(monkeypatch, installed):
    def fail(command, **kwargs):
        raise subprocess.CalledProcessError(23, command)

    monkeypatch.setattr(bootstrap.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        bootstrap.ensure_ci_dependencies()
