from __future__ import annotations

import os
import subprocess
from fnmatch import fnmatch
from pathlib import Path

from packaging.requirements import Requirement


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
INSTALLER = (
    REPOSITORY_ROOT / "scripts/ci/hcu/install_vllm_runtime_dependencies.sh"
)
REQUIREMENTS = REPOSITORY_ROOT / "docker/requirements-vllm-runtime.txt"
TEST_REQUIREMENTS = REPOSITORY_ROOT / "requirements-test.txt"
DOCKERFILE = REPOSITORY_ROOT / "docker/Dockerfile"
DOCKERIGNORE = REPOSITORY_ROOT / ".dockerignore"


def _recording_python(tmp_path: Path) -> tuple[Path, Path]:
    executable = tmp_path / "record-python"
    argv_log = tmp_path / "argv"
    executable.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\0' \"$@\" > \"$RECORDED_ARGV\"\n"
        "exit \"${FAKE_PIP_EXIT:-0}\"\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return executable, argv_log


def _run_installer(
    tmp_path: Path,
    *arguments: str,
    pip_exit: int = 0,
) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    python, argv_log = _recording_python(tmp_path)
    env = os.environ.copy()
    env.update(
        {
            "PYTHON": str(python),
            "RECORDED_ARGV": str(argv_log),
            "FAKE_PIP_EXIT": str(pip_exit),
        }
    )
    result = subprocess.run(
        ["bash", str(INSTALLER), *arguments],
        cwd=REPOSITORY_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    argv = []
    if argv_log.exists():
        argv = argv_log.read_bytes().rstrip(b"\0").decode().split("\0")
    return result, argv


def test_installer_uses_active_pip_configuration(tmp_path: Path) -> None:
    supplied_requirements = tmp_path / "runtime requirements.txt"
    supplied_requirements.touch()

    result, argv = _run_installer(tmp_path, str(supplied_requirements))

    assert result.returncode == 0, result.stderr
    assert argv == [
        "-m",
        "pip",
        "install",
        "--no-cache-dir",
        "--requirement",
        str(supplied_requirements),
    ]
    assert "--index-url" not in argv
    assert "--extra-index-url" not in argv


def test_installer_defaults_to_repository_runtime_requirements(
    tmp_path: Path,
) -> None:
    result, argv = _run_installer(tmp_path)

    assert result.returncode == 0, result.stderr
    assert argv[-2:] == ["--requirement", str(REQUIREMENTS)]


def test_installer_propagates_pip_failure(tmp_path: Path) -> None:
    supplied_requirements = tmp_path / "runtime.txt"
    supplied_requirements.touch()

    result, _ = _run_installer(
        tmp_path,
        str(supplied_requirements),
        pip_exit=37,
    )

    assert result.returncode == 37


def test_runtime_requirements_pin_ray_extras_and_pandas() -> None:
    requirements = {
        requirement.name: requirement
        for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines()
        if (stripped := line.strip()) and not stripped.startswith("#")
        for requirement in [Requirement(stripped)]
    }

    ray = requirements["ray"]
    assert ray.extras == {"data", "serve", "train", "tune"}
    assert str(ray.specifier) == "==2.58.0"
    assert str(requirements["pandas"].specifier) == "==2.3.3"


def test_test_requirements_pin_ci_runner_and_official_evalscope() -> None:
    requirements = {
        requirement.name: requirement
        for line in TEST_REQUIREMENTS.read_text(encoding="utf-8").splitlines()
        if (stripped := line.strip()) and not stripped.startswith("#")
        for requirement in [Requirement(stripped)]
    }

    assert str(requirements["pytest"].specifier) == "==9.1.1"
    assert str(requirements["evalscope"].specifier) == "==1.9.1"


def _dockerfile_instructions() -> list[str]:
    instructions: list[str] = []
    current: list[str] = []
    for raw_line in DOCKERFILE.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not current and (not line or line.startswith("#")):
            continue
        current.append(line.removesuffix("\\").strip())
        if not line.endswith("\\"):
            instructions.append(" ".join(current))
            current = []
    assert not current, "Dockerfile ends inside a continued instruction"
    return instructions


def test_dockerfile_copies_runtime_installer_before_framework_layer() -> None:
    instructions = _dockerfile_instructions()
    requirements_copy = next(
        index
        for index, instruction in enumerate(instructions)
        if instruction.startswith("COPY ")
        and "requirements-vllm-runtime.txt" in instruction
    )
    installer_copy = next(
        index
        for index, instruction in enumerate(instructions)
        if instruction.startswith("COPY ")
        and "install_vllm_runtime_dependencies.sh" in instruction
    )
    framework_run = next(
        index
        for index, instruction in enumerate(instructions)
        if instruction.startswith("RUN ")
        and "pip install --no-cache-dir ninja wheel setuptools" in instruction
    )

    assert requirements_copy < framework_run
    assert installer_copy < framework_run
    assert "/tmp/install_vllm_runtime_dependencies.sh" in instructions[framework_run]
    assert "/tmp/requirements-vllm-runtime.txt" in instructions[framework_run]


def test_dockerfile_installs_registered_test_dependencies() -> None:
    instructions = _dockerfile_instructions()
    requirements_copy = next(
        index
        for index, instruction in enumerate(instructions)
        if instruction.startswith("COPY ") and "requirements-test.txt" in instruction
    )
    framework_run = next(
        index
        for index, instruction in enumerate(instructions)
        if instruction.startswith("RUN ")
        and "pip install --no-cache-dir ninja wheel setuptools" in instruction
    )

    assert requirements_copy < framework_run
    assert "pip install --no-cache-dir -r /tmp/requirements-test.txt" in instructions[
        framework_run
    ]


def test_dockerfile_does_not_override_the_index_for_ray() -> None:
    ray_instructions = [
        instruction
        for instruction in _dockerfile_instructions()
        if "ray[" in instruction.lower()
    ]

    assert not ray_instructions


def _docker_context_excludes(relative_path: str) -> bool:
    excluded = False
    for raw_pattern in DOCKERIGNORE.read_text(encoding="utf-8").splitlines():
        pattern = raw_pattern.strip()
        if not pattern or pattern.startswith("#"):
            continue
        included = pattern.startswith("!")
        if included:
            pattern = pattern[1:]
        pattern = pattern.lstrip("/").rstrip("/")
        if (
            relative_path == pattern
            or relative_path.startswith(f"{pattern}/")
            or fnmatch(relative_path, pattern)
        ):
            excluded = not included
    return excluded


def test_docker_runtime_artifacts_are_available_in_build_context() -> None:
    assert not _docker_context_excludes("docker/requirements-vllm-runtime.txt")
    assert not _docker_context_excludes(
        "scripts/ci/hcu/install_vllm_runtime_dependencies.sh"
    )
