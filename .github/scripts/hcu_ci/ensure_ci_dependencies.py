# SPDX-License-Identifier: Apache-2.0
"""Prepare CI dependencies inside the disposable test container."""

from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
import tempfile
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from ensure_evalscope import PROTECTED_DISTRIBUTIONS, _installed


ROOT = Path(__file__).resolve().parents[3]


def ensure_ci_dependencies(*, user: bool = False) -> None:
    requested = []
    for path in (ROOT / "docker/requirements-vllm-runtime.txt", ROOT / "requirements-test.txt"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            requirement = Requirement(line)
            if canonicalize_name(requirement.name) != "evalscope":
                requested.append(requirement)

    def satisfied(requirement: Requirement) -> bool:
        version = _installed(requirement.name)
        return version is not None and version in requirement.specifier

    explicit = {canonicalize_name(requirement.name) for requirement in requested}
    before = {
        name: version
        for name in PROTECTED_DISTRIBUTIONS
        if canonicalize_name(name) not in explicit
        and (version := _installed(name)) is not None
    }
    with tempfile.TemporaryDirectory(prefix="hcu-ci-dependencies-") as temporary:
        constraints = Path(temporary) / "runtime-constraints.txt"
        constraints.write_text(
            "".join(f"{name}=={version}\n" for name, version in before.items()),
            encoding="utf-8",
        )
        command = [
            sys.executable, "-m", "pip", "install",
            "--quiet", "--no-input", "--no-cache-dir", "--disable-pip-version-check",
            "--constraint", str(constraints),
        ]
        if user:
            command.append("--user")
        command.extend(str(requirement) for requirement in requested)
        print("Installing CI dependencies in the temporary container", flush=True)
        subprocess.run(command, check=True)

    if user:
        # This directory may have been created by pip after Python started.
        import site

        sys.path.insert(0, site.getusersitepackages())
        importlib.invalidate_caches()

    missing = [str(requirement) for requirement in requested if not satisfied(requirement)]
    drift = {
        name: (version, _installed(name))
        for name, version in before.items()
        if _installed(name) != version
    }
    if missing or drift:
        raise RuntimeError(
            f"CI dependency setup failed: missing={missing}; protected runtime drift={drift}"
        )
    print("CI container dependencies ready; protected runtime versions unchanged")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user", action="store_true")
    args = parser.parse_args()
    try:
        ensure_ci_dependencies(user=args.user)
    except (ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(str(error), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
