# SPDX-License-Identifier: Apache-2.0
"""Install EvalScope only for selected evaluation jobs, without replacing runtime pins."""

from __future__ import annotations

import importlib.metadata as metadata
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

PROTECTED_DISTRIBUTIONS = (
    "torch", "vllm", "vllm-hcu", "aiter", "numpy", "scipy", "sympy",
    "transformers", "tokenizers", "pandas", "ray", "pytest",
    "pydantic", "fastapi", "starlette", "openai", "httpx", "aiohttp",
)


def _installed(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def ensure_evalscope(requirements: list[dict]) -> bool:
    requested = [
        item for item in requirements
        if item.get("kind") == "distribution" and item.get("name") == "evalscope"
    ]
    if not requested:
        print("EvalScope is not required by this HCU job")
        return False
    if len(requested) != 1 or requested[0].get("match") != "exact":
        raise ValueError("EvalScope bootstrap requires one exact version requirement")
    required = requested[0]["version"]
    from packaging.version import Version

    Version(required)
    if _installed("evalscope") == required:
        print(f"Reusing evalscope=={required}")
        return False

    before = {
        name: installed
        for name in PROTECTED_DISTRIBUTIONS
        if (installed := _installed(name)) is not None
    }
    with tempfile.TemporaryDirectory(prefix="hcu-ci-evalscope-") as temporary:
        constraints = Path(temporary) / "runtime-constraints.txt"
        constraints.write_text(
            "".join(f"{name}=={version}\n" for name, version in before.items())
        )
        print(f"Installing evalscope=={required} for the selected HCU job", flush=True)
        subprocess.run(
            [
                sys.executable, "-m", "pip", "install",
                "--quiet", "--no-input", "--disable-pip-version-check",
                "--constraint", str(constraints),
                f"evalscope=={required}",
            ],
            check=True,
        )

    if _installed("evalscope") != required:
        raise RuntimeError(f"EvalScope bootstrap did not install evalscope=={required}")
    drift = {
        name: (version, _installed(name))
        for name, version in before.items()
        if _installed(name) != version
    }
    if drift:
        raise RuntimeError(f"EvalScope installation changed protected runtime versions: {drift}")
    print(f"evalscope=={required}; protected runtime versions unchanged")
    return True


def main() -> int:
    try:
        requirements = json.loads(os.environ.get("HCU_CI_REQUIREMENTS_JSON", "[]"))
        ensure_evalscope(requirements)
    except (ValueError, KeyError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"HCU evaluation dependency setup failed: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
