# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Resolve release-specific inputs for HCU CI workflows."""

import argparse
import os
import sys
from pathlib import Path
from typing import Mapping, NamedTuple, Optional, Sequence


ENVIRONMENT_LOCK = ".github/workflows/configs/hcu-runner-environment.json"
V0281_REF = "v0.28.1-dev"
V0281_IMAGE_ENV = "HCU_CI_IMAGE_V0281"


class ReleaseEnvironmentError(RuntimeError):
    """Raised when a release has no usable HCU CI environment."""


class ReleaseEnvironment(NamedTuple):
    image: str
    environment_lock: str


def resolve_release_environment(
    ref: str,
    environment: Mapping[str, str],
) -> ReleaseEnvironment:
    """Return the explicitly configured HCU CI environment for ``ref``."""

    if ref != V0281_REF:
        raise ReleaseEnvironmentError(f"unsupported HCU CI release ref: {ref}")
    image = environment.get(V0281_IMAGE_ENV, "").strip()
    if not image:
        raise ReleaseEnvironmentError(
            f"{V0281_IMAGE_ENV} is required for {V0281_REF}"
        )
    return ReleaseEnvironment(image=image, environment_lock=ENVIRONMENT_LOCK)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", required=True)
    parser.add_argument("--github-output", required=True, type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        resolved = resolve_release_environment(args.ref, os.environ)
        rendered = (
            f"image={resolved.image}\n"
            f"environment_lock={resolved.environment_lock}\n"
        )
        with args.github_output.open("a", encoding="utf-8") as output:
            output.write(rendered)
    except (OSError, ReleaseEnvironmentError) as exc:
        print(f"HCU CI environment resolution failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
