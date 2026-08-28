# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Install exactly one locally built vLLM-HCU wheel into an isolated venv."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import venv
from pathlib import Path
from typing import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel-dir", required=True, type=Path)
    parser.add_argument("--venv", required=True, type=Path)
    args = parser.parse_args(argv)

    wheels = sorted(args.wheel_dir.glob("vllm_hcu-*.whl"))
    if len(wheels) != 1:
        print(
            f"expected exactly one vllm_hcu wheel in {args.wheel_dir}, got {wheels}",
            file=sys.stderr,
        )
        return 2
    checksum_path = args.wheel_dir / "SHA256SUMS"
    try:
        checksum_fields = checksum_path.read_text(encoding="utf-8").split()
    except OSError as exc:
        print(f"cannot read wheel checksum {checksum_path}: {exc}", file=sys.stderr)
        return 2
    if len(checksum_fields) != 2 or checksum_fields[1].lstrip("*") != wheels[0].name:
        print(f"invalid wheel checksum manifest: {checksum_fields}", file=sys.stderr)
        return 2
    digest = hashlib.sha256()
    with wheels[0].open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    actual_checksum = digest.hexdigest()
    if actual_checksum != checksum_fields[0]:
        print(
            f"wheel checksum mismatch: expected {checksum_fields[0]}, "
            f"got {actual_checksum}",
            file=sys.stderr,
        )
        return 2
    venv.EnvBuilder(
        clear=True,
        with_pip=True,
        system_site_packages=True,
    ).create(args.venv)
    python = args.venv / "bin" / "python"
    result = subprocess.run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--force-reinstall",
            "--no-deps",
            str(wheels[0].resolve()),
        ],
        check=False,
    )
    if result.returncode:
        return result.returncode
    print(python)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
