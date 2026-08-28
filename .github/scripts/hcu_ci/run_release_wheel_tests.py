# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Run server or multi-HCU smoke tests against an installed wheel."""

from __future__ import annotations

import argparse
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Sequence


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _junit_counts(path: Path) -> tuple[int, int, int]:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    tests = sum(int(suite.attrib.get("tests", "0")) for suite in suites)
    skipped = sum(int(suite.attrib.get("skipped", "0")) for suite in suites)
    failures = sum(
        int(suite.attrib.get("failures", "0"))
        + int(suite.attrib.get("errors", "0"))
        for suite in suites
    )
    return tests, skipped, failures


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True, type=Path)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=("server", "multi-card"))
    parser.add_argument("--collect-only", action="store_true")
    args = parser.parse_args(argv)

    repository = args.repository.resolve()
    sys.path[:] = [
        entry
        for entry in sys.path
        if not entry or Path(entry).resolve() != repository
    ]
    import vllm_hcu

    module_path = Path(vllm_hcu.__file__).resolve()
    if _is_relative_to(module_path, repository):
        print(f"release smoke imported checkout source: {module_path}", file=sys.stderr)
        return 2
    sys.path.append(str(repository))
    import pytest

    args.artifact_root.mkdir(parents=True, exist_ok=True)
    os.environ.pop("PYTHONPATH", None)
    os.environ["VLLM_HCU_RELEASE_WHEEL"] = "1"
    os.environ["VLLM_HCU_TEST_STRICT_RESOURCES"] = "1"
    os.environ["VLLM_HCU_INTEGRATION_LOG_DIR"] = str(
        args.artifact_root / "integration"
    )
    junit = args.artifact_root / "pytest.xml"
    if args.mode == "server":
        selected = [
            str(
                repository
                / "tests/integration/server/test_qwen25_server_smoke.py"
            ),
            "-k",
            "qwen25_15b_openai_server_smoke",
        ]
    else:
        selected = [
            str(repository / "tests/integration/parallel/test_tp_ep_models.py"),
            "-k",
            (
                "qwen35_35b_a3b_tp_ep_smoke and tp4-ep4 "
                "and aiter-tuned-shuffle"
            ),
        ]
    pytest_args = [
        "-q",
        "--import-mode=importlib",
        "-m",
        "model and hcu and not multi_node",
        *selected,
        "-rsxX",
        f"--junitxml={junit}",
    ]
    if args.collect_only:
        pytest_args.insert(1, "--collect-only")
    print("release wheel pytest:", " ".join(pytest_args), flush=True)
    returncode = int(pytest.main(pytest_args))
    if args.collect_only:
        return returncode
    if not junit.is_file():
        print(f"release smoke did not produce JUnit: {junit}", file=sys.stderr)
        return 2
    tests, skipped, failures = _junit_counts(junit)
    if returncode or tests != 1 or skipped or failures:
        print(
            f"release smoke failed closed: rc={returncode} tests={tests} "
            f"skipped={skipped} failures={failures}",
            file=sys.stderr,
        )
        return returncode or 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
