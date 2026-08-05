# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Run deterministic vLLM-HCU patch test tiers."""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
import sysconfig
from pathlib import Path
from typing import Sequence

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from tools.check_patch_test_coverage import audit_repository


INVENTORY_TESTS = (
    "tests/patch/test_patch_coverage.py",
    "tests/patch/test_import_coordinator.py",
    "tests/patch/test_module_exchange.py",
    "tests/patch/test_platform_dispatcher.py",
    "tests/patch/test_runtime_callbacks.py",
    "tests/patch/test_runtime_state.py",
    "tests/patch/test_worker_dispatcher.py",
)
CONTRACT_TESTS = (
    "tests/patch",
    "tests/runtime_patch",
    "tests/accuracy",
    "tests/gemma4_test",
)
HCU_CONTRACT_TESTS = (
    "tests/patch",
    "tests/runtime_patch",
    "tests/gemma4_test",
)
INTEGRATION_TESTS = ("tests/integration",)
SINGLE_NODE_DISTRIBUTED_TESTS = ("tests/distributed/single_node",)
MULTI_NODE_DISTRIBUTED_TESTS = ("tests/distributed/multi_node",)
DISTRIBUTED_TESTS = ("tests/distributed",)
STRESS_TESTS = ("tests/stress",)
SUITES = {
    "inventory": INVENTORY_TESTS,
    "accuracy": ("tests/accuracy",),
    "accuracy-hcu": ("tests/accuracy",),
    "contract": CONTRACT_TESTS,
    "contract-hcu": HCU_CONTRACT_TESTS,
    "integration-smoke": INTEGRATION_TESTS,
    "model": INTEGRATION_TESTS,
    "distributed-single-node": SINGLE_NODE_DISTRIBUTED_TESTS,
    "distributed-multi-node": MULTI_NODE_DISTRIBUTED_TESTS,
    "distributed": DISTRIBUTED_TESTS,
    "stress": STRESS_TESTS,
    "nightly": ("tests",),
    "full": ("tests",),
}
SUITE_PYTEST_ARGS = {
    "inventory": (),
    "accuracy": ("-m", "not hcu"),
    "accuracy-hcu": ("-m", "hcu"),
    "contract": ("-m", "not hcu"),
    "contract-hcu": ("-m", "hcu and not model and not multi_node"),
    "integration-smoke": (
        "-m",
        "not slow and not multi_hcu and not multi_node",
    ),
    "model": ("-m", "model and hcu and not multi_node"),
    "distributed-single-node": ("-m", "not multi_node"),
    "distributed-multi-node": ("-m", "multi_node"),
    "distributed": (),
    "stress": (),
    "nightly": ("-m", "hcu or model or nightly"),
    "full": (),
}
EMPTY_SCAFFOLD_SUITES = {
    "integration-smoke",
    "model",
    "distributed-single-node",
    "distributed-multi-node",
    "distributed",
    "stress",
}


def _valid_vllm_root(path: Path) -> bool:
    return (path / "vllm" / "__init__.py").is_file()


def _resolve_vllm_root(value: Path | None) -> Path:
    installed_roots = [
        Path(path)
        for key in ("platlib", "purelib")
        if (path := sysconfig.get_path(key))
    ]
    candidates = [
        value,
        Path(os.environ["VLLM_V0251_SOURCE_ROOT"])
        if "VLLM_V0251_SOURCE_ROOT" in os.environ
        else None,
        *installed_roots,
        REPOSITORY.parent / "vllm_0251",
    ]
    for candidate in candidates:
        if candidate is not None:
            resolved = candidate.expanduser().resolve()
            if _valid_vllm_root(resolved):
                return resolved
    rendered = ", ".join(str(path) for path in candidates if path is not None)
    raise SystemExit(
        "No vLLM 0.25.1 source tree found. Pass --vllm-source or set "
        f"VLLM_V0251_SOURCE_ROOT. Checked: {rendered}"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        choices=tuple(SUITES),
        default="contract",
        help=(
            "inventory, portable/HCU accuracy, portable contract, model "
            "integration, single/multi-node distributed, stress, nightly, "
            "or complete hardware-aware suite"
        ),
    )
    parser.add_argument(
        "--vllm-source",
        type=Path,
        help="directory containing the target vllm/ package",
    )
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help="collect selected pytest nodes without executing them",
    )
    parser.add_argument(
        "--target",
        action="append",
        default=[],
        help=(
            "registered repository-relative pytest file or nodeid; repeat to "
            "replace the suite's default filesystem targets"
        ),
    )
    return parser


def _validated_targets(values: Sequence[str]) -> tuple[str, ...]:
    targets: list[str] = []
    for value in values:
        path_text = value.split("::", 1)[0]
        path = Path(path_text)
        if path.is_absolute() or ".." in path.parts or not path_text.startswith("tests/"):
            raise SystemExit(f"invalid registered pytest target: {value}")
        resolved = REPOSITORY / path
        if not resolved.is_file() or not resolved.name.startswith("test_"):
            raise SystemExit(f"registered pytest target does not exist: {value}")
        targets.append(value)
    return tuple(targets)


def _contains_test_files(entries: Sequence[str]) -> bool:
    """Return whether a scaffold suite currently contains pytest test files."""
    for entry in entries:
        path_text = entry.split("::", 1)[0]
        path = REPOSITORY / path_text
        if path.is_file() and path.name.startswith("test_"):
            return True
        if path.is_dir() and any(path.rglob("test_*.py")):
            return True
    return False


def main(argv: Sequence[str] | None = None) -> int:
    args, pytest_args = _parser().parse_known_args(argv)
    if pytest_args[:1] == ["--"]:
        pytest_args = pytest_args[1:]

    audit = audit_repository(REPOSITORY)
    if not audit.ok:
        for module_name, missing in audit.missing_contract.items():
            print(
                f"invalid patch adapter {module_name}: "
                f"missing {', '.join(missing)}",
                file=sys.stderr,
            )
        for module_name in audit.untested_modules:
            print(f"patch has no direct test reference: {module_name}", file=sys.stderr)
        return 1

    if importlib.util.find_spec("pytest") is None:
        print(
            "pytest is not installed. Run: "
            f"{sys.executable} -m pip install -r requirements-test.txt",
            file=sys.stderr,
        )
        return 2

    if (
        args.suite in EMPTY_SCAFFOLD_SUITES
        and not _contains_test_files(SUITES[args.suite])
    ):
        print(
            f"patch test suite={args.suite}: scaffold exists but contains "
            "no test_*.py files yet",
            flush=True,
        )
        return 0

    vllm_root = _resolve_vllm_root(args.vllm_source)
    environment = dict(os.environ)
    environment["VLLM_V0251_SOURCE_ROOT"] = str(vllm_root)
    environment.setdefault("VLLM_V0251_PYTHON", sys.executable)
    environment["VLLM_PLUGINS"] = "__disabled__"
    python_path = [str(vllm_root), str(REPOSITORY)]
    if environment.get("PYTHONPATH"):
        python_path.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(python_path)

    command = [sys.executable, "-m", "pytest", "-q"]
    if args.collect_only:
        command.append("--collect-only")
    command.extend(SUITE_PYTEST_ARGS[args.suite])
    command.extend(_validated_targets(args.target) or SUITES[args.suite])
    command.extend(pytest_args)
    print(
        f"patch test suite={args.suite} vllm_source={vllm_root}",
        flush=True,
    )
    print("command:", " ".join(command), flush=True)
    result = subprocess.run(
        command,
        cwd=REPOSITORY,
        env=environment,
        check=False,
    )
    if result.returncode == 5 and args.suite in EMPTY_SCAFFOLD_SUITES:
        print(
            f"patch test suite={args.suite}: no tests selected by current "
            "markers/arguments",
            flush=True,
        )
        return 0
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
