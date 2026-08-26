# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Fail-closed HCU CI resource and dependency preflight."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any, Sequence


REPOSITORY = Path(__file__).resolve().parents[3]
DEFAULT_ENVIRONMENT_LOCK = (
    REPOSITORY
    / ".github"
    / "workflows"
    / "configs"
    / "hcu-runner-environment.json"
)


class PreflightError(RuntimeError):
    """Raised when the selected runner cannot execute its assigned job."""


def _version_specification(value: object, *, name: str) -> dict[str, str]:
    if isinstance(value, str):
        return {"match": "exact", "version": value}
    if not isinstance(value, dict):
        raise PreflightError(f"{name} must be a version string or mapping")
    match = value.get("match")
    version = value.get("version")
    if match not in {"exact", "prefix"} or not isinstance(version, str):
        raise PreflightError(
            f"{name} must declare match=exact|prefix and a version string"
        )
    return {"match": match, "version": version}


def _matches_version(actual: str | None, specification: dict[str, str]) -> bool:
    if actual is None:
        return False
    if specification["match"] == "prefix":
        return actual.startswith(specification["version"])
    return actual == specification["version"]


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _resolve_requirement(item: dict[str, Any], model_root: Path | None) -> Path:
    env_name = item.get("env")
    relative = item.get("relative")
    if not isinstance(env_name, str) or not env_name:
        raise PreflightError(f"invalid requirement env: {item!r}")
    if not isinstance(relative, str) or not relative:
        raise PreflightError(f"invalid requirement relative path: {item!r}")
    override = os.environ.get(env_name)
    if override:
        return Path(override).expanduser().resolve()
    if model_root is None:
        raise PreflightError(
            f"{env_name} is unset and VLLM_HCU_TEST_MODEL_ROOT is unavailable"
        )
    return (model_root / relative).resolve()


def _check_requirements(
    requirements: list[dict[str, Any]],
    model_root: Path | None,
) -> list[dict[str, str]]:
    resolved: list[dict[str, str]] = []
    for item in requirements:
        kind = item.get("kind")
        if kind == "distribution":
            name = item.get("name")
            if not isinstance(name, str) or not name:
                raise PreflightError(
                    f"invalid distribution requirement name: {item!r}"
                )
            specification = _version_specification(
                {
                    "match": item.get("match"),
                    "version": item.get("version"),
                },
                name=f"distribution requirement {name!r}",
            )
            actual = _distribution_version(name)
            if actual is None:
                raise PreflightError(
                    f"required distribution is missing: {name}"
                )
            if not _matches_version(actual, specification):
                raise PreflightError(
                    f"required distribution drift for {name}: expected "
                    f"{specification['match']} {specification['version']}, "
                    f"got {actual}"
                )
            resolved.append(
                {
                    "kind": "distribution",
                    "name": name,
                    "version": actual,
                }
            )
            continue

        if kind not in {"model", "path"}:
            raise PreflightError(f"unsupported requirement kind: {kind!r}")
        path = _resolve_requirement(item, model_root)
        if not path.exists():
            raise PreflightError(f"required resource is unavailable: {path}")
        if kind == "model":
            if not path.is_dir() or not (
                (path / "config.json").is_file()
                or (path / "params.json").is_file()
            ):
                raise PreflightError(
                    f"model resource is not loadable: {path}; expected "
                    "config.json or params.json"
                )
        resolved.append(
            {
                "env": str(item["env"]),
                "kind": str(kind),
                "path": str(path),
            }
        )
    return resolved


def _load_environment_lock(path: Path) -> dict[str, Any]:
    try:
        lock = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PreflightError(f"cannot load environment lock {path}: {exc}") from exc
    if not isinstance(lock, dict) or lock.get("schema_version") != 1:
        raise PreflightError("environment lock must be a schema_version=1 mapping")
    if not isinstance(lock.get("python"), str):
        raise PreflightError("environment lock must declare an exact Python version")
    lock["torch_hip"] = _version_specification(
        lock.get("torch_hip"),
        name="environment lock torch_hip",
    )
    distributions = lock.get("distributions")
    if not isinstance(distributions, dict) or not distributions:
        raise PreflightError(
            "environment lock distributions must be a non-empty mapping"
        )
    for name, specification in distributions.items():
        if not isinstance(name, str) or not name:
            raise PreflightError("environment lock distribution names must be strings")
        distributions[name] = _version_specification(
            specification,
            name=f"environment lock entry {name!r}",
        )
    rocm = lock.get("rocm")
    if not isinstance(rocm, dict) or not all(
        isinstance(rocm.get(name), str) and rocm[name]
        for name in ("environment", "version_file", "version")
    ):
        raise PreflightError("environment lock must declare the DTK/ROCm version file")
    return lock


def _check_environment_lock(
    path: Path,
    *,
    versions: dict[str, str | None],
    torch_hip: str | None,
) -> dict[str, Any]:
    lock = _load_environment_lock(path)
    actual_python = platform.python_version()
    expected_python = lock["python"]
    if actual_python != expected_python:
        raise PreflightError(
            f"runner Python drift: expected {expected_python}, got {actual_python}"
        )
    expected_hip = lock["torch_hip"]
    if not _matches_version(torch_hip, expected_hip):
        raise PreflightError(
            "runner torch HIP drift: expected "
            f"{expected_hip['match']} {expected_hip['version']}, got {torch_hip}"
        )

    for name, specification in lock["distributions"].items():
        actual = versions.get(name)
        expected = specification["version"]
        if actual is None:
            raise PreflightError(f"locked distribution is missing: {name}")
        if not _matches_version(actual, specification):
            raise PreflightError(
                f"runner distribution drift for {name}: expected "
                f"{specification['match']} {expected}, got {actual}"
            )

    rocm = lock["rocm"]
    root_text = os.environ.get(rocm["environment"])
    if not root_text:
        raise PreflightError(
            f"runner environment is missing {rocm['environment']}"
        )
    version_path = Path(root_text).expanduser().resolve() / rocm["version_file"]
    try:
        actual_rocm = version_path.read_text(encoding="utf-8").splitlines()[0].strip()
    except (OSError, IndexError) as exc:
        raise PreflightError(
            f"cannot read DTK/ROCm version from {version_path}: {exc}"
        ) from exc
    if actual_rocm != rocm["version"]:
        raise PreflightError(
            f"runner DTK/ROCm drift: expected {rocm['version']}, got {actual_rocm}"
        )
    return {
        "path": str(path.resolve()),
        "python": expected_python,
        "torch_hip": expected_hip["version"],
        "rocm": actual_rocm,
        "distributions": lock["distributions"],
    }


def _collect_runtime_versions(torch: Any) -> tuple[dict[str, str | None], str | None]:
    versions = {
        name: _distribution_version(name)
        for name in (
            "torch",
            "vllm",
            "vllm-hcu",
            "aiter",
            "evalscope",
            "pytest",
        )
    }
    for mandatory in ("torch", "vllm"):
        if versions[mandatory] is None:
            raise PreflightError(f"required distribution is missing: {mandatory}")
    return versions, getattr(getattr(torch, "version", None), "hip", None)


def run_preflight(
    *,
    expected_arch: str,
    required_cards: int,
    requirements: list[dict[str, Any]],
    environment_lock: Path | None = None,
) -> dict[str, Any]:
    try:
        import torch
    except Exception:
        raise PreflightError(
            "HCU runtime dependency initialization failed."
        ) from None

    try:
        if not torch.cuda.is_available():
            raise PreflightError("torch reports no live HCU/ROCm device")
        actual_cards = int(torch.cuda.device_count())
        if actual_cards < required_cards:
            raise PreflightError(
                f"requires {required_cards} visible HCU devices, "
                f"got {actual_cards}"
            )

        device_arches: list[str] = []
        for index in range(required_cards):
            properties = torch.cuda.get_device_properties(index)
            raw_arch = getattr(properties, "gcnArchName", None)
            if not isinstance(raw_arch, str):
                raise PreflightError(
                    f"device {index} does not expose gcnArchName"
                )
            device_arches.append(raw_arch.split(":", 1)[0])
        wrong = [
            f"{index}:{arch}"
            for index, arch in enumerate(device_arches)
            if arch != expected_arch
        ]
        if wrong:
            raise PreflightError(
                f"requires {expected_arch}, incompatible devices: "
                f"{', '.join(wrong)}"
            )
    except PreflightError:
        raise
    except Exception:
        raise PreflightError("HCU device inspection failed.") from None

    model_root_text = os.environ.get("VLLM_HCU_TEST_MODEL_ROOT")
    model_root = (
        Path(model_root_text).expanduser().resolve()
        if model_root_text
        else None
    )
    resolved_requirements = _check_requirements(requirements, model_root)
    versions, torch_hip = _collect_runtime_versions(torch)
    lock_report = (
        _check_environment_lock(
            environment_lock,
            versions=versions,
            torch_hip=torch_hip,
        )
        if environment_lock is not None
        else None
    )
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "expected_arch": expected_arch,
        "device_arches": device_arches,
        "required_cards": required_cards,
        "visible_cards": actual_cards,
        "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "versions": versions,
        "resources": resolved_requirements,
        "environment_lock": lock_report,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=("gfx936", "gfx938"))
    parser.add_argument("--cards", type=int)
    parser.add_argument("--requirements-json", default="[]")
    parser.add_argument(
        "--environment-lock",
        type=Path,
        default=DEFAULT_ENVIRONMENT_LOCK,
    )
    parser.add_argument("--check-lock-only", action="store_true")
    parser.add_argument("--check-environment-only", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.check_lock_only:
            print(json.dumps(_load_environment_lock(args.environment_lock), indent=2))
            return 0
        if args.check_environment_only:
            try:
                import torch
            except Exception:
                raise PreflightError(
                    "HCU runtime dependency initialization failed."
                ) from None
            versions, torch_hip = _collect_runtime_versions(torch)
            print(
                json.dumps(
                    _check_environment_lock(
                        args.environment_lock,
                        versions=versions,
                        torch_hip=torch_hip,
                    ),
                    indent=2,
                )
            )
            return 0
        if args.arch is None or args.cards is None:
            raise PreflightError(
                "--arch and --cards are required for hardware preflight"
            )
        raw_requirements = json.loads(args.requirements_json)
        if not isinstance(raw_requirements, list) or not all(
            isinstance(item, dict) for item in raw_requirements
        ):
            raise PreflightError("requirements JSON must be a list of mappings")
        report = run_preflight(
            expected_arch=args.arch,
            required_cards=args.cards,
            requirements=raw_requirements,
            environment_lock=args.environment_lock,
        )
        rendered = json.dumps(report, indent=2, sort_keys=True)
        print(rendered)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")
    except (OSError, json.JSONDecodeError, PreflightError) as exc:
        print(f"HCU CI preflight failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
