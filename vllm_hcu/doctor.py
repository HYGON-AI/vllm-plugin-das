# SPDX-License-Identifier: Apache-2.0
"""Read-only installation/runtime diagnostics for vLLM-HCU."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from vllm_hcu.compatibility import inspect_vllm_compatibility


@dataclass(frozen=True, slots=True)
class Diagnostic:
    check: str
    ok: bool
    detail: str


def _vllm_location() -> tuple[str | None, Path | None]:
    try:
        version = importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError:
        return None, None
    spec = importlib.util.find_spec("vllm")
    if spec is None:
        return version, None
    locations = spec.submodule_search_locations
    if locations:
        return version, Path(next(iter(locations))).resolve()
    if spec.origin:
        return version, Path(spec.origin).resolve().parent
    return version, None


def _source_integrity_checks(vllm_root: Path) -> list[Diagnostic]:
    patched_markers: list[str] = []
    hcu_symlinks: list[str] = []
    for path in vllm_root.rglob("*.py"):
        try:
            if path.is_symlink():
                target = path.resolve(strict=False)
                if "vllm_hcu" in target.parts or "vllm-hcu" in target.parts:
                    hcu_symlinks.append(str(path.relative_to(vllm_root)))
                continue
            if "PATCHED_" in path.read_text(encoding="utf-8", errors="ignore"):
                patched_markers.append(str(path.relative_to(vllm_root)))
        except OSError as exc:
            return [
                Diagnostic(
                    "vllm_source_readable",
                    False,
                    f"cannot inspect {path}: {type(exc).__name__}: {exc}",
                )
            ]
    return [
        Diagnostic(
            "no_source_patch_markers",
            not patched_markers,
            "none" if not patched_markers else ", ".join(patched_markers[:20]),
        ),
        Diagnostic(
            "no_hcu_source_symlinks",
            not hcu_symlinks,
            "none" if not hcu_symlinks else ", ".join(hcu_symlinks[:20]),
        ),
    ]


def collect_diagnostics(*, arm_platform: bool = True) -> list[Diagnostic]:
    """Collect diagnostics without writing to either package tree."""

    checks: list[Diagnostic] = []
    compatibility = inspect_vllm_compatibility()
    version, vllm_root = _vllm_location()
    checks.append(
        Diagnostic(
            "vllm_installed",
            version is not None and vllm_root is not None,
            f"version={version!r}, root={str(vllm_root) if vllm_root else None}",
        )
    )
    checks.append(
        Diagnostic(
            "vllm_compatible",
            compatibility.compatible,
            compatibility.detail(),
        )
    )
    if vllm_root is not None:
        checks.extend(_source_integrity_checks(vllm_root))

    package_root = Path(__file__).resolve().parent
    legacy_files = sorted((package_root / "patches").glob("vllm*.patch.py"))
    checks.append(
        Diagnostic(
            "legacy_string_patches_removed",
            not legacy_files,
            "none" if not legacy_files else f"{len(legacy_files)} files remain",
        )
    )
    legacy_engine = package_root / "patch_utils.py"
    checks.append(
        Diagnostic(
            "legacy_source_engine_removed",
            not legacy_engine.exists(),
            "absent" if not legacy_engine.exists() else str(legacy_engine),
        )
    )

    if arm_platform:
        try:
            from vllm_hcu.patch import apply_platform_patches, patch_report

            apply_platform_patches()
            report = patch_report()
            patches = report.get("patches", {})
            failures = [
                patch_id
                for patch_id, record in patches.items()
                if isinstance(record, dict) and record.get("status") == "failed"
            ]
            checks.append(
                Diagnostic(
                    "platform_runtime_patches_armed",
                    bool(patches) and not failures,
                    f"count={len(patches)}, failed={failures}",
                )
            )
        except Exception as exc:
            checks.append(
                Diagnostic(
                    "platform_runtime_patches_armed",
                    False,
                    f"{type(exc).__name__}: {exc}",
                )
            )
    return checks


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vllm-hcu-doctor",
        description="Read-only vLLM-HCU compatibility diagnostics.",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON")
    parser.add_argument(
        "--no-arm",
        action="store_true",
        help="skip process-local platform callback/replacement registration",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    checks = collect_diagnostics(arm_platform=not args.no_arm)
    if args.json:
        print(json.dumps([asdict(check) for check in checks], indent=2, sort_keys=True))
    else:
        for check in checks:
            state = "PASS" if check.ok else "FAIL"
            print(f"[{state}] {check.check}: {check.detail}")
    return 0 if all(check.ok for check in checks) else 1


__all__ = ["Diagnostic", "collect_diagnostics", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
