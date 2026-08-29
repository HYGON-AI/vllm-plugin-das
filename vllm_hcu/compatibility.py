# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Dependency-light vLLM distribution compatibility checks.

The runtime integration is audited against the vLLM release series encoded by
``vllm_hcu.version.__version_tuple__``.  Check the installed distribution
metadata before any process-local patch registration so a mismatched wheel
cannot leave a partially armed registry behind.
"""

from __future__ import annotations

import importlib.metadata as importlib_metadata
from dataclasses import dataclass
from pathlib import Path

from vllm_hcu.version import (
    __hcu_version__,
    __version_tuple__,
)


class VllmCompatibilityError(RuntimeError):
    """The installed vLLM distribution is absent, invalid, or unsupported."""


@dataclass(frozen=True, slots=True)
class VllmCompatibility:
    """A read-only observation of the vLLM/vLLM-HCU version contract."""

    expected_series: tuple[int, int]
    actual_version: str | None
    vllm_location: str | None
    vllm_hcu_version: str
    vllm_hcu_location: str
    compatible: bool
    reason: str

    @property
    def expected(self) -> str:
        return f"{self.expected_series[0]}.{self.expected_series[1]}.x"

    def detail(self) -> str:
        actual = (
            self.actual_version
            if self.actual_version is not None
            else "not installed"
        )
        location = self.vllm_location or "not installed"
        return (
            f"expected={self.expected}; actual={actual!r}; "
            f"vllm_hcu={self.vllm_hcu_version!r}; "
            f"vllm_location={location!r}; "
            f"vllm_hcu_location={self.vllm_hcu_location!r}; "
            f"reason={self.reason}"
        )


def _supported_series() -> tuple[int, int]:
    series = tuple(__version_tuple__[:2])
    if len(series) != 2 or not all(type(value) is int for value in series):
        raise RuntimeError(
            "vllm_hcu.version.__version_tuple__ must begin with integer "
            "major/minor components"
        )
    return series


def _distribution_location(
    distribution: importlib_metadata.Distribution,
) -> str | None:
    try:
        return str(Path(distribution.locate_file("")).resolve())
    except (OSError, TypeError, ValueError):
        return None


def inspect_vllm_compatibility() -> VllmCompatibility:
    """Inspect the installed vLLM metadata without importing vLLM itself."""

    expected_series = _supported_series()
    hcu_location = str(Path(__file__).resolve().parent)
    try:
        distribution = importlib_metadata.distribution("vllm")
    except importlib_metadata.PackageNotFoundError:
        return VllmCompatibility(
            expected_series=expected_series,
            actual_version=None,
            vllm_location=None,
            vllm_hcu_version=__hcu_version__,
            vllm_hcu_location=hcu_location,
            compatible=False,
            reason="the vLLM distribution is not installed",
        )

    actual = distribution.version
    location = _distribution_location(distribution)
    from packaging.version import InvalidVersion, Version

    try:
        parsed = Version(actual)
    except (InvalidVersion, TypeError) as exc:
        return VllmCompatibility(
            expected_series=expected_series,
            actual_version=actual,
            vllm_location=location,
            vllm_hcu_version=__hcu_version__,
            vllm_hcu_location=hcu_location,
            compatible=False,
            reason=f"invalid vLLM distribution version: {exc}",
        )

    actual_series = parsed.release[:2]
    compatible = parsed.epoch == 0 and actual_series == expected_series
    if compatible:
        reason = "installed vLLM release series is supported"
    else:
        reason = f"installed vLLM release series {actual_series!r} is unsupported"
    return VllmCompatibility(
        expected_series=expected_series,
        actual_version=actual,
        vllm_location=location,
        vllm_hcu_version=__hcu_version__,
        vllm_hcu_location=hcu_location,
        compatible=compatible,
        reason=reason,
    )


def ensure_vllm_compatible(
    *,
    check_target_api: bool = True,
) -> VllmCompatibility:
    """Fail before mutation unless release and target API contracts match.

    check_target_api=False is reserved for metadata-only diagnostics and unit
    tests that intentionally provide a synthetic distribution.
    """

    result = inspect_vllm_compatibility()
    if not result.compatible:
        raise VllmCompatibilityError(
            f"vLLM compatibility check failed: {result.detail()}"
        )
    if check_target_api:
        if result.vllm_location is None:
            raise VllmCompatibilityError(
                "vLLM target API fingerprint failed: distribution location "
                "is unavailable"
            )
        from vllm_hcu.target_api import inspect_target_api

        mismatches = inspect_target_api(result.vllm_location)
        if mismatches:
            details = "\n- ".join(mismatches)
            raise VllmCompatibilityError(
                "vLLM target API fingerprint failed:\n- " + details
            )
    return result


__all__ = [
    "VllmCompatibility",
    "VllmCompatibilityError",
    "ensure_vllm_compatible",
    "inspect_vllm_compatibility",
]
