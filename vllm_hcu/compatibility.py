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
    __vllm_opendas_sha__,
    __vllm_target_version__,
    __vllm_upstream_sha__,
)


class VllmCompatibilityError(RuntimeError):
    """The installed vLLM distribution is absent, invalid, or unsupported."""


@dataclass(frozen=True, slots=True)
class VllmCompatibility:
    """A read-only observation of the vLLM/vLLM-HCU version contract."""

    expected_version: str
    upstream_sha: str
    opendas_sha: str
    actual_version: str | None
    vllm_location: str | None
    vllm_hcu_version: str
    vllm_hcu_location: str
    compatible: bool
    reason: str

    @property
    def expected(self) -> str:
        return self.expected_version

    def detail(self) -> str:
        actual = (
            self.actual_version
            if self.actual_version is not None
            else "not installed"
        )
        location = self.vllm_location or "not installed"
        return (
            f"expected={self.expected}; actual={actual!r}; "
            f"upstream_sha={self.upstream_sha}; "
            f"opendas_sha={self.opendas_sha}; "
            f"vllm_hcu={self.vllm_hcu_version!r}; "
            f"vllm_location={location!r}; "
            f"vllm_hcu_location={self.vllm_hcu_location!r}; "
            f"reason={self.reason}"
        )


def _distribution_location(
    distribution: importlib_metadata.Distribution,
) -> str | None:
    try:
        return str(Path(distribution.locate_file("")).resolve())
    except (OSError, TypeError, ValueError):
        return None


def inspect_vllm_compatibility() -> VllmCompatibility:
    """Inspect the installed vLLM metadata without importing vLLM itself."""

    # Keep platform-plugin discovery importable under ``python -S``.  The
    # platform probe latches a missing runtime dependency before any patch
    # mutation, while normal compatibility checks still use packaging's full
    # PEP 440 parser.
    from packaging.version import InvalidVersion, Version

    try:
        expected = Version(__vllm_target_version__)
    except (InvalidVersion, TypeError) as exc:
        raise RuntimeError(
            "vllm_hcu.version.__vllm_target_version__ is invalid: "
            f"{exc}"
        ) from exc
    hcu_location = str(Path(__file__).resolve().parent)
    try:
        distribution = importlib_metadata.distribution("vllm")
    except importlib_metadata.PackageNotFoundError:
        return VllmCompatibility(
            expected_version=__vllm_target_version__,
            upstream_sha=__vllm_upstream_sha__,
            opendas_sha=__vllm_opendas_sha__,
            actual_version=None,
            vllm_location=None,
            vllm_hcu_version=__hcu_version__,
            vllm_hcu_location=hcu_location,
            compatible=False,
            reason="the vLLM distribution is not installed",
        )

    actual = distribution.version
    location = _distribution_location(distribution)
    try:
        parsed = Version(actual)
    except (InvalidVersion, TypeError) as exc:
        return VllmCompatibility(
            expected_version=__vllm_target_version__,
            upstream_sha=__vllm_upstream_sha__,
            opendas_sha=__vllm_opendas_sha__,
            actual_version=actual,
            vllm_location=location,
            vllm_hcu_version=__hcu_version__,
            vllm_hcu_location=hcu_location,
            compatible=False,
            reason=f"invalid vLLM distribution version: {exc}",
        )

    compatible = parsed == expected
    if compatible:
        reason = "installed vLLM build matches the frozen OpenDAS artifact"
    else:
        reason = "installed vLLM build does not match the frozen OpenDAS artifact"
    return VllmCompatibility(
        expected_version=__vllm_target_version__,
        upstream_sha=__vllm_upstream_sha__,
        opendas_sha=__vllm_opendas_sha__,
        actual_version=actual,
        vllm_location=location,
        vllm_hcu_version=__hcu_version__,
        vllm_hcu_location=hcu_location,
        compatible=compatible,
        reason=reason,
    )


def ensure_vllm_compatible() -> VllmCompatibility:
    """Return compatibility details or fail before runtime patch mutation."""

    result = inspect_vllm_compatibility()
    if not result.compatible:
        raise VllmCompatibilityError(
            f"vLLM compatibility check failed: {result.detail()}"
        )
    return result


__all__ = [
    "VllmCompatibility",
    "VllmCompatibilityError",
    "ensure_vllm_compatible",
    "inspect_vllm_compatibility",
]
