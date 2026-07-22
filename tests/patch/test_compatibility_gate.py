# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest

import vllm_hcu as plugin
from vllm_hcu import compatibility
from vllm_hcu import version as hcu_version
from vllm_hcu.patch import (
    IMPORT_COORDINATOR,
    PATCH_REGISTRY,
    apply_platform_patches,
    apply_worker_patches,
)


class _Distribution:
    def __init__(self, version: str, location: Path) -> None:
        self.version = version
        self._location = location

    def locate_file(self, path: str) -> Path:
        assert path == ""
        return self._location


@pytest.fixture(autouse=True)
def _clean_process_state(monkeypatch: pytest.MonkeyPatch):
    IMPORT_COORDINATOR.reset_for_tests()
    monkeypatch.setattr(plugin, "_PLATFORM_INIT_FAILURE", None)
    yield
    IMPORT_COORDINATOR.reset_for_tests()


def _install_version(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    value: str,
) -> None:
    location = tmp_path / "site-packages"
    monkeypatch.setattr(
        compatibility.importlib_metadata,
        "distribution",
        lambda name: _Distribution(value, location),
    )


@pytest.mark.parametrize(
    "value",
    (
        "0.25.0",
        "0.25.9",
        "0.25.0+das.5bf5c5f.dtk2604",
        "0.25.1.post2",
    ),
)
def test_supported_vllm_series_accepts_pep440_variants(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    value: str,
) -> None:
    _install_version(monkeypatch, tmp_path, value)

    result = compatibility.ensure_vllm_compatible()

    assert result.compatible
    assert result.expected == "0.25.x"
    assert result.actual_version == value
    assert result.vllm_location == str((tmp_path / "site-packages").resolve())


@pytest.mark.parametrize(
    "value",
    (
        "0.20.2",
        "0.22.0",
        "0.24.1+das.local",
        "1!0.25.0",
        "not a version",
    ),
)
def test_unsupported_or_invalid_vllm_metadata_has_actionable_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    value: str,
) -> None:
    _install_version(monkeypatch, tmp_path, value)

    with pytest.raises(compatibility.VllmCompatibilityError) as raised:
        compatibility.ensure_vllm_compatible()

    message = str(raised.value)
    assert "expected=0.25.x" in message
    assert f"actual={value!r}" in message
    assert "vllm_hcu=" in message
    assert "vllm_location=" in message
    assert "vllm_hcu_location=" in message


def test_missing_vllm_distribution_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(name: str):
        raise compatibility.importlib_metadata.PackageNotFoundError(name)

    monkeypatch.setattr(compatibility.importlib_metadata, "distribution", missing)

    with pytest.raises(
        compatibility.VllmCompatibilityError,
        match="actual='not installed'",
    ):
        compatibility.ensure_vllm_compatible()


def test_runtime_hcu_version_prefers_metadata_with_source_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        hcu_version.importlib_metadata,
        "version",
        lambda name: "0.25.0+das.abcdef0.dtk2604",
    )
    assert hcu_version.get_hcu_version() == "0.25.0+das.abcdef0.dtk2604"

    def missing(name: str):
        raise hcu_version.importlib_metadata.PackageNotFoundError(name)

    monkeypatch.setattr(hcu_version.importlib_metadata, "version", missing)
    assert hcu_version.get_hcu_version() == hcu_version.__version__


def test_every_apply_boundary_fails_before_registry_mutation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import vllm_hcu.patch.platform as platform_dispatcher
    import vllm_hcu.patch.worker as worker_dispatcher

    _install_version(monkeypatch, tmp_path, "0.22.0")

    for apply in (
        apply_platform_patches,
        apply_worker_patches,
        platform_dispatcher.apply_platform_patches,
        worker_dispatcher.apply_worker_patches,
    ):
        with pytest.raises(compatibility.VllmCompatibilityError):
            apply()
        assert IMPORT_COORDINATOR.registrations() == ()
        assert PATCH_REGISTRY.report()["patches"] == {}


def test_all_plugin_entries_gate_and_platform_probe_latches_compatibility(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_version(monkeypatch, tmp_path, "0.24.0")

    # vLLM v0.25 probes the platform plugin under a broad exception handler.
    # Preserve selection once, then surface the exact compatibility class.
    assert plugin.hcu_platform_plugin() == plugin._PLATFORM_CLASS_PATH
    with pytest.raises(
        compatibility.VllmCompatibilityError,
        match="previously failed",
    ):
        plugin.hcu_platform_plugin()

    for entry in (
        plugin.hcu_platform_register_model,
        plugin.hcu_platform_register_ops,
    ):
        with pytest.raises(compatibility.VllmCompatibilityError):
            entry()

    assert IMPORT_COORDINATOR.registrations() == ()
    assert PATCH_REGISTRY.report()["patches"] == {}
