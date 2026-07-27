# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from __future__ import annotations

import builtins
import importlib
import importlib.abc
import importlib.util
import os
import sys
import threading
from pathlib import Path
from types import ModuleType

import pytest

from vllm_hcu.patch.import_coordinator import (
    ExactImportCoordinator,
    LateModuleReplacementError,
    ModuleReloadBlockedError,
    OptionalPatchUnavailable,
)
from vllm_hcu.patch.runtime_state import (
    LatchedPatchError,
    PatchRegistry,
    PatchStatus,
    ProcessRole,
)


def _write_module(root: Path, name: str, source: str) -> None:
    parts = name.split(".")
    package = root
    for part in parts[:-1]:
        package /= part
        package.mkdir(exist_ok=True)
        init = package / "__init__.py"
        if not init.exists():
            init.write_text("", encoding="utf-8")
    (package / f"{parts[-1]}.py").write_text(source, encoding="utf-8")
    importlib.invalidate_caches()


@pytest.fixture
def coordinator() -> ExactImportCoordinator:
    value = ExactImportCoordinator(registry=PatchRegistry())
    yield value
    value.reset_for_tests()


@pytest.fixture
def module_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.syspath_prepend(str(tmp_path))
    yield tmp_path
    for name in tuple(sys.modules):
        if name.startswith("hcu_coord_test_"):
            sys.modules.pop(name, None)


class _BatchProbeFinder(importlib.abc.MetaPathFinder):
    def __init__(self, names: tuple[str, ...]) -> None:
        self.reached = {name: threading.Event() for name in names}

    def find_spec(self, fullname, path=None, target=None):
        event = self.reached.get(fullname)
        if event is not None:
            event.set()
        return None


class _CountingOfficialLoader(importlib.abc.Loader):
    def __init__(self, owner: "_CountingOfficialFinder", fullname: str) -> None:
        self.owner = owner
        self.fullname = fullname

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        self.owner.executed.append(self.fullname)
        module.VALUE = "official"


class _CountingOfficialFinder(importlib.abc.MetaPathFinder):
    def __init__(self, names: tuple[str, ...]) -> None:
        self.names = frozenset(names)
        self.consulted: list[str] = []
        self.executed: list[str] = []

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in self.names:
            return None
        self.consulted.append(fullname)
        return importlib.util.spec_from_loader(
            fullname,
            _CountingOfficialLoader(self, fullname),
            origin="registration-batch-official-test",
        )


def _thread_import(
    name: str,
    loaded: dict[str, ModuleType],
    errors: list[BaseException],
) -> None:
    try:
        loaded[name] = importlib.import_module(name)
    except BaseException as exc:  # pragma: no cover - asserted in parent thread
        errors.append(exc)


def test_exact_post_load_callback_does_not_match_ancestors_or_siblings(
    coordinator: ExactImportCoordinator,
    module_path: Path,
) -> None:
    package = "hcu_coord_test_exact"
    target = f"{package}.target"
    sibling = f"{package}.sibling"
    _write_module(module_path, target, "READY = True\n")
    _write_module(module_path, sibling, "READY = True\n")
    observations: list[bool] = []
    coordinator.register_callback(
        "exact.callback",
        target,
        lambda module: observations.append(module.READY),
    )

    original_import = builtins.__import__
    coordinator.install()
    assert builtins.__import__ is original_import
    importlib.import_module(package)
    importlib.import_module(sibling)
    assert observations == []
    imported = importlib.import_module(target)
    assert imported.READY is True
    assert observations == [True]


def test_loaded_module_callback_is_immediate_and_reload_fails_closed(
    coordinator: ExactImportCoordinator,
    module_path: Path,
) -> None:
    name = "hcu_coord_test_loaded"
    _write_module(module_path, name, "VALUE = 3\n")
    module = importlib.import_module(name)
    calls: list[int] = []

    coordinator.register_callback("loaded.callback", name, lambda value: calls.append(value.VALUE))
    coordinator.install()
    assert calls == [3]
    with pytest.raises(ModuleReloadBlockedError, match="reload could discard"):
        importlib.reload(module)
    assert calls == [3]
    assert module.VALUE == 3


def test_partial_module_callback_waits_for_import_completion_and_drains(
    coordinator: ExactImportCoordinator,
) -> None:
    name = "hcu_coord_test_partial_module"
    module = ModuleType(name)
    spec = importlib.machinery.ModuleSpec(name, loader=None)
    spec._initializing = True
    module.__spec__ = spec
    sys.modules[name] = module
    calls: list[ModuleType] = []

    def callback(value: ModuleType) -> None:
        calls.append(value)

    registration = coordinator.register_callback(
        "partial.callback",
        name,
        callback,
    )
    assert registration.status == PatchStatus.ARMED.value
    assert calls == []

    module.READY = True
    spec._initializing = False
    repeated = coordinator.register_callback(
        "partial.callback",
        name,
        callback,
    )
    assert repeated.status == PatchStatus.APPLIED.value
    assert calls == [module]


def test_failed_loaded_callback_reentry_raises_latched_error(
    coordinator: ExactImportCoordinator,
) -> None:
    name = "hcu_coord_test_failed_loaded_reentry"
    module = ModuleType(name)
    sys.modules[name] = module
    calls = 0

    def fail(value: ModuleType) -> None:
        nonlocal calls
        assert value is module
        calls += 1
        raise RuntimeError("incompatible loaded target")

    with pytest.raises(RuntimeError, match="incompatible loaded target"):
        coordinator.register_callback("loaded.failed", name, fail)
    with pytest.raises(LatchedPatchError, match="previously failed"):
        coordinator.register_callback("loaded.failed", name, fail)
    assert calls == 1


def test_callback_reports_explicit_target_symbols_and_reuses_declaration(
    coordinator: ExactImportCoordinator,
    module_path: Path,
) -> None:
    name = "hcu_coord_test_symbol_targets"
    target = f"{name}.Target.method"
    _write_module(module_path, name, "VALUE = 7\n")
    module = importlib.import_module(name)

    coordinator._registry.declare("symbol.callback", target)
    registration = coordinator.register_callback(
        "symbol.callback",
        name,
        lambda value: setattr(value, "PATCHED", True),
        targets=target,
    )

    assert registration.targets == (target,)
    assert module.PATCHED is True
    report = coordinator._registry.report()["patches"]["symbol.callback"]
    assert report["targets"] == [target]


def test_install_closes_registration_to_import_race(
    coordinator: ExactImportCoordinator,
    module_path: Path,
) -> None:
    name = "hcu_coord_test_install_race"
    _write_module(module_path, name, "VALUE = 11\n")
    calls: list[int] = []
    coordinator.register_callback(
        "install.race", name, lambda module: calls.append(module.VALUE)
    )

    # No finder is installed yet, so this is the canonical import.  install()
    # must immediately process it instead of leaving an armed patch behind.
    importlib.import_module(name)
    assert calls == []
    coordinator.install()
    assert calls == [11]


def test_concurrent_import_runs_callback_once(
    coordinator: ExactImportCoordinator,
    module_path: Path,
) -> None:
    name = "hcu_coord_test_concurrent"
    _write_module(module_path, name, "VALUE = 5\n")
    calls: list[int] = []
    coordinator.register_callback(
        "concurrent.callback", name, lambda module: calls.append(module.VALUE)
    )
    coordinator.install()

    modules: list[ModuleType] = []
    threads = [
        threading.Thread(target=lambda: modules.append(importlib.import_module(name)))
        for _ in range(8)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert all(not thread.is_alive() for thread in threads)
    assert len(modules) == 8
    assert calls == [5]


def test_registration_batch_hides_partial_replacement_inventory_from_imports(
    coordinator: ExactImportCoordinator,
) -> None:
    names = (
        "hcu_coord_test_batch_first",
        "hcu_coord_test_batch_second",
    )
    replacements = {
        name: ModuleType(f"{name}_hcu")
        for name in names
    }
    for replacement in replacements.values():
        replacement.VALUE = "hcu"

    probe = _BatchProbeFinder(names)
    official = _CountingOfficialFinder(names)
    loaded: dict[str, ModuleType] = {}
    errors: list[BaseException] = []
    threads: list[threading.Thread] = []
    try:
        with coordinator.registration_batch():
            coordinator.install()
            # Nested batches are required by platform -> module-exchange
            # composition and must retain the outer fence.
            with coordinator.registration_batch():
                coordinator.register_replacement(
                    "batch.first",
                    names[0],
                    replacements[names[0]],
                )

            coordinator_index = sys.meta_path.index(coordinator)
            sys.meta_path.insert(coordinator_index, probe)
            sys.meta_path.insert(coordinator_index + 2, official)
            # CPython serializes portions of finder traversal, so one import
            # thread is sufficient and deterministic: it targets the alias
            # that has deliberately not been registered yet.
            threads = [
                threading.Thread(
                    target=_thread_import,
                    args=(names[1], loaded, errors),
                )
            ]
            for thread in threads:
                thread.start()
            assert probe.reached[names[1]].wait(timeout=5)
            assert all(thread.is_alive() for thread in threads)

            # The second alias is intentionally registered only after both
            # imports have reached the coordinator fence.  Neither thread may
            # continue through the partially visible inventory.
            coordinator.register_replacement(
                "batch.second",
                names[1],
                replacements[names[1]],
            )

        for thread in threads:
            thread.join(timeout=5)
        assert all(not thread.is_alive() for thread in threads)
        assert errors == []
        loaded[names[0]] = importlib.import_module(names[0])
        assert loaded == replacements
        assert official.consulted == []
        assert official.executed == []
        assert {
            item.patch_id: item.status
            for item in coordinator.registrations()
        } == {
            "batch.first": PatchStatus.APPLIED.value,
            "batch.second": PatchStatus.APPLIED.value,
        }
    finally:
        while probe in sys.meta_path:
            sys.meta_path.remove(probe)
        while official in sys.meta_path:
            sys.meta_path.remove(official)
        for thread in threads:
            thread.join(timeout=5)


def test_registration_batch_exception_releases_import_fence(
    coordinator: ExactImportCoordinator,
) -> None:
    name = "hcu_coord_test_batch_exception"
    probe = _BatchProbeFinder((name,))
    official = _CountingOfficialFinder((name,))
    loaded: dict[str, ModuleType] = {}
    errors: list[BaseException] = []
    thread: threading.Thread | None = None
    try:
        with pytest.raises(RuntimeError, match="batch setup failed"):
            with coordinator.registration_batch():
                coordinator.install()
                coordinator_index = sys.meta_path.index(coordinator)
                sys.meta_path.insert(coordinator_index, probe)
                sys.meta_path.insert(coordinator_index + 2, official)
                thread = threading.Thread(
                    target=_thread_import,
                    args=(name, loaded, errors),
                )
                thread.start()
                assert probe.reached[name].wait(timeout=5)
                assert thread.is_alive()
                raise RuntimeError("batch setup failed")

        assert thread is not None
        thread.join(timeout=5)
        assert not thread.is_alive()
        assert errors == []
        assert loaded[name].VALUE == "official"
        assert official.consulted == [name]
        assert official.executed == [name]

        # The RLock was released, not merely bypassed by the import thread.
        with coordinator.registration_batch():
            pass
    finally:
        while probe in sys.meta_path:
            sys.meta_path.remove(probe)
        while official in sys.meta_path:
            sys.meta_path.remove(official)
        if thread is not None:
            thread.join(timeout=5)


def test_callback_failure_is_latched_and_blocks_fallback_without_retry(
    coordinator: ExactImportCoordinator,
    module_path: Path,
) -> None:
    name = "hcu_coord_test_failed_callback"
    _write_module(module_path, name, "READY = True\n")
    calls = 0

    def fail(module: ModuleType) -> None:
        nonlocal calls
        assert module.READY
        calls += 1
        raise RuntimeError("callback failed")

    coordinator.register_callback("failed.callback", name, fail)
    coordinator.install()
    with pytest.raises(RuntimeError, match="callback failed"):
        importlib.import_module(name)
    assert name not in sys.modules
    with pytest.raises(LatchedPatchError, match="import blocked"):
        importlib.import_module(name)
    assert calls == 1


def test_feature_on_callback_import_failure_preserves_truthful_report(
    coordinator: ExactImportCoordinator,
    module_path: Path,
) -> None:
    name = "hcu_coord_test_feature_on_failed_callback"
    target = f"{name}.RequiredBackend.initialize"
    patch_id = "feature.on.failed.callback"
    _write_module(module_path, name, "READY = True\n")
    coordinator._registry.set_process_role(ProcessRole.WORKER)

    def fail(module: ModuleType) -> None:
        assert module.READY is True
        raise RuntimeError("requested backend target is incompatible")

    coordinator.register_callback(
        patch_id,
        name,
        fail,
        targets=target,
        feature_enabled=True,
    )
    coordinator.install()
    with pytest.raises(RuntimeError, match="requested backend target"):
        importlib.import_module(name)

    report = coordinator._registry.report()
    patch = report["patches"][patch_id]
    assert report["pid"] == os.getpid()
    assert report["process_role"] == "Worker"
    assert patch["pid"] == os.getpid()
    assert patch["process_role"] == "Worker"
    assert patch["status"] == "failed"
    assert patch["targets"] == [target]
    assert patch["failure_reason"] == (
        "RuntimeError: requested backend target is incompatible"
    )
    assert patch["feature_enabled"] is True
    assert isinstance(patch["updated_at"], float)


def test_optional_unavailable_callback_is_reported_skipped_without_breaking_import(
    coordinator: ExactImportCoordinator,
    module_path: Path,
) -> None:
    name = "hcu_coord_test_optional_unavailable"
    _write_module(module_path, name, "VALUE = 'official'\n")
    calls = 0

    def probe(module: ModuleType) -> None:
        nonlocal calls
        assert module.VALUE == "official"
        calls += 1
        raise OptionalPatchUnavailable("RCCL ncclAllToAll is unavailable")

    registration = coordinator.register_callback(
        "optional.callback", name, probe, feature_enabled=False
    )
    assert registration.status == "armed"
    assert registration.feature_enabled is False
    coordinator.install()
    module = importlib.import_module(name)
    assert module.VALUE == "official"
    assert calls == 1
    registration = coordinator.registrations()[0]
    assert registration.status == "skipped"
    assert "ncclAllToAll" in (
        coordinator._registry.get("optional.callback").failure_reason or ""
    )

    # Config may arrive after the capability probe; the report must reveal
    # that a user subsequently requested the unavailable optional feature.
    updated = coordinator.set_feature_enabled("optional.callback", True)
    assert updated.feature_enabled is True
    assert coordinator._registry.get("optional.callback").feature_enabled is True
    assert importlib.import_module(name) is module
    assert calls == 1


def test_replacement_preempts_canonical_module_and_is_reused(
    coordinator: ExactImportCoordinator,
    module_path: Path,
) -> None:
    canonical = "hcu_coord_test_canonical"
    replacement = "hcu_coord_test_replacement"
    _write_module(module_path, canonical, "raise AssertionError('canonical executed')\n")
    _write_module(module_path, replacement, "VALUE = 'hcu'\n")
    coordinator.register_replacement("module.swap", canonical, replacement)
    coordinator.install()

    first = importlib.import_module(canonical)
    assert first is sys.modules[replacement]
    assert first.VALUE == "hcu"
    assert coordinator.registrations()[0].status == "applied"

    with pytest.raises(ModuleReloadBlockedError, match="custom op twice"):
        importlib.reload(first)

    # Evicting the alias simulates a fresh import/reload boundary.  The cached
    # replacement is reused; neither implementation is executed a second time.
    sys.modules.pop(canonical)
    second = importlib.import_module(canonical)
    assert second is first
    assert coordinator.registrations()[0].status == "applied"


def test_replacement_factory_failure_is_latched_and_not_retried(
    coordinator: ExactImportCoordinator,
    module_path: Path,
) -> None:
    canonical = "hcu_coord_test_factory_failure"
    _write_module(module_path, canonical, "VALUE = 'official'\n")
    calls = 0

    def fail() -> ModuleType:
        nonlocal calls
        calls += 1
        raise RuntimeError("replacement failed")

    coordinator.register_replacement("module.failed_swap", canonical, fail)
    coordinator.install()
    with pytest.raises(RuntimeError, match="replacement failed"):
        importlib.import_module(canonical)
    with pytest.raises(LatchedPatchError, match="import blocked"):
        importlib.import_module(canonical)
    assert calls == 1


def test_replacement_cannot_recursively_load_canonical_module(
    coordinator: ExactImportCoordinator,
    module_path: Path,
) -> None:
    canonical = "hcu_coord_test_recursive_replacement"
    _write_module(module_path, canonical, "VALUE = 'official'\n")

    def recursive_factory() -> ModuleType:
        return importlib.import_module(canonical)

    coordinator.register_replacement("module.recursive", canonical, recursive_factory)
    coordinator.install()
    with pytest.raises(LatchedPatchError, match="re-entrant import"):
        importlib.import_module(canonical)
    record = coordinator.registrations()[0]
    assert record.status == "failed"
    assert canonical not in sys.modules


def test_late_replacement_fails_by_default_and_explicit_swap_can_be_reset(
    coordinator: ExactImportCoordinator,
    module_path: Path,
) -> None:
    canonical = "hcu_coord_test_late"
    replacement = "hcu_coord_test_late_replacement"
    _write_module(module_path, canonical, "VALUE = 'official'\n")
    _write_module(module_path, replacement, "VALUE = 'hcu'\n")
    original = importlib.import_module(canonical)

    with pytest.raises(LateModuleReplacementError, match="already imported"):
        coordinator.register_replacement("late.strict", canonical, replacement)
    assert sys.modules[canonical] is original
    assert coordinator.registrations()[0].status == PatchStatus.FAILED.value

    other = "hcu_coord_test_late_allowed"
    _write_module(module_path, other, "VALUE = 'official-2'\n")
    original_other = importlib.import_module(other)
    coordinator.register_replacement(
        "late.allowed",
        other,
        replacement,
        late_policy="replace",
    )
    assert sys.modules[other].VALUE == "hcu"
    coordinator.reset_for_tests()
    assert sys.modules[other] is original_other


@pytest.mark.parametrize("name", [".relative", "bad.*", "bad-name", "trailing."])
def test_invalid_or_non_exact_names_are_rejected(
    coordinator: ExactImportCoordinator,
    name: str,
) -> None:
    with pytest.raises(ValueError):
        coordinator.register_callback("invalid", name, lambda module: None)


def test_replacement_name_must_differ_from_canonical(
    coordinator: ExactImportCoordinator,
) -> None:
    with pytest.raises(ValueError, match="must differ"):
        coordinator.register_replacement("recursive", "package.module", "package.module")
