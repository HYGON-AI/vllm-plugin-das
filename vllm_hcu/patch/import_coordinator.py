# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Exact-name import coordination for runtime HCU patches.

Unlike the legacy hook, this module never replaces ``builtins.__import__`` and
never scans ancestor module names.  Only explicitly registered full module
names are intercepted.  A callback or replacement is applied at most once;
failures are latched in the process-local patch registry.

Reload is deliberately fail-closed after application: replaying an arbitrary
wrapper is not generally idempotent, while replaying a custom-op module can be
fatal.  A process restart is the supported way to obtain a fresh import graph.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import sys
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from types import ModuleType
from typing import Callable, Literal

from .runtime_state import PATCH_REGISTRY, LatchedPatchError, PatchRegistry, PatchStatus


ModuleCallback = Callable[[ModuleType], None]
ModuleFactory = Callable[[], ModuleType]
ModuleReplacement = str | ModuleType | ModuleFactory
LateReplacementPolicy = Literal["fail", "replace"]


class ImportAction(str, Enum):
    CALLBACK = "callback"
    REPLACEMENT = "replacement"


class LateModuleReplacementError(RuntimeError):
    """Raised when strict module replacement is registered after import."""


class ModuleReloadBlockedError(RuntimeError):
    """Raised when reload could discard or duplicate an applied patch."""


class OptionalPatchUnavailable(RuntimeError):
    """Signal that an optional capability probe did not find its dependency.

    Exact imports must continue with the official module when an optional,
    disabled capability is unavailable.  The coordinator records ``skipped``
    with this reason instead of incorrectly reporting the callback as applied.
    """

    def __init__(self, reason: str) -> None:
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("optional patch reason must be a non-empty string")
        super().__init__(reason.strip())


@dataclass(frozen=True, slots=True)
class ImportRegistration:
    patch_id: str
    module_name: str
    targets: tuple[str, ...]
    action: ImportAction
    status: str
    feature_enabled: bool


@dataclass(slots=True)
class _Entry:
    patch_id: str
    module_name: str
    targets: tuple[str, ...]
    action: ImportAction
    payload: ModuleCallback | ModuleReplacement
    feature_enabled: bool
    late_policy: LateReplacementPolicy = "fail"
    replacement_module: ModuleType | None = None
    previous_module: object | None = None
    previous_replacement_spec: object | None = None
    previous_replacement_loader: object | None = None
    reload_guard_installed: bool = False


_MISSING = object()


class _PostLoadLoader(importlib.abc.Loader):
    def __init__(
        self,
        coordinator: "ExactImportCoordinator",
        fullname: str,
        delegate: importlib.abc.Loader | None,
    ) -> None:
        self._coordinator = coordinator
        self._fullname = fullname
        self._delegate = delegate

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> ModuleType | None:
        if self._delegate is not None and hasattr(self._delegate, "create_module"):
            return self._delegate.create_module(spec)  # type: ignore[no-any-return]
        return None

    def exec_module(self, module: ModuleType) -> None:
        try:
            if self._delegate is not None:
                self._delegate.exec_module(module)
        except BaseException as exc:
            self._coordinator._module_load_failed(self._fullname, exc)
            raise
        self._coordinator._module_loaded(self._fullname, module)

    def __getattr__(self, name: str) -> object:
        # Preserve optional loader APIs such as get_code/get_resource_reader.
        if self._delegate is None:
            raise AttributeError(name)
        return getattr(self._delegate, name)


class _ReplacementLoader(importlib.abc.Loader):
    def __init__(self, coordinator: "ExactImportCoordinator", fullname: str) -> None:
        self._coordinator = coordinator
        self._fullname = fullname

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> ModuleType:
        return self._coordinator._create_replacement(self._fullname)

    def exec_module(self, module: ModuleType) -> None:
        self._coordinator._replacement_loaded(self._fullname, module)


class _BlockedLoader(importlib.abc.Loader):
    def __init__(self, error_type: type[RuntimeError], message: str) -> None:
        self._error_type = error_type
        self._message = message

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> ModuleType:
        raise self._error_type(self._message)

    def exec_module(self, module: ModuleType) -> None:
        # Reload skips create_module and executes into the existing module.
        raise self._error_type(self._message)


class ExactImportCoordinator(importlib.abc.MetaPathFinder):
    """Coordinate callbacks and whole-module replacements by exact name."""

    def __init__(self, *, registry: PatchRegistry = PATCH_REGISTRY) -> None:
        self._registry = registry
        self._lock = threading.RLock()
        self._entries: dict[str, _Entry] = {}
        self._by_module: dict[str, list[str]] = {}
        self._installed = False

    @staticmethod
    def _validate_module_name(module_name: str) -> str:
        if not isinstance(module_name, str) or not module_name:
            raise ValueError("module_name must be a non-empty absolute module name")
        if module_name.startswith(".") or module_name.endswith("."):
            raise ValueError(f"relative or incomplete module name is not allowed: {module_name!r}")
        parts = module_name.split(".")
        if any(not part.isidentifier() for part in parts):
            raise ValueError(f"invalid exact module name: {module_name!r}")
        return module_name

    def install(self) -> None:
        """Install the narrow meta-path finder once.

        Registrations are normally made after installation.  To close the
        race where a target loads between registration and installation, this
        method also processes exact targets already present in ``sys.modules``.
        """

        with self._lock:
            if self in sys.meta_path:
                self._installed = True
                return
            sys.meta_path.insert(0, self)
            self._installed = True
            module_names = tuple(self._by_module)

        for module_name in module_names:
            loaded = sys.modules.get(module_name)
            if not isinstance(loaded, ModuleType) or self._module_is_initializing(
                loaded
            ):
                continue
            with self._lock:
                entries = [
                    self._entries[patch_id]
                    for patch_id in self._by_module.get(module_name, ())
                ]
            replacement = self._active_replacement(entries)
            if replacement is not None:
                if replacement.late_policy == "fail":
                    error = LateModuleReplacementError(
                        f"cannot replace already imported module {module_name!r}; "
                        "the module loaded between HCU registration and coordinator install"
                    )
                    self._registry.mark_failed(replacement.patch_id, error)
                    raise error
                self._replace_loaded_module(replacement, loaded)
            else:
                self._module_loaded(module_name, loaded)

    def uninstall(self) -> None:
        with self._lock:
            while self in sys.meta_path:
                sys.meta_path.remove(self)
            self._installed = False

    @property
    def installed(self) -> bool:
        with self._lock:
            return self._installed and self in sys.meta_path

    @contextmanager
    def registration_batch(self) -> Iterator["ExactImportCoordinator"]:
        """Fence a group of exact registrations from concurrent imports.

        ``find_spec`` uses the same re-entrant lock.  A dispatcher can
        therefore install this finder and register every cold replacement
        while holding one batch: an importing thread either linearizes before
        the batch or waits until the complete inventory is visible.  Nested
        batches and registration calls from the owner thread are supported.

        The fence is always released when registration raises.  Patch records
        already declared before an error retain their normal latched state;
        this context controls import visibility and deliberately does not
        erase diagnostic history.
        """

        with self._lock:
            yield self

    def register_callback(
        self,
        patch_id: str,
        module_name: str,
        callback: ModuleCallback,
        *,
        targets: str | Sequence[str] | None = None,
        feature_enabled: bool = True,
    ) -> ImportRegistration:
        """Run ``callback(module)`` once, immediately if already imported."""

        if not callable(callback):
            raise TypeError("callback must be callable")
        entry = _Entry(
            patch_id=patch_id,
            module_name=self._validate_module_name(module_name),
            targets=self._normalise_targets(module_name, targets),
            action=ImportAction.CALLBACK,
            payload=callback,
            feature_enabled=self._validate_feature_enabled(feature_enabled),
        )
        existing = self._register(entry)
        if existing is not entry:
            self._raise_if_failed(existing)
            loaded = sys.modules.get(existing.module_name)
            if (
                isinstance(loaded, ModuleType)
                and not self._module_is_initializing(loaded)
                and self._is_armed(existing)
            ):
                # A previous registration may have observed the module while
                # its loader was still executing.  Re-entry after import
                # completion is the safe point to run the deferred callback.
                self._apply_callback(existing, loaded)
            return self._view(existing)

        loaded = sys.modules.get(entry.module_name)
        if isinstance(loaded, ModuleType) and not self._module_is_initializing(loaded):
            self._apply_callback(entry, loaded)
        return self._view(entry)

    def register_replacement(
        self,
        patch_id: str,
        module_name: str,
        replacement: ModuleReplacement,
        *,
        targets: str | Sequence[str] | None = None,
        feature_enabled: bool = True,
        late_policy: LateReplacementPolicy = "fail",
    ) -> ImportRegistration:
        """Replace an exact module before its official implementation loads.

        ``late_policy='fail'`` is the safe default for modules that register
        Torch custom ops at import time.  ``late_policy='replace'`` supports an
        explicit late ``sys.modules`` swap for modules known to be side-effect
        free; existing references cannot be retroactively changed.
        """

        normalized_name = self._validate_module_name(module_name)
        self._validate_replacement(replacement)
        if isinstance(replacement, str) and replacement == normalized_name:
            raise ValueError("replacement module must differ from the canonical module name")
        if late_policy not in {"fail", "replace"}:
            raise ValueError("late_policy must be 'fail' or 'replace'")
        entry = _Entry(
            patch_id=patch_id,
            module_name=normalized_name,
            targets=self._normalise_targets(normalized_name, targets),
            action=ImportAction.REPLACEMENT,
            payload=replacement,
            feature_enabled=self._validate_feature_enabled(feature_enabled),
            late_policy=late_policy,
        )
        existing = self._register(entry)
        if existing is not entry:
            self._raise_if_failed(existing)
            return self._view(existing)

        loaded = sys.modules.get(entry.module_name, _MISSING)
        if loaded is not _MISSING:
            if late_policy == "fail":
                error = LateModuleReplacementError(
                    f"cannot replace already imported module {entry.module_name!r}; "
                    "register this replacement during platform plugin initialization"
                )
                self._registry.mark_failed(entry.patch_id, error)
                raise error
            self._replace_loaded_module(entry, loaded)
        return self._view(entry)

    def find_spec(
        self,
        fullname: str,
        path: object = None,
        target: ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        # Import machinery calls this concurrently for unrelated modules.  The
        # exact dictionary lookup ensures ancestors and descendants are ignored.
        with self._lock:
            entries = [self._entries[key] for key in self._by_module.get(fullname, ())]
        if not entries:
            return None

        if target is not None:
            applied_entry = None
            for entry in entries:
                record = self._registry.get(entry.patch_id)
                if record is not None and record.status is PatchStatus.APPLIED:
                    applied_entry = entry
                    break
            if applied_entry is not None:
                entry = applied_entry
                message = (
                    f"reload of {fullname!r} is blocked because HCU patch "
                    f"{entry.patch_id!r} is already applied; reload could discard "
                    "a wrapper or register a custom op twice"
                )
                return importlib.util.spec_from_loader(
                    fullname,
                    _BlockedLoader(ModuleReloadBlockedError, message),
                    origin=f"vllm-hcu reload guard:{entry.patch_id}",
                )

        for entry in entries:
            record = self._registry.get(entry.patch_id)
            if record is not None and record.status is PatchStatus.APPLYING:
                # Most importantly, prevent a replacement factory from
                # recursively importing the canonical module and loading both
                # custom-op implementations in the same process.
                return importlib.util.spec_from_loader(
                    fullname,
                    _BlockedLoader(
                        LatchedPatchError,
                        f"re-entrant import of {fullname!r} blocked while HCU patch "
                        f"{entry.patch_id!r} is applying",
                    ),
                    origin=f"vllm-hcu applying patch:{entry.patch_id}",
                )

        for entry in entries:
            record = self._registry.get(entry.patch_id)
            if record is not None and record.status is PatchStatus.FAILED:
                # Do not silently fall back to the canonical implementation and
                # do not rerun a failed callback/replacement.
                return importlib.util.spec_from_loader(
                    fullname,
                    _BlockedLoader(
                        LatchedPatchError,
                        f"import blocked by failed HCU patch {entry.patch_id!r}: "
                        f"{record.failure_reason or 'unknown failure'}",
                    ),
                    origin=f"vllm-hcu failed patch:{entry.patch_id}",
                )

        replacement = self._active_replacement(entries)
        if replacement is not None:
            return importlib.util.spec_from_loader(
                fullname,
                _ReplacementLoader(self, fullname),
                origin=f"vllm-hcu replacement:{replacement.patch_id}",
            )

        if not any(
            self._is_armed(entry)
            for entry in entries
            if entry.action is ImportAction.CALLBACK
        ):
            return None
        spec = self._find_delegate_spec(fullname, path, target)
        if spec is None:
            return None
        spec.loader = _PostLoadLoader(self, fullname, spec.loader)
        return spec

    def registrations(self) -> tuple[ImportRegistration, ...]:
        with self._lock:
            entries = tuple(self._entries[key] for key in sorted(self._entries))
        return tuple(self._view(entry) for entry in entries)

    def drain_ready_callbacks(self) -> tuple[ImportRegistration, ...]:
        """Apply armed callbacks whose target imports have now completed.

        A plugin can be discovered recursively from inside one of its target
        modules.  Such a module is already present in ``sys.modules`` but its
        public classes/functions are not necessarily defined yet.  Applying a
        post-import adapter at that point produces a false compatibility
        failure.  Registrations therefore remain armed while
        ``ModuleSpec._initializing`` is true and are drained at the next safe
        plugin/lifecycle boundary.
        """

        with self._lock:
            entries = tuple(
                entry
                for entry in self._entries.values()
                if entry.action is ImportAction.CALLBACK
            )
        for entry in entries:
            self._raise_if_failed(entry)
            if not self._is_armed(entry):
                continue
            loaded = sys.modules.get(entry.module_name)
            if not isinstance(loaded, ModuleType):
                continue
            if self._module_is_initializing(loaded):
                continue
            self._apply_callback(entry, loaded)
        return self.registrations()

    def set_feature_enabled(
        self, patch_id: str, enabled: bool
    ) -> ImportRegistration:
        """Update a registration after process-local config is available."""

        enabled = self._validate_feature_enabled(enabled)
        with self._lock:
            try:
                entry = self._entries[patch_id]
            except KeyError as exc:
                raise KeyError(f"unknown import registration {patch_id!r}") from exc
            entry.feature_enabled = enabled
            self._registry.set_feature_enabled(patch_id, enabled)
            return self._view(entry)

    def reset_for_tests(self, *, reset_registry: bool = True) -> None:
        """Uninstall, restore module aliases, and clear registrations."""

        self.uninstall()
        with self._lock:
            replacements = [
                entry
                for entry in self._entries.values()
                if entry.action is ImportAction.REPLACEMENT
            ]
            self._entries.clear()
            self._by_module.clear()
        for entry in replacements:
            if entry.replacement_module is None:
                continue
            current = sys.modules.get(entry.module_name, _MISSING)
            if current is entry.replacement_module:
                if entry.previous_module is _MISSING or entry.previous_module is None:
                    sys.modules.pop(entry.module_name, None)
                else:
                    sys.modules[entry.module_name] = (  # type: ignore[assignment]
                        entry.previous_module
                    )
            if entry.reload_guard_installed:
                entry.replacement_module.__spec__ = (  # type: ignore[assignment]
                    entry.previous_replacement_spec
                )
                entry.replacement_module.__loader__ = (  # type: ignore[assignment]
                    entry.previous_replacement_loader
                )
        if reset_registry:
            self._registry.reset_for_tests()

    def _register(self, entry: _Entry) -> _Entry:
        with self._lock:
            existing = self._entries.get(entry.patch_id)
            if existing is not None:
                if not self._same_registration(existing, entry):
                    raise ValueError(
                        f"conflicting import registration for patch {entry.patch_id!r}"
                    )
                self._raise_if_failed(existing)
                if existing.feature_enabled != entry.feature_enabled:
                    existing.feature_enabled = entry.feature_enabled
                    self._registry.set_feature_enabled(
                        existing.patch_id, entry.feature_enabled
                    )
                return existing
            if entry.action is ImportAction.REPLACEMENT:
                for patch_id in self._by_module.get(entry.module_name, ()):
                    other = self._entries[patch_id]
                    if other.action is ImportAction.REPLACEMENT:
                        raise ValueError(
                            f"module {entry.module_name!r} already has replacement "
                            f"{other.patch_id!r}"
                        )
            self._registry.declare(entry.patch_id, entry.targets)
            self._registry.set_feature_enabled(
                entry.patch_id, entry.feature_enabled
            )
            self._entries[entry.patch_id] = entry
            self._by_module.setdefault(entry.module_name, []).append(entry.patch_id)
            return entry

    def _raise_if_failed(self, entry: _Entry) -> None:
        record = self._registry.get(entry.patch_id)
        if record is not None and record.status is PatchStatus.FAILED:
            raise LatchedPatchError(
                f"patch {entry.patch_id!r} previously failed: "
                f"{record.failure_reason or 'unknown failure'}"
            )

    @staticmethod
    def _module_is_initializing(module: ModuleType) -> bool:
        spec = getattr(module, "__spec__", None)
        return bool(getattr(spec, "_initializing", False))

    @staticmethod
    def _same_registration(left: _Entry, right: _Entry) -> bool:
        return (
            left.module_name == right.module_name
            and left.targets == right.targets
            and left.action is right.action
            and left.payload is right.payload
            and left.late_policy == right.late_policy
        )

    def _active_replacement(self, entries: list[_Entry]) -> _Entry | None:
        for entry in entries:
            if entry.action is not ImportAction.REPLACEMENT:
                continue
            record = self._registry.get(entry.patch_id)
            # An applied replacement remains authoritative if its alias was
            # manually evicted.  A failed replacement must never be retried.
            if record is not None and record.status in {PatchStatus.ARMED, PatchStatus.APPLIED}:
                return entry
        return None

    def _is_armed(self, entry: _Entry) -> bool:
        record = self._registry.get(entry.patch_id)
        return record is not None and record.status is PatchStatus.ARMED

    def _find_delegate_spec(
        self,
        fullname: str,
        path: object,
        target: ModuleType | None,
    ) -> importlib.machinery.ModuleSpec | None:
        # Calling importlib.util.find_spec here would recurse into this finder.
        # Delegate directly through the remaining meta-path chain instead.
        for finder in tuple(sys.meta_path):
            if finder is self:
                continue
            find_spec = getattr(finder, "find_spec", None)
            if find_spec is None:
                continue
            spec = find_spec(fullname, path, target)
            if spec is not None:
                return spec
        return None

    def _module_loaded(self, fullname: str, module: ModuleType) -> None:
        with self._lock:
            entries = [self._entries[key] for key in self._by_module.get(fullname, ())]
        first_error: BaseException | None = None
        for entry in entries:
            if entry.action is not ImportAction.CALLBACK or not self._is_armed(entry):
                continue
            try:
                self._apply_callback(entry, module)
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error

    def _module_load_failed(self, fullname: str, error: BaseException) -> None:
        with self._lock:
            entries = [self._entries[key] for key in self._by_module.get(fullname, ())]
        for entry in entries:
            if entry.action is ImportAction.CALLBACK and self._is_armed(entry):
                self._registry.mark_failed(entry.patch_id, error)

    def _apply_callback(self, entry: _Entry, module: ModuleType) -> None:
        if not self._registry.begin(entry.patch_id):
            return
        callback = entry.payload
        assert callable(callback)
        try:
            callback(module)
        except OptionalPatchUnavailable as exc:
            self._registry.mark_skipped(entry.patch_id, str(exc))
            return
        except BaseException as exc:
            self._registry.mark_failed(entry.patch_id, exc)
            raise
        self._registry.mark_applied(
            entry.patch_id,
            feature_enabled=entry.feature_enabled,
        )

    def _create_replacement(self, fullname: str) -> ModuleType:
        entry = self._replacement_for(fullname)
        record = self._registry.get(entry.patch_id)
        if record is not None and record.status is PatchStatus.APPLIED:
            if entry.replacement_module is None:
                error = RuntimeError(
                    f"replacement {entry.patch_id!r} is applied but its module reference is missing"
                )
                raise LatchedPatchError(str(error))
            return entry.replacement_module

        self._registry.begin(entry.patch_id)
        try:
            module = self._resolve_replacement(entry.payload)
        except BaseException as exc:
            self._registry.mark_failed(entry.patch_id, exc)
            raise
        entry.previous_module = _MISSING
        entry.replacement_module = module
        return module

    def _replacement_loaded(self, fullname: str, module: ModuleType) -> None:
        entry = self._replacement_for(fullname)
        record = self._registry.get(entry.patch_id)
        if record is not None and record.status is PatchStatus.APPLYING:
            self._registry.mark_applied(
                entry.patch_id,
                feature_enabled=entry.feature_enabled,
            )
        self._install_reload_guard(entry, module)
        self._module_loaded(fullname, module)

    def _replace_loaded_module(self, entry: _Entry, previous: object) -> None:
        if not self._registry.begin(entry.patch_id):
            return
        try:
            replacement = self._resolve_replacement(entry.payload)
            with self._lock:
                entry.previous_module = previous
                entry.replacement_module = replacement
                sys.modules[entry.module_name] = replacement
        except BaseException as exc:
            self._registry.mark_failed(entry.patch_id, exc)
            raise
        self._registry.mark_applied(
            entry.patch_id,
            feature_enabled=entry.feature_enabled,
        )
        self._install_reload_guard(entry, replacement)
        self._module_loaded(entry.module_name, replacement)

    def _install_reload_guard(self, entry: _Entry, module: ModuleType) -> None:
        if not entry.reload_guard_installed:
            entry.previous_replacement_spec = getattr(module, "__spec__", None)
            entry.previous_replacement_loader = getattr(module, "__loader__", None)
            entry.reload_guard_installed = True
        guarded_loader = _ReplacementLoader(self, entry.module_name)
        module.__spec__ = importlib.util.spec_from_loader(
            entry.module_name,
            guarded_loader,
            origin=f"vllm-hcu replacement:{entry.patch_id}",
            is_package=hasattr(module, "__path__"),
        )
        module.__loader__ = guarded_loader

    def _replacement_for(self, fullname: str) -> _Entry:
        with self._lock:
            for patch_id in self._by_module.get(fullname, ()):
                entry = self._entries[patch_id]
                if entry.action is ImportAction.REPLACEMENT:
                    return entry
        raise ImportError(f"no HCU replacement registered for {fullname!r}")

    @staticmethod
    def _resolve_replacement(replacement: ModuleReplacement) -> ModuleType:
        if isinstance(replacement, ModuleType):
            module = replacement
        elif isinstance(replacement, str):
            module = importlib.import_module(replacement)
        else:
            module = replacement()
        if not isinstance(module, ModuleType):
            raise TypeError(
                "module replacement must resolve to ModuleType, "
                f"got {type(module).__name__}"
            )
        return module

    @staticmethod
    def _normalise_targets(
        module_name: str,
        targets: str | Sequence[str] | None,
    ) -> tuple[str, ...]:
        if targets is None:
            values = (module_name,)
        elif isinstance(targets, str):
            values = (targets,)
        else:
            values = tuple(targets)
        if not values or any(
            not isinstance(target, str) or not target.strip()
            for target in values
        ):
            raise ValueError("targets must contain at least one non-empty symbol name")
        return tuple(target.strip() for target in values)

    @staticmethod
    def _validate_replacement(replacement: ModuleReplacement) -> None:
        if isinstance(replacement, str):
            ExactImportCoordinator._validate_module_name(replacement)
            return
        if not isinstance(replacement, ModuleType) and not callable(replacement):
            raise TypeError("replacement must be an absolute module name, ModuleType, or factory")

    @staticmethod
    def _validate_feature_enabled(value: bool) -> bool:
        if not isinstance(value, bool):
            raise TypeError("feature_enabled must be bool")
        return value

    def _view(self, entry: _Entry) -> ImportRegistration:
        record = self._registry.get(entry.patch_id)
        status = record.status.value if record is not None else "unknown"
        enabled = record.feature_enabled if record is not None else False
        return ImportRegistration(
            patch_id=entry.patch_id,
            module_name=entry.module_name,
            targets=entry.targets,
            action=entry.action,
            status=status,
            feature_enabled=enabled,
        )


IMPORT_COORDINATOR = ExactImportCoordinator()


def install_import_coordinator() -> None:
    IMPORT_COORDINATOR.install()


def uninstall_import_coordinator() -> None:
    IMPORT_COORDINATOR.uninstall()


def register_post_import(
    patch_id: str,
    module_name: str,
    callback: ModuleCallback,
    *,
    targets: str | Sequence[str] | None = None,
    feature_enabled: bool = True,
) -> ImportRegistration:
    return IMPORT_COORDINATOR.register_callback(
        patch_id,
        module_name,
        callback,
        targets=targets,
        feature_enabled=feature_enabled,
    )


def register_module_replacement(
    patch_id: str,
    module_name: str,
    replacement: ModuleReplacement,
    *,
    targets: str | Sequence[str] | None = None,
    feature_enabled: bool = True,
    late_policy: LateReplacementPolicy = "fail",
) -> ImportRegistration:
    return IMPORT_COORDINATOR.register_replacement(
        patch_id,
        module_name,
        replacement,
        targets=targets,
        feature_enabled=feature_enabled,
        late_policy=late_policy,
    )


__all__ = [
    "IMPORT_COORDINATOR",
    "ExactImportCoordinator",
    "ImportAction",
    "ImportRegistration",
    "LateModuleReplacementError",
    "ModuleReloadBlockedError",
    "OptionalPatchUnavailable",
    "install_import_coordinator",
    "register_module_replacement",
    "register_post_import",
    "uninstall_import_coordinator",
]
