# SPDX-License-Identifier: Apache-2.0
"""Explicit, process-owned publication of the five temporary GDN aliases."""

from __future__ import annotations

import hashlib
import inspect
import os
from contextlib import contextmanager
from dataclasses import dataclass, replace
from importlib import import_module
from pathlib import Path
from threading import RLock

from hcu_adapter.extensions import ExtensionCommitError, ExtensionSpec

from .environment import gdn_policy
from .extension import _require_vllm_033_release_line
from .gdn import (
    DECODE,
    PREFILL,
    boltops_contract,
    make_wrapper,
    prepare_routes,
    require_signature,
)
from .gdn_contracts import BOLTOPS_IMPLEMENTATIONS, OFFICIAL, OWNERS

CAPABILITY = "hcu-gdn-fla"
FACTOR = "VLLM_HCU_DAS_GDN_PROVIDER_IDENTITY"
CHUNK = "vllm.third_party.flash_linear_attention.ops.chunk"
QWEN = "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn"
_lock = RLock()
_published = None
_failure = None
_publishing = False


@dataclass(frozen=True)
class Publication:
    pid: int
    policy: tuple
    bindings: tuple
    providers: tuple
    envs: object
    files: tuple
    identity: str


def _fatal(reason, error=None):
    global _failure
    # Latch before invoking arbitrary exception formatting or user cleanup.
    if _failure is None:
        _failure = reason
    raise ExtensionCommitError(_failure) from error


def _after_fork_child():
    global _lock, _failure
    _lock = RLock()
    if _publishing and _failure is None:
        _failure = "fork inherited incomplete GDN preparation/publication"


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork_child)


@contextmanager
def _attempt():
    global _publishing
    if _failure is not None:
        _fatal(_failure)
    if _publishing:
        _fatal("recursive GDN activation")
    _publishing = True
    try:
        yield
    finally:
        _publishing = False


def _owned(p):
    return (
        gdn_policy() == p.policy
        and import_module("vllm.envs") is p.envs
        and p.envs.environment_variables.get(FACTOR) is compile_identity
        and all(
            import_module(module.__name__) is module
            and getattr(module, name) is wrapper
            and import_module(OWNERS[name]) is owner
            and getattr(owner, name) is original
            for module, name, owner, original, wrapper in p.bindings
        )
        and all(
            import_module(module.__name__) is module
            and getattr(module, name) is function
            for module, name, function in p.providers
        )
    )


def check_health():
    """Check published owners without importing the Adapter host or repairing."""
    with _lock:
        if _failure is not None:
            _fatal(_failure)
        if _publishing:
            _fatal("GDN consumer re-entered incomplete publication")
        p = _published
        if p is None:
            raise RuntimeError("GDN provider is not published")
        if p.pid != os.getpid():
            raise RuntimeError("GDN requires current-process activation after fork")
        try:
            valid = _owned(p)
        except BaseException as error:
            _fatal("GDN ownership check failed", error)
        if not valid:
            _fatal("GDN provider, policy or alias owner changed; restart the process")


def compile_identity():
    """Effective routes and source bytes enter both official compilation keys.

    This is source identity, not numerical qualification or an exact native
    wheel lock. Providers retain ownership of their native/JIT cache keys.
    """
    check_health()
    try:
        for filename, expected in _published.files:
            if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != expected:
                _fatal("GDN provider source changed; restart the process")
    except BaseException as error:
        _fatal("GDN provider source identity check failed", error)
    return _published.identity


def _provider_owners(routes):
    owners = {}
    for name, candidates in routes.items():
        for provider, module, attribute, function in candidates:
            owners[module, attribute] = function
            if (
                provider == "hcu-boltops"
                and module.__name__ == "boltops.fla.gdn"
                and function.__module__ == module.__name__
            ):
                # Lazy public shims import these implementations on each call.
                # Track their owners and bytes before creating compile keys.
                for suffix, impl_name in BOLTOPS_IMPLEMENTATIONS[name]:
                    implementation = import_module(module.__name__ + "." + suffix)
                    leaf = getattr(implementation, impl_name)
                    require_signature(
                        leaf,
                        boltops_contract(name, leaf),
                        implementation.__name__ + "." + impl_name,
                    )
                    owners[implementation, impl_name] = leaf
    return tuple(
        (module, name, function) for (module, name), function in owners.items()
    )


def _source_identity(routes, policy, providers):
    paths = {
        Path(__file__),
        Path(__file__).with_name("gdn.py"),
        Path(__file__).with_name("gdn_contracts.py"),
        Path(__file__).with_name("environment.py"),
    }
    for module, _, function in providers:
        for filename in (
            getattr(module, "__file__", None),
            inspect.getsourcefile(function),
        ):
            if filename:
                paths.add(Path(filename))
    files = tuple(
        (str(p.resolve()), hashlib.sha256(p.read_bytes()).hexdigest())
        for p in sorted(paths)
    )
    # Paths aid drift detection, but are deliberately absent from the key.
    selection = tuple(
        (name, tuple(entry[0] for entry in entries)) for name, entries in routes.items()
    )
    identity = hashlib.sha256(
        repr((policy, selection, tuple(digest for _, digest in files))).encode()
    ).hexdigest()
    return files, identity


def activate(context):
    global _published
    with _lock, _attempt():
        attempt_pid = os.getpid()
        try:
            _require_vllm_033_release_line(context.vllm_version)
            if CAPABILITY not in context.requested_capabilities:
                raise RuntimeError("GDN requires explicit hcu-gdn-fla capability")
            policy = gdn_policy()
        except BaseException as error:
            if _published is not None:
                _fatal("published GDN activation policy changed", error)
            raise
        if _published is not None:
            try:
                valid = _owned(_published)
            except BaseException as error:
                _fatal("published GDN owners unavailable", error)
            if not valid:
                _fatal("published GDN owner changed; no automatic recovery")
            # Inherited complete aliases need fresh Adapter runtime admission.
            _published = replace(_published, pid=os.getpid())
            return

        bindings = []
        for module_name, names in ((CHUNK, PREFILL), (QWEN, DECODE)):
            module = import_module(module_name)
            for name in names:
                owner = import_module(OWNERS[name])
                original = getattr(module, name)
                if original is not getattr(owner, name):
                    raise RuntimeError(
                        "GDN consumer already has another owner: " + name
                    )
                require_signature(original, OFFICIAL[name], module_name + "." + name)
                bindings.append((module, name, owner, original))
        envs = import_module("vllm.envs")
        if not isinstance(
            getattr(envs, "environment_variables", None), dict
        ) or not callable(getattr(envs, "compile_factors", None)):
            raise RuntimeError(
                "official GDN compilation environment registry unavailable"
            )
        if FACTOR in envs.environment_variables:
            raise RuntimeError("GDN compilation factor already has another owner")
        routes = prepare_routes(policy)
        providers = _provider_owners(routes)
        files, identity = _source_identity(routes, policy, providers)
        publication = Publication(
            os.getpid(),
            policy,
            tuple(
                (
                    *binding,
                    make_wrapper(
                        binding[1], binding[3], routes[binding[1]], check_health
                    ),
                )
                for binding in bindings
            ),
            providers,
            envs,
            files,
            identity,
        )
        # Validate again after optional imports; no alias has changed yet.
        if _failure is not None or os.getpid() != attempt_pid:
            _fatal("GDN preparation was interrupted by reentry or fork")
        if (
            gdn_policy() != policy
            or FACTOR in envs.environment_variables
            or any(
                getattr(module, name) is not original
                or getattr(owner, name) is not original
                for module, name, owner, original in bindings
            )
        ):
            raise RuntimeError("GDN owner/policy changed during preparation")
        try:
            # Consumers cannot execute a partial commit: every wrapper checks
            # the publishing latch before invoking even an official fallback.
            envs.environment_variables[FACTOR] = compile_identity
            for module, name, _, _, wrapper in publication.bindings:
                setattr(module, name, wrapper)
            _published = publication
            if not _owned(publication):
                _fatal("GDN publication verification failed")
        except BaseException as error:
            _fatal("GDN publication failed; restart the process", error)


def extension_spec():
    return ExtensionSpec(
        name="das-hcu-gdn-fla",
        api_version=1,
        priority=70,
        capabilities=frozenset({CAPABILITY}),
        activate=activate,
        check_health=check_health,
    )
