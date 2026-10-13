# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Stamp draft attention metadata as explicitly non-causal.

The wheel (0.25.1) ``MLACommonMetadata`` carries no ``causal`` flag (the fork
carries one, ``mla_attention.py: causal``), so the HCU triton MLA backend
cannot tell a non-causal DSpark draft block (``num_query_per_req > 1``, every
query row reads the whole committed KV prefix) from a causal speculative
verify batch (same uniform row shape, but query row ``i`` must NOT see rows
committed after its own position).  Keying the backend flattening on row
counts alone poisons causal verify batches (review 2026-10-12, P1: early
queries read future KV -> wrong logits for any model, not just the draft).

This patch wraps the spec-side draft metadata producer
(``Speculator._build_draft_attn_metadata`` and every subclass override in the
spec_decode package) and stamps per-group metadata right where the fork
passes ``causal=False`` (fork 877e6996).  The backend flatten then keys on
this explicit origin stamp instead of row counts; unstamped multi-token
batches keep exact causal semantics via per-row causal lengths (always
correct, only slower).

Producer shapes handled (fork-shaped, review rounds 2-4):
- ``causal`` is resolved through ``signature.bind`` + ``apply_defaults`` so
  per-class defaults (base ``True``, ``DFlash`` subclass ``False``), kw-only
  and ``*args``-shifted call shapes all resolve correctly; a CAUSAL producer
  stays UNSTAMPED.  Truthiness matches the fork (`if not layer_causal`):
  config values pass through verbatim, so round-4 N-4 stamps every FALSY
  value (False, 0, np.False_) and leaves truthy ones unstamped; ambiguous
  truthiness (a multi-element tensor raises on bool()) stays unstamped too
  -- the safe direction for that shape.
``causal`` may be a ``Mapping[gid, bool]`` (``DFlashSpeculator._group_causal``
when the model declares per-group causality, fork speculator.py:196-215).
Group ids are ints while the RESULT mapping is keyed by layer name
(attn_utils.py:578): round-3 finding A.  The trusted alignment maps gid ->
layer names through ``kv_cache_config.kv_cache_groups[gid].layer_names``
(propose path: the speculator's own ``kv_cache_config`` attribute; capture
path: the helper's ``kv_cache_config`` input) and stamps exactly the FALSY
groups' layers; a mapped group with NO metadata in the result is unaudited
and raises (round-4 N-2).  Without a resolvable ``kv_cache_config``,
disjoint key spaces are UNALIGNABLE and refused: an all-falsy map blankets
(assumption N-3: the fork always declares the FULL group map -- a partial
map must resolve through kv_cache_config, blanket-stamping an undeclared
group would contradict its causal default), any causal group present raises
loudly -- a silent skip on an unaudited shape was round-2 finding N2.  The
FULL cudagraph capture path
(``dflash/cudagraph.py: _prepare_dflash_inputs_to_capture``) calls the base
builder directly and returns an ``AttentionState(attn_metadata=..., ...)``;
that module is a SECOND, optional target: present -> its helper is wrapped
and only the ``attn_metadata`` mapping's values are stamped (never
``slot_mappings``); absent -> skipped; present but shapeless -> fail loudly.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Mapping
from types import ModuleType

from ._common import (
    PatchCompatibilityError,
    already_applied,
    load_exact_module,
)

TARGET_MODULE = "vllm.v1.worker.gpu.spec_decode.speculator"
PATCH_ID = "worker.framework_opt.spec_decode.non_causal_stamp"
# Declared target: the draft metadata producer on the shared speculator base.
TARGETS = (f"{TARGET_MODULE}.DraftModelSpeculator._build_draft_attn_metadata",)
_MARKER = "_vllm_hcu_non_causal_stamp_applied"
_WRAPPER = "_vllm_hcu_non_causal_stamp_wrapper"
_METHOD = "_build_draft_attn_metadata"

# Dependency-free leaf module: importing the backend here would violate the
# worker framework's lazy-arming contract (no patch target module may be
# imported while registering adapters).
from vllm_hcu.v1.attention.backends.mla._stamps import (
    NON_CAUSAL_DRAFT_STAMP as _STAMP,
)


def _is_falsy(value: object) -> bool:
    """Fork-matching truthiness for a ``causal`` flag.

    The fork gates on ``if not layer_causal`` and passes config values
    through verbatim, so ``0`` means NON-causal exactly like ``False``
    (round-4 N-4).  ``None`` (parameter absent) also stamps: only draft
    producers are wrapped.  Ambiguous truthiness -- a multi-element tensor
    raises on bool() -- stays CAUSAL: refusing to flatten an undecidable
    shape keeps per-row causal semantics, which is always correct.
    """

    if value is None or value is False:
        return True
    try:
        return not value
    except Exception:  # noqa: BLE001 - ambiguous (array-like) truthiness
        return False


def _bind_arguments(signature, args, kwargs):
    try:
        binding = signature.bind(*args, **kwargs)
    except TypeError:
        # The call does not fit its own signature; invoking the original
        # raises the same TypeError, so this mapping is never consumed.
        # Return None and let the caller stay correctness-preserving.
        return None
    binding.apply_defaults()
    return binding


def _make_signature_resolver(original, parameter):
    """Resolve one parameter of a call through its real signature.

    Returns ``None`` when the callable lacks the parameter.  The returned
    resolver returns the bound value, or the CONSERVATIVE default when the
    call cannot be bound (``causal`` -> True keeps semantics; unknown
    ``kv_cache_config`` -> None disables group alignment).
    """

    try:
        signature = inspect.signature(original)
    except (TypeError, ValueError):
        return None
    if parameter not in signature.parameters:
        return None

    def resolve(args, kwargs):
        binding = _bind_arguments(signature, args, kwargs)
        if binding is None:
            return True if parameter == "causal" else None
        return binding.arguments.get(parameter)

    return resolve


def _group_layer_names(kv_cache_config, gid):
    """gid -> layer names via ``kv_cache_config.kv_cache_groups[gid]``."""

    groups = getattr(kv_cache_config, "kv_cache_groups", None)
    if groups is None:
        return None
    try:
        group = groups[gid]
    except (KeyError, IndexError, TypeError):
        return None
    names = getattr(group, "layer_names", None)
    if names is None:
        return None
    return list(names)


def _align_group_causal(
    causal_map: Mapping, container: Mapping, kv_cache_config: object
) -> list:
    """Metadata objects of every group declared non-causal (falsy).

    Trusted shape: gid -> layer names through ``kv_cache_config``.  A mapped
    group whose layers are ALL absent from the result is an unaudited shape
    -> raise (round-4 N-2; fork containers always carry at least one draft
    layer per attn group).  Without ``kv_cache_config``, only a SHARED key
    space aligns; disjoint keys are unalignable: all-falsy blankets, any
    truthy group present raises (round-3 finding A).
    """

    false_gids = [gid for gid, flag in causal_map.items() if _is_falsy(flag)]
    true_gids = [gid for gid, flag in causal_map.items() if not _is_falsy(flag)]

    def fallback() -> list:
        # N-3: blanket assumes the map DECLARES EVERY group in the container
        # (fork builds _group_causal from all draft attn groups).  Partial
        # maps must resolve through kv_cache_config instead.
        if false_gids and true_gids:
            raise PatchCompatibilityError(
                "mixed per-group causal map cannot be aligned onto the draft "
                "metadata result; extend patch_dspark_non_causal_stamp"
            )
        if false_gids:
            return list(container.values())
        return []

    if kv_cache_config is not None:
        selected: list = []
        seen: set = set()
        for gid in false_gids:
            names = _group_layer_names(kv_cache_config, gid)
            if names is None:
                # gid unmappable: same refusal rules as the bare fallback.
                return fallback()
            hits = 0
            for name in names:
                if name in seen:
                    continue
                seen.add(name)
                meta = container.get(name)
                if meta is not None:
                    selected.append(meta)
                    hits += 1
            if hits == 0:
                raise PatchCompatibilityError(
                    f"causal group {gid!r} maps to layers {names} but none of "
                    "them appear in the draft metadata result; the alignment "
                    "is unaudited -- extend patch_dspark_non_causal_stamp"
                )
        return selected

    if set(causal_map.keys()) & set(container.keys()):
        # Defensive shared key space (not the fork shape, but alignable).
        return [
            meta
            for key, meta in container.items()
            if _is_falsy(causal_map.get(key, True))
        ]
    return fallback()


def _stamp_metas_for(
    result: object, causal: object, kv_cache_config: object = None
) -> list:
    """Select the per-group metadata objects eligible for the stamp.

    A truthy scalar ``causal`` means a causal producer -> nothing to stamp;
    falsy (False/0/...) stamps.  ``AttentionState``-shaped returns (capture
    path) are unwrapped to their ``attn_metadata`` mapping only -- field
    ``slot_mappings`` is never touched; ``attn_metadata=None`` legitimately
    stamps nothing.
    """

    if isinstance(causal, Mapping):
        if not causal:
            # No group declared non-causal -> nothing may be stamped.
            return []
    elif not _is_falsy(causal):
        return []

    container = result
    attention = getattr(result, "attn_metadata", None)
    if attention is not None or (
        hasattr(result, "attn_metadata") and hasattr(result, "_fields")
    ):
        # NamedTuple AttentionState (capture path).  ``attn_metadata`` may
        # legitimately be None (no metadata for the captured phase).
        if attention is None:
            return []
        container = attention

    if isinstance(causal, Mapping):
        if isinstance(container, Mapping):
            return _align_group_causal(causal, container, kv_cache_config)
        if all(_is_falsy(value) for value in causal.values()):
            return _flatten_metas(container)
        if not any(_is_falsy(value) for value in causal.values()):
            return []
        raise PatchCompatibilityError(
            "mixed per-group causal map cannot be aligned onto the draft "
            "metadata result; extend patch_dspark_non_causal_stamp"
        )
    return _flatten_metas(container)


def _flatten_metas(container: object) -> list:
    if container is None:
        return []
    if isinstance(container, Mapping):
        return list(container.values())
    if isinstance(container, (tuple, list)):
        return [
            item
            for item in container
            if item is not None and not isinstance(item, (int, float, str, bytes, bool))
        ]
    if isinstance(container, (int, float, str, bytes, bool)):
        return []
    return [container]


def _apply_stamp(
    result: object, causal: object, kv_cache_config: object = None
) -> object:
    for meta in _stamp_metas_for(result, causal, kv_cache_config):
        try:
            setattr(meta, _STAMP, True)  # noqa: B010 - crossing the wheel
        except AttributeError as exc:  # __slots__ without the attr
            raise PatchCompatibilityError(
                f"draft attn metadata {type(meta).__name__} rejects the "
                "non-causal stamp (__slots__); cannot disambiguate "
                "causal verify from non-causal draft blocks"
            ) from exc
    return result


def _self_kv_cache_config(args) -> object:
    return getattr(args[0], "kv_cache_config", None) if args else None


def _wrap_class(cls: type, target: str, require_causal: bool = False) -> bool:
    own = vars(cls).get(_METHOD)
    if own is None:
        return False
    if getattr(own, _WRAPPER, False):
        # Idempotency checks the FUNCTION attribute (vars(cls)), not the
        # class: getattr(cls, ...) would see a user attribute named like the
        # marker and never the method's marker.
        return False
    if not callable(own):
        raise PatchCompatibilityError(f"{target} is not callable")
    original = own
    resolve_causal = _make_signature_resolver(original, "causal")
    if require_causal and resolve_causal is None:
        # Auto-wrapped strangers (hook path / late-arm walk) whose override
        # has NO ``causal`` parameter: their causal semantics are unknown.
        # Defaulting to False would silently stamp CAUSAL verify builds as
        # non-causal drafts and mis-flatten their attention rows (review R3
        # P2 #1).  Leave them unwrapped: an unlabeled builder falls back to
        # the per-row causal handling -- correct, only slower.
        return False

    @functools.wraps(original)
    def hcu_build_draft_attn_metadata(*args, **kwargs):
        causal = (
            resolve_causal(args, kwargs) if resolve_causal is not None else False
        )
        kvcc = _self_kv_cache_config(args)
        return _apply_stamp(original(*args, **kwargs), causal, kvcc)

    setattr(hcu_build_draft_attn_metadata, _WRAPPER, True)
    setattr(cls, "_vllm_hcu_original_build_draft_attn_metadata", original)
    setattr(cls, _METHOD, hcu_build_draft_attn_metadata)
    return True


def apply_to_module(module: ModuleType) -> bool:
    spec = load_exact_module(TARGET_MODULE, module)
    # Wrap every class in the module that defines the producer, so subclass
    # overrides in THIS module cannot shadow the stamp.
    candidates = [
        (name, obj)
        for name, obj in vars(spec).items()
        if isinstance(obj, type) and _METHOD in vars(obj)
    ]
    if not candidates:
        raise PatchCompatibilityError(
            f"no class defines {TARGET_MODULE}.{_METHOD}; cannot install the "
            "non-causal draft stamp"
        )
    wrapped = tuple(
        (cls, _METHOD, f"{TARGET_MODULE}.{name}.{_METHOD}", _WRAPPER)
        for name, cls in candidates
    )
    if already_applied(spec, _MARKER, wrapped):
        return False
    for name, cls in candidates:
        _wrap_class(cls, f"{TARGET_MODULE}.{name}.{_METHOD}")
    _arm_subclass_override_guard(spec)
    _wrap_optional_capture_path()
    setattr(spec, _MARKER, True)
    return True


# USER MODULES (review 2026-10-13 P1 #1): DFlash overrides
# ``_build_draft_attn_metadata`` in its OWN module -- at fork 877e6996
# dflash/speculator.py:276 is the only such override (autoregressive/
# eagle/gemma4/mtp do not override); the base scan above only sees the
# shared base module, so dispatch through a subclass override bypassed the
# stamp entirely.  A factory-setattr fix is NOT viable: model_runner.py
# imports init_speculator at module top, so by the time the patch arms,
# the runner already holds the ORIGINAL function object and re-binding the
# module attribute changes nothing.  This hook instead rides Python
# itself: when the patch's post-import callback fires for the base
# module, every descendant class is created LATER (its module imports the
# base first), and the (implicit-classmethod) hook fires at class-body
# creation.  It is installed under the literal dunder name so subclass
# creation resolves it through the MRO without any conversion; chained
# overrides call super().__init_subclass__ so the guard propagates
# through every inheritance level.  Known limits (accepted, documented):
# an intermediate class defining a custom __init_subclass__ WITHOUT
# calling super() cuts descendants off the hook, and a producer setattr'ed
# onto a class AFTER creation never fires anything.  Overrides without a
# ``causal`` parameter are left unwrapped (unknown semantics must not be
# stamped non-causal by default -- review R3 P2 #1).
_HOOK_NAME = "__init_subclass__"


def _make_override_hook(host: type, previous: object | None):
    """Per-host __init_subclass__ guard (classmethod payload).

    A naive ``super(cls, cls).__init_subclass__`` forward recurses forever
    when the resolved entry is ANOTHER armed host on the same MRO.  The
    forward order instead: (1) the payload UNSEALED at arm time (a
    third-party hook already living in the host dict) keeps the chain as
    the next link -- it must not be silently discarded; (2) with nothing
    pre-existing, walk cls.__mro__ after ``host``, skip armed payloads, and
    invoke the first third-party hook found; (3) fall through to object's.
    """

    def _hcu_override_hook(cls: type, **kwargs: object) -> None:
        own = vars(cls).get(_METHOD)
        if own is not None:
            target = f"{cls.__module__}.{cls.__qualname__}.{_METHOD}"
            # require_causal: auto-wrapped overrides without a ``causal``
            # parameter keep their original (unlabeled) behavior instead of
            # being guessed non-causal -- see _wrap_class (review R3 P2 #1).
            _wrap_class(cls, target, require_causal=True)
        if previous is not None:
            fn = getattr(previous, "__func__", previous)
            if not getattr(fn, _WRAPPER, False):
                fn(cls, **kwargs)
                return
        mro = cls.__mro__
        try:
            start = mro.index(host) + 1
        except ValueError:  # host no longer in the MRO: scan from the top
            start = 0
        for base in mro[start:]:
            entry = vars(base).get(_HOOK_NAME)
            if entry is None:
                continue
            fn = getattr(entry, "__func__", entry)
            if getattr(fn, _WRAPPER, False):
                continue  # another armed host: forward past it
            fn(cls, **kwargs)
            return
        # mro[start:] always contains object, whose ``__init_subclass__``
        # descriptor the loop resolves and calls above; reaching here means
        # nothing was found -- a no-op terminal is correct (review R2).

    setattr(_hcu_override_hook, _WRAPPER, True)
    return _hcu_override_hook


def _arm_subclass_override_guard(spec: ModuleType) -> bool:
    armed_any = False
    roots: list[type] = []
    for obj in vars(spec).values():
        if not isinstance(obj, type) or _METHOD not in vars(obj):
            continue
        current = vars(obj).get(_HOOK_NAME)
        if getattr(getattr(current, "__func__", None), _WRAPPER, False):
            continue  # already armed on this class
        # The conversion to an implicit classmethod happens ONLY for a
        # __init_subclass__ entry in a class NAMESPACE at creation; a
        # setattr'ed plain function would be resolved UNBOUND by subclass
        # creation (TypeError: missing cls).  Store the classmethod form.
        setattr(obj, _HOOK_NAME, classmethod(_make_override_hook(obj, current)))
        armed_any = True
        roots.append(obj)
    # Late arm (review follow-up R2 #1a): the post-import callback ALSO
    # fires immediately when the base module is already imported, and the
    # subclass modules may be imported by then too -- those live overrides
    # predate the hook, so class creation will never re-fire for them and
    # they would stay unguarded forever.  Wrap every descendant override in
    # place.  ``_wrap_class`` is idempotent and ``__subclasses__`` holds
    # only live classes, so the walk is safe to repeat.
    seen = {id(root) for root in roots}
    stack = [child for root in roots for child in root.__subclasses__()]
    while stack:
        cls = stack.pop()
        if id(cls) in seen:
            continue
        seen.add(id(cls))
        if _METHOD in vars(cls):
            # Same require_causal rationale as the hook path above.
            _wrap_class(
                cls, f"{cls.__module__}.{cls.__qualname__}.{_METHOD}",
                require_causal=True,
            )
        stack.extend(cls.__subclasses__())
    return armed_any


# The FULL cudagraph capture path builds draft metadata through the dflash
# capture helper, which calls the base ``build_attn_metadata`` directly and
# never routes through ``_build_draft_attn_metadata`` (fork
# dflash/cudagraph.py:46-88).  Unstamped capture metadata would bake the
# CAUSAL expand into the recorded graph and silently downgrade draft
# semantics on replay.  The module is OPTIONAL: deployments without the
# dflash draft backend legitimately lack it, but a present module MUST
# expose the helper or the deployment shape is unaudited -> fail loudly.
_CAPTURE_MODULE = "vllm.v1.worker.gpu.spec_decode.dflash.cudagraph"
_CAPTURE_HELPER = "_prepare_dflash_inputs_to_capture"


def _wrap_optional_capture_path() -> list[str]:
    import importlib

    try:
        capture_module = importlib.import_module(_CAPTURE_MODULE)
    except ModuleNotFoundError as exc:
        missing = getattr(exc, "name", "") or ""
        # ONLY a genuinely absent module may be skipped; a broken import of
        # an installed module must surface.
        if missing == _CAPTURE_MODULE or _CAPTURE_MODULE.startswith(missing + "."):
            return []
        raise
    helper = getattr(capture_module, _CAPTURE_HELPER, None)
    if helper is None or not callable(helper):
        raise PatchCompatibilityError(
            f"{_CAPTURE_MODULE} exists but lacks {_CAPTURE_HELPER}; the "
            "cudagraph capture shape is unaudited -- extend the stamp patch"
        )
    if getattr(helper, _WRAPPER, False):
        return []
    resolve_causal = _make_signature_resolver(helper, "causal")
    # The helper is a MODULE-LEVEL function: its kv_cache_config comes as an
    # input (fork passes the same object it forwards to build_attn_metadata).
    # There is no `self` to fall back to; a missing parameter simply leaves
    # the stamp on the causal=True-safe path.
    resolve_kvcc = _make_signature_resolver(helper, "kv_cache_config")

    @functools.wraps(helper)
    def hcu_prepare_dflash_inputs_to_capture(*args, **kwargs):
        causal = (
            resolve_causal(args, kwargs) if resolve_causal is not None else False
        )
        kvcc = resolve_kvcc(args, kwargs) if resolve_kvcc is not None else None
        return _apply_stamp(helper(*args, **kwargs), causal, kvcc)

    setattr(hcu_prepare_dflash_inputs_to_capture, _WRAPPER, True)
    setattr(capture_module, _CAPTURE_HELPER, hcu_prepare_dflash_inputs_to_capture)
    return [_CAPTURE_HELPER]


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
