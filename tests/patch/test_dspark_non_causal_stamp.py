# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Non-causal draft stamp: producer patch unit contract (fake spec module).

The patch must stamp the draft metadata wherever the fork passes
``causal=False`` -- including subclass overrides AND the FULL cudagraph
capture helper -- while leaving causal producers unstamped.  Shapes follow
the fork: per-group dicts keyed by LAYER NAME, ``Mapping[int_gid, bool]``
causal maps (``DFlashSpeculator._group_causal`` maps kv group ids, fork
speculator.py:209-214 -- round-3 finding A: the key spaces are disjoint and
alignment must go through ``kv_cache_config.kv_cache_groups[gid].layer_names``),
and the capture path's ``AttentionState(attn_metadata=..., slot_mappings=...)``.
"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import NamedTuple

import pytest

from vllm_hcu.patch.worker.framework_opt import patch_dspark_non_causal_stamp
from vllm_hcu.patch.worker.framework_opt._common import PatchCompatibilityError
from vllm_hcu.v1.attention.backends.mla._stamps import NON_CAUSAL_DRAFT_STAMP

CAPTURE_MODULE = patch_dspark_non_causal_stamp._CAPTURE_MODULE
CAPTURE_HELPER = patch_dspark_non_causal_stamp._CAPTURE_HELPER


class _Meta:
    pass


class _Group:
    def __init__(self, layer_names):
        self.layer_names = layer_names


class _KVCC:
    def __init__(self, *groups):
        self.kv_cache_groups = list(groups)


class _AttentionState(NamedTuple):
    """fork cudagraph_utils.AttentionState: (attn_metadata, slot_mappings)."""

    attn_metadata: dict | None
    slot_mappings: dict


def _capture_state(metas: dict | None) -> NamedTuple:
    return _AttentionState(
        attn_metadata=metas, slot_mappings={"g0": _Meta(), "g1": _Meta()}
    )


class BaseSpeculator:
    def _build_draft_attn_metadata(self, *args, **kwargs):
        return {"g0": _Meta(), "g1": _Meta()}


class SubSpeculator(BaseSpeculator):
    def _build_draft_attn_metadata(self, *args, **kwargs):  # override shadow
        return [_Meta()]


def _fake_module() -> ModuleType:
    module = ModuleType(patch_dspark_non_causal_stamp.TARGET_MODULE)
    module.DraftModelSpeculator = BaseSpeculator
    module.SubSpeculator = SubSpeculator
    return module


def test_wraps_every_producer_and_stamps_the_return_shape() -> None:
    module = _fake_module()

    assert patch_dspark_non_causal_stamp.apply_to_module(module) is True
    stamped = BaseSpeculator()._build_draft_attn_metadata()
    assert all(getattr(m, NON_CAUSAL_DRAFT_STAMP) is True for m in stamped.values())
    # subclass overrides cannot shadow the stamp
    for m in SubSpeculator()._build_draft_attn_metadata():
        assert getattr(m, NON_CAUSAL_DRAFT_STAMP) is True


def test_explicit_causal_true_producers_are_left_unstamped() -> None:
    class CausalSpeculator:
        def _build_draft_attn_metadata(self, causal=True):
            return _Meta()

    module = _fake_module()
    module.CausalSpeculator = CausalSpeculator
    assert patch_dspark_non_causal_stamp.apply_to_module(module) is True

    assert (
        getattr(
            CausalSpeculator()._build_draft_attn_metadata(),
            NON_CAUSAL_DRAFT_STAMP,
            None,
        )
        is None
    )
    stamped = CausalSpeculator()._build_draft_attn_metadata(causal=False)
    assert getattr(stamped, NON_CAUSAL_DRAFT_STAMP) is True
    stamped = CausalSpeculator()._build_draft_attn_metadata(False)
    assert getattr(stamped, NON_CAUSAL_DRAFT_STAMP) is True


def test_signature_defaults_resolve_the_causal_flag() -> None:
    """bind + apply_defaults: subclass default False, kw-only, *args, N4."""

    class SubDefaultFalse:
        def _build_draft_attn_metadata(self, *args, causal=False):
            return _Meta()

    class KwOnlyCausal:
        def _build_draft_attn_metadata(self, *, num_tokens=0, causal=True):
            return _Meta()

    module = _fake_module()
    module.SubDefaultFalse = SubDefaultFalse
    module.KwOnlyCausal = KwOnlyCausal
    assert patch_dspark_non_causal_stamp.apply_to_module(module) is True

    # fork DFlash subclass default: False -> stamped without passing causal
    assert getattr(SubDefaultFalse()._build_draft_attn_metadata(), NON_CAUSAL_DRAFT_STAMP) is True
    assert (
        getattr(
            SubDefaultFalse()._build_draft_attn_metadata(causal=True),
            NON_CAUSAL_DRAFT_STAMP,
            None,
        )
        is None
    )
    assert (
        getattr(
            KwOnlyCausal()._build_draft_attn_metadata(num_tokens=3),
            NON_CAUSAL_DRAFT_STAMP,
            None,
        )
        is None
    )
    assert (
        getattr(
            KwOnlyCausal()._build_draft_attn_metadata(num_tokens=3, causal=False),
            NON_CAUSAL_DRAFT_STAMP,
        )
        is True
    )


def test_mapping_causal_shared_key_space_stamps_per_key() -> None:
    """Defensive shared key space (not the fork shape) still aligns."""

    class GroupedSpeculator:
        def _build_draft_attn_metadata(self, causal=None):
            return {"g0": _Meta(), "g1": _Meta(), "g2": _Meta()}

    module = _fake_module()
    module.GroupedSpeculator = GroupedSpeculator
    assert patch_dspark_non_causal_stamp.apply_to_module(module) is True

    groups = GroupedSpeculator()._build_draft_attn_metadata(
        causal={"g0": False, "g1": True, "g2": False}
    )
    assert getattr(groups["g0"], NON_CAUSAL_DRAFT_STAMP) is True
    assert getattr(groups["g1"], NON_CAUSAL_DRAFT_STAMP, None) is None
    assert getattr(groups["g2"], NON_CAUSAL_DRAFT_STAMP) is True


def test_mapping_causal_by_gid_via_kv_cache_config() -> None:
    """Fork shape A: int gid map, str-keyed result, kv_cache_config aligns."""

    class DSparkStyle:
        def __init__(self):
            self.kv_cache_config = _KVCC(_Group(["l0", "l1"]), _Group(["l2"]))

        def _build_draft_attn_metadata(self, causal=None):
            return {"l0": _Meta(), "l1": _Meta(), "l2": _Meta()}

    module = _fake_module()
    module.DSparkStyle = DSparkStyle
    assert patch_dspark_non_causal_stamp.apply_to_module(module) is True

    groups = DSparkStyle()._build_draft_attn_metadata(causal={0: False, 1: True})
    assert getattr(groups["l0"], NON_CAUSAL_DRAFT_STAMP) is True
    assert getattr(groups["l1"], NON_CAUSAL_DRAFT_STAMP) is True
    assert getattr(groups["l2"], NON_CAUSAL_DRAFT_STAMP, None) is None

    groups = DSparkStyle()._build_draft_attn_metadata(causal={0: True, 1: False})
    assert getattr(groups["l0"], NON_CAUSAL_DRAFT_STAMP, None) is None
    assert getattr(groups["l2"], NON_CAUSAL_DRAFT_STAMP) is True


def test_mapping_causal_disjoint_without_kvcc_refuses_to_guess() -> None:
    """Fork shape A without kv_cache_config: never silently stamp/skip."""

    class OrphanGroups:
        # NOTE: no kv_cache_config attribute on purpose
        def _build_draft_attn_metadata(self, causal=None):
            return {"l0": _Meta(), "l1": _Meta()}

    module = _fake_module()
    module.OrphanGroups = OrphanGroups
    assert patch_dspark_non_causal_stamp.apply_to_module(module) is True

    # int gid map never intersects the str result keys -> unalignable:
    # all-falsy may blanket (safe), mixed must fail loudly.
    blanket = OrphanGroups()._build_draft_attn_metadata(causal={0: 0, 1: False})
    assert all(getattr(m, NON_CAUSAL_DRAFT_STAMP) is True for m in blanket.values())
    with pytest.raises(PatchCompatibilityError, match="per-group causal map"):
        OrphanGroups()._build_draft_attn_metadata(causal={0: False, 1: True})
    # all-truthy: causal groups -> stamp nothing, still no silent blanket
    quiet = OrphanGroups()._build_draft_attn_metadata(causal={0: True, 1: 1})
    assert all(getattr(m, NON_CAUSAL_DRAFT_STAMP, None) is None for m in quiet.values())


def test_mapping_causal_mapped_gid_without_layers_raises() -> None:
    """N-2: a mapped group with zero layers in the result is unaudited."""

    class HoleGroups:
        def __init__(self):
            self.kv_cache_config = _KVCC(_Group(["l_missing"]), _Group(["l1"]))

        def _build_draft_attn_metadata(self, causal=None):
            return {"l1": _Meta()}

    module = _fake_module()
    module.HoleGroups = HoleGroups
    assert patch_dspark_non_causal_stamp.apply_to_module(module) is True
    with pytest.raises(PatchCompatibilityError, match="l_missing"):
        HoleGroups()._build_draft_attn_metadata(causal={0: False, 1: False})


def test_falsy_causal_values_stamp_like_false() -> None:
    """N-4: fork passes config values verbatim; 0 must stamp like False."""

    class NumericCausal:
        def _build_draft_attn_metadata(self, causal=1):
            return _Meta()

    module = _fake_module()
    module.NumericCausal = NumericCausal
    assert patch_dspark_non_causal_stamp.apply_to_module(module) is True
    assert getattr(NumericCausal()._build_draft_attn_metadata(0), NON_CAUSAL_DRAFT_STAMP) is True
    assert (
        getattr(
            NumericCausal()._build_draft_attn_metadata(1),
            NON_CAUSAL_DRAFT_STAMP,
            None,
        )
        is None
    )


def test_empty_causal_map_stamps_nothing() -> None:
    class EmptyMap:
        def _build_draft_attn_metadata(self, causal=None):
            return {"l0": _Meta()}

    module = _fake_module()
    module.EmptyMap = EmptyMap
    assert patch_dspark_non_causal_stamp.apply_to_module(module) is True
    meta = EmptyMap()._build_draft_attn_metadata(causal={})
    assert getattr(meta["l0"], NON_CAUSAL_DRAFT_STAMP, None) is None


def test_mapping_causal_without_aligned_result_failures() -> None:
    class UnalignedSpeculator:
        def _build_draft_attn_metadata(self, causal=None):
            return [_Meta()]

    module = _fake_module()
    module.UnalignedSpeculator = UnalignedSpeculator
    assert patch_dspark_non_causal_stamp.apply_to_module(module) is True

    # all-False blanket -> stamped even on a list result
    meta = UnalignedSpeculator()._build_draft_attn_metadata(
        causal={"g0": False, "g1": False}
    )
    assert getattr(meta[0], NON_CAUSAL_DRAFT_STAMP) is True
    # mixed causal on an unalignable result -> fail loudly
    with pytest.raises(PatchCompatibilityError, match="per-group causal map"):
        UnalignedSpeculator()._build_draft_attn_metadata(
            causal={"g0": False, "g1": True}
        )


def test_idempotent_and_wrapper_marked() -> None:
    module = _fake_module()
    assert patch_dspark_non_causal_stamp.apply_to_module(module) is True
    method_owner = vars(BaseSpeculator)["_build_draft_attn_metadata"]
    assert (
        getattr(method_owner, "_vllm_hcu_non_causal_stamp_wrapper", False) is True
    )
    assert patch_dspark_non_causal_stamp.apply_to_module(module) is False


def test_rejects_slot_only_metadata_loudly() -> None:
    class SlottedMeta:
        __slots__ = ()

    class SlottedSpeculator:
        def _build_draft_attn_metadata(self):
            return SlottedMeta()

    module = _fake_module()
    module.SlottedSpeculator = SlottedSpeculator
    assert patch_dspark_non_causal_stamp.apply_to_module(module) is True
    with pytest.raises(PatchCompatibilityError, match="__slots__"):
        SlottedSpeculator()._build_draft_attn_metadata()


def test_requires_at_least_one_producer() -> None:
    module = ModuleType(patch_dspark_non_causal_stamp.TARGET_MODULE)
    with pytest.raises(PatchCompatibilityError, match="no class defines"):
        patch_dspark_non_causal_stamp.apply_to_module(module)


def test_capture_path_stamps_attn_metadata_only(monkeypatch) -> None:
    """N-1: REAL NamedTuple state; slot_mappings/others stay unstamped."""

    import sys

    capture = ModuleType(CAPTURE_MODULE)
    m0, m1 = _Meta(), _Meta()
    state = _capture_state({"g0": m0, "g1": m1})
    capture._prepare_dflash_inputs_to_capture = lambda **kwargs: state
    monkeypatch.setitem(sys.modules, CAPTURE_MODULE, capture)

    assert patch_dspark_non_causal_stamp.apply_to_module(_fake_module()) is True
    returned = capture._prepare_dflash_inputs_to_capture(for_cudagraph_capture=True)
    assert getattr(returned.attn_metadata["g0"], NON_CAUSAL_DRAFT_STAMP) is True
    assert getattr(returned.attn_metadata["g1"], NON_CAUSAL_DRAFT_STAMP) is True
    # the companion field must remain untouched -- this assertion CAN fail
    # now that slot_mappings values are stampable _Meta objects
    assert all(
        getattr(meta, NON_CAUSAL_DRAFT_STAMP, None) is None
        for meta in returned.slot_mappings.values()
    )


def test_capture_path_maps_gid_causal_via_kvcc(monkeypatch) -> None:
    """Fork shape A on the capture path: kv_cache_config is a helper input."""

    import sys

    capture = ModuleType(CAPTURE_MODULE)
    m0, m1 = _Meta(), _Meta()
    state = _capture_state({"l0": m0, "l1": m1})

    def _prepare_dflash_inputs_to_capture(kv_cache_config=None, causal=None):
        assert kv_cache_config is not None
        return state

    capture._prepare_dflash_inputs_to_capture = _prepare_dflash_inputs_to_capture
    monkeypatch.setitem(sys.modules, CAPTURE_MODULE, capture)

    assert patch_dspark_non_causal_stamp.apply_to_module(_fake_module()) is True
    kvcc = _KVCC(_Group(["l0"]), _Group(["l1"]))
    returned = capture._prepare_dflash_inputs_to_capture(
        kv_cache_config=kvcc, causal={0: False, 1: True}
    )
    assert getattr(returned.attn_metadata["l0"], NON_CAUSAL_DRAFT_STAMP) is True
    assert getattr(returned.attn_metadata["l1"], NON_CAUSAL_DRAFT_STAMP, None) is None


def test_capture_path_skips_empty_attention_state(monkeypatch) -> None:
    """attn_metadata=None stamps NOTHING: not container values, not the
    NamedTuple state itself (the SimpleNamespace fixture used to mask this
    branch -- N-1)."""

    import sys

    capture = ModuleType(CAPTURE_MODULE)
    state = _capture_state(None)
    capture._prepare_dflash_inputs_to_capture = lambda: state
    monkeypatch.setitem(sys.modules, CAPTURE_MODULE, capture)
    assert patch_dspark_non_causal_stamp.apply_to_module(_fake_module()) is True

    capture._prepare_dflash_inputs_to_capture()  # no crash
    assert getattr(state, NON_CAUSAL_DRAFT_STAMP, None) is None
    assert all(
        getattr(meta, NON_CAUSAL_DRAFT_STAMP, None) is None
        for meta in state.slot_mappings.values()
    )


def test_capture_path_module_optional(monkeypatch) -> None:
    """Absent module skips; the fallback must not recurse into itself (C1)."""

    real_import_module = importlib.import_module

    def _raiser(name, *args, **kwargs):
        if name == CAPTURE_MODULE:
            raise ModuleNotFoundError(name=name)
        return real_import_module(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", _raiser)
    assert patch_dspark_non_causal_stamp.apply_to_module(_fake_module()) is True


def test_capture_path_skip_covers_only_its_own_family(monkeypatch) -> None:
    """N5/C2: parent-package absence skips; unrelated absence surfaces."""

    real_import_module = importlib.import_module

    def _parent_missing(name, *args, **kwargs):
        if name == CAPTURE_MODULE:
            raise ModuleNotFoundError(name="vllm.v1.worker.gpu.spec_decode")
        return real_import_module(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", _parent_missing)
    assert patch_dspark_non_causal_stamp.apply_to_module(_fake_module()) is True

    def _unrelated_missing(name, *args, **kwargs):
        if name == CAPTURE_MODULE:
            raise ModuleNotFoundError(name="flash_attn")
        return real_import_module(name, *args, **kwargs)

    fresh = _fake_module()
    monkeypatch.setattr(importlib, "import_module", _unrelated_missing)
    with pytest.raises(ModuleNotFoundError) as excinfo:
        patch_dspark_non_causal_stamp.apply_to_module(fresh)
    # bare ModuleNotFoundError(name=...) has an empty str(): assert .name
    assert excinfo.value.name == "flash_attn"


def test_capture_path_broken_import_surfaces(monkeypatch) -> None:
    real_import_module = importlib.import_module

    def _raiser(name, *args, **kwargs):
        if name == CAPTURE_MODULE:
            raise RuntimeError("dependency boom")
        return real_import_module(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", _raiser)
    with pytest.raises(RuntimeError, match="dependency boom"):
        patch_dspark_non_causal_stamp.apply_to_module(_fake_module())


def test_capture_path_present_but_shapeless_fails_loudly(monkeypatch) -> None:
    import sys

    capture = ModuleType(CAPTURE_MODULE)
    monkeypatch.setitem(sys.modules, CAPTURE_MODULE, capture)
    with pytest.raises(PatchCompatibilityError, match=CAPTURE_HELPER):
        patch_dspark_non_causal_stamp.apply_to_module(_fake_module())


# --- review 2026-10-13 P1 #1: subclass overrides in OTHER modules --------


class _HookMeta:
    pass


def test_override_defined_after_apply_is_wrapped_by_hook() -> None:
    """A dflash-style override created AFTER the patch still gets stamped.

    The base-module scan cannot see subclasses defined in their own modules;
    the __init_subclass__ guard must wrap them at class creation.  The
    override carries the fork's ``causal`` parameter shape.
    """

    base = _fake_module()
    assert patch_dspark_non_causal_stamp.apply_to_module(base) is True

    class LateDFlashStyle(SubSpeculator):
        def _build_draft_attn_metadata(
            self, *args, causal=False, **kwargs
        ):  # own-module dflash shape (fork speculator.py:276)
            return [_HookMeta()]

    stamped = LateDFlashStyle()._build_draft_attn_metadata()
    assert len(stamped) == 1
    assert getattr(stamped[0], NON_CAUSAL_DRAFT_STAMP) is True


def test_hook_skips_overrides_without_a_causal_parameter() -> None:
    """R3 #1: the guard auto-wraps strangers; an override with NO ``causal``
    parameter has unknown causal semantics and must stay UNLABELED (falls
    back to the causal rows) rather than being guessed non-causal."""

    base = _fake_module()
    assert patch_dspark_non_causal_stamp.apply_to_module(base) is True

    class NoCausalOverride(SubSpeculator):
        def _build_draft_attn_metadata(self, *args, **kwargs):
            return [_HookMeta()]

    own = vars(NoCausalOverride)["_build_draft_attn_metadata"]
    assert not getattr(
        own, patch_dspark_non_causal_stamp._WRAPPER, False
    ), "stranger override without causal was wrapped anyway"
    stamped = NoCausalOverride()._build_draft_attn_metadata()
    assert getattr(stamped[0], NON_CAUSAL_DRAFT_STAMP, None) is None


def test_hook_chain_wraps_grandchild_override_too() -> None:
    """Two-level chains (dspark extends dflash) keep working."""

    base = _fake_module()
    assert patch_dspark_non_causal_stamp.apply_to_module(base) is True

    class LateDFlash(BaseSpeculator):
        def _build_draft_attn_metadata(self, *args, causal=False, **kwargs):
            return [_HookMeta()]

    class LateDSpark(LateDFlash):
        pass  # inherits the override

    assert getattr(LateDSpark()._build_draft_attn_metadata()[0], NON_CAUSAL_DRAFT_STAMP) is True
    # adding an override at the grandchild level is also guarded
    class LateDSparkOverride(LateDFlash):
        def _build_draft_attn_metadata(self, *args, causal=False, **kwargs):
            return [_HookMeta()]

    assert getattr(LateDSparkOverride()._build_draft_attn_metadata()[0], NON_CAUSAL_DRAFT_STAMP) is True


def test_hook_default_and_explicit_causal_semantics() -> None:
    """Fork shape: dflash override defaults to causal=False (stamped); an
    explicit causal=True through the hook path stays unstamped."""

    base = _fake_module()
    assert patch_dspark_non_causal_stamp.apply_to_module(base) is True

    class LateDefaultFalse(BaseSpeculator):
        def _build_draft_attn_metadata(self, *args, causal=False, **kwargs):
            return _HookMeta()

    assert getattr(LateDefaultFalse()._build_draft_attn_metadata(), NON_CAUSAL_DRAFT_STAMP) is True
    assert getattr(LateDefaultFalse()._build_draft_attn_metadata(causal=True), NON_CAUSAL_DRAFT_STAMP, None) is None
    assert getattr(LateDefaultFalse()._build_draft_attn_metadata(causal=False), NON_CAUSAL_DRAFT_STAMP) is True


def test_hook_stores_classmethod_form_and_is_idempotent() -> None:
    """A setattr'ed plain function resolves UNBOUND on subclass creation
    (TypeError: missing cls); the stored payload is a classmethod, and a
    second apply leaves the installed guard untouched."""

    base = _fake_module()
    assert patch_dspark_non_causal_stamp.apply_to_module(base) is True
    guard = vars(BaseSpeculator)["__init_subclass__"]
    assert isinstance(guard, classmethod)
    assert patch_dspark_non_causal_stamp.apply_to_module(base) is False
    assert vars(BaseSpeculator)["__init_subclass__"] is guard


def test_third_party_init_subclass_stays_in_the_chain() -> None:
    """A grandparent-defined custom hook (not ours) keeps receiving the
    chain through the forward logic."""

    module = ModuleType(patch_dspark_non_causal_stamp.TARGET_MODULE)

    class PartnerSpeculator:
        seen = []

        def __init_subclass__(cls, tag=None, **kwargs):
            PartnerSpeculator.seen.append((cls, tag))
            super().__init_subclass__(**kwargs)

        def _build_draft_attn_metadata(self, *args, **kwargs):
            return {"g": _Meta()}

    module.DraftModelSpeculator = PartnerSpeculator
    assert patch_dspark_non_causal_stamp.apply_to_module(module) is True

    class PartnerChild(PartnerSpeculator, tag="x"):
        def _build_draft_attn_metadata(self, *args, causal=False, **kwargs):
            return [_HookMeta()]

    assert PartnerSpeculator.seen == [(PartnerChild, "x")]
    stamped = PartnerChild()._build_draft_attn_metadata()
    assert getattr(stamped[0], NON_CAUSAL_DRAFT_STAMP) is True


def test_late_arm_wraps_already_created_subclass_override() -> None:
    """R2 #1a: the post-import callback also fires for an ALREADY-imported
    base module; subclass modules may have been imported by then, so their
    overrides predate the hook.  The arm must wrap live descendants in
    place.  Uses a PRIVATE class hierarchy: the shared module-level fakes
    are armed by earlier tests and would mask the late-arm path.  The
    overrides carry the fork's ``causal`` shape so the guard accepts them
    (R3 #1: parameterless overrides stay unwrapped)."""

    module = ModuleType(patch_dspark_non_causal_stamp.TARGET_MODULE)

    class PrivateBase:
        def _build_draft_attn_metadata(self, *args, **kwargs):
            return {"g0": _Meta()}

    class PrivateEarlyOverride(PrivateBase):  # created BEFORE the arm
        def _build_draft_attn_metadata(self, *args, causal=False, **kwargs):
            return [_HookMeta()]

    module.DraftModelSpeculator = PrivateBase
    assert patch_dspark_non_causal_stamp.apply_to_module(module) is True

    stamped = PrivateEarlyOverride()._build_draft_attn_metadata()
    assert len(stamped) == 1
    assert getattr(stamped[0], NON_CAUSAL_DRAFT_STAMP) is True
    # the in-place wrap is marked, so a re-arm stays idempotent
    own = vars(PrivateEarlyOverride)["_build_draft_attn_metadata"]
    assert getattr(
        own, patch_dspark_non_causal_stamp._WRAPPER, False
    ), "in-place late-arm wrap missing its marker"

    # a subclass created afterwards that DEFINES its own override rides
    # the hook: the class-creation fires the armed guard and wraps it
    class PrivateLate(PrivateEarlyOverride):
        def _build_draft_attn_metadata(self, *args, causal=False, **kwargs):
            return [_HookMeta()]

    own_late = vars(PrivateLate)["_build_draft_attn_metadata"]
    assert getattr(
        own_late, patch_dspark_non_causal_stamp._WRAPPER, False
    ), "post-arm override was not wrapped by the hook"
    late = PrivateLate()._build_draft_attn_metadata()
    assert getattr(late[0], NON_CAUSAL_DRAFT_STAMP) is True

    # a STRANGER created after the arm without a causal parameter must be
    # left alone by both the hook and the idempotent re-arm walk (R3 #1)
    class PrivateStranger(PrivateBase):
        def _build_draft_attn_metadata(self, *args, **kwargs):
            return [_HookMeta()]

    assert not getattr(
        vars(PrivateStranger)["_build_draft_attn_metadata"],
        patch_dspark_non_causal_stamp._WRAPPER,
        False,
    )
    assert getattr(
        PrivateStranger()._build_draft_attn_metadata()[0],
        NON_CAUSAL_DRAFT_STAMP,
        None,
    ) is None
