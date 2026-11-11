# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""CPU tests for the SP-MoE DeepEP-LL 3584 admission needles (review P1).

The combine family ships LogFMT-on/off twins (--Lb0E/--Lb1E); a wheel
may carry both legacy 2048 variants but only a partial 3584
specialization set.  Admission must hold PER VARIANT: the exact missing
twin is reported, never a family-level pass.
"""

from __future__ import annotations

import importlib
import pytest

MOD = "vllm_hcu.patch.worker.op_opt.patch_kimi_k3_sp_moe"

C0 = b"internode_ll7combineILb0ELi"
C1 = b"internode_ll7combineILb1ELi"
D8 = b"internode_ll8dispatchILi"


def _patch_mod():
    try:
        return importlib.import_module(MOD)
    except ModuleNotFoundError as exc:
        if (exc.name or "").startswith("vllm_hcu.patch"):
            raise
        pytest.skip(f"torch/heavy deps unavailable: {exc}")


def test_partial_combine_specialization_refused_per_variant():
    """The bot's precise scenario: both legacy variants in the wheel,
    only the LogFMT-off 3584 twin present -> the missing Lb1E twin must
    be reported (a family-level any() would expose 3584 anyway)."""
    mod = _patch_mod()
    data = (
        C0 + b"2048E" + C1 + b"2048E"
        + C0 + b"3584E" + D8 + b"2048E" + D8 + b"3584E"
    )
    assert mod._scan_ll_needles(data) == [C1 + b"3584E"]


def test_fully_provisioned_wheel_admits():
    mod = _patch_mod()
    data = (
        C0 + b"2048E" + C1 + b"2048E"
        + C0 + b"3584E" + C1 + b"3584E"
        + D8 + b"2048E" + D8 + b"3584E"
    )
    assert mod._scan_ll_needles(data) == []


def test_single_variant_family_admits_on_its_own_twin():
    """A wheel providing only one legacy variant is honest: its twin
    pair is complete, admission holds."""
    mod = _patch_mod()
    data = C0 + b"2048E" + C0 + b"3584E"
    assert mod._scan_ll_needles(data) == []


def test_present_variant_with_foreign_twin_only_is_refused():
    """Lb0 legacy present but only the Lb1 3584 twin in the binary: the
    Lb0 demand is still missing."""
    mod = _patch_mod()
    data = C0 + b"2048E" + C1 + b"3584E"
    assert mod._scan_ll_needles(data) == [C0 + b"3584E"]


def test_no_legacy_binary_abstains():
    mod = _patch_mod()
    assert mod._scan_ll_needles(b"\x00unrelated bytes") is None


def test_dispatch_family_scopes_to_its_needs():
    """A binary with dispatch-only traffic does not demand combine
    needles -- families absent stay auto-skipped."""
    mod = _patch_mod()
    data = D8 + b"2048E"
    assert mod._scan_ll_needles(data) == [D8 + b"3584E"]
    data = D8 + b"2048E" + D8 + b"3584E"
    assert mod._scan_ll_needles(data) == []
