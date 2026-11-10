# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""CPU tests for the Triton fast-path custom-op master gate (review P2).

With VLLM_HCU_USE_CUSTOM_OPS=0 the platform selects HcuTritonMLABackend
precisely to bypass the FlashMLA provider (hcu.py
_use_triton_mla_master_fallback); the fast path ends in the vendor
FlashMLA ops, so the master switch must keep it off or the fallback
invokes the provider it was meant to avoid.
"""

from __future__ import annotations

import pytest


def _gate():
    triton_mla = pytest.importorskip(
        "vllm_hcu.v1.attention.backends.mla.triton_mla"
    )
    return triton_mla._flash_gate_enabled()


def test_master_on_keeps_flash_fast_path(monkeypatch):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    assert _gate() is True


def test_master_unset_defaults_on(monkeypatch):
    monkeypatch.delenv("VLLM_HCU_USE_CUSTOM_OPS", raising=False)
    assert _gate() is True


def test_master_off_turns_the_fast_path_off(monkeypatch):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "0")
    assert _gate() is False


def test_master_read_per_call_not_cached(monkeypatch):
    """The env master is consulted on every gate read, so a policy flip
    is visible without re-importing the module (matches
    envs.optional_custom_op_enabled's contract)."""
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "0")
    assert _gate() is False
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    assert _gate() is True


def test_platform_gate_present_in_both_fast_path_consumers():
    """Review P2: the quantized-query fast path is a gfx938 kernel
    contract (the ops dispatch falls to fwd_kvcache_quantization_mla
    elsewhere, which takes no descale_q). Both consumers -- the builder
    readiness chain and the impl's forward entry -- must consult
    on_gfx938(); the builder clause sits after the env master in the
    same readiness chain."""
    import inspect

    mod = pytest.importorskip(
        "vllm_hcu.v1.attention.backends.mla.triton_mla"
    )
    init_src = inspect.getsource(mod.HcuTritonMLAMetadataBuilder.__init__)
    fwd_src = inspect.getsource(mod.HcuTritonMLAImpl.forward_mqa)
    assert "on_gfx938()" in init_src
    assert "on_gfx938()" in fwd_src
    assert (
        init_src.index("_flash_gate_enabled")
        < init_src.index("on_gfx938()")
    )
