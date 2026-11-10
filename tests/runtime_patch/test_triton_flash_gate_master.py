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


def test_pool_bound_scales_by_manager_kernel_ratio():
    """Review P1: num_gpu_blocks counts MANAGER blocks (Kimi: 1536
    tokens) while the clamp bounds 64-token kernel-page ids -- the fill
    must scale the bound by the manager/kernel ratio (the FlashMLA
    builder's _pages_per_block, added after the same junk-page
    incident), never by the layer-spec subpage count (which is 1 when
    the layer page is 64 tokens)."""
    import inspect
    import re

    mod = pytest.importorskip(
        "vllm_hcu.v1.attention.backends.mla.triton_mla"
    )
    src = inspect.getsource(mod.HcuTritonMLAMetadataBuilder._fill_flash_sched)
    bound = src[src.index("_mgr_block") : src.index("if pool_pages <= 0")]
    assert "_pages_per_block" in bound
    assert re.search(r"_mgr_block\s*//\s*_FLASH_PAGE_BLOCK", bound)
    # The pool total multiplies by the ratio (the source breaks the
    # multiplication onto its own line).
    assert re.search(r"\*\s*_pages_per_block", bound)
    # The old buggy scaling must be gone from the bound expression.
    assert not re.search(r"\*\s*subpage", bound)


def test_ledger_ready_construction_reads_block_count(monkeypatch):
    """Bot P1 regression: the LEDGER snapshot inside the flash-ready
    constructor arm must read num_gpu_blocks itself -- a reference to an
    undefined name raises NameError at builder construction and kills
    every explicitly-selected TRITON_MLA gfx938 NC FP8 service.  The
    builder is really constructed here; the upstream parent's __init__
    is stubbed at the source module (layout-safe) and the device knobs
    are faked to arm the fast path on CPU."""
    import re
    from types import SimpleNamespace

    torch = pytest.importorskip("torch")
    mod = pytest.importorskip(
        "vllm_hcu.v1.attention.backends.mla.triton_mla"
    )
    parent_mod = pytest.importorskip(
        "vllm.v1.attention.backends.mla.triton_mla"
    )
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.setattr(mod, "on_gfx938", lambda: True)
    monkeypatch.setattr(mod, "_dense_flash_ok", lambda: True)
    monkeypatch.setattr(mod, "num_compute_units", lambda device: 8)

    def _fake_parent_init(
        self, kv_cache_spec, layer_names, vllm_config, device
    ):
        self.non_causal_multi_token_decode = getattr(
            kv_cache_spec, "non_causal_multi_token_decode", False
        )
        self.vllm_config = vllm_config

    monkeypatch.setattr(
        parent_mod.TritonMLAMetadataBuilder,
        "__init__",
        _fake_parent_init,
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=2048),
        parallel_config=SimpleNamespace(),
        scheduler_config=SimpleNamespace(max_num_seqs=4),
        cache_config=SimpleNamespace(
            cache_dtype="fp8", num_gpu_blocks=123, block_size=1536
        ),
    )
    spec = SimpleNamespace(
        non_causal_multi_token_decode=True, block_size=64
    )
    # Fresh registry so this construction registers and emits the
    # VA-LEDGER line; capture the module logger wholesale (the vllm
    # logger object is replaced -- the %-format contract stays intact).
    monkeypatch.setattr(mod, "_FLASH_BUF_REGISTRY", {})
    info_calls = []

    def _capture_info(*args, **kwargs):
        info_calls.append(args)

    monkeypatch.setattr(
        mod,
        "_vllogger",
        SimpleNamespace(
            info=_capture_info,
            debug=lambda *a, **k: None,
            warning=lambda *a, **k: None,
            warning_once=lambda *a, **k: None,
            info_once=lambda *a, **k: None,
        ),
    )
    builder = mod.HcuTritonMLAMetadataBuilder(
        spec, ["k3.self_attn"], vllm_config, torch.device("cpu")
    )
    assert builder._hcu_flash_ready is True
    # Manager/kernel ratio: 1536/64 = 24 kernel pages per manager block.
    assert builder._hcu_pool_pages == 123 * 24
    # Review P3: every ledger extent must be read from the tensor's own
    # shape -- the hardcoded "+64 rows x8" was only true at num_sms=16,
    # and the nsplits "max_seqs" under-reported the allocation by its
    # spare row, so a fault there read as out-of-bounds.
    ledger = [c for c in info_calls if "VA-LEDGER" in str(c[0])] if info_calls else []
    assert ledger, "VA-LEDGER line missing from construction"
    msg = ledger[0][0] % tuple(ledger[0][1:])
    m = re.search(
        r"sched=\[0x[0-9a-f]+ \+(\d+) rows x(\d+)\]"
        r" nsplits=\[0x[0-9a-f]+ \+(\d+)\]",
        msg,
    )
    assert m, f"ledger shape not parseable: {msg}"
    sched_rows, sched_cols, nsplits_rows = (int(g) for g in m.groups())
    assert sched_rows == builder._hcu_sched_buf.shape[0]
    assert sched_cols == builder._hcu_sched_buf.shape[1]
    assert nsplits_rows == builder._hcu_nsplits_buf.numel()
