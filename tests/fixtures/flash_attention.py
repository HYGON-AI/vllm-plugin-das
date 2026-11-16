# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""FlashAttention doubles for CPU-only backend contracts."""

from __future__ import annotations

import importlib
import sys
from importlib.machinery import ModuleSpec
from types import ModuleType

import pytest


def stub_flash_attn_native_extension(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Keep CPU contracts from loading vendor native binaries."""
    module = ModuleType("flash_attn_2_cuda")
    module.__spec__ = ModuleSpec(module.__name__, loader=None)

    def unexpected_native_call(*args, **kwargs):
        raise AssertionError("CPU attention contracts must not execute native FlashAttention")

    module.varlen_fwd = unexpected_native_call
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return module


def isolate_flash_attn_backend_imports(monkeypatch: pytest.MonkeyPatch) -> None:
    """Restore both module and parent-package caches after CPU doubles are used."""
    package = importlib.import_module("vllm_hcu.v1.attention.backends")
    for child in ("flash_attn", "fa_utils"):
        name = f"{package.__name__}.{child}"
        # Record absent entries too, so newly imported modules are removed on undo.
        monkeypatch.setitem(sys.modules, name, sys.modules.get(name))
        monkeypatch.delitem(sys.modules, name)
        monkeypatch.setattr(package, child, getattr(package, child, None), raising=False)
        monkeypatch.delattr(package, child)


def stub_flash_attn_version_check(monkeypatch: pytest.MonkeyPatch) -> None:
    """Kernel orchestration tests do not exercise distribution compatibility."""
    from vllm_hcu.v1.attention import _flash_attn_compat

    monkeypatch.setattr(_flash_attn_compat, "require_compatible_flash_attn", lambda: None)


def stub_flash_attn_extensions(monkeypatch: pytest.MonkeyPatch) -> None:
    isolate_flash_attn_backend_imports(monkeypatch)
    stub_flash_attn_version_check(monkeypatch)
    stub_flash_attn_native_extension(monkeypatch)
    hcu_ops = ModuleType("vllm_hcu.hcu_ops")
    hcu_ops.__spec__ = ModuleSpec(hcu_ops.__name__, loader=None)
    monkeypatch.setitem(sys.modules, hcu_ops.__name__, hcu_ops)

    flash_attn_extension = ModuleType("flash_attn")
    flash_attn_extension.__spec__ = ModuleSpec(
        flash_attn_extension.__name__,
        loader=None,
    )

    def layout_entrypoint(
        q=None,
        k=None,
        v=None,
        *,
        layout="bshd",
        **kwargs,
    ):
        del q, k, v, layout, kwargs

    for symbol in (
        "flash_attn_varlen_func",
        "hg_flash_attn_varlen_func",
        "varlen_fwd_unified",
    ):
        setattr(flash_attn_extension, symbol, layout_entrypoint)
    flash_attn_extension.vllm_flash_attn_varlen_func = (
        lambda *args, **kwargs: None
    )
    monkeypatch.setitem(sys.modules, flash_attn_extension.__name__, flash_attn_extension)


def load_hcu_flash_attention_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    stub_flash_attn_extensions(monkeypatch)
    return importlib.import_module("vllm_hcu.v1.attention.backends.flash_attn")
