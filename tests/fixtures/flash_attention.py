# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""FlashAttention doubles for CPU-only backend contracts."""

from __future__ import annotations

import importlib
import importlib.metadata
import sys
from importlib.machinery import ModuleSpec
from types import ModuleType

import pytest


def stub_flash_attn_native_extension(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Keep CPU contracts from loading vendor native binaries."""
    module = ModuleType("flash_attn_2_cuda")
    module.__spec__ = ModuleSpec(module.__name__, loader=None)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return module


FLASH_ATTN_TEST_VERSION = "2.8.4+dtk2604.torch2110.2609241509.g624d7b"


def stub_flash_attn_distribution(
    monkeypatch: pytest.MonkeyPatch,
    flash_attn_extension: ModuleType,
    version: str = FLASH_ATTN_TEST_VERSION,
) -> None:
    """Keep CPU kernel doubles independent of the installed vendor wheel."""

    flash_attn_extension.__version__ = version
    distribution_version = importlib.metadata.version

    def version_for_test(distribution_name: str) -> str:
        if distribution_name == "flash_attn":
            return version
        return distribution_version(distribution_name)

    monkeypatch.setattr(importlib.metadata, "version", version_for_test)


def stub_flash_attn_extensions(monkeypatch: pytest.MonkeyPatch) -> None:
    stub_flash_attn_native_extension(monkeypatch)
    hcu_ops = ModuleType("vllm_hcu.hcu_ops")
    hcu_ops.__spec__ = ModuleSpec(hcu_ops.__name__, loader=None)
    monkeypatch.setitem(sys.modules, hcu_ops.__name__, hcu_ops)

    flash_attn_extension = ModuleType("flash_attn")
    flash_attn_extension.__spec__ = ModuleSpec(
        flash_attn_extension.__name__,
        loader=None,
    )
    stub_flash_attn_distribution(monkeypatch, flash_attn_extension)

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
