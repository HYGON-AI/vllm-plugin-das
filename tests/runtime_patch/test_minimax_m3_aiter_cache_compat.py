# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from __future__ import annotations

from types import ModuleType, SimpleNamespace

import pytest

from vllm_hcu.patch.worker.core_fix import patch_minimax_m3_aiter_cache
from vllm_hcu.patch.worker.core_fix._common import PatchCompatibilityError


def _target_module() -> ModuleType:
    module = ModuleType(patch_minimax_m3_aiter_cache.TARGET_MODULE)

    class MiniMaxM3SparseAttention:
        def _insert_aiter_sparse_pa_kv(self):
            return None

    module.MiniMaxM3SparseAttention = MiniMaxM3SparseAttention
    return module


def test_exposes_nested_aiter_reshape_and_cache(monkeypatch: pytest.MonkeyPatch):
    def expected():
        return None

    aiter = SimpleNamespace()
    cache = SimpleNamespace(reshape_and_cache=expected)

    def import_module(name: str):
        return {"aiter": aiter, "aiter.ops.cache": cache}[name]

    monkeypatch.setattr(
        patch_minimax_m3_aiter_cache.importlib, "import_module", import_module
    )
    module = _target_module()

    assert patch_minimax_m3_aiter_cache.apply_to_module(module) is True
    assert aiter.reshape_and_cache is expected
    assert patch_minimax_m3_aiter_cache.apply_to_module(module) is False


def test_preserves_native_top_level_export(monkeypatch: pytest.MonkeyPatch):
    def native():
        return None

    aiter = SimpleNamespace(reshape_and_cache=native)

    def import_module(name: str):
        assert name == "aiter"
        return aiter

    monkeypatch.setattr(
        patch_minimax_m3_aiter_cache.importlib, "import_module", import_module
    )

    assert patch_minimax_m3_aiter_cache.apply_to_module(_target_module()) is True
    assert aiter.reshape_and_cache is native


def test_missing_nested_export_fails_closed(monkeypatch: pytest.MonkeyPatch):
    modules = {
        "aiter": SimpleNamespace(),
        "aiter.ops.cache": SimpleNamespace(),
    }
    monkeypatch.setattr(
        patch_minimax_m3_aiter_cache.importlib,
        "import_module",
        lambda name: modules[name],
    )

    with pytest.raises(PatchCompatibilityError, match="reshape_and_cache"):
        patch_minimax_m3_aiter_cache.apply_to_module(_target_module())
