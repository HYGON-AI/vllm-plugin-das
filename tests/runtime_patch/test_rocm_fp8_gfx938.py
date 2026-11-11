# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from __future__ import annotations

import os
import sys
from types import ModuleType

os.environ.setdefault("VLLM_PLUGINS", "__disabled__")

from vllm_hcu.patch.worker.op_opt import patch_rocm_fp8_gfx938


def _target_module(result: bool) -> ModuleType:
    module = ModuleType(patch_rocm_fp8_gfx938.TARGET_MODULE)

    def _rocm_torch_fp8_scaled_mm_supported() -> bool:
        return result

    module._rocm_torch_fp8_scaled_mm_supported = (
        _rocm_torch_fp8_scaled_mm_supported
    )
    return module


def _install_rocm_module(monkeypatch, arch: str) -> None:
    vllm = ModuleType("vllm")
    platforms = ModuleType("vllm.platforms")
    rocm = ModuleType("vllm.platforms.rocm")
    rocm._GCN_ARCH = arch
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.platforms", platforms)
    monkeypatch.setitem(sys.modules, "vllm.platforms.rocm", rocm)


def test_rocm_torch_fp8_scaled_mm_allows_gfx938(monkeypatch):
    _install_rocm_module(monkeypatch, "gfx938")
    module = _target_module(False)

    assert patch_rocm_fp8_gfx938.apply_to_module(module) is True
    assert patch_rocm_fp8_gfx938.apply_to_module(module) is False
    assert module._rocm_torch_fp8_scaled_mm_supported() is True


def test_rocm_torch_fp8_scaled_mm_preserves_upstream_result(monkeypatch):
    _install_rocm_module(monkeypatch, "gfx90a")

    upstream_supported = _target_module(True)
    assert patch_rocm_fp8_gfx938.apply_to_module(upstream_supported) is True
    assert upstream_supported._rocm_torch_fp8_scaled_mm_supported() is True

    upstream_unsupported = _target_module(False)
    assert patch_rocm_fp8_gfx938.apply_to_module(upstream_unsupported) is True
    assert upstream_unsupported._rocm_torch_fp8_scaled_mm_supported() is False
