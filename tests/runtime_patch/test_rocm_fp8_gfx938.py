# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from __future__ import annotations

import os
import sys
from types import ModuleType

os.environ.setdefault("VLLM_PLUGINS", "__disabled__")

from vllm_hcu.patch.worker.op_opt import patch_rocm_fp8_gfx938

# Same tail order as vLLM's ROCm FP8 candidate list.
_CANDIDATE_ORDER = (
    "AiterPerTokenFp8ScaledMMLinearKernel",
    "PerTensorTorchFP8ScaledMMLinearKernel",
    "RowWiseTorchFP8ScaledMMLinearKernel",
    "ChannelWiseTorchFP8ScaledMMLinearKernel",
)


def _kernel(name: str, *, supported, can_implement) -> type:
    def is_supported(cls, compute_capability=None):
        del cls, compute_capability
        return supported()

    def can_implement_config(cls, per_tensor: bool):
        del cls
        return can_implement(per_tensor)

    return type(
        name,
        (),
        {
            "is_supported": classmethod(is_supported),
            "can_implement": classmethod(can_implement_config),
        },
    )


def _target_module(*, upstream_supported: bool, aiter_supported: bool) -> ModuleType:
    module = ModuleType(patch_rocm_fp8_gfx938.TARGET_MODULE)

    def _rocm_torch_fp8_scaled_mm_supported() -> bool:
        return upstream_supported

    module._rocm_torch_fp8_scaled_mm_supported = _rocm_torch_fp8_scaled_mm_supported

    def torch_support():
        if _rocm_torch_fp8_scaled_mm_supported():
            return True, None
        return False, "requires a platform with torch FP8 scaled-MM support."

    module.PerTensorTorchFP8ScaledMMLinearKernel = _kernel(
        "PerTensorTorchFP8ScaledMMLinearKernel",
        supported=torch_support,
        can_implement=lambda per_tensor: (
            (True, None)
            if per_tensor
            else (False, "requires per tensor activation and weight scales.")
        ),
    )
    module.RowWiseTorchFP8ScaledMMLinearKernel = _kernel(
        "RowWiseTorchFP8ScaledMMLinearKernel",
        supported=torch_support,
        can_implement=lambda per_tensor: (
            (False, "cannot be used with per tensor activation and weight scales.")
            if per_tensor
            else (True, None)
        ),
    )
    module.ChannelWiseTorchFP8ScaledMMLinearKernel = _kernel(
        "ChannelWiseTorchFP8ScaledMMLinearKernel",
        supported=lambda: (True, None),
        can_implement=lambda per_tensor: (
            (False, "cannot be used with per tensor activation and weight scales.")
            if per_tensor
            else (True, None)
        ),
    )
    module.AiterPerTokenFp8ScaledMMLinearKernel = _kernel(
        "AiterPerTokenFp8ScaledMMLinearKernel",
        supported=lambda: (
            (True, None) if aiter_supported else (False, "requires AITER linear.")
        ),
        can_implement=lambda per_tensor: (True, None),
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


def _choose(module: ModuleType, per_tensor: bool) -> str | None:
    for name in _CANDIDATE_ORDER:
        kernel = getattr(module, name)
        supported, _ = kernel.is_supported()
        if not supported:
            continue
        can_implement, _ = kernel.can_implement(per_tensor)
        if can_implement:
            return name
    return None


def test_gfx938_enables_per_tensor_without_selecting_rowwise(monkeypatch):
    _install_rocm_module(monkeypatch, "gfx938")
    module = _target_module(upstream_supported=False, aiter_supported=False)

    assert patch_rocm_fp8_gfx938.apply_to_module(module) is True
    assert patch_rocm_fp8_gfx938.apply_to_module(module) is False
    assert module._rocm_torch_fp8_scaled_mm_supported() is False

    per_tensor = module.PerTensorTorchFP8ScaledMMLinearKernel
    row_wise = module.RowWiseTorchFP8ScaledMMLinearKernel
    assert per_tensor.is_supported() == (True, None)
    assert row_wise.is_supported()[0] is False
    assert _choose(module, per_tensor=True) == "PerTensorTorchFP8ScaledMMLinearKernel"
    assert _choose(module, per_tensor=False) == "ChannelWiseTorchFP8ScaledMMLinearKernel"


def test_other_arch_keeps_upstream_per_tensor_result(monkeypatch):
    _install_rocm_module(monkeypatch, "gfx90a")
    unsupported = _target_module(upstream_supported=False, aiter_supported=False)
    assert patch_rocm_fp8_gfx938.apply_to_module(unsupported) is True
    assert unsupported.PerTensorTorchFP8ScaledMMLinearKernel.is_supported()[0] is False
    assert unsupported._rocm_torch_fp8_scaled_mm_supported() is False
    assert _choose(unsupported, per_tensor=False) == (
        "ChannelWiseTorchFP8ScaledMMLinearKernel"
    )

    supported = _target_module(upstream_supported=True, aiter_supported=False)
    assert patch_rocm_fp8_gfx938.apply_to_module(supported) is True
    assert supported.PerTensorTorchFP8ScaledMMLinearKernel.is_supported() == (
        True,
        None,
    )
    assert supported.RowWiseTorchFP8ScaledMMLinearKernel.is_supported() == (True, None)


def test_aiter_keeps_priority_when_available(monkeypatch):
    _install_rocm_module(monkeypatch, "gfx938")
    module = _target_module(upstream_supported=False, aiter_supported=True)

    assert patch_rocm_fp8_gfx938.apply_to_module(module) is True
    assert _choose(module, per_tensor=True) == "AiterPerTokenFp8ScaledMMLinearKernel"
    assert _choose(module, per_tensor=False) == "AiterPerTokenFp8ScaledMMLinearKernel"
