# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

"""Contracts for the narrow LightOp Qwen gated-RMSNorm route."""

from __future__ import annotations

from contextlib import nullcontext
import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from vllm_hcu.ops import rms_norm_gated
from vllm_hcu.patch.worker.op_opt import patch_gdn_rms_norm_gated
from vllm_hcu.platforms import envs as henvs


class _TensorMetadata:
    def __init__(
        self,
        shape: tuple[int, ...],
        *,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
        contiguous: bool = True,
    ) -> None:
        self.shape = shape
        self.ndim = len(shape)
        self.dtype = dtype
        self.device = torch.device(device)
        self._contiguous = contiguous

    def is_contiguous(self) -> bool:
        return self._contiguous


def _layer(width: int = 128, **overrides):
    values = {
        "weight": _TensorMetadata((width,)),
        "bias": None,
        "group_size": None,
        "norm_before_gate": True,
        "activation": "silu",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_qwen_module_captured_binding_uses_hcu_oot_class() -> None:
    target = ModuleType(patch_gdn_rms_norm_gated.TARGET_MODULE)
    target.RMSNormGated = rms_norm_gated.RMSNormGated

    assert patch_gdn_rms_norm_gated.apply_to_module(target)
    assert target.RMSNormGated is rms_norm_gated.HcuRMSNormGated
    assert (
        target._vllm_hcu_original_rms_norm_gated
        is rms_norm_gated.RMSNormGated
    )
    assert not patch_gdn_rms_norm_gated.apply_to_module(target)


def test_qwen_module_captured_binding_rejects_signature_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = ModuleType(patch_gdn_rms_norm_gated.TARGET_MODULE)
    target.RMSNormGated = rms_norm_gated.RMSNormGated
    monkeypatch.setattr(
        rms_norm_gated.RMSNormGated,
        "forward_cuda",
        lambda self, x: x,
    )

    with pytest.raises(
        patch_gdn_rms_norm_gated.PatchCompatibilityError,
        match="incompatible signature",
    ):
        patch_gdn_rms_norm_gated.apply_to_module(target)


def test_qwen_gated_rmsnorm_eligibility_is_strict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(henvs, "VLLM_HCU_USE_CUSTOM_OPS", True)
    monkeypatch.setattr(
        henvs, "VLLM_HCU_USE_LIGHTOP_QWEN_RMSNORM_GATED", True
    )
    monkeypatch.setattr(
        rms_norm_gated,
        "_lightop_layer_norm_fwd_1pass_opt",
        lambda: object(),
    )
    x = _TensorMetadata((32, 128))
    z = _TensorMetadata((32, 128))

    assert rms_norm_gated._is_qwen_gated_rmsnorm_eligible(_layer(), x, z)
    x_256 = _TensorMetadata((32, 256))
    assert rms_norm_gated._is_qwen_gated_rmsnorm_eligible(
        _layer(256), x_256, _TensorMetadata((32, 256))
    )
    assert not rms_norm_gated._is_qwen_gated_rmsnorm_eligible(
        _layer(), _TensorMetadata((3, 128)), _TensorMetadata((3, 128))
    )
    assert not rms_norm_gated._is_qwen_gated_rmsnorm_eligible(
        _layer(),
        _TensorMetadata((32, 128), dtype=torch.float32),
        _TensorMetadata((32, 128), dtype=torch.float32),
    )
    assert not rms_norm_gated._is_qwen_gated_rmsnorm_eligible(
        _layer(weight=_TensorMetadata((128,), dtype=torch.float32)), x, z
    )
    assert not rms_norm_gated._is_qwen_gated_rmsnorm_eligible(
        _layer(), _TensorMetadata((32, 128), contiguous=False), z
    )
    assert not rms_norm_gated._is_qwen_gated_rmsnorm_eligible(
        _layer(), _TensorMetadata((32, 128), device="cpu"), z
    )
    assert not rms_norm_gated._is_qwen_gated_rmsnorm_eligible(
        _layer(weight=_TensorMetadata((127,))), x, z
    )
    assert not rms_norm_gated._is_qwen_gated_rmsnorm_eligible(_layer(), x, None)
    assert not rms_norm_gated._is_qwen_gated_rmsnorm_eligible(
        _layer(group_size=64), x, z
    )
    assert not rms_norm_gated._is_qwen_gated_rmsnorm_eligible(
        _layer(activation="sigmoid"), x, z
    )

    monkeypatch.setattr(henvs, "VLLM_HCU_USE_CUSTOM_OPS", False)
    assert not rms_norm_gated._is_qwen_gated_rmsnorm_eligible(_layer(), x, z)


@pytest.mark.parametrize("export", (None, object()))
def test_qwen_gated_rmsnorm_missing_categorized_export_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    export: object | None,
) -> None:
    module = ModuleType("lightop.norm")
    if export is not None:
        module.layer_norm_fwd_1pass_opt = export
    monkeypatch.setattr(importlib, "import_module", lambda _name: module)
    rms_norm_gated._lightop_layer_norm_fwd_1pass_opt.cache_clear()
    try:
        assert rms_norm_gated._lightop_layer_norm_fwd_1pass_opt() is None
    finally:
        rms_norm_gated._lightop_layer_norm_fwd_1pass_opt.cache_clear()


def test_qwen_gated_rmsnorm_missing_lightop_package_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(_name: str):
        raise ModuleNotFoundError("lightop")

    monkeypatch.setattr(importlib, "import_module", missing)
    rms_norm_gated._lightop_layer_norm_fwd_1pass_opt.cache_clear()
    try:
        assert rms_norm_gated._lightop_layer_norm_fwd_1pass_opt() is None
    finally:
        rms_norm_gated._lightop_layer_norm_fwd_1pass_opt.cache_clear()


def test_qwen_gated_rmsnorm_missing_lightop_falls_back_to_vllm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(henvs, "VLLM_HCU_USE_CUSTOM_OPS", True)
    monkeypatch.setattr(
        henvs, "VLLM_HCU_USE_LIGHTOP_QWEN_RMSNORM_GATED", True
    )
    monkeypatch.setattr(
        rms_norm_gated,
        "_lightop_layer_norm_fwd_1pass_opt",
        lambda: None,
    )
    sentinel = object()
    layer = _layer(forward_cuda=lambda x, z: sentinel)
    result = rms_norm_gated.HcuRMSNormGated.forward_hip(
        layer,
        _TensorMetadata((32, 128)),
        _TensorMetadata((32, 128)),
    )

    assert result is sentinel


def test_qwen_gated_rmsnorm_uses_categorized_lightop_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def layer_norm_fwd_1pass_opt(*args):
        calls.append(args)
        args[1].copy_(args[0])
        args[6].fill_(1.0)

    lightop = ModuleType("lightop")
    lightop.__path__ = []
    norm = ModuleType("lightop.norm")
    norm.layer_norm_fwd_1pass_opt = layer_norm_fwd_1pass_opt
    lightop.norm = norm
    monkeypatch.setitem(sys.modules, "lightop", lightop)
    monkeypatch.setitem(sys.modules, "lightop.norm", norm)
    rms_norm_gated._lightop_layer_norm_fwd_1pass_opt.cache_clear()
    monkeypatch.setattr(torch.cuda, "device", lambda _device: nullcontext())
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(multi_processor_count=64),
    )

    x = torch.arange(6 * 128, dtype=torch.bfloat16).reshape(6, 128)
    z = torch.ones_like(x)
    weight = torch.ones(128, dtype=torch.bfloat16)
    output = rms_norm_gated._hcu_lightop_qwen_rmsnorm_gated_impl(
        x, z, weight, 1e-6
    )

    assert output is calls[0][1]
    assert torch.equal(output, x)
    assert calls[0][0] is x
    assert calls[0][2] is weight
    assert calls[0][3] is None
    assert calls[0][4] is z
    assert calls[0][5] is None
    assert calls[0][10:13] == (6, 128, 1e-6)
    assert calls[0][15:] == (False, True, True, True, "silu")
    rms_norm_gated._lightop_layer_norm_fwd_1pass_opt.cache_clear()
