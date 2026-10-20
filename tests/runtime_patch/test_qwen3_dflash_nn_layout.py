# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from __future__ import annotations

import importlib
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from vllm_hcu.patch.worker.core_fix._common import PatchCompatibilityError


TARGET_MODULE = "vllm.model_executor.models.qwen3_dflash"


def _upstream_module() -> ModuleType:
    module = ModuleType(TARGET_MODULE)

    class DFlashQwen3Model:
        def _build_context_kv_buffers(self, layers_attn, has_bias):
            self._hidden_norm_weight = self.hidden_norm.weight.data
            kv_weights = [a.qkv_proj.weight[a.q_size :] for a in layers_attn]
            self._fused_kv_weight = torch.cat(kv_weights, dim=0)
            self._fused_kv_bias = (
                torch.cat(
                    [a.qkv_proj.bias[a.q_size :] for a in layers_attn], dim=0
                )
                if has_bias
                else None
            )
            self._k_norm_weights = torch.stack(
                [a.k_norm.weight.data for a in layers_attn], dim=0
            ).contiguous()

    module.DFlashQwen3Model = DFlashQwen3Model
    return module


def _attention(
    weight: torch.Tensor,
    *,
    is_quantization: bool = False,
    input_size: int = 4,
) -> SimpleNamespace:
    return SimpleNamespace(
        q_size=2,
        kv_size=2,
        qkv_proj=SimpleNamespace(
            weight=weight,
            bias=torch.arange(6, dtype=torch.float32),
            input_size=input_size,
            is_quantization=is_quantization,
        ),
        k_norm=SimpleNamespace(weight=torch.tensor([0.5, 1.5])),
    )


def _patch_module():
    try:
        return importlib.import_module(
            "vllm_hcu.patch.worker.core_fix.patch_qwen3_dflash_nn_layout"
        )
    except ModuleNotFoundError:
        pytest.fail("Qwen3 DSpark NN-layout compatibility patch is missing")


def test_nn_layout_ignores_disabled_custom_ops_master_switch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch = _patch_module()
    module = _upstream_module()
    monkeypatch.setenv("VLLM_USE_NN", "1")
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "0")
    assert patch.apply_to_module(module) is True

    model = module.DFlashQwen3Model()
    model.hidden_norm = SimpleNamespace(weight=torch.arange(4))
    attention = _attention(
        torch.tensor(
            [
                [1.0, 2.0, 10.0, 11.0, 12.0, 13.0],
                [3.0, 4.0, 20.0, 21.0, 22.0, 23.0],
                [5.0, 6.0, 30.0, 31.0, 32.0, 33.0],
                [7.0, 8.0, 40.0, 41.0, 42.0, 43.0],
            ]
        )
    )

    model._build_context_kv_buffers([attention], has_bias=True)

    assert model._fused_kv_weight.shape == (4, 4)
    projected = F.linear(
        torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
        model._fused_kv_weight,
        model._fused_kv_bias,
    )
    torch.testing.assert_close(
        projected,
        torch.tensor([[302.0, 313.0, 324.0, 335.0]]),
    )


def test_non_nn_layout_delegates_and_patch_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch = _patch_module()
    module = _upstream_module()
    original = module.DFlashQwen3Model._build_context_kv_buffers
    monkeypatch.setattr(patch, "_use_nn_layout", lambda: False)

    assert patch.apply_to_module(module) is True
    wrapped = module.DFlashQwen3Model._build_context_kv_buffers
    assert wrapped is not original
    assert patch.apply_to_module(module) is False

    model = module.DFlashQwen3Model()
    model.hidden_norm = SimpleNamespace(weight=torch.ones(4))
    attention = _attention(torch.arange(24).reshape(6, 4).float())
    model._build_context_kv_buffers([attention], has_bias=False)

    torch.testing.assert_close(
        model._fused_kv_weight,
        attention.qkv_proj.weight[attention.q_size :],
    )


def test_nn_layout_delegates_quantized_output_major_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch = _patch_module()
    module = _upstream_module()
    monkeypatch.setattr(patch, "_use_nn_layout", lambda: True)
    assert patch.apply_to_module(module) is True

    model = module.DFlashQwen3Model()
    model.hidden_norm = SimpleNamespace(weight=torch.ones(4))
    attention = _attention(
        torch.arange(24).reshape(6, 4).float(), is_quantization=True
    )
    model._build_context_kv_buffers([attention], has_bias=False)

    torch.testing.assert_close(
        model._fused_kv_weight,
        attention.qkv_proj.weight[attention.q_size :],
    )


def test_nn_layout_rejects_mixed_quantization_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch = _patch_module()
    module = _upstream_module()
    monkeypatch.setattr(patch, "_use_nn_layout", lambda: True)
    assert patch.apply_to_module(module) is True

    model = module.DFlashQwen3Model()
    model.hidden_norm = SimpleNamespace(weight=torch.ones(4))
    unquantized = _attention(torch.arange(24).reshape(4, 6).float())
    quantized = _attention(
        torch.arange(24).reshape(6, 4).float(), is_quantization=True
    )

    with pytest.raises(PatchCompatibilityError, match="quantization.*inconsistent"):
        model._build_context_kv_buffers(
            [unquantized, quantized], has_bias=False
        )


def test_nn_layout_rejects_missing_quantization_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch = _patch_module()
    module = _upstream_module()
    monkeypatch.setattr(patch, "_use_nn_layout", lambda: True)
    assert patch.apply_to_module(module) is True

    model = module.DFlashQwen3Model()
    model.hidden_norm = SimpleNamespace(weight=torch.ones(4))
    attention = _attention(torch.arange(24).reshape(4, 6).float())
    del attention.qkv_proj.is_quantization

    with pytest.raises(PatchCompatibilityError, match="metadata.*missing"):
        model._build_context_kv_buffers([attention], has_bias=False)


def test_nn_layout_rejects_unexpected_weight_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch = _patch_module()
    module = _upstream_module()
    monkeypatch.setattr(patch, "_use_nn_layout", lambda: True)
    assert patch.apply_to_module(module) is True

    model = module.DFlashQwen3Model()
    model.hidden_norm = SimpleNamespace(weight=torch.ones(4))
    attention = _attention(torch.arange(24).reshape(6, 4).float())

    with pytest.raises(PatchCompatibilityError, match="expected .* got"):
        model._build_context_kv_buffers([attention], has_bias=False)


def test_qwen3_dflash_nn_layout_patch_is_registered() -> None:
    from vllm_hcu.patch.worker import worker_callback_names

    assert (
        "worker.core_fix.qwen3_dflash.nn_layout",
        TARGET_MODULE,
    ) in worker_callback_names()
