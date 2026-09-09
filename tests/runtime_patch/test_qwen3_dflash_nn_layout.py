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
            if has_bias:
                self._fused_kv_bias = torch.cat(
                    [a.qkv_proj.bias[a.q_size :] for a in layers_attn], dim=0
                )
            else:
                self._fused_kv_bias = None
            self._k_norm_weights = torch.stack(
                [a.k_norm.weight.data for a in layers_attn], dim=0
            ).contiguous()

    module.DFlashQwen3Model = DFlashQwen3Model
    return module


def _attention(
    weight: torch.Tensor,
    bias: torch.Tensor,
    k_norm: torch.Tensor,
    *,
    input_size: int | None = None,
    is_quantization: bool = False,
):
    return SimpleNamespace(
        q_size=2,
        kv_size=2,
        qkv_proj=SimpleNamespace(
            weight=weight,
            bias=bias,
            input_size=weight.shape[0] if input_size is None else input_size,
            is_quantization=is_quantization,
        ),
        k_norm=SimpleNamespace(weight=k_norm),
    )


def test_nn_layout_builds_output_major_context_kv_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    try:
        patch = importlib.import_module(
            "vllm_hcu.patch.worker.core_fix.patch_qwen3_dflash_nn_layout"
        )
    except ModuleNotFoundError:
        pytest.fail("Qwen3 DSpark NN-layout compatibility patch is missing")

    module = _upstream_module()
    monkeypatch.setattr(patch, "_use_nn_layout", lambda: True)
    assert patch.apply_to_module(module) is True

    model = module.DFlashQwen3Model()
    hidden_norm = torch.tensor([1.0, 2.0, 3.0, 4.0])
    model.hidden_norm = SimpleNamespace(weight=hidden_norm)
    first = _attention(
        torch.tensor(
            [
                [1.0, 2.0, 10.0, 11.0, 12.0, 13.0],
                [3.0, 4.0, 20.0, 21.0, 22.0, 23.0],
                [5.0, 6.0, 30.0, 31.0, 32.0, 33.0],
                [7.0, 8.0, 40.0, 41.0, 42.0, 43.0],
            ]
        ),
        torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
        torch.tensor([0.5, 1.5]),
    )
    second = _attention(
        torch.tensor(
            [
                [101.0, 102.0, 110.0, 111.0, 112.0, 113.0],
                [103.0, 104.0, 120.0, 121.0, 122.0, 123.0],
                [105.0, 106.0, 130.0, 131.0, 132.0, 133.0],
                [107.0, 108.0, 140.0, 141.0, 142.0, 143.0],
            ]
        ),
        torch.tensor([10.0, 11.0, 12.0, 13.0, 14.0, 15.0]),
        torch.tensor([2.5, 3.5]),
    )

    model._build_context_kv_buffers([first, second], has_bias=True)

    assert model._fused_kv_weight.shape == (8, 4)
    projected = F.linear(
        torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
        model._fused_kv_weight,
        model._fused_kv_bias,
    )
    torch.testing.assert_close(
        projected,
        torch.tensor(
            [[302.0, 313.0, 324.0, 335.0, 1312.0, 1323.0, 1334.0, 1345.0]]
        ),
    )
    assert model._hidden_norm_weight.data_ptr() == hidden_norm.data_ptr()
    torch.testing.assert_close(
        model._k_norm_weights,
        torch.tensor([[0.5, 1.5], [2.5, 3.5]]),
    )


def test_non_nn_layout_delegates_to_upstream_and_patch_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch = importlib.import_module(
        "vllm_hcu.patch.worker.core_fix.patch_qwen3_dflash_nn_layout"
    )
    module = _upstream_module()
    original = module.DFlashQwen3Model._build_context_kv_buffers
    monkeypatch.setattr(patch, "_use_nn_layout", lambda: False)

    assert patch.apply_to_module(module) is True
    wrapped = module.DFlashQwen3Model._build_context_kv_buffers
    assert wrapped is not original
    assert patch.apply_to_module(module) is False
    assert module.DFlashQwen3Model._build_context_kv_buffers is wrapped

    model = module.DFlashQwen3Model()
    model.hidden_norm = SimpleNamespace(weight=torch.ones(4))
    attention = _attention(
        torch.arange(24, dtype=torch.float32).reshape(6, 4),
        torch.arange(6, dtype=torch.float32),
        torch.ones(2),
    )

    model._build_context_kv_buffers([attention], has_bias=False)

    torch.testing.assert_close(
        model._fused_kv_weight,
        attention.qkv_proj.weight[attention.q_size :],
    )
    assert model._fused_kv_bias is None


def test_nn_layout_delegates_output_major_quantized_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch = importlib.import_module(
        "vllm_hcu.patch.worker.core_fix.patch_qwen3_dflash_nn_layout"
    )
    module = _upstream_module()
    monkeypatch.setattr(patch, "_use_nn_layout", lambda: True)
    assert patch.apply_to_module(module) is True

    model = module.DFlashQwen3Model()
    model.hidden_norm = SimpleNamespace(weight=torch.ones(4))
    attention = _attention(
        torch.arange(24, dtype=torch.float32).reshape(6, 4),
        torch.arange(6, dtype=torch.float32),
        torch.ones(2),
        input_size=4,
        is_quantization=True,
    )

    model._build_context_kv_buffers([attention], has_bias=False)

    torch.testing.assert_close(
        model._fused_kv_weight,
        attention.qkv_proj.weight[attention.q_size :],
    )


def test_nn_layout_rejects_unexpected_unquantized_weight_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch = importlib.import_module(
        "vllm_hcu.patch.worker.core_fix.patch_qwen3_dflash_nn_layout"
    )
    module = _upstream_module()
    monkeypatch.setattr(patch, "_use_nn_layout", lambda: True)
    assert patch.apply_to_module(module) is True

    model = module.DFlashQwen3Model()
    model.hidden_norm = SimpleNamespace(weight=torch.ones(4))
    attention = _attention(
        torch.arange(24, dtype=torch.float32).reshape(6, 4),
        torch.arange(6, dtype=torch.float32),
        torch.ones(2),
        input_size=4,
    )

    with pytest.raises(PatchCompatibilityError, match="expected .* got"):
        model._build_context_kv_buffers([attention], has_bias=False)
