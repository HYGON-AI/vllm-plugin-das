# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Regression contracts for HY3 KV-cache scale loading on vLLM 0.25.1."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.model_executor.layers.quantization.base_config import QuantizationConfig


class _MapperOnlyQuantConfig:
    """Expose the v0.25.1 API without the removed get_cache_scale API."""

    @staticmethod
    def get_cache_scale_mapper():
        return QuantizationConfig.get_cache_scale_mapper()


def _blank_module(module_type: type[nn.Module]) -> nn.Module:
    module = object.__new__(module_type)
    nn.Module.__init__(module)
    return module


def _scalar_parameter() -> nn.Parameter:
    return nn.Parameter(torch.zeros(()), requires_grad=False)


@pytest.fixture(scope="module")
def hy_v3_types():
    patch = pytest.MonkeyPatch()
    import vllm.model_executor.layers.attention as attention_package
    from vllm_hcu.model_executor.layers.attention_runtime import (
        FusedQkvSplitRmsNormRopeAttention,
    )

    # Production installs this export before model discovery. Install the same
    # concrete class only for the lifetime of this focused CPU test module.
    patch.setattr(
        attention_package,
        "FusedQkvSplitRmsNormRopeAttention",
        FusedQkvSplitRmsNormRopeAttention,
        raising=False,
    )
    from vllm_hcu.models.hy_v3 import HYV3Model
    from vllm_hcu.models.hy_v3_mtp import HYV3MTP

    yield HYV3Model, HYV3MTP
    patch.undo()


def test_hy_v3_model_loads_pre_mapped_cache_scale_without_legacy_api(
    hy_v3_types,
) -> None:
    HYV3Model, _ = hy_v3_types
    model = _blank_module(HYV3Model)
    model.config = SimpleNamespace(tie_word_embeddings=False)
    model.quant_config = _MapperOnlyQuantConfig()
    model.get_expert_mapping = lambda: []

    target_name = "model.layers.0.self_attn.attn.k_scale"
    target = _scalar_parameter()
    model.named_parameters = lambda: iter(((target_name, target),))

    loaded = model.load_weights(((target_name, torch.tensor([3.0])),))

    assert loaded == {target_name}
    torch.testing.assert_close(target, torch.tensor(3.0))


def test_hy_v3_mtp_maps_checkpoint_cache_scale_with_v0251_mapper(
    hy_v3_types,
) -> None:
    _, HYV3MTP = hy_v3_types
    model = _blank_module(HYV3MTP)
    model.config = SimpleNamespace(
        tie_word_embeddings=False,
        num_hidden_layers=80,
        num_nextn_predict_layers=1,
        num_attention_heads=64,
        num_key_value_heads=8,
    )
    model.quant_config = _MapperOnlyQuantConfig()
    model.use_pp = False

    target_name = "model.layers.80.mtp_block.self_attn.attn.k_scale"
    target = _scalar_parameter()
    model.named_parameters = lambda: iter(((target_name, target),))

    model.load_weights(
        (("model.layers.80.self_attn.k_cache.scale", torch.tensor([5.0])),)
    )

    torch.testing.assert_close(target, torch.tensor(5.0))
