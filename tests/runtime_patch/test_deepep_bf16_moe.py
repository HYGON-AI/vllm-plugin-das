# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
from unittest.mock import Mock

import pytest
import torch

from tests.runtime_patch.test_lightop_w16a16_moe import _config, _target_module
from vllm_hcu.model_executor.layers.fused_moe.experts.deepep_bf16_moe import (
    DeepEPAutoBF16Experts,
    DeepEPBF16ContiguousExperts,
)
from vllm_hcu.patch.worker.op_opt.moe import patch_unquantized_oracle


def _auto_config():
    config = _config(backend="deep_gemm")
    config.moe_parallel_config.use_deepep_auto_kernels = True
    return config


def test_explicit_deepgemm_selects_auto_bf16_and_preserves_ht_weights(monkeypatch):
    target, calls = _target_module()
    target.logger = Mock()
    monkeypatch.setattr(
        DeepEPBF16ContiguousExperts,
        "_supports_current_device",
        staticmethod(lambda: True),
    )
    patch_unquantized_oracle.apply_to_module(target)
    config = _auto_config()
    backend, experts = target.select_unquantized_moe_backend(config)
    assert backend.name == "HCU_DEEPEP_BF16"
    assert experts is DeepEPAutoBF16Experts
    assert target.map_unquantized_backend("deep_gemm") is backend
    w13, w2 = torch.empty(4, 1024, 2048), torch.empty(4, 2048, 512)
    converted = target.convert_to_unquantized_kernel_format(backend, config, w13, w2)
    assert converted[0] is w13 and converted[1] is w2
    assert not calls  # Explicit BF16 request never delegates to AITER or Triton.


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"in_dtype": torch.float16}, "bfloat16"),
        ({"has_bias": True}, "bias"),
        ({"hidden_dim": 3000}, "divisible"),
        ({"intermediate_size_per_partition": 1000}, "divisible"),
        ({"is_lora_enabled": True}, "LoRA"),
        ({"apply_router_weight_on_input": True}, "router weighting"),
    ],
)
def test_unsupported_bf16_request_fails_before_loading(monkeypatch, change, reason):
    target, _ = _target_module()
    target.logger = Mock()
    monkeypatch.setattr(
        DeepEPBF16ContiguousExperts,
        "_supports_current_device",
        staticmethod(lambda: True),
    )
    patch_unquantized_oracle.apply_to_module(target)
    config = _auto_config()
    for key, value in change.items():
        setattr(config, key, value)
    with pytest.raises(ValueError, match=reason):
        target.select_unquantized_moe_backend(config)


def test_bf16_rejects_fixed_all2all_instead_of_building_auto(monkeypatch):
    config = _auto_config()
    config.moe_parallel_config.use_deepep_auto_kernels = False
    monkeypatch.setattr(
        DeepEPBF16ContiguousExperts,
        "_supports_current_device",
        staticmethod(lambda: True),
    )
    supported, reason = DeepEPAutoBF16Experts.is_supported_config(
        DeepEPAutoBF16Experts, config, None, None, None
    )
    assert not supported
    assert "parallel config" in reason


def test_bf16_does_not_request_deferred_quantization():
    experts = object.__new__(DeepEPBF16ContiguousExperts)
    assert experts.expects_unquantized_inputs is False
