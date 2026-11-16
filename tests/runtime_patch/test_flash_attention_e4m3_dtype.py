# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""E4M3 cache format selection on HCU devices."""
from types import SimpleNamespace

import pytest
import torch

from tests.fixtures.flash_attention import load_hcu_flash_attention_module


@pytest.mark.parametrize("arch", ["gfx92a:sramecc+:xnack-", "gfx938:sramecc+:xnack-"])
@pytest.mark.parametrize("cache_dtype", ["fp8", "fp8_e4m3"])
def test_e4m3_cache_uses_ocp_format(monkeypatch, arch, cache_dtype):
    backend = load_hcu_flash_attention_module(monkeypatch).HcuFlashAttentionBackend
    monkeypatch.setattr(
        torch.cuda, "get_device_properties",
        lambda device: SimpleNamespace(gcnArchName=arch),
    )
    assert backend.get_fp8_dtype_for_flashattn(cache_dtype) == torch.float8_e4m3fn


def test_e4m3_cache_rejects_unverified_architecture(monkeypatch):
    backend = load_hcu_flash_attention_module(monkeypatch).HcuFlashAttentionBackend
    monkeypatch.setattr(
        torch.cuda, "get_device_properties",
        lambda device: SimpleNamespace(gcnArchName="gfx936"),
    )
    with pytest.raises(ValueError, match="gfx936"):
        backend.get_fp8_dtype_for_flashattn("fp8_e4m3")


@pytest.mark.parametrize("cache_dtype", ["fp8", "fp8_e4m3"])
@pytest.mark.parametrize("arch", ["gfx92a", "gfx938"])
def test_e4m3_keeps_fa_quantized_queries(monkeypatch, cache_dtype, arch):
    module = load_hcu_flash_attention_module(monkeypatch)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: SimpleNamespace(gcnArchName=arch))
    monkeypatch.setattr(module, "get_current_vllm_config_or_none", lambda: None)
    impl = module.FlashAttentionImpl(
        num_heads=24, head_size=256, scale=0.0625, num_kv_heads=4,
        alibi_slopes=None, sliding_window=None, kv_cache_dtype=cache_dtype,
    )
    assert impl.supports_quant_query_input
