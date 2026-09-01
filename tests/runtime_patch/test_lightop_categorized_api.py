# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

import sys
from importlib import import_module, invalidate_caches
from types import ModuleType, SimpleNamespace

import pytest
import torch


REQUIRED_EXPORTS = {
    "lightop.activation": {
        "fuse_silu_mul_fp8_quant",
        "fuse_silu_mul_fp8_quant_ep",
        "fuse_silu_mul_per_token_quant",
        "fuse_silu_mul_quant",
        "fuse_silu_mul_quant_ep",
        "silu_and_mul_opt",
    },
    "lightop.attention": {
        "fused_deepseek_v4_qnorm_rope_kvnorm_rope_quant_insert_int32",
        "get_paged_mqa_logits_metadata",
        "mqa_logits",
        "paged_mqa_logits",
        "split_qkv_rms_rotary_embedding_fuse_with_kv_store_quant",
        "top_k_per_row_decode",
        "top_k_per_row_prefill",
    },
    "lightop.gemm_ops": {
        "hipblaslt_w8a8_gemm",
        "hipblaslt_w8a8_channelwise_gemm",
        "m_grouped_w8a8_gemm_nt_contig_asm",
        "m_grouped_w8a8_gemm_nt_masked",
    },
    "lightop.moe": {
        "ep_gather",
        "ep_scatter",
        "fused_experts_impl_fp8_marlin",
        "fused_experts_impl_int8_marlin",
        "moe_align_block_size_out",
        "moe_fused_gate",
    },
    "lightop.norm": {
        "fused_add_rms_norm",
        "gemma_fused_add_rmsnorm",
        "gemma_rmsnorm",
        "rms_norm_dynamic_per_token_quant",
        "rmsnorm_forward_autograd",
    },
    "lightop.quant": {"per_token_quant_fp8", "per_token_quant_int8"},
    "lightop.tensor": {"ds_cat"},
}


@pytest.fixture
def isolated_lightop_modules():
    original_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "lightop" or name.startswith("lightop.")
    }
    for name in tuple(original_modules):
        sys.modules.pop(name, None)
    invalidate_caches()
    try:
        yield
    finally:
        for name in tuple(sys.modules):
            if name == "lightop" or name.startswith("lightop."):
                sys.modules.pop(name, None)
        invalidate_caches()
        sys.modules.update(original_modules)


def test_isolated_lightop_modules_evicts_stale_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cached test double must not satisfy the installed-wheel contract."""
    stale_lightop = ModuleType("lightop")
    stale_activation = ModuleType("lightop.activation")
    stale_lightop.activation = stale_activation  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "lightop", stale_lightop)
    monkeypatch.setitem(sys.modules, "lightop.activation", stale_activation)

    lifecycle = isolated_lightop_modules.__wrapped__()
    next(lifecycle)
    try:
        assert "lightop" not in sys.modules
        assert "lightop.activation" not in sys.modules
    finally:
        with pytest.raises(StopIteration):
            next(lifecycle)


@pytest.mark.usefixtures("isolated_lightop_modules")
@pytest.mark.hcu
@pytest.mark.parametrize("module_name", sorted(REQUIRED_EXPORTS))
def test_categorized_lightop_exports(
    module_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device_properties = SimpleNamespace(
        gcnArchName="gfx936:sramecc+:xnack-",
        multi_processor_count=80,
        name="HYGON HCU",
        major=9,
        minor=3,
        total_memory=64 << 30,
    )
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda *args, **kwargs: device_properties,
    )
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)

    import_module("lightop")
    module = import_module(module_name)
    missing = sorted(
        name for name in REQUIRED_EXPORTS[module_name] if not hasattr(module, name)
    )
    assert not missing, f"{module_name} is missing required exports: {missing}"
