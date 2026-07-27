# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Runtime contracts for PP partitioning and HCU attention mode routing."""

from __future__ import annotations

import sys
from types import ModuleType

import pytest

from vllm_hcu.patch.platform.framework_opt import patch_distributed_utils
from vllm_hcu.patch.worker.op_opt import patch_triton_unified_attention
from vllm_hcu.platforms import envs as hcu_envs


def test_pp_size_one_ignores_invalid_manual_partition(monkeypatch: pytest.MonkeyPatch):
    calls: list[tuple[int, int, int]] = []

    def get_pp_indices(num_hidden_layers: int, pp_rank: int, pp_size: int):
        calls.append((num_hidden_layers, pp_rank, pp_size))
        raise ValueError("invalid VLLM_PP_LAYER_PARTITION")

    module = ModuleType(patch_distributed_utils.TARGET_MODULE)
    module.get_pp_indices = get_pp_indices

    assert patch_distributed_utils.apply_to_module(module) is True
    assert patch_distributed_utils.apply_to_module(module) is False
    assert module.get_pp_indices(61, 0, 1) == (0, 61)
    assert calls == []
    with pytest.raises(ValueError, match="invalid VLLM_PP_LAYER_PARTITION"):
        module.get_pp_indices(61, 0, 2)
    assert calls == [(61, 0, 2)]


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        (None, "custom"),
        ("VLLM_HCU_USE_FLASH_ATTN", "classic"),
        ("VLLM_HCU_USE_FLASH_ATTN_UNIFIED", "cutlass"),
        ("VLLM_HCU_USE_CUSTOM_FLASH_ATTN", "custom"),
    ],
)
def test_flash_attention_mode_legacy_priority_and_default(
    monkeypatch: pytest.MonkeyPatch, name: str | None, expected: str
):
    for variable in (
        "VLLM_HCU_USE_FLASH_ATTN",
        "VLLM_HCU_USE_FLASH_ATTN_UNIFIED",
        "VLLM_HCU_USE_CUSTOM_FLASH_ATTN",
    ):
        monkeypatch.delenv(variable, raising=False)
    if name is not None:
        monkeypatch.setenv(name, "1")
    assert hcu_envs.resolve_hcu_flash_attn_mode(None) == expected


def test_explicit_flash_attention_mode_wins_over_legacy_environment(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_FLASH_ATTN", "1")
    assert hcu_envs.resolve_hcu_flash_attn_mode("classic") == "classic"
    assert hcu_envs.resolve_hcu_flash_attn_mode("unified") == "cutlass"


@pytest.mark.parametrize(
    ("payload", "error"),
    [
        ({"hcu_flash_attn_mode": 1}, TypeError),
        ({"hcu_flash_attn_mode": "future"}, ValueError),
    ],
)
def test_platform_flash_attention_mode_rejects_invalid_sidecar(
    monkeypatch: pytest.MonkeyPatch,
    payload: dict[str, object],
    error: type[Exception],
):
    import torch
    import vllm.config as vllm_config_module

    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: type("Properties", (), {"gcnArchName": "gfx936"})(),
    )
    import vllm_hcu.platforms.hcu as hcu_module

    monkeypatch.setattr(
        vllm_config_module,
        "get_current_vllm_config_or_none",
        lambda: {"additional_config": {"hcu": payload}},
    )
    with pytest.raises(error):
        hcu_module.get_hcu_flash_attn_mode()


@pytest.mark.parametrize(
    ("mode", "expected_shape", "expected_block_dim"),
    [
        (
            "custom",
            ((3, 4, 64, 128), (3, 4, 128, 64)),
            0,
        ),
        ("cutlass", (2, 3, 64, 4, 128), 1),
    ],
)
def test_flash_attention_kv_cache_contract_follows_resolved_mode(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    expected_shape: object,
    expected_block_dim: int,
):
    monkeypatch.setitem(
        sys.modules,
        "vllm_hcu.hcu_ops",
        ModuleType("vllm_hcu.hcu_ops"),
    )
    flash_attn_extension = ModuleType("flash_attn")
    for symbol in (
        "flash_attn_varlen_func",
        "vllm_flash_attn_varlen_func",
        "hg_flash_attn_varlen_func",
        "varlen_fwd_unified",
    ):
        setattr(flash_attn_extension, symbol, lambda *args, **kwargs: None)
    monkeypatch.setitem(sys.modules, "flash_attn", flash_attn_extension)
    from vllm_hcu.v1.attention.backends import flash_attn

    monkeypatch.setattr(flash_attn, "_get_flash_attn_mode", lambda: mode)
    monkeypatch.setattr(flash_attn, "get_kv_cache_layout", lambda: "NHD")
    backend = flash_attn.HcuFlashAttentionBackend

    assert backend.get_kv_cache_shape(3, 64, 4, 128) == expected_shape
    assert backend.get_kv_cache_block_dim(64, 4, 128) == expected_block_dim
    if mode == "custom":
        assert backend.get_kv_cache_stride_order() == (
            (0, 1, 2, 3),
            (0, 1, 2, 3),
        )
    else:
        assert backend.get_kv_cache_stride_order() == (0, 1, 2, 3, 4)


class _FakeKernel:
    def __init__(self) -> None:
        self.launches: list[tuple[object, tuple[object, ...], dict[str, object]]] = []

    def __getitem__(self, grid: object):
        def launch(*args: object, **kwargs: object):
            self.launches.append((grid, args, kwargs))
            return "launched"

        return launch


def _unified_attention(
    q,
    k,
    v,
    out,
    cu_seqlens_q,
    max_seqlen_q,
    seqused_k,
    max_seqlen_k,
    softmax_scale,
    causal,
    window_size,
    block_table,
    softcap,
    q_descale,
    k_descale,
    v_descale,
    seq_threshold_3D=None,
    num_par_softmax_segments=None,
    softmax_segm_output=None,
    softmax_segm_max=None,
    softmax_segm_expsum=None,
    alibi_slopes=None,
    output_scale=None,
    qq_bias=None,
    sinks=None,
    mm_prefix_range=None,
    rswa_prefix_lens=None,
    rswa_window=None,
    use_alibi_sqrt=False,
    kv_quant_mode=None,
    k_scale_cache=None,
    v_scale_cache=None,
    chunk_lookback=-1,
    use_td=False,
    mm_prefix_clamp_sliding_window=False,
):
    del (
        q,
        k,
        v,
        out,
        cu_seqlens_q,
        max_seqlen_q,
        seqused_k,
        max_seqlen_k,
        softmax_scale,
        causal,
        window_size,
        block_table,
        softcap,
        q_descale,
        k_descale,
        v_descale,
        seq_threshold_3D,
        num_par_softmax_segments,
        softmax_segm_output,
        softmax_segm_max,
        softmax_segm_expsum,
        alibi_slopes,
        output_scale,
        qq_bias,
        sinks,
        mm_prefix_range,
        rswa_prefix_lens,
        rswa_window,
        use_alibi_sqrt,
        kv_quant_mode,
        k_scale_cache,
        v_scale_cache,
        chunk_lookback,
        use_td,
        mm_prefix_clamp_sliding_window,
    )


def test_unified_attention_proxy_forces_single_stage_and_preserves_other_kwargs():
    kernel = _FakeKernel()
    module = ModuleType(patch_triton_unified_attention.TARGET_MODULE)
    module.unified_attention = _unified_attention
    module.kernel_unified_attention = kernel

    assert patch_triton_unified_attention.apply_to_module(module) is True
    assert patch_triton_unified_attention.apply_to_module(module) is False
    launcher = module.kernel_unified_attention[(3, 7)]
    assert launcher("q", num_stages=2, num_warps=8) == "launched"
    assert kernel.launches == [
        ((3, 7), ("q",), {"num_stages": 1, "num_warps": 8})
    ]
