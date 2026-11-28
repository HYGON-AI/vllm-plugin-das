# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Unit coverage for HCU-specific AITER MoE dispatch guards."""

from __future__ import annotations

from types import ModuleType, SimpleNamespace

import pytest
import torch

from vllm_hcu.model_executor.layers.fused_moe import aiter_moe_dispatch
from vllm_hcu.model_executor.layers.fused_moe.aiter_moe_dispatch import (
    HcuAiterMoeDispatchError,
)


def _fake_moe_c_module(*, omit_selected_key: bool = False) -> ModuleType:
    module = ModuleType("aiter.fused_moe_c")
    top_configs = {
        256: {"MODE": 10001101, "BLOCK_SIZE_M": 16},
        512: {"MODE": 10202101, "BLOCK_SIZE_M": 32},
    }
    bottom_configs = {
        256: {"MODE": 20001103, "BLOCK_SIZE_M": 16},
        512: {"MODE": 20202102, "BLOCK_SIZE_M": 32},
    }

    def try_get_optimal_moe_config_marlin(
        w1_shape,
        w2_shape,
        top_k,
        dtype,
        M,
        is_marlin=False,
        block_shape=None,
        is_bottom=False,
        use_moe_wna16_cuda=False,
        indtype=None,
    ):
        del (
            w1_shape,
            w2_shape,
            top_k,
            dtype,
            M,
            is_marlin,
            block_shape,
            use_moe_wna16_cuda,
            indtype,
        )
        selected = dict((bottom_configs if is_bottom else top_configs)[512])
        if not omit_selected_key:
            selected["selected_m_key"] = 512
        return selected

    def get_moe_configs_marlin(
        experts,
        intermediate,
        dtype,
        device_id,
        num_cu,
        is_bottom,
        use_moe_wna16_cuda,
        *,
        K,
        top_k,
        indtype,
    ):
        del (
            experts,
            intermediate,
            dtype,
            device_id,
            num_cu,
            use_moe_wna16_cuda,
            K,
            top_k,
            indtype,
        )
        return bottom_configs if is_bottom else top_configs

    module.try_get_optimal_moe_config_marlin = try_get_optimal_moe_config_marlin
    module.get_moe_configs_marlin = get_moe_configs_marlin
    return module


def _select_config(module: ModuleType, *, is_bottom: bool) -> dict[str, int]:
    return module.try_get_optimal_moe_config_marlin(
        (256, 512, 6144),
        (256, 6144, 256),
        8,
        "fp8_w8a8",
        512,
        is_bottom=is_bottom,
    )


def test_gfx938_fp8_guard_replaces_both_stages_with_same_safe_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _fake_moe_c_module()
    monkeypatch.setattr(aiter_moe_dispatch, "import_module", lambda name: module)

    aiter_moe_dispatch._install_gfx938_fp8_moe_c_config_guard()

    assert _select_config(module, is_bottom=False) == {
        "MODE": 10001101,
        "BLOCK_SIZE_M": 16,
        "selected_m_key": 256,
    }
    assert _select_config(module, is_bottom=True) == {
        "MODE": 20001103,
        "BLOCK_SIZE_M": 16,
        "selected_m_key": 256,
    }


def test_gfx938_fp8_guard_rejects_broken_config_without_pair_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _fake_moe_c_module(omit_selected_key=True)
    monkeypatch.setattr(aiter_moe_dispatch, "import_module", lambda name: module)
    aiter_moe_dispatch._install_gfx938_fp8_moe_c_config_guard()

    with pytest.raises(HcuAiterMoeDispatchError, match="without a paired table key"):
        _select_config(module, is_bottom=False)


def _execute_fp8_moe(
    monkeypatch: pytest.MonkeyPatch,
    *,
    block_shape: list[int] | None,
) -> torch.Tensor:
    expected = torch.ones((2, 4))
    moe_module = SimpleNamespace(aiter_moe=lambda **kwargs: expected)
    monkeypatch.setattr(
        aiter_moe_dispatch,
        "import_module",
        lambda name: moe_module,
    )
    monkeypatch.setattr(
        aiter_moe_dispatch,
        "_install_gfx938_fp8_moe_c_config_guard",
        lambda: pytest.fail("config guard must not be installed for this route"),
    )
    config = SimpleNamespace(solution_type="moe_c", quant_type="fp8_w8a8")
    hidden_states = torch.ones((2, 4))
    return aiter_moe_dispatch.execute_aiter_moe(
        config,
        hidden_states=hidden_states,
        w1=torch.ones((2, 8, 4)),
        w2=torch.ones((2, 4, 4)),
        topk_weights=torch.ones((2, 1)),
        topk_ids=torch.zeros((2, 1), dtype=torch.int32),
        block_shape=block_shape,
    )


def test_fp8_moe_on_non_gfx938_does_not_install_gfx938_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actual = _execute_fp8_moe(monkeypatch, block_shape=None)

    torch.testing.assert_close(actual, torch.ones((2, 4)))


def test_block_fp8_moe_does_not_install_channel_fp8_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(aiter_moe_dispatch, "_is_gfx938_device", lambda device: True)

    actual = _execute_fp8_moe(monkeypatch, block_shape=[128, 128])

    torch.testing.assert_close(actual, torch.ones((2, 4)))
