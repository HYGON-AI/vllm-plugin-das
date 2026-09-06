# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from __future__ import annotations

import sys
from types import ModuleType

import pytest
import torch

from vllm_hcu.model_executor.layers.fused_moe.sqrtsoftplus_routing import (
    can_use_lightop_sqrtsoftplus,
)
from vllm_hcu.patch.worker.op_opt.moe import patch_fused_topk_bias_router
from vllm_hcu.platforms import envs as henvs


class _CudaTensorMetadata:
    def __init__(self, shape: tuple[int, ...], dtype: torch.dtype) -> None:
        self.shape = shape
        self.dtype = dtype
        self.device = torch.device("cuda")

    def is_contiguous(self) -> bool:
        return True


def _module(name: str, **attributes: object) -> ModuleType:
    module = ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


def _install_lightop_moe(
    monkeypatch: pytest.MonkeyPatch,
    **exports: object,
) -> ModuleType:
    lightop = _module("lightop")
    lightop.__path__ = []
    moe = _module("lightop.moe", **exports)
    lightop.moe = moe
    monkeypatch.setitem(sys.modules, "lightop", lightop)
    monkeypatch.setitem(sys.modules, "lightop.moe", moe)
    return moe


def _target_module(official_result: object) -> ModuleType:
    def original(
        topk_weights,
        topk_indices,
        token_expert_indices,
        gating_output,
        renormalize=False,
        e_score_correction_bias=None,
        input_tokens=None,
        hash_indices_table=None,
        routed_scaling_factor=1.0,
    ):
        del (
            topk_weights,
            topk_indices,
            token_expert_indices,
            gating_output,
            renormalize,
            e_score_correction_bias,
            input_tokens,
            hash_indices_table,
            routed_scaling_factor,
        )
        return official_result

    return _module(
        patch_fused_topk_bias_router.TARGET_MODULE,
        vllm_topk_softplus_sqrt=original,
    )


def _call_patched(
    module: ModuleType,
    *,
    gating_output: object,
    correction_bias: object,
    input_tokens: torch.Tensor | None = None,
    hash_indices_table: torch.Tensor | None = None,
):
    indices = torch.empty((2, 6), dtype=torch.int64)
    weights = torch.empty((2, 6), dtype=torch.float32)
    return module.vllm_topk_softplus_sqrt(
        weights,
        indices,
        torch.empty((2, 6), dtype=torch.int32),
        gating_output,
        True,
        correction_bias,
        input_tokens,
        hash_indices_table,
        1.5,
    )


def test_sqrtsoftplus_eligibility_accepts_supported_non_hash_metadata() -> None:
    logits = _CudaTensorMetadata((2, 256), torch.float32)
    bias = _CudaTensorMetadata((256,), torch.float32)

    assert can_use_lightop_sqrtsoftplus(
        logits,
        bias,
        topk=6,
        input_tokens=None,
        hash_indices_table=None,
    )


@pytest.mark.parametrize(
    "logit_shape, bias_shape, topk, input_tokens, hash_table",
    (
        ((2, 128), (128,), 6, None, None),
        ((2, 256), (256,), 17, None, None),
        ((2, 256), (255,), 6, None, None),
        ((2, 256), (256,), 6, torch.tensor([1]), torch.tensor([[1]])),
    ),
)
def test_sqrtsoftplus_eligibility_rejects_unsupported_metadata(
    logit_shape: tuple[int, ...],
    bias_shape: tuple[int, ...],
    topk: int,
    input_tokens: torch.Tensor | None,
    hash_table: torch.Tensor | None,
) -> None:
    assert not can_use_lightop_sqrtsoftplus(
        _CudaTensorMetadata(logit_shape, torch.float32),
        _CudaTensorMetadata(bias_shape, torch.float32),
        topk=topk,
        input_tokens=input_tokens,
        hash_indices_table=hash_table,
    )


def test_non_hash_sqrtsoftplus_uses_lightop_when_both_switches_are_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_weights = torch.tensor(
        [[0.75, 0.25, 0.0, 0.0, 0.0, 0.0]] * 2,
        dtype=torch.float32,
    )
    expected_ids = torch.tensor([[3, 7, 0, 0, 0, 0]] * 2, dtype=torch.int32)

    def lightop_gate(*args, **kwargs):
        del args, kwargs
        return expected_weights, expected_ids

    _install_lightop_moe(
        monkeypatch,
        moe_fused_gate_sqrtsoftplus=lightop_gate,
    )
    monkeypatch.setattr(henvs, "VLLM_HCU_USE_CUSTOM_OPS", True)
    monkeypatch.setattr(
        henvs,
        "VLLM_HCU_USE_LIGHTOP_SQRTSOFTPLUS_GATE",
        True,
        raising=False,
    )
    official_result = object()
    module = _target_module(official_result)
    patch_fused_topk_bias_router.apply_to_module(module)

    actual_weights, actual_ids = _call_patched(
        module,
        gating_output=_CudaTensorMetadata((2, 256), torch.float32),
        correction_bias=_CudaTensorMetadata((256,), torch.float32),
    )

    assert actual_weights is expected_weights
    assert torch.equal(actual_ids, expected_ids.to(torch.int64))
    assert actual_ids.dtype is torch.int64


@pytest.mark.parametrize("master, leaf", ((False, True), (True, False)))
def test_sqrtsoftplus_switches_preserve_official_fallback(
    monkeypatch: pytest.MonkeyPatch,
    master: bool,
    leaf: bool,
) -> None:
    _install_lightop_moe(
        monkeypatch,
        moe_fused_gate_sqrtsoftplus=lambda *args, **kwargs: pytest.fail(
            "disabled route must not execute LightOp"
        ),
    )
    monkeypatch.setattr(henvs, "VLLM_HCU_USE_CUSTOM_OPS", master)
    monkeypatch.setattr(
        henvs,
        "VLLM_HCU_USE_LIGHTOP_SQRTSOFTPLUS_GATE",
        leaf,
        raising=False,
    )
    official_result = object()
    module = _target_module(official_result)
    patch_fused_topk_bias_router.apply_to_module(module)

    assert _call_patched(
        module,
        gating_output=_CudaTensorMetadata((2, 256), torch.float32),
        correction_bias=_CudaTensorMetadata((256,), torch.float32),
    ) is official_result


def test_hash_routing_never_calls_lightop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_lightop_moe(
        monkeypatch,
        moe_fused_gate_sqrtsoftplus=lambda *args, **kwargs: pytest.fail(
            "hash routing must remain on the official vLLM operator"
        ),
    )
    monkeypatch.setattr(henvs, "VLLM_HCU_USE_CUSTOM_OPS", True)
    monkeypatch.setattr(
        henvs,
        "VLLM_HCU_USE_LIGHTOP_SQRTSOFTPLUS_GATE",
        True,
        raising=False,
    )
    official_result = object()
    module = _target_module(official_result)
    patch_fused_topk_bias_router.apply_to_module(module)

    assert _call_patched(
        module,
        gating_output=_CudaTensorMetadata((2, 256), torch.float32),
        correction_bias=_CudaTensorMetadata((256,), torch.float32),
        input_tokens=torch.tensor([1, 2], dtype=torch.int64),
        hash_indices_table=torch.tensor([[1], [2], [3]], dtype=torch.int64),
    ) is official_result


@pytest.mark.parametrize("missing_export", (None, object()))
def test_enabled_sqrtsoftplus_missing_operator_uses_official_fallback(
    monkeypatch: pytest.MonkeyPatch,
    missing_export: object,
) -> None:
    exports = (
        {}
        if missing_export is None
        else {"moe_fused_gate_sqrtsoftplus": missing_export}
    )
    _install_lightop_moe(monkeypatch, **exports)
    monkeypatch.setattr(henvs, "VLLM_HCU_USE_CUSTOM_OPS", True)
    monkeypatch.setattr(
        henvs,
        "VLLM_HCU_USE_LIGHTOP_SQRTSOFTPLUS_GATE",
        True,
        raising=False,
    )
    official_result = object()
    module = _target_module(official_result)
    patch_fused_topk_bias_router.apply_to_module(module)

    assert _call_patched(
        module,
        gating_output=_CudaTensorMetadata((2, 256), torch.float32),
        correction_bias=_CudaTensorMetadata((256,), torch.float32),
    ) is official_result
