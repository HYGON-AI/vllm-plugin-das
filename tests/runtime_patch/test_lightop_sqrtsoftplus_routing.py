# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from vllm_hcu.model_executor.layers.fused_moe.sqrtsoftplus_routing import (
    can_use_lightop_sqrtsoftplus,
)
from vllm_hcu.patch.worker.op_opt.moe import patch_fused_topk_bias_router
from vllm_hcu.patch.worker.op_opt.moe._common import PatchCompatibilityError
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
    logits = _CudaTensorMetadata((1024, 256), torch.float32)
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
        ((0, 256), (256,), 6, None, None),
        ((2, 256), (256,), 6, None, None),
        ((256, 256), (256,), 6, None, None),
        ((2, 256), (256,), 17, None, None),
        ((1024, 256), (256,), 16, None, None),
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
        gating_output=_CudaTensorMetadata((1024, 256), torch.float32),
        correction_bias=_CudaTensorMetadata((256,), torch.float32),
    )

    assert actual_weights is expected_weights
    assert torch.equal(actual_ids, expected_ids.to(torch.int64))
    assert actual_ids.dtype is torch.int64


def test_sqrtsoftplus_patch_rejects_stale_wrapper() -> None:
    module = _target_module(object())
    assert patch_fused_topk_bias_router.apply_to_module(module)
    module.vllm_topk_softplus_sqrt = lambda *_args, **_kwargs: None

    with pytest.raises(PatchCompatibilityError, match="stale HCU MoE marker"):
        patch_fused_topk_bias_router.apply_to_module(module)


def test_sqrtsoftplus_kernel_failure_does_not_emit_success_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_hcu.model_executor.layers.fused_moe import sqrtsoftplus_routing

    messages: list[str] = []
    _install_lightop_moe(
        monkeypatch,
        moe_fused_gate_sqrtsoftplus=lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("kernel failed")
        ),
    )
    monkeypatch.setattr(
        sqrtsoftplus_routing,
        "logger",
        SimpleNamespace(warning_once=messages.append),
    )

    with pytest.raises(RuntimeError, match="kernel failed"):
        sqrtsoftplus_routing.run_lightop_sqrtsoftplus(
            torch.empty((2, 256), dtype=torch.float32),
            torch.empty((256,), dtype=torch.float32),
            topk=6,
            renormalize=True,
            routed_scaling_factor=1.0,
            indices_dtype=torch.int32,
        )
    assert messages == []


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


def test_ineligible_hash_route_does_not_probe_optional_lightop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_hcu.model_executor.layers.fused_moe import sqrtsoftplus_routing

    monkeypatch.setattr(henvs, "VLLM_HCU_USE_CUSTOM_OPS", True)
    monkeypatch.setattr(
        henvs,
        "VLLM_HCU_USE_LIGHTOP_SQRTSOFTPLUS_GATE",
        True,
        raising=False,
    )
    monkeypatch.setattr(
        sqrtsoftplus_routing,
        "is_lightop_sqrtsoftplus_available",
        lambda: pytest.fail("ineligible input must not probe LightOp"),
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


def test_hash_routing_uses_official_torch_fallback_when_compiled_op_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_lightop_moe(
        monkeypatch,
        moe_fused_gate_sqrtsoftplus=lambda *args, **kwargs: pytest.fail(
            "hash routing must not execute LightOp"
        ),
    )
    monkeypatch.setattr(henvs, "VLLM_HCU_USE_CUSTOM_OPS", True)
    monkeypatch.setattr(
        henvs,
        "VLLM_HCU_USE_LIGHTOP_SQRTSOFTPLUS_GATE",
        True,
        raising=False,
    )
    native_result = object()
    native_calls: list[tuple[object, ...]] = []

    def unavailable_original(
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
        pytest.fail("missing compiled operator must not be invoked")

    def native_fallback(*args):
        native_calls.append(args)
        return native_result

    module = _module(
        patch_fused_topk_bias_router.TARGET_MODULE,
        vllm_topk_softplus_sqrt=unavailable_original,
        _topk_softplus_sqrt_torch=native_fallback,
        torch=SimpleNamespace(ops=SimpleNamespace(_moe_C=SimpleNamespace())),
    )
    patch_fused_topk_bias_router.apply_to_module(module)

    assert _call_patched(
        module,
        gating_output=_CudaTensorMetadata((2, 256), torch.float32),
        correction_bias=_CudaTensorMetadata((256,), torch.float32),
        input_tokens=torch.tensor([1, 2], dtype=torch.int64),
        hash_indices_table=torch.tensor([[1], [2], [3]], dtype=torch.int64),
    ) is native_result
    assert len(native_calls) == 1
    assert native_calls[0][6].dtype is torch.int64
    assert native_calls[0][7].dtype is torch.int64


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
