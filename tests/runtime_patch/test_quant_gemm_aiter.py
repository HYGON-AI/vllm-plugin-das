# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from __future__ import annotations

import ast
import enum
import os
import subprocess
import sys
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from vllm_hcu.model_executor.layers.fused_moe import aiter_runtime
from vllm_hcu.model_executor.layers.quantization import (
    compressed_tensors_moe_runtime,
    int8_runtime,
    lightop_fp8_runtime,
)
from vllm_hcu.patch.worker.op_opt import (
    patch_activation,
    patch_aiter_ops,
    patch_compressed_tensors,
    patch_compressed_tensors_moe_w8a8_fp8,
    patch_compressed_tensors_moe_wna16,
    patch_compressed_tensors_scheme,
    patch_compressed_tensors_w8a8_fp8,
    patch_compressed_tensors_w8a8_int8,
    patch_deep_gemm,
    patch_input_quant_fp8,
    patch_layers_utils,
    patch_scaled_mm_linear_kernel,
    patch_w8a8_utils,
)


def _module(name: str, **attributes: object) -> ModuleType:
    module = ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


def _package(name: str, **attributes: object) -> ModuleType:
    module = _module(name, **attributes)
    module.__package__ = name
    module.__path__ = []  # type: ignore[attr-defined]
    return module


def _fp8_quant_abi_stub(
    x,
    scale=None,
    quant_dtype=torch.int8,
    num_rows=None,
    num_rows_factor=1,
):
    del quant_dtype, num_rows, num_rows_factor
    return x, scale


def test_aiter_asm_int8_quant_context_routes_only_enabled_calls(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[tuple[str, object]] = []

    def native_quant(x):
        calls.append(("aiter", x))
        return "native"

    def boltops_quant(x):
        calls.append(("boltops", x))
        return "aligned"

    asm_module = _module(
        "aiter.fused_moe_asm_wna16",
        per_token_quant_int8=native_quant,
    )
    monkeypatch.setitem(
        sys.modules,
        "aiter.fused_moe_asm_wna16",
        asm_module,
    )
    monkeypatch.setitem(
        sys.modules,
        "boltops.fused_moe.triton.moe_compat",
        _module(
            "boltops.fused_moe.triton.moe_compat",
            per_token_quant_hip=boltops_quant,
        ),
    )

    assert asm_module.per_token_quant_int8("before") == "native"
    with aiter_runtime.aiter_asm_boltops_int8_quant_context(enabled=True):
        assert asm_module.per_token_quant_int8("inside") == "aligned"
    assert asm_module.per_token_quant_int8("after") == "native"
    assert calls == [
        ("aiter", "before"),
        ("boltops", "inside"),
        ("aiter", "after"),
    ]


def test_aiter_asm_boltops_fp8_quant_context_aligns_both_fp8_quant_calls(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[tuple[str, object, object]] = []

    def native_quant(
        x,
        scale=None,
        quant_dtype=torch.int8,
        num_rows=None,
        num_rows_factor=1,
    ):
        del scale, num_rows, num_rows_factor
        calls.append(("aiter", x, quant_dtype))
        return "aiter_quant"

    boltops_output = torch.ones((1, 1), dtype=torch.float8_e4m3fn)
    boltops_scale = torch.ones((1, 1), dtype=torch.float32)

    def boltops_quant(x, scale=None, quant_dtype=torch.int8, **kwargs):
        assert scale is None
        assert kwargs == {}
        calls.append(("boltops", x, quant_dtype))
        return boltops_output, boltops_scale

    asm_module = _module(
        "aiter.fused_moe_asm_wna16",
        per_token_quant_hip=native_quant,
    )
    boltops_module = _module(
        "boltops.fused_moe.triton.moe_compat",
        per_token_quant_hip=boltops_quant,
    )
    monkeypatch.setitem(sys.modules, "aiter.fused_moe_asm_wna16", asm_module)
    monkeypatch.setitem(
        sys.modules,
        "boltops.fused_moe.triton.moe_compat",
        boltops_module,
    )

    with aiter_runtime.aiter_asm_boltops_fp8_quant_context(enabled=True):
        gemm1_output = asm_module.per_token_quant_hip(
            "gemm1_input", quant_dtype=torch.float8_e4m3fn
        )
        gemm2_output = asm_module.per_token_quant_hip(
            "gemm2_input", quant_dtype=torch.float8_e4m3fn
        )
        torch.testing.assert_close(gemm1_output[0], boltops_output)
        torch.testing.assert_close(gemm1_output[1], boltops_scale)
        torch.testing.assert_close(gemm2_output[0], boltops_output)
        torch.testing.assert_close(gemm2_output[1], boltops_scale)
        assert asm_module.per_token_quant_hip(
            "int8_input", quant_dtype=torch.int8
        ) == "aiter_quant"
    assert asm_module.per_token_quant_hip(
        "outside", quant_dtype=torch.float8_e4m3fn
    ) == "aiter_quant"
    assert calls == [
        ("boltops", "gemm1_input", torch.float8_e4m3fn),
        ("boltops", "gemm2_input", torch.float8_e4m3fn),
        ("aiter", "int8_input", torch.int8),
        ("aiter", "outside", torch.float8_e4m3fn),
    ]


def test_aiter_asm_boltops_fp8_quant_repairs_zero_scale_rows(
    monkeypatch: pytest.MonkeyPatch,
):
    quantized = torch.full((2, 4), float("nan"), dtype=torch.float8_e4m3fn)
    scales = torch.zeros((2, 1), dtype=torch.float32)
    monkeypatch.setitem(
        sys.modules,
        "boltops.fused_moe.triton.moe_compat",
        _module(
            "boltops.fused_moe.triton.moe_compat",
            per_token_quant_hip=lambda x, **kwargs: (quantized, scales),
        ),
    )

    output, output_scales = aiter_runtime._boltops_per_token_quant_fp8(
        torch.zeros((2, 4), dtype=torch.bfloat16)
    )

    torch.testing.assert_close(output.float(), torch.zeros((2, 4)))
    assert torch.isfinite(output.float()).all()
    torch.testing.assert_close(
        output_scales,
        torch.full_like(scales, 1.0e-10)
        * (1.0 / torch.finfo(torch.float8_e4m3fn).max),
    )


def test_aiter_asm_boltops_fp8_quant_context_preserves_native_activation(
    monkeypatch: pytest.MonkeyPatch,
):
    def native_activation(
        activation,
        is_gated,
        activated_out,
        ffn1_out_2d,
        gemm1_alpha,
        gemm1_limit,
    ):
        del activation, is_gated, ffn1_out_2d, gemm1_alpha, gemm1_limit
        activated_out.fill_(1)

    asm_module = _module(
        "aiter.fused_moe_asm_wna16",
        _apply_activation=native_activation,
        per_token_quant_hip=_fp8_quant_abi_stub,
    )
    monkeypatch.setitem(sys.modules, "aiter.fused_moe_asm_wna16", asm_module)
    monkeypatch.setitem(
        sys.modules,
        "boltops.fused_moe.triton.moe_compat",
        _module(
            "boltops.fused_moe.triton.moe_compat",
            per_token_quant_hip=lambda x, **kwargs: (x, kwargs),
        ),
    )
    output = torch.empty((2, 4))

    with aiter_runtime.aiter_asm_boltops_fp8_quant_context(enabled=True):
        assert asm_module._apply_activation is native_activation
        asm_module._apply_activation(
            "silu", True, output, torch.empty((2, 8)), None, None
        )
    torch.testing.assert_close(output, torch.ones_like(output))


def test_aiter_dynamo_metrics_patch_preserves_nonserializable_runtime_config(
    monkeypatch: pytest.MonkeyPatch,
):
    import torch._dynamo.utils as dynamo_utils

    calls = 0

    def nonserializable_config():
        nonlocal calls
        calls += 1
        raise TypeError("Object of type function is not JSON serializable")

    monkeypatch.setattr(
        dynamo_utils,
        "_get_dynamo_config_for_logging",
        nonserializable_config,
    )

    aiter_runtime._install_aiter_dynamo_metrics_logging_patch()
    patched = dynamo_utils._get_dynamo_config_for_logging

    assert patched() == "Dynamo Config is not JSON serializable"
    assert calls == 1
    aiter_runtime._install_aiter_dynamo_metrics_logging_patch()
    assert dynamo_utils._get_dynamo_config_for_logging is patched


def test_aiter_asm_boltops_fp8_quant_context_nested_disable_restores_state(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[str] = []

    def native_quant(
        x,
        scale=None,
        quant_dtype=torch.int8,
        num_rows=None,
        num_rows_factor=1,
    ):
        del x, scale, quant_dtype, num_rows, num_rows_factor
        calls.append("aiter")
        return "aiter"

    boltops_output = torch.ones((1, 1), dtype=torch.float8_e4m3fn)
    boltops_scale = torch.ones((1, 1), dtype=torch.float32)

    def boltops_quant(x, **kwargs):
        del x, kwargs
        calls.append("boltops")
        return boltops_output, boltops_scale

    asm_module = _module(
        "aiter.fused_moe_asm_wna16",
        per_token_quant_hip=native_quant,
    )
    monkeypatch.setitem(sys.modules, "aiter.fused_moe_asm_wna16", asm_module)
    monkeypatch.setitem(
        sys.modules,
        "boltops.fused_moe.triton.moe_compat",
        _module(
            "boltops.fused_moe.triton.moe_compat",
            per_token_quant_hip=boltops_quant,
        ),
    )

    with aiter_runtime.aiter_asm_boltops_fp8_quant_context(enabled=True):
        torch.testing.assert_close(
            asm_module.per_token_quant_hip(
                "outer", quant_dtype=torch.float8_e4m3fn
            )[0],
            boltops_output,
        )
        with aiter_runtime.aiter_asm_boltops_fp8_quant_context(enabled=False):
            assert asm_module.per_token_quant_hip(
                "disabled", quant_dtype=torch.float8_e4m3fn
            ) == "aiter"
        with pytest.raises(RuntimeError, match="cleanup"):
            with aiter_runtime.aiter_asm_boltops_fp8_quant_context(enabled=True):
                raise RuntimeError("cleanup")
        torch.testing.assert_close(
            asm_module.per_token_quant_hip(
                "outer-again", quant_dtype=torch.float8_e4m3fn
            )[0],
            boltops_output,
        )

    assert asm_module.per_token_quant_hip(
        "outside", quant_dtype=torch.float8_e4m3fn
    ) == "aiter"
    assert calls == ["boltops", "aiter", "boltops", "aiter"]


@pytest.mark.parametrize(
    "quant_kwargs",
    [
        {"quant_dtype": torch.int8},
        {"quant_dtype": torch.float8_e4m3fn, "scale": torch.ones(1)},
        {"quant_dtype": torch.float8_e4m3fn, "num_rows": torch.ones(1)},
        {"quant_dtype": torch.float8_e4m3fn, "num_rows_factor": 2},
    ],
)
def test_aiter_asm_boltops_fp8_quant_context_preserves_unsupported_quant_modes(
    monkeypatch: pytest.MonkeyPatch,
    quant_kwargs: dict[str, object],
):
    calls: list[str] = []

    def native_quant(
        x,
        scale=None,
        quant_dtype=torch.int8,
        num_rows=None,
        num_rows_factor=1,
    ):
        del x, scale, quant_dtype, num_rows, num_rows_factor
        calls.append("aiter")
        return "aiter"

    asm_module = _module(
        "aiter.fused_moe_asm_wna16",
        per_token_quant_hip=native_quant,
    )
    monkeypatch.setitem(sys.modules, "aiter.fused_moe_asm_wna16", asm_module)
    monkeypatch.setitem(
        sys.modules,
        "boltops.fused_moe.triton.moe_compat",
        _module(
            "boltops.fused_moe.triton.moe_compat",
            per_token_quant_hip=lambda x, **kwargs: "boltops",
        ),
    )

    with aiter_runtime.aiter_asm_boltops_fp8_quant_context(enabled=True):
        assert asm_module.per_token_quant_hip("input", **quant_kwargs) == "aiter"
    assert calls == ["aiter"]


def test_aiter_asm_boltops_fp8_quant_context_rejects_incompatible_quant_abi(
    monkeypatch: pytest.MonkeyPatch,
):
    def incompatible_quant(x):
        return x

    asm_module = _module(
        "aiter.fused_moe_asm_wna16",
        per_token_quant_hip=incompatible_quant,
    )
    monkeypatch.setitem(sys.modules, "aiter.fused_moe_asm_wna16", asm_module)

    with pytest.raises(
        aiter_runtime.HcuAiterRuntimeError,
        match="per_token_quant_hip exposes unsupported arguments",
    ):
        with aiter_runtime.aiter_asm_boltops_fp8_quant_context(enabled=True):
            pass
    assert asm_module.per_token_quant_hip is incompatible_quant


@pytest.mark.parametrize("abi_change", ["extra_parameter", "changed_default"])
def test_aiter_asm_boltops_fp8_quant_context_rejects_subtle_abi_drift(
    monkeypatch: pytest.MonkeyPatch,
    abi_change: str,
):
    if abi_change == "extra_parameter":

        def incompatible_quant(
            x,
            scale=None,
            quant_dtype=torch.int8,
            num_rows=None,
            num_rows_factor=1,
            stochastic=False,
        ):
            del x, scale, quant_dtype, num_rows, num_rows_factor, stochastic

    else:

        def incompatible_quant(
            x,
            scale=None,
            quant_dtype=torch.int8,
            num_rows=None,
            num_rows_factor=2,
        ):
            del x, scale, quant_dtype, num_rows, num_rows_factor

    asm_module = _module(
        "aiter.fused_moe_asm_wna16",
        per_token_quant_hip=incompatible_quant,
    )
    monkeypatch.setitem(sys.modules, "aiter.fused_moe_asm_wna16", asm_module)

    with pytest.raises(
        aiter_runtime.HcuAiterRuntimeError,
        match="per_token_quant_hip exposes unsupported",
    ):
        with aiter_runtime.aiter_asm_boltops_fp8_quant_context(enabled=True):
            pass
    assert asm_module.per_token_quant_hip is incompatible_quant


def test_int8_aiter_oracle_maps_explicit_backend_and_keeps_canonical_weights(
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm_hcu.patch.worker.op_opt.moe import patch_int8_oracle

    class Int8MoeBackend(enum.Enum):
        TRITON = "TRITON"
        HUMMING = "HUMMING"
        CPU = "CPU"

    class AiterExperts:
        pass

    def backend_to_kernel_cls(backend):
        return [f"original:{backend.value}"]

    def map_int8_backend(runner_backend):
        if runner_backend == "triton":
            return Int8MoeBackend.TRITON
        raise ValueError(runner_backend)

    def convert_to_int8_moe_kernel_format(
        int8_backend,
        w13,
        w2,
        layer=None,
        w13_scale=None,
    ):
        del layer, w13_scale
        if int8_backend != Int8MoeBackend.TRITON:
            raise ValueError(int8_backend)
        return w13 + 1, w2 + 1

    def make_int8_moe_quant_config(
        int8_backend,
        w1_scale,
        w2_scale,
        a1_scale=None,
        a2_scale=None,
        w1_bias=None,
        w2_bias=None,
        per_act_token_quant=False,
        layer=None,
    ):
        del (
            int8_backend,
            w1_scale,
            w2_scale,
            a1_scale,
            a2_scale,
            w1_bias,
            w2_bias,
            per_act_token_quant,
            layer,
        )
        return "original-w8a16"

    def int8_w8a8_moe_quant_config(
        w1_scale,
        w2_scale,
        a1_scale,
        a2_scale,
        w1_bias=None,
        w2_bias=None,
        per_act_token_quant=False,
    ):
        return SimpleNamespace(
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            a1_scale=a1_scale,
            a2_scale=a2_scale,
            w1_bias=w1_bias,
            w2_bias=w2_bias,
            per_act_token_quant=per_act_token_quant,
            use_int8_w8a8=True,
        )

    target = _module(
        patch_int8_oracle.TARGET_MODULE,
        Enum=enum.Enum,
        Int8MoeBackend=Int8MoeBackend,
        backend_to_kernel_cls=backend_to_kernel_cls,
        map_int8_backend=map_int8_backend,
        convert_to_int8_moe_kernel_format=convert_to_int8_moe_kernel_format,
        make_int8_moe_quant_config=make_int8_moe_quant_config,
        int8_w8a8_moe_quant_config=int8_w8a8_moe_quant_config,
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe",
        _module(
            "vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe",
            AiterExperts=AiterExperts,
        ),
    )

    assert patch_int8_oracle.apply_to_module(target) is True
    assert target.map_int8_backend("aiter") == target.Int8MoeBackend.AITER
    assert target.map_int8_backend("triton").value == "TRITON"
    assert target.backend_to_kernel_cls(target.Int8MoeBackend.AITER) == [
        AiterExperts
    ]

    w13 = torch.zeros((2, 4, 3), dtype=torch.int8)
    w2 = torch.zeros((2, 3, 2), dtype=torch.int8)
    converted_w13, converted_w2 = target.convert_to_int8_moe_kernel_format(
        target.Int8MoeBackend.AITER,
        w13,
        w2,
    )
    assert converted_w13 is w13
    assert converted_w2 is w2

    w1_scale = torch.ones((2, 4, 1))
    w2_scale = torch.ones((2, 3, 1))
    quant_config = target.make_int8_moe_quant_config(
        target.Int8MoeBackend.AITER,
        w1_scale,
        w2_scale,
        per_act_token_quant=True,
    )
    assert quant_config.use_int8_w8a8 is True
    assert quant_config.per_act_token_quant is True
    assert quant_config.a1_scale is None
    assert quant_config.a2_scale is None


def test_worker_registers_int8_aiter_oracle_before_quantized_methods():
    from vllm_hcu.patch import worker

    callbacks = worker.worker_callback_names()
    int8_entry = (
        "worker.op_opt.moe.oracle.int8_aiter",
        "vllm.model_executor.layers.fused_moe.oracle.int8",
    )
    fp8_method_entry = (
        "worker.op_opt.compressed_tensors.moe_w8a8_fp8",
        "vllm.model_executor.layers.quantization.compressed_tensors."
        "compressed_tensors_moe.compressed_tensors_moe_w8a8_fp8",
    )
    assert int8_entry in callbacks
    assert callbacks.index(int8_entry) < callbacks.index(fp8_method_entry)


def _install_fake_vllm_envs(
    monkeypatch: pytest.MonkeyPatch,
    **attributes: object,
) -> ModuleType:
    """Install a complete fake package edge for ``import vllm.envs``.

    Adding only the child to ``sys.modules`` still makes Python import the real
    parent package.  That both defeats isolation and can leave a partially
    initialized ``vllm`` behind for later tests when the child is deliberately
    minimal.
    """

    envs = _module("vllm.envs", **attributes)
    vllm = _package("vllm", envs=envs)
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.envs", envs)
    return envs


def _install_fake_vllm_torch_utils(
    monkeypatch: pytest.MonkeyPatch,
) -> object:
    """Provide only the custom-op registration dependency under test."""

    def direct_register_custom_op(**kwargs):
        return kwargs

    torch_utils = _module(
        "vllm.utils.torch_utils",
        direct_register_custom_op=direct_register_custom_op,
    )
    utils = _package("vllm.utils", torch_utils=torch_utils)
    vllm = _package("vllm", utils=utils)
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.utils", utils)
    monkeypatch.setitem(sys.modules, "vllm.utils.torch_utils", torch_utils)
    return direct_register_custom_op


def test_aiter_gfx93x_capability_extends_upstream(monkeypatch: pytest.MonkeyPatch):
    fake_hcu = _module("vllm_hcu.platforms.hcu", on_gfx93x=lambda: True)
    monkeypatch.setitem(sys.modules, "vllm_hcu.platforms.hcu", fake_hcu)
    platform = SimpleNamespace(is_rocm=lambda: True)
    assert aiter_runtime.is_aiter_found_and_supported(
        lambda: False, platform, True
    )
    assert not aiter_runtime.is_aiter_found_and_supported(
        lambda: False, platform, False
    )


class _QuantType(enum.IntEnum):
    No = 0
    Other = 1


def _install_fake_aiter(
    monkeypatch: pytest.MonkeyPatch, **attributes: object
) -> ModuleType:
    module = _module("aiter", QuantType=_QuantType, **attributes)
    module.__path__ = []  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "aiter", module)
    return module


def test_aiter_w8a8_tuning_capability_requires_runtime_and_target_config(
    monkeypatch: pytest.MonkeyPatch,
):
    aiter = _install_fake_aiter(
        monkeypatch,
        gemm_a8w8_bpreshuffle=lambda: None,
        gemm_a8w8_CK=lambda: None,
    )
    configs = SimpleNamespace(
        AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE_FILE="/configs/preshuffle.csv",
        AITER_CONFIG_GEMM_A8W8_FILE="/configs/per-token.csv",
    )
    monkeypatch.setitem(sys.modules, "aiter.ops", _package("aiter.ops"))
    monkeypatch.setitem(
        sys.modules,
        "aiter.ops.gemm_op_a8w8",
        _module("aiter.ops.gemm_op_a8w8", AITER_CONFIGS=configs),
    )

    assert aiter_runtime.get_w8a8_tuned_config_path(
        "gemm_a8w8_bpreshuffle",
        "AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE_FILE",
    ) == "/configs/preshuffle.csv"
    assert aiter_runtime.get_w8a8_tuned_config_path(
        "gemm_a8w8_CK",
        "AITER_CONFIG_GEMM_A8W8_FILE",
    ) == "/configs/per-token.csv"

    delattr(aiter, "gemm_a8w8_bpreshuffle")
    assert (
        aiter_runtime.get_w8a8_tuned_config_path(
            "gemm_a8w8_bpreshuffle",
            "AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE_FILE",
        )
        is None
    )


def test_aiter_w8a8_tuning_capability_fails_closed_on_expected_api_drift(
    monkeypatch: pytest.MonkeyPatch,
):
    _install_fake_aiter(monkeypatch, gemm_a8w8_CK=lambda: None)
    monkeypatch.setitem(sys.modules, "aiter.ops", _package("aiter.ops"))

    for name, gemm_module in (
        ("missing-config-owner", _module("aiter.ops.gemm_op_a8w8")),
        (
            "missing-config-attribute",
            _module(
                "aiter.ops.gemm_op_a8w8",
                AITER_CONFIGS=SimpleNamespace(),
            ),
        ),
        (
            "invalid-empty-path",
            _module(
                "aiter.ops.gemm_op_a8w8",
                AITER_CONFIGS=SimpleNamespace(AITER_CONFIG_GEMM_A8W8_FILE=""),
            ),
        ),
    ):
        monkeypatch.setitem(sys.modules, "aiter.ops.gemm_op_a8w8", gemm_module)
        assert (
            aiter_runtime.get_w8a8_tuned_config_path(
                "gemm_a8w8_CK",
                "AITER_CONFIG_GEMM_A8W8_FILE",
            )
            is None
        ), name


def test_aiter_w8a8_tuning_capability_does_not_hide_unexpected_abi_error(
    monkeypatch: pytest.MonkeyPatch,
):
    aiter = _module("aiter", gemm_a8w8_CK=lambda: None)

    for error in (
        OSError("unexpected AITER loader ABI error"),
        ImportError("undefined symbol: proprietary_aiter_abi"),
        AttributeError("unexpected module initialization failure"),
    ):
        def fake_import(name: str, *, failure=error):
            if name == "aiter":
                return aiter
            raise failure

        monkeypatch.setattr(aiter_runtime, "import_module", fake_import)
        with pytest.raises(type(error), match=str(error)):
            aiter_runtime.get_w8a8_tuned_config_path(
                "gemm_a8w8_CK",
                "AITER_CONFIG_GEMM_A8W8_FILE",
            )


def test_aiter_w8a8_tuning_capability_short_circuits_broken_submodule(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[str] = []

    def fake_import(name: str):
        calls.append(name)
        if name == "aiter":
            return _module("aiter")
        raise AssertionError("the unavailable runtime must short-circuit")

    monkeypatch.setattr(aiter_runtime, "import_module", fake_import)
    assert (
        aiter_runtime.get_w8a8_tuned_config_path(
            "gemm_a8w8_CK",
            "AITER_CONFIG_GEMM_A8W8_FILE",
        )
        is None
    )
    assert calls == ["aiter"]


def test_optional_aiter_module_distinguishes_absence_from_transitive_failure(
    monkeypatch: pytest.MonkeyPatch,
):
    requested = "aiter.ops.gemm_op_a8w8"

    def missing_requested(name: str):
        raise ModuleNotFoundError(
            f"No module named {name!r}",
            name=name,
        )

    monkeypatch.setattr(aiter_runtime, "import_module", missing_requested)
    assert aiter_runtime._import_optional_aiter_module(requested) is None

    def missing_transitive(name: str):
        raise ModuleNotFoundError(
            "No module named 'proprietary_abi_dependency'",
            name="proprietary_abi_dependency",
        )

    monkeypatch.setattr(aiter_runtime, "import_module", missing_transitive)
    with pytest.raises(ModuleNotFoundError, match="proprietary_abi_dependency"):
        aiter_runtime._import_optional_aiter_module(requested)


def test_aiter_triton_fp8_bmm_capability_is_symbol_aware(
    monkeypatch: pytest.MonkeyPatch,
):
    symbol = aiter_runtime._AITER_TRITON_FP8_BMM_SYMBOL
    module = _module(
        aiter_runtime._AITER_TRITON_FP8_BMM_MODULE,
        **{symbol: lambda: None},
    )
    monkeypatch.setattr(aiter_runtime, "import_module", lambda name: module)
    assert aiter_runtime.has_triton_fp8_bmm()

    delattr(module, symbol)
    assert not aiter_runtime.has_triton_fp8_bmm()

    def missing_requested(name: str):
        raise ModuleNotFoundError(f"No module named {name!r}", name=name)

    monkeypatch.setattr(aiter_runtime, "import_module", missing_requested)
    assert not aiter_runtime.has_triton_fp8_bmm()

    def missing_transitive(name: str):
        raise ModuleNotFoundError(
            "No module named 'triton_runtime_abi'",
            name="triton_runtime_abi",
        )

    monkeypatch.setattr(aiter_runtime, "import_module", missing_transitive)
    with pytest.raises(ModuleNotFoundError, match="triton_runtime_abi"):
        aiter_runtime.has_triton_fp8_bmm()


def test_aiter_triton_fp8_bmm_env_gates_short_circuit_capability_probe(
    monkeypatch: pytest.MonkeyPatch,
):
    def unexpected_probe():
        raise AssertionError("disabled environment gates must short-circuit")

    monkeypatch.setattr(aiter_runtime, "has_triton_fp8_bmm", unexpected_probe)
    assert not aiter_runtime.is_triton_fp8_bmm_enabled(False, False)
    assert not aiter_runtime.is_triton_fp8_bmm_enabled(False, True)
    assert not aiter_runtime.is_triton_fp8_bmm_enabled(True, False)

    monkeypatch.setattr(aiter_runtime, "has_triton_fp8_bmm", lambda: False)
    assert not aiter_runtime.is_triton_fp8_bmm_enabled(True, True)
    monkeypatch.setattr(aiter_runtime, "has_triton_fp8_bmm", lambda: True)
    assert aiter_runtime.is_triton_fp8_bmm_enabled(True, True)


def _operator_schema(*names: str) -> SimpleNamespace:
    return SimpleNamespace(
        arguments=[SimpleNamespace(name=name) for name in names]
    )


@pytest.mark.parametrize(
    ("op_name", "legacy_arguments"),
    tuple(aiter_runtime._AITER_RMSNORM_DYNAMIC_QUANT_ARGUMENTS.items()),
)
@pytest.mark.parametrize(
    ("profile", "suffix"),
    (
        ("legacy-default", ()),
        (
            "model-sensitive",
            (aiter_runtime._AITER_MODEL_SENSITIVE_RMSNORM_ARGUMENT,),
        ),
    ),
)
def test_aiter_rmsnorm_dynamic_quant_abi_requires_exact_known_schema(
    monkeypatch: pytest.MonkeyPatch,
    op_name: str,
    legacy_arguments: tuple[str, ...],
    profile: str,
    suffix: tuple[str, ...],
):
    overload = SimpleNamespace(
        _schema=_operator_schema(*(legacy_arguments + suffix))
    )
    namespace = SimpleNamespace(
        **{op_name: SimpleNamespace(default=overload)}
    )
    monkeypatch.setattr(
        aiter_runtime,
        "torch",
        SimpleNamespace(ops=SimpleNamespace(aiter=namespace)),
    )

    assert aiter_runtime._aiter_rmsnorm_dynamic_quant_abi(op_name) == profile


@pytest.mark.parametrize(
    "arguments",
    (
        (),
        ("a0", "a1", "a2", "a3", "a4"),
        ("input", "out", "yscale", "weight", "epsilon"),
        ("out", "input", "yscale", "weight", "epsilon", "unexpected"),
    ),
)
def test_aiter_rmsnorm_dynamic_quant_abi_fails_closed_on_unknown_schema(
    monkeypatch: pytest.MonkeyPatch,
    arguments: tuple[str, ...],
):
    op_name = "rmsnorm2d_fwd_with_dynamicquant"
    overload = SimpleNamespace(_schema=_operator_schema(*arguments))
    namespace = SimpleNamespace(
        **{op_name: SimpleNamespace(default=overload)}
    )
    monkeypatch.setattr(
        aiter_runtime,
        "torch",
        SimpleNamespace(ops=SimpleNamespace(aiter=namespace)),
    )

    with pytest.raises(
        aiter_runtime.HcuAiterRuntimeError,
        match="no readable operator schema|unsupported arguments",
    ):
        aiter_runtime._aiter_rmsnorm_dynamic_quant_abi(op_name)


def test_aiter_rmsnorm_dynamic_quant_abi_fails_closed_when_op_is_missing(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        aiter_runtime,
        "torch",
        SimpleNamespace(ops=SimpleNamespace(aiter=SimpleNamespace())),
    )
    with pytest.raises(
        aiter_runtime.HcuAiterRuntimeError,
        match="no readable operator schema",
    ):
        aiter_runtime._aiter_rmsnorm_dynamic_quant_abi(
            "rmsnorm2d_fwd_with_dynamicquant"
        )


@pytest.mark.parametrize("fused_add", [False, True])
@pytest.mark.parametrize(
    ("profile", "expected_kwargs"),
    (
        ("legacy-default", {}),
        ("model-sensitive", {"use_model_sensitive_rmsnorm": 0}),
    ),
)
def test_aiter_rmsnorm_int8_calls_each_supported_abi_once(
    monkeypatch: pytest.MonkeyPatch,
    fused_add: bool,
    profile: str,
    expected_kwargs: dict[str, int],
):
    monkeypatch.setattr(
        aiter_runtime,
        "_aiter_rmsnorm_dynamic_quant_abi",
        lambda op_name: profile,
    )
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def operation(*args, **kwargs):
        calls.append((args, kwargs))

    x = torch.ones(2, 4)
    weight = torch.ones(4)
    if fused_add:
        result = aiter_runtime.rmsnorm_add_dynamic_quant_impl(
            operation,
            x,
            torch.full_like(x, 2),
            weight,
            1e-6,
            torch.int8,
        )
        assert len(result) == 3
        assert len(calls[0][0]) == 7
    else:
        result = aiter_runtime.rmsnorm_dynamic_quant_impl(
            operation, x, weight, 1e-6, torch.int8
        )
        assert len(result) == 2
        assert len(calls[0][0]) == 5
    assert len(calls) == 1
    assert calls[0][1] == expected_kwargs


@pytest.mark.parametrize("fused_add", [False, True])
def test_legacy_aiter_rmsnorm_fp8_uses_vllm_native_fallback_only(
    monkeypatch: pytest.MonkeyPatch,
    fused_add: bool,
):
    monkeypatch.setattr(
        aiter_runtime,
        "_aiter_rmsnorm_dynamic_quant_abi",
        lambda op_name: "legacy-default",
    )
    native_calls: list[tuple[object, ...]] = []

    def native(x, weight, epsilon, quant_dtype, residual=None):
        native_calls.append((x, weight, epsilon, quant_dtype, residual))
        output = torch.empty_like(x, dtype=quant_dtype)
        scale = torch.empty(x.shape[0], 1)
        residual_out = residual.clone() if residual is not None else None
        return output, scale, residual_out

    monkeypatch.setattr(
        aiter_runtime,
        "_vllm_native_rmsnorm_dynamic_quant",
        native,
    )

    def unexpected_aiter(*args, **kwargs):
        raise AssertionError("legacy AITER FP8 kernel must not run")

    x = torch.ones(2, 4)
    weight = torch.ones(4)
    if fused_add:
        residual = torch.full_like(x, 2)
        output, residual_out, scale = (
            aiter_runtime.rmsnorm_add_dynamic_quant_impl(
                unexpected_aiter,
                x,
                residual,
                weight,
                1e-6,
                torch.float8_e4m3fn,
            )
        )
        assert native_calls[0][4] is residual
        assert residual_out is not residual
        torch.testing.assert_close(residual_out, residual)
    else:
        output, scale = aiter_runtime.rmsnorm_dynamic_quant_impl(
            unexpected_aiter,
            x,
            weight,
            1e-6,
            torch.float8_e4m3fn,
        )
        assert native_calls[0][4] is None
    assert output.dtype is torch.float8_e4m3fn
    assert scale.shape == (2, 1)
    assert len(native_calls) == 1


def test_vllm_native_rmsnorm_fallback_validates_schema_and_clones_residual(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[tuple[object, ...]] = []

    class NativeOperation:
        _schema = _operator_schema(
            *aiter_runtime._VLLM_NATIVE_RMSNORM_DYNAMIC_QUANT_ARGUMENTS
        )

        def __call__(self, *args):
            calls.append(args)
            output, x, _weight, scale, _epsilon, scale_ub, residual = args
            output.fill_(1)
            scale.fill_(2)
            assert scale_ub is None
            if residual is not None:
                residual.add_(x)

    torch_proxy = SimpleNamespace(
        empty=torch.empty,
        float32=torch.float32,
        ops=SimpleNamespace(
            _C=SimpleNamespace(
                rms_norm_dynamic_per_token_quant=SimpleNamespace(
                    default=NativeOperation()
                )
            )
        ),
    )
    monkeypatch.setattr(aiter_runtime, "torch", torch_proxy)
    monkeypatch.setattr(aiter_runtime, "import_module", lambda name: object())

    x = torch.ones(2, 4)
    residual = torch.full_like(x, 2)
    original_x = x.clone()
    original_residual = residual.clone()
    output, scale, residual_out = (
        aiter_runtime._vllm_native_rmsnorm_dynamic_quant(
            x,
            torch.ones(4),
            1e-6,
            torch.float8_e4m3fn,
            residual,
        )
    )

    assert len(calls) == 1
    assert output.dtype is torch.float8_e4m3fn
    assert torch.all(scale == 2)
    assert residual_out is not None and residual_out is not residual
    torch.testing.assert_close(residual_out, original_x + original_residual)
    torch.testing.assert_close(x, original_x)
    torch.testing.assert_close(residual, original_residual)


@pytest.mark.parametrize("fused_add", [False, True])
def test_aiter_rmsnorm_kernel_errors_propagate_without_retry(
    monkeypatch: pytest.MonkeyPatch,
    fused_add: bool,
):
    monkeypatch.setattr(
        aiter_runtime,
        "_aiter_rmsnorm_dynamic_quant_abi",
        lambda op_name: "model-sensitive",
    )
    calls = 0

    def broken_operation(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError("AITER kernel execution failed")

    x = torch.ones(2, 4)
    if fused_add:
        call = lambda: aiter_runtime.rmsnorm_add_dynamic_quant_impl(
            broken_operation,
            x,
            torch.ones_like(x),
            torch.ones(4),
            1e-6,
            torch.int8,
        )
    else:
        call = lambda: aiter_runtime.rmsnorm_dynamic_quant_impl(
            broken_operation,
            x,
            torch.ones(4),
            1e-6,
            torch.int8,
        )
    with pytest.raises(RuntimeError, match="AITER kernel execution failed"):
        call()
    assert calls == 1


def test_aiter_replacement_rmsnorm_wrappers_delegate_without_retry_logic():
    source = (
        Path(__file__).resolve().parents[2]
        / "vllm_hcu/model_executor/layers/fused_moe/aiter_ops.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }
    expected = {
        "_rocm_aiter_rmsnorm_fused_dynamic_quant_impl": (
            "rmsnorm_dynamic_quant_impl"
        ),
        "_rocm_aiter_rmsnorm_fused_add_dynamic_quant_impl": (
            "rmsnorm_add_dynamic_quant_impl"
        ),
    }
    for function_name, runtime_name in expected.items():
        function = functions[function_name]
        calls = [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == runtime_name
        ]
        assert len(calls) == 1, function_name
        assert not any(isinstance(node, ast.Try) for node in ast.walk(function))


def test_aiter_replacement_maps_each_optional_capability_exactly():
    source = (
        Path(__file__).resolve().parents[2]
        / "vllm_hcu/model_executor/layers/fused_moe/aiter_ops.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    owner = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "rocm_aiter_ops"
    )
    methods = {
        node.name: node
        for node in owner.body
        if isinstance(node, ast.FunctionDef)
    }
    expected = {
        "is_shuffled_per_token_w8a8_gemm_tuned": (
            "gemm_a8w8_bpreshuffle",
            "AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE_FILE",
        ),
        "is_per_token_w8a8_gemm_tuned": (
            "gemm_a8w8_CK",
            "AITER_CONFIG_GEMM_A8W8_FILE",
        ),
    }
    for method_name, expected_arguments in expected.items():
        calls = [
            node
            for node in ast.walk(methods[method_name])
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get_w8a8_tuned_config_path"
        ]
        assert len(calls) == 1, method_name
        assert tuple(ast.literal_eval(arg) for arg in calls[0].args) == expected_arguments
        assert any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_check_kernel_tuned"
            for node in ast.walk(methods[method_name])
        ), method_name

    fp8_bmm_source = ast.unparse(methods["is_fp8bmm_enabled"])
    assert "_hcu_runtime.is_triton_fp8_bmm_enabled" in fp8_bmm_source
    fused_moe_source = ast.unparse(methods["is_fused_moe_enabled"])
    assert "_hcu_runtime.is_aiter_moe_requested()" in fused_moe_source


def test_aiter_replacement_uses_workspace_aiter_module_layout():
    source = (
        Path(__file__).resolve().parents[2]
        / "vllm_hcu/model_executor/layers/fused_moe/aiter_ops.py"
    ).read_text(encoding="utf-8")

    assert "from aiter.fused_moe_asm import asm_moe_tkw1" in source
    assert "from aiter.ops.triton.rope import (" in source
    assert "_hcu_runtime.triton_rope_and_cache_impl" in source
    assert "aiter.fused_moe_bf16_asm" not in source
    assert "aiter.ops.triton.rope.rope" not in source
    assert "aiter.ops.triton.fused_kv_cache" not in source
    assert "aiter.ops.triton.fused_fp8_quant" not in source
    assert "aiter.ops.triton.fused_add_rmsnorm_pad" not in source
    assert "aiter.ops.fused_qk_rmsnorm_group_quant" not in source


def test_workspace_aiter_rope_and_cache_composes_public_ops(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def rope(*args, **kwargs):
        calls.append(("rope", args, kwargs))

    def cache_flash(*args, **kwargs):
        calls.append(("cache_flash", args, kwargs))

    def cache(*args, **kwargs):
        calls.append(("cache", args, kwargs))

    aiter = _package("aiter")
    ops = _package("aiter.ops")
    triton = _package("aiter.ops.triton")
    cache_module = _module(
        "aiter.ops.cache",
        reshape_and_cache=cache,
        reshape_and_cache_flash=cache_flash,
    )
    rope_module = _module(
        "aiter.ops.triton.rope",
        rope_cached_thd_positions_2c_fwd_inplace=rope,
    )
    monkeypatch.setitem(sys.modules, "aiter", aiter)
    monkeypatch.setitem(sys.modules, "aiter.ops", ops)
    monkeypatch.setitem(sys.modules, "aiter.ops.triton", triton)
    monkeypatch.setitem(sys.modules, "aiter.ops.cache", cache_module)
    monkeypatch.setitem(sys.modules, "aiter.ops.triton.rope", rope_module)

    query = torch.zeros(2, 8)
    key = torch.zeros(2, 4)
    value = torch.zeros(2, 1, 4)
    positions = torch.tensor([0, 1])
    cos_sin_cache = torch.zeros(4, 8)
    key_cache = torch.zeros(1)
    value_cache = torch.zeros(1)
    slots = torch.tensor([0, 1])
    k_scale = torch.tensor(2.0)
    v_scale = torch.tensor(3.0)

    aiter_runtime.triton_rope_and_cache_impl(
        query,
        key,
        value,
        positions,
        cos_sin_cache,
        True,
        key_cache,
        value_cache,
        slots,
        k_scale,
        v_scale,
        True,
        True,
    )
    assert [call[0] for call in calls] == ["rope", "cache_flash"]
    assert calls[0][1][0].shape == (2, 2, 4)
    assert calls[0][1][1].shape == (2, 1, 4)
    assert calls[0][1][5] == 0
    assert calls[1][1][5] == "fp8"

    calls.clear()
    aiter_runtime.triton_rope_and_cache_impl(
        query,
        key,
        value,
        positions,
        cos_sin_cache,
        False,
        key_cache,
        value_cache,
        slots,
        k_scale,
        v_scale,
        False,
        False,
    )
    assert [call[0] for call in calls] == ["rope", "cache"]
    assert calls[0][1][5] == 1
    assert calls[1][1][5:] == ("auto", 2.0, 3.0, False)


def test_hcu_aiter_moe_uses_v0251_out_of_place_contract():
    repo = Path(__file__).resolve().parents[2]
    sources = (
        repo
        / "vllm_hcu/model_executor/layers/quantization/"
        "compressed_tensors_moe_runtime.py",
        repo
        / "vllm_hcu/model_executor/layers/quantization/compressed_tensors/"
        "compressed_tensors_moe_marlin.py",
    )

    for source_path in sources:
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        assert not any(
            isinstance(node, ast.Attribute) and node.attr == "disable_inplace"
            for node in ast.walk(tree)
        ), source_path
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "aiter_moe"
        ]
        assert calls, source_path
        for call in calls:
            inplace = next(
                (keyword.value for keyword in call.keywords if keyword.arg == "inplace"),
                None,
            )
            assert isinstance(inplace, ast.Constant), source_path
            assert inplace.value is False, source_path


def test_aiter_gelu_tanh_and_feature_off_delegation(
    monkeypatch: pytest.MonkeyPatch,
):
    class ActivationType:
        GeluTanh = object()

    _install_fake_aiter(monkeypatch, ActivationType=ActivationType)
    fused_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fused_moe(*args, **kwargs):
        fused_calls.append((args, kwargs))
        return "gelu_tanh"

    monkeypatch.setitem(
        sys.modules,
        "aiter.fused_moe",
        _module("aiter.fused_moe", fused_moe=fused_moe),
    )
    _install_fake_vllm_envs(
        monkeypatch,
        VLLM_ROCM_USE_AITER=False,
        VLLM_ROCM_USE_AITER_MOE=False,
    )
    calls: list[tuple[object, ...]] = []

    def original(*args):
        calls.append(args)
        return "upstream"

    x = torch.ones(1, 2)
    w1 = torch.ones(1, 4, 2)
    w2 = torch.ones(1, 2, 2)
    topk_weight = torch.ones(1, 1)
    topk_ids = torch.zeros(1, 1, dtype=torch.int64)
    assert (
        aiter_runtime.fused_moe_impl(
            original, x, w1, w2, topk_weight, topk_ids, activation_method=0
        )
        == "upstream"
    )
    assert len(calls) == 1
    assert (
        aiter_runtime.fused_moe_impl(
            original, x, w1, w2, topk_weight, topk_ids, activation_method=3
        )
        == "gelu_tanh"
    )
    assert fused_calls[0][0][6] == "gelu_tanh"
    assert aiter_runtime.get_gelu_tanh_activation_type() is ActivationType.GeluTanh


def test_aiter_activation_string_mapping(monkeypatch: pytest.MonkeyPatch):
    sentinel = object()

    class ActivationType:
        GeluTanh = sentinel

    _install_fake_aiter(monkeypatch, ActivationType=ActivationType)
    original = lambda value: {"silu": "silu"}.get(value)
    assert (
        aiter_runtime.get_aiter_activation_type(original, "gelu_tanh")
        is sentinel
    )
    assert (
        aiter_runtime.get_aiter_activation_type(
            original, "GELU_PYTORCH_TANH"
        )
        is sentinel
    )
    assert aiter_runtime.get_aiter_activation_type(original, "silu") == "silu"


def test_aiter_gate_mode_requires_compatible_abi():
    assert aiter_runtime.aiter_gate_mode_kwargs(
        "",
        supports_gate_mode=False,
    ) == {}
    assert aiter_runtime.aiter_gate_mode_kwargs(
        "interleave",
        supports_gate_mode=True,
    ) == {"gate_mode": "interleave"}

    with pytest.raises(
        aiter_runtime.HcuAiterRuntimeError,
        match="does not support.*gate_mode='separated'",
    ):
        aiter_runtime.aiter_gate_mode_kwargs(
            "separated",
            supports_gate_mode=False,
        )


def test_aiter_w16a16_asm_selection(monkeypatch: pytest.MonkeyPatch):
    _install_fake_aiter(monkeypatch)
    _install_fake_vllm_envs(
        monkeypatch,
        VLLM_ROCM_USE_AITER=True,
        VLLM_ROCM_USE_AITER_MOE=True,
    )
    asm_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def asm(*args, **kwargs):
        asm_calls.append((args, kwargs))
        return "asm"

    monkeypatch.setitem(
        sys.modules,
        "aiter.fused_moe_asm_wna16",
        _module("aiter.fused_moe_asm_wna16", fused_experts_asm_impl=asm),
    )
    from vllm_hcu.platforms import envs as henvs

    monkeypatch.setattr(henvs, "VLLM_HCU_USE_AITER_W16A16_MOE_SHUFFLE", False)
    monkeypatch.setattr(henvs, "VLLM_HCU_USE_AITER_MOE_CONFIG", False)
    x = torch.ones(2, 4)
    w1 = torch.ones(3, 8, 4)
    w2 = torch.ones(3, 4, 4)
    topk_weight = torch.ones(2, 2)
    topk_ids = torch.zeros(2, 2, dtype=torch.int64)

    def upstream(*args):
        raise AssertionError("ASM selection must not delegate")

    assert (
        aiter_runtime.fused_moe_impl(
            upstream, x, w1, w2, topk_weight, topk_ids, activation_method=3
        )
        == "asm"
    )
    assert asm_calls[0][1] == {
        "activation": "gelu_tanh",
        "global_num_experts": 3,
        "expert_map": None,
        "use_shuffle": 0,
    }

    assert (
        aiter_runtime.fused_moe_impl(
            upstream,
            x,
            w1,
            w2,
            topk_weight,
            topk_ids,
            activation_method=3,
            swiglu_limit=7.5,
        )
        == "asm"
    )
    assert asm_calls[1][1]["gemm1_limit"] == 7.5

    with pytest.raises(
        aiter_runtime.HcuAiterRuntimeError,
        match="has no gate_mode ABI.*gate_mode",
    ):
        aiter_runtime.fused_moe_impl(
            upstream,
            x,
            w1,
            w2,
            topk_weight,
            topk_ids,
            gate_mode="interleave",
        )
    with pytest.raises(
        aiter_runtime.HcuAiterRuntimeError,
        match="has no sorting-dispatch ABI.*moe_sorting_dispatch_policy",
    ):
        aiter_runtime.fused_moe_impl(
            upstream,
            x,
            w1,
            w2,
            topk_weight,
            topk_ids,
            moe_sorting_dispatch_policy=2,
        )


@pytest.mark.parametrize("use_shuffle", [False, True])
def test_aiter_w16a16_weight_preparation_bypasses_rocm_padding(
    monkeypatch: pytest.MonkeyPatch,
    use_shuffle: bool,
):
    from vllm_hcu.model_executor.layers.fused_moe import (
        unquantized_fused_moe_method as hcu_unquantized,
    )

    w13 = torch.nn.Parameter(torch.ones(2, 8, 4), requires_grad=False)
    w2 = torch.nn.Parameter(torch.ones(2, 4, 4), requires_grad=False)
    layer = SimpleNamespace(
        w13_weight=w13,
        w2_weight=w2,
        w13_bias=None,
        w2_bias=None,
        activation=SimpleNamespace(value="silu"),
        apply_router_weight_on_input=False,
        layer_name="model.layers.0.mlp.experts",
    )
    method = object.__new__(hcu_unquantized.HcuUnquantizedFusedMoEMethod)
    method.moe = SimpleNamespace()
    method.unquantized_backend = hcu_unquantized.UnquantizedMoeBackend.AITER
    method.experts_cls = object
    method.get_fused_moe_quant_config = lambda unused_layer: object()

    def emulate_rocm_padding(weight: torch.Tensor) -> torch.Tensor:
        padded = torch.nn.functional.pad(weight, (0, 1))[..., :-1]
        assert not padded.is_contiguous()
        return padded

    method._maybe_pad_weight = emulate_rocm_padding
    monkeypatch.setattr(
        hcu_unquantized,
        "_is_hcu_aiter_moe_asm_requested",
        lambda method=None: True,
    )
    monkeypatch.setattr(
        hcu_unquantized,
        "_raise_if_aiter_moe_asm_blocked",
        lambda *args: None,
    )
    monkeypatch.setattr(
        hcu_unquantized.henvs,
        "VLLM_HCU_USE_AITER_W16A16_MOE_SHUFFLE",
        use_shuffle,
    )
    monkeypatch.setattr(
        hcu_unquantized,
        "make_unquantized_moe_kernel",
        lambda **kwargs: object(),
    )

    shuffle_inputs: list[tuple[torch.Tensor, torch.Tensor]] = []

    def shuffle_weights(w1, w2, unused_config):
        shuffle_inputs.append((w1, w2))
        return w1.contiguous(), w2.contiguous()

    monkeypatch.setitem(
        sys.modules,
        "aiter.moe",
        _module(
            "aiter.moe",
            AiterMoeConfig=lambda **kwargs: SimpleNamespace(**kwargs),
            MoeQuantType=SimpleNamespace(W16A16="w16a16"),
            MoeSolutionType=SimpleNamespace(ASM="asm"),
            aiter_moe_shfl_weight=shuffle_weights,
        ),
    )

    method.process_weights_after_loading(layer)

    if use_shuffle:
        assert len(shuffle_inputs) == 1
        assert all(weight.is_contiguous() for weight in shuffle_inputs[0])
    else:
        assert layer.w13_weight is w13
        assert layer.w2_weight is w2
        assert layer.w13_weight.is_contiguous()
        assert layer.w2_weight.is_contiguous()


def test_explicit_aiter_backend_selects_asm_without_auto_env_gate(
    monkeypatch: pytest.MonkeyPatch,
):
    _install_fake_aiter(monkeypatch)
    _install_fake_vllm_envs(
        monkeypatch,
        VLLM_ROCM_USE_AITER=False,
        VLLM_ROCM_USE_AITER_MOE=False,
    )
    monkeypatch.setitem(
        sys.modules,
        "aiter.fused_moe_asm_wna16",
        _module(
            "aiter.fused_moe_asm_wna16",
            fused_experts_asm_impl=lambda *args, **kwargs: (args, kwargs),
        ),
    )
    from vllm_hcu.platforms import envs as henvs

    monkeypatch.setattr(henvs, "VLLM_HCU_USE_AITER_W16A16_MOE_SHUFFLE", False)
    monkeypatch.setattr(henvs, "VLLM_HCU_USE_AITER_MOE_CONFIG", False)
    x = torch.ones(2, 4)
    w1 = torch.ones(3, 8, 4)
    w2 = torch.ones(3, 4, 4)
    topk_weight = torch.ones(2, 2)
    topk_ids = torch.zeros(2, 2, dtype=torch.int64)

    with aiter_runtime.aiter_moe_request_context(
        SimpleNamespace(moe_backend="aiter")
    ):
        args, kwargs = aiter_runtime.fused_moe_impl(
            lambda *unused: pytest.fail("explicit AITER must not delegate"),
            x,
            w1,
            w2,
            topk_weight,
            topk_ids,
        )
    assert all(
        actual is expected
        for actual, expected in zip(
            args[:5], (x, w1, w2, topk_weight, topk_ids)
        )
    )
    assert kwargs["activation"] == "silu"


def test_explicit_aiter_backend_enables_mask_construction_from_current_config(
    monkeypatch: pytest.MonkeyPatch,
):
    _install_fake_vllm_envs(
        monkeypatch,
        VLLM_ROCM_USE_AITER=False,
        VLLM_ROCM_USE_AITER_MOE=False,
    )
    config = _module(
        "vllm.config",
        get_current_vllm_config_or_none=lambda: SimpleNamespace(
            kernel_config=SimpleNamespace(moe_backend="aiter")
        ),
    )
    setattr(sys.modules["vllm"], "config", config)
    monkeypatch.setitem(sys.modules, "vllm.config", config)

    assert aiter_runtime.is_aiter_moe_requested()


def test_configured_w16a16_prefers_direct_asm(
    monkeypatch: pytest.MonkeyPatch,
):
    _install_fake_aiter(monkeypatch)
    _install_fake_vllm_envs(
        monkeypatch,
        VLLM_ROCM_USE_AITER=True,
        VLLM_ROCM_USE_AITER_MOE=True,
    )
    from vllm_hcu.platforms import envs as henvs

    monkeypatch.setattr(henvs, "VLLM_HCU_USE_AITER_W16A16_MOE_SHUFFLE", True)
    monkeypatch.setattr(henvs, "VLLM_HCU_USE_AITER_MOE_CONFIG", True)
    monkeypatch.setattr(
        aiter_runtime,
        "get_w16a16_moe_config",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        aiter_runtime,
        "get_w16a16_moe_solution_id",
        lambda *args, **kwargs: "4+9",
    )

    def unified_aiter(**kwargs):
        pytest.fail("configured W16A16 must not call aiter.moe.aiter_moe")

    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def direct_asm(*args, **kwargs):
        calls.append((args, kwargs))
        return "direct-asm"

    monkeypatch.setitem(
        sys.modules,
        "aiter.moe",
        _module("aiter.moe", aiter_moe=unified_aiter),
    )
    monkeypatch.setitem(
        sys.modules,
        "aiter.fused_moe_asm_wna16",
        _module(
            "aiter.fused_moe_asm_wna16",
            fused_experts_asm_impl=direct_asm,
        ),
    )

    hidden_states = torch.ones(2, 6, dtype=torch.bfloat16)
    w1 = torch.ones(3, 8, 6, dtype=torch.bfloat16)
    w2 = torch.ones(3, 6, 4, dtype=torch.bfloat16)
    topk_weight = torch.ones(2, 2)
    topk_ids = torch.zeros(2, 2, dtype=torch.int64)

    assert (
        aiter_runtime.fused_moe_impl(
            lambda *unused: pytest.fail("AITER ASM must not delegate"),
            hidden_states,
            w1,
            w2,
            topk_weight,
            topk_ids,
        )
        == "direct-asm"
    )
    assert calls[0][1] == {
        "activation": "silu",
        "global_num_experts": 3,
        "expert_map": None,
        "use_shuffle": 1,
        "solution_id": "4+9",
    }


def test_aiter_w16a16_solution_lookup_uses_w2_output_dim(
    monkeypatch: pytest.MonkeyPatch,
):
    _install_fake_aiter(monkeypatch)
    _install_fake_vllm_envs(
        monkeypatch,
        VLLM_ROCM_USE_AITER=True,
        VLLM_ROCM_USE_AITER_MOE=True,
    )
    from vllm_hcu.platforms import envs as henvs

    monkeypatch.setattr(henvs, "VLLM_HCU_USE_AITER_W16A16_MOE_SHUFFLE", True)
    monkeypatch.setattr(henvs, "VLLM_HCU_USE_AITER_MOE_CONFIG", True)
    captured: dict[str, object] = {}
    calls: list[dict[str, object]] = []

    def get_solution(**kwargs):
        captured.update(kwargs)
        return "4+9"

    def direct_asm(*args, **kwargs):
        calls.append(kwargs)
        return "direct-aiter"

    monkeypatch.setattr(
        aiter_runtime, "get_w16a16_moe_solution_id", get_solution
    )
    monkeypatch.setitem(
        sys.modules,
        "aiter.fused_moe_asm_wna16",
        _module(
            "aiter.fused_moe_asm_wna16",
            fused_experts_asm_impl=direct_asm,
        ),
    )
    x = torch.ones(2, 6, dtype=torch.bfloat16)
    w1 = torch.ones(3, 8, 6, dtype=torch.bfloat16)
    w2 = torch.ones(3, 6, 4, dtype=torch.bfloat16)
    topk_weight = torch.ones(2, 2)
    topk_ids = torch.zeros(2, 2, dtype=torch.int64)

    assert (
        aiter_runtime.fused_moe_impl(
            lambda *unused: pytest.fail("AITER ASM must not delegate"),
            x,
            w1,
            w2,
            topk_weight,
            topk_ids,
        )
        == "direct-aiter"
    )

    assert captured["N1"] == 8
    assert captured["E"] == 3
    assert captured["N2"] == 6
    assert captured["K"] == 6
    assert calls[0]["solution_id"] == "4+9"
    assert calls[0]["use_shuffle"] == 1

    with pytest.raises(ValueError, match="unexpected MoE weight layout"):
        aiter_runtime.fused_moe_impl(
            lambda *unused: pytest.fail("invalid layout must fail early"),
            x,
            torch.ones(3, 10, 6, dtype=torch.bfloat16),
            w2,
            topk_weight,
            topk_ids,
        )


def test_aiter_w16a16_solution_lookup_uses_global_expert_count_for_ep(
    monkeypatch: pytest.MonkeyPatch,
):
    _install_fake_aiter(monkeypatch)
    _install_fake_vllm_envs(
        monkeypatch,
        VLLM_ROCM_USE_AITER=True,
        VLLM_ROCM_USE_AITER_MOE=True,
    )
    from vllm_hcu.platforms import envs as henvs

    monkeypatch.setattr(henvs, "VLLM_HCU_USE_AITER_W16A16_MOE_SHUFFLE", True)
    monkeypatch.setattr(henvs, "VLLM_HCU_USE_AITER_MOE_CONFIG", True)
    captured: dict[str, object] = {}
    calls: list[dict[str, object]] = []

    def get_solution(**kwargs):
        captured.update(kwargs)
        return "4+9"

    def direct_asm(*args, **kwargs):
        calls.append(kwargs)
        return "direct-aiter"

    monkeypatch.setattr(
        aiter_runtime, "get_w16a16_moe_solution_id", get_solution
    )
    monkeypatch.setitem(
        sys.modules,
        "aiter.fused_moe_asm_wna16",
        _module(
            "aiter.fused_moe_asm_wna16",
            fused_experts_asm_impl=direct_asm,
        ),
    )
    x = torch.ones(2, 6, dtype=torch.bfloat16)
    w1 = torch.ones(2, 8, 6, dtype=torch.bfloat16)
    w2 = torch.ones(2, 6, 4, dtype=torch.bfloat16)
    topk_weight = torch.ones(2, 2)
    topk_ids = torch.zeros(2, 2, dtype=torch.int64)
    expert_mask = torch.tensor([1, 0, 1, 0, 0], dtype=torch.int32)

    with aiter_runtime.aiter_moe_request_context(
        SimpleNamespace(moe_backend="aiter", num_experts=4)
    ):
        assert (
            aiter_runtime.fused_moe_impl(
                lambda *unused: pytest.fail("AITER ASM must not delegate"),
                x,
                w1,
                w2,
                topk_weight,
                topk_ids,
                expert_mask=expert_mask,
            )
            == "direct-aiter"
        )

    assert captured["E"] == 4
    assert calls[0]["global_num_experts"] == 4
    assert calls[0]["expert_map"] is expert_mask
    assert calls[0]["solution_id"] == "4+9"


def test_aiter_w16a16_asm_requires_mask_sentinel_and_rejects_expert_map(
    monkeypatch: pytest.MonkeyPatch,
):
    _install_fake_aiter(monkeypatch)
    _install_fake_vllm_envs(
        monkeypatch,
        VLLM_ROCM_USE_AITER=True,
        VLLM_ROCM_USE_AITER_MOE=True,
    )
    asm_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def asm(*args, **kwargs):
        asm_calls.append((args, kwargs))
        return "asm"

    monkeypatch.setitem(
        sys.modules,
        "aiter.fused_moe_asm_wna16",
        _module("aiter.fused_moe_asm_wna16", fused_experts_asm_impl=asm),
    )
    from vllm_hcu.platforms import envs as henvs

    monkeypatch.setattr(henvs, "VLLM_HCU_USE_AITER_W16A16_MOE_SHUFFLE", False)
    monkeypatch.setattr(henvs, "VLLM_HCU_USE_AITER_MOE_CONFIG", False)
    x = torch.ones(2, 4)
    w1 = torch.ones(2, 8, 4)
    w2 = torch.ones(2, 4, 4)
    topk_weight = torch.ones(2, 2)
    topk_ids = torch.zeros(2, 2, dtype=torch.int64)
    expert_mask = torch.tensor([1, 0, 1, 0, 0], dtype=torch.int32)

    moe_config = SimpleNamespace(moe_backend="aiter", num_experts=4)
    with aiter_runtime.aiter_moe_request_context(moe_config):
        assert (
            aiter_runtime.fused_moe_impl(
                lambda *unused: pytest.fail("AITER ASM must not delegate"),
                x,
                w1,
                w2,
                topk_weight,
                topk_ids,
                expert_mask=expert_mask,
            )
            == "asm"
        )

    assert asm_calls[0][1]["global_num_experts"] == 4
    assert asm_calls[0][1]["expert_map"] is expert_mask

    global_to_local_map = torch.tensor([0, 1, -1, -1], dtype=torch.int32)
    with aiter_runtime.aiter_moe_request_context(moe_config):
        with pytest.raises(
            aiter_runtime.HcuAiterRuntimeError,
            match="trailing sentinel.*global-to-local expert map",
        ):
            aiter_runtime.fused_moe_impl(
                lambda *unused: pytest.fail("invalid expert map must fail early"),
                x,
                w1,
                w2,
                topk_weight,
                topk_ids,
                expert_mask=global_to_local_map,
            )

    with aiter_runtime.aiter_moe_request_context(moe_config):
        with pytest.raises(
            aiter_runtime.HcuAiterRuntimeError,
            match="requires an expert mask for EP",
        ):
            aiter_runtime.fused_moe_impl(
                lambda *unused: pytest.fail("missing EP mask must fail early"),
                x,
                w1,
                w2,
                topk_weight,
                topk_ids,
            )


def test_aiter_feature_off_delegates_v0251_fused_moe_contract(
    monkeypatch: pytest.MonkeyPatch,
):
    _install_fake_aiter(monkeypatch)
    _install_fake_vllm_envs(
        monkeypatch,
        VLLM_ROCM_USE_AITER=False,
        VLLM_ROCM_USE_AITER_MOE=False,
    )
    calls: list[tuple[object, ...]] = []

    def original(*args):
        calls.append(args)
        return "target"

    tensors = (
        torch.ones(2, 4),
        torch.ones(3, 8, 4),
        torch.ones(3, 4, 4),
        torch.ones(2, 2),
        torch.zeros(2, 2, dtype=torch.int64),
    )
    assert (
        aiter_runtime.fused_moe_impl(
            original,
            *tensors,
            gate_mode="separated",
            moe_sorting_dispatch_policy=3,
            swiglu_limit=4.5,
        )
        == "target"
    )
    assert calls[0][17] == "separated"
    assert calls[0][20] == 3
    assert calls[0][21] == 4.5


def test_aiter_solution_lookup_success_and_failures(monkeypatch: pytest.MonkeyPatch):
    class MoeQuantType:
        W16A16 = "w16a16"

    class MoeSolutionType:
        ASM = "asm"

    captured: dict[str, object] = {}

    def get_config(**kwargs):
        captured.update(kwargs)
        return True, SimpleNamespace(
            solution_type=MoeSolutionType.ASM,
            config={"SOL_ID1": 4, "SOL_ID2": 9},
        )

    monkeypatch.setitem(
        sys.modules,
        "aiter.moe",
        _module(
            "aiter.moe",
            MoeQuantType=MoeQuantType,
            MoeSolutionType=MoeSolutionType,
            get_aiter_moe_config=get_config,
        ),
    )
    aiter_runtime.get_w16a16_moe_config.cache_clear()
    aiter_runtime.get_w16a16_moe_solution_id.cache_clear()
    assert (
        aiter_runtime.get_w16a16_moe_solution_id(
            1, 2, 3, 4, 5, 6, torch.bfloat16, "silu", 1
        )
        == "4+9"
    )
    assert captured["spec_sol_type"] == MoeSolutionType.ASM


@pytest.mark.parametrize("extended", [False, True])
def test_aiter_topk_supports_old_and_new_abi(
    monkeypatch: pytest.MonkeyPatch, extended: bool
):
    calls: list[tuple[object, ...]] = []
    if extended:

        def topk(a, b, c, d, e, num_shared_experts=0, shared_expert_scoring_func=""):
            calls.append(
                (a, b, c, d, e, num_shared_experts, shared_expert_scoring_func)
            )

    else:

        def topk(a, b, c, d, e):
            calls.append((a, b, c, d, e))

    _install_fake_aiter(monkeypatch, topk_softmax=topk)
    tensors = [torch.empty(1) for _ in range(4)]
    aiter_runtime.topk_softmax_impl(*tensors, True, 2, "sigmoid")
    assert len(calls[0]) == (7 if extended else 5)


def test_scaled_mm_prequantized_input_bypasses_quantizer():
    calls: list[tuple[object, ...]] = []

    class FP8ScaledMMLinearKernel:
        _hcu_fp8_patch_applied = True
        _hcu_fp8_backend = "target-triton"

        def apply_weights(self, layer, x, bias=None):
            calls.append(("original", layer, x, bias))
            return "original"

    module = _module(
        patch_scaled_mm_linear_kernel.TARGET_MODULE,
        FP8ScaledMMLinearKernel=FP8ScaledMMLinearKernel,
    )
    patch_scaled_mm_linear_kernel.apply_to_module(module)
    kernel = FP8ScaledMMLinearKernel()
    kernel.config = SimpleNamespace(out_dtype=None)
    weight = torch.ones(3, 5, dtype=torch.int8)
    weight_scale = torch.ones(5)
    kernel._get_layer_params = lambda layer: (weight, weight_scale, None, None)
    kernel.apply_scaled_mm = lambda **kwargs: calls.append(("scaled", kwargs)) or kwargs
    x = torch.ones(2, 3)
    x_q = torch.ones(2, 3, dtype=torch.int8)
    x_scale = torch.ones(2, 1)
    result = kernel.apply_weights(object(), x, x_and_scale_quanted=(x_q, x_scale))
    assert calls[0][0] == "scaled"
    assert result["A"] is x_q and result["As"] is x_scale
    assert kernel.supports_quanted_inputs() is True
    assert kernel.apply_weights(object(), x) == "original"


def test_scaled_mm_prequantized_scale_shape_preserves_eager_value_error():
    class FP8ScaledMMLinearKernel:
        _hcu_fp8_patch_applied = True
        _hcu_fp8_backend = "target-triton"

        def apply_weights(self, layer, x, bias=None):
            return x

    module = _module(
        patch_scaled_mm_linear_kernel.TARGET_MODULE,
        FP8ScaledMMLinearKernel=FP8ScaledMMLinearKernel,
    )
    patch_scaled_mm_linear_kernel.apply_to_module(module)
    kernel = FP8ScaledMMLinearKernel()
    kernel.config = SimpleNamespace(out_dtype=None)
    weight = torch.ones(3, 5, dtype=torch.int8)
    weight_scale = torch.ones(5)
    kernel._get_layer_params = lambda layer: (weight, weight_scale, None, None)
    kernel.apply_scaled_mm = lambda **kwargs: kwargs

    with pytest.raises(ValueError, match="scale must be scalar or per-token"):
        kernel.apply_weights(
            object(),
            torch.ones(2, 3),
            x_and_scale_quanted=(
                torch.ones(2, 3, dtype=torch.int8),
                torch.ones(2, 2),
            ),
        )


def test_scaled_mm_prequantized_scale_shape_is_one_strict_dynamic_graph(monkeypatch):
    from sympy.logic.boolalg import Boolean

    # Some ROCm-enabled PyTorch builds report accelerator support to Dynamo
    # even on a CPU-only runner, which makes compilation snapshot a CUDA RNG
    # state and initialize HIP.  This graph is deliberately CPU-only.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    class FP8ScaledMMLinearKernel:
        _hcu_fp8_patch_applied = True
        _hcu_fp8_backend = "target-triton"

        def apply_weights(self, layer, x, bias=None):
            return x

        def apply_scaled_mm(
            self,
            *,
            A,
            B,
            out_dtype,
            As,
            Bs,
            bias,
            output_shape,
        ):
            # Keep this CPU-only graph shape dependent while avoiding any
            # device kernel.  The regression target is the wrapper's symbolic
            # scale-shape validation before this target-owned call.
            return A.to(out_dtype) * As

    module = _module(
        patch_scaled_mm_linear_kernel.TARGET_MODULE,
        FP8ScaledMMLinearKernel=FP8ScaledMMLinearKernel,
    )
    patch_scaled_mm_linear_kernel.apply_to_module(module)
    kernel = FP8ScaledMMLinearKernel()
    kernel.config = SimpleNamespace(out_dtype=torch.float32)
    weight = torch.ones(3, 5, dtype=torch.int8)
    weight_scale = torch.ones(5)
    kernel._get_layer_params = lambda layer: (weight, weight_scale, None, None)
    layer = object()
    compile_count = 0
    captured_graphs = []

    def counting_backend(graph_module, example_inputs):
        nonlocal compile_count
        compile_count += 1
        captured_graphs.append(graph_module)
        return graph_module.forward

    def apply_prequantized(x, x_2d_q, x_scale):
        return kernel.apply_weights(
            layer,
            x,
            x_and_scale_quanted=(x_2d_q, x_scale),
        )

    compiled = torch.compile(
        apply_prequantized,
        backend=counting_backend,
        dynamic=True,
        fullgraph=True,
    )
    for num_tokens in (2, 33, 65, 129):
        x = torch.ones(num_tokens, 3)
        x_2d_q = torch.ones(num_tokens, 3, dtype=torch.int8)
        x_scale = torch.ones(num_tokens, 1)
        for tensor in (x, x_2d_q, x_scale):
            torch._dynamo.mark_dynamic(
                tensor,
                0,
                min=1,
                max=10240,
            )

        result = compiled(x, x_2d_q, x_scale)
        assert result.shape == (num_tokens, 3)
        torch.testing.assert_close(result, torch.ones(num_tokens, 3))

    assert compile_count == 1
    symbolic_boolean_nodes = []
    for graph_module in captured_graphs:
        for node in graph_module.graph.nodes:
            value = node.meta.get("example_value")
            if isinstance(value, (torch.SymBool, Boolean)):
                symbolic_boolean_nodes.append((node.name, repr(value)))
    assert symbolic_boolean_nodes == [], symbolic_boolean_nodes


def test_clamp_swiglu_enforces_rocm_custom_op():
    sentinel = object()

    class CustomOp:
        def __init__(self, *, enforce_enable=False, compile_native=False):
            self.base_args = (enforce_enable, compile_native)
            self._forward_method = "dispatched"

    class SiluAndMulWithClamp(CustomOp):
        def __init__(
            self,
            swiglu_limit: float,
            alpha: float = 1.0,
            beta: float = 0.0,
            *,
            compile_native: bool = True,
        ):
            super().__init__(compile_native=compile_native)
            self.swiglu_limit = float(swiglu_limit)
            self.alpha = float(alpha)
            self.beta = float(beta)

        def forward_native(self, x):
            return x

    platform = SimpleNamespace(
        is_rocm=lambda: True,
        is_cuda_alike=lambda: False,
        is_xpu=lambda: False,
        is_cpu=lambda: False,
    )
    module = _module(
        patch_activation.TARGET_MODULE,
        SiluAndMulWithClamp=SiluAndMulWithClamp,
        current_platform=platform,
        torch=SimpleNamespace(
            ops=SimpleNamespace(
                _C=SimpleNamespace(silu_and_mul_with_clamp=sentinel)
            )
        ),
    )
    patch_activation.apply_to_module(module)
    instance = SiluAndMulWithClamp(7.0, 1.5, 0.25, compile_native=False)
    assert instance.base_args == (True, False)
    assert instance.op is sentinel
    assert instance.alpha == 1.5
    assert instance.beta == 0.25


def test_compressed_linear_only_forwards_supported_prequantized_input():
    class CompressedTensorsLinearMethod:
        def apply(self, layer, x, bias=None):
            return layer.scheme.apply_weights(layer, x, bias=bias)

    module = _module(
        patch_compressed_tensors.TARGET_MODULE,
        CompressedTensorsLinearMethod=CompressedTensorsLinearMethod,
    )
    patch_compressed_tensors.apply_to_module(module)
    calls: list[dict[str, object]] = []
    scheme = SimpleNamespace(
        supports_quanted_inputs=lambda: True,
        apply_weights=lambda layer, x, **kwargs: calls.append(kwargs) or "quantized",
    )
    method = CompressedTensorsLinearMethod()
    pair = (torch.ones(1, 2, dtype=torch.int8), torch.ones(1, 1))
    assert (
        method.apply(
            SimpleNamespace(scheme=scheme),
            torch.ones(1, 2),
            x_and_scale_quanted=pair,
        )
        == "quantized"
    )
    assert calls == [{"bias": None, "x_and_scale_quanted": pair}]
    assert method.supports_quanted_inputs() is True


def test_compressed_scheme_inactive_anchor_is_corrected_at_runtime():
    class CompressedTensorsScheme:
        def process_weights_after_loading(self, layer):
            raise NotImplementedError()

    module = _module(
        patch_compressed_tensors_scheme.TARGET_MODULE,
        CompressedTensorsScheme=CompressedTensorsScheme,
    )
    assert patch_compressed_tensors_scheme.apply_to_module(module) is True
    assert CompressedTensorsScheme().supports_quanted_inputs() is False
    assert patch_compressed_tensors_scheme.apply_to_module(module) is False


@pytest.mark.hcu
def test_slimquant_w4a8_moe_quant_config_uses_int4_weight_contract(
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm_hcu.model_executor.layers.quantization import slimquant_w4a8

    calls: list[dict[str, object]] = []

    class FusedMoEQuantConfig:
        @staticmethod
        def make(*args, **kwargs):
            calls.append({"args": args, **kwargs})
            return SimpleNamespace(kind="w4a8_quant_config")

    monkeypatch.setattr(
        slimquant_w4a8, "FusedMoEQuantConfig", FusedMoEQuantConfig
    )
    method = object.__new__(slimquant_w4a8.SlimQuantW4A8Int8AiterMoEMethod)
    layer = SimpleNamespace(
        w13_weight_scale=torch.ones(2, 4, 1),
        w2_weight_scale=torch.ones(2, 2, 1),
        w13_input_scale=None,
        w2_input_scale=None,
    )

    quant_config = method.get_fused_moe_quant_config(layer)

    assert quant_config.kind == "w4a8_quant_config"
    assert method.moe_quant_config is quant_config
    assert len(calls) == 1
    call = calls[0]
    assert call["args"] == (torch.int8,)
    assert call["w1_scale"] is layer.w13_weight_scale
    assert call["w2_scale"] is layer.w2_weight_scale
    assert call["a1_scale"] is None
    assert call["a2_scale"] is None
    assert call["per_act_token_quant"] is True
    assert call["per_out_ch_quant"] is False
    assert call["block_shape"] is None
    assert call["weight_dtype"] == "int4"


@pytest.mark.hcu
def test_slimquant_w4a8_moe_method_is_a_direct_fused_moe_method():
    from vllm_hcu.model_executor.layers.quantization import slimquant_w4a8

    method = slimquant_w4a8.SlimQuantW4A8Int8AiterMoEMethod(
        quant_config=object(),
        moe=object(),
    )

    assert type(method) is slimquant_w4a8.SlimQuantW4A8Int8AiterMoEMethod
    assert isinstance(method, slimquant_w4a8.FusedMoEMethodBase)


@pytest.mark.hcu
def test_slimquant_w4a8_apply_uses_triton_channelwise_raw_packed_weights(
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm_hcu.model_executor.layers.quantization import slimquant_w4a8

    calls: list[dict[str, object]] = []

    def fake_fused_experts_impl(*args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})
        hidden_states = args[0]
        return hidden_states + 1

    monkeypatch.setattr(
        slimquant_w4a8.aiter_triton_fused_moe,
        "fused_experts_impl",
        fake_fused_experts_impl,
    )

    method = object.__new__(slimquant_w4a8.SlimQuantW4A8Int8AiterMoEMethod)
    w13_weight = torch.arange(2 * 4 * 2, dtype=torch.int8).reshape(2, 4, 2)
    w2_weight = torch.arange(2 * 2 * 2, dtype=torch.int8).reshape(2, 2, 2)
    w13_parameter = torch.nn.Parameter(w13_weight.clone(), requires_grad=False)
    w2_parameter = torch.nn.Parameter(w2_weight.clone(), requires_grad=False)
    layer = SimpleNamespace(
        w13_weight=w13_parameter,
        w2_weight=w2_parameter,
        w13_weight_scale=torch.nn.Parameter(
            torch.ones(2, 4, 1), requires_grad=False
        ),
        w2_weight_scale=torch.nn.Parameter(
            torch.ones(2, 2, 1), requires_grad=False
        ),
    )

    method.process_weights_after_loading(layer)

    assert layer.w13_weight is w13_parameter
    assert layer.w2_weight is w2_parameter
    torch.testing.assert_close(layer.w13_weight.data, w13_weight)
    torch.testing.assert_close(layer.w2_weight.data, w2_weight)

    x = torch.zeros(3, 2, dtype=torch.float16)
    topk_weights = torch.ones(3, 1, dtype=torch.float16)
    topk_ids = torch.zeros(3, 1, dtype=torch.int32)
    result = method.apply(layer, x, topk_weights, topk_ids, None)

    torch.testing.assert_close(result, x + 1)
    assert len(calls) == 1
    call = calls[0]
    assert call["args"][:4] == (
        x,
        layer.w13_weight,
        layer.w2_weight,
        topk_weights,
    )
    assert call["args"][4] is topk_ids
    kwargs = call["kwargs"]
    assert kwargs["output_dtype"] is x.dtype
    assert kwargs["use_int4_w4a8"] is True
    assert kwargs["per_channel_quant"] is True
    assert kwargs["global_num_experts"] == 2
    assert kwargs["expert_map"] is None
    assert kwargs["w1_scale"] is layer.w13_weight_scale
    assert kwargs["w2_scale"] is layer.w2_weight_scale
    assert kwargs["activation"] == "silu"
    assert kwargs["is_gated"] is True


@pytest.mark.hcu
def test_slimquant_w4a8_apply_fails_closed_for_unsupported_inputs(
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm_hcu.model_executor.layers.quantization import slimquant_w4a8

    method = object.__new__(slimquant_w4a8.SlimQuantW4A8Int8AiterMoEMethod)
    layer = SimpleNamespace()
    x = torch.zeros(3, 2, dtype=torch.float16)
    topk_weights = torch.ones(3, 1, dtype=torch.float16)
    topk_ids = torch.zeros(3, 1, dtype=torch.int32)

    with pytest.raises(ValueError, match="rank-2 hidden states"):
        method.apply(layer, x.unsqueeze(0), topk_weights, topk_ids, None)
    with pytest.raises(ValueError, match="shared_experts_input"):
        method.apply(layer, x, topk_weights, topk_ids, x)
    with pytest.raises(ValueError, match="same shape"):
        method.apply(layer, x, topk_weights[:, :0], topk_ids, None)
    with pytest.raises(ValueError, match="same token count"):
        method.apply(layer, x, topk_weights[:2], topk_ids[:2], None)

    import_error = ImportError("missing aiter Triton fused MoE")
    monkeypatch.setattr(slimquant_w4a8, "aiter_triton_fused_moe", None)
    monkeypatch.setattr(slimquant_w4a8, "_AITER_TRITON_IMPORT_ERROR", import_error)
    with pytest.raises(RuntimeError, match="requires aiter.ops.triton.fused_moe"):
        method.apply(layer, x, topk_weights, topk_ids, None)


def _fake_moe_fp8_module():
    channel = object()
    token = object()
    tensor = object()

    class CompressedTensorsW8A8Fp8MoEMethod:
        init_calls: list[object] = []
        selected_backend = "TRITON"

        def __init__(self, weight_quant, input_quant, moe, layer_name=None):
            type(self).init_calls.append(moe)
            self.weight_quant = weight_quant
            self.input_quant = input_quant
            self.moe = moe
            self.layer_name = layer_name
            self.fp8_backend = SimpleNamespace(value=type(self).selected_backend)

        def process_weights_after_loading(self, layer):
            layer.upstream_processed = True

        def apply(
            self,
            layer,
            x,
            topk_weights,
            topk_ids,
            shared_experts,
            shared_experts_input,
        ):
            return (
                "upstream",
                layer,
                x,
                topk_weights,
                topk_ids,
                shared_experts,
                shared_experts_input,
            )

    return _module(
        patch_compressed_tensors_moe_w8a8_fp8.TARGET_MODULE,
        CompressedTensorsW8A8Fp8MoEMethod=CompressedTensorsW8A8Fp8MoEMethod,
        QuantizationStrategy=SimpleNamespace(
            CHANNEL=channel,
            TOKEN=token,
            TENSOR=tensor,
        ),
    )


def _channel_fp8_moe_args(module: ModuleType) -> tuple[object, object]:
    strategy = module.QuantizationStrategy
    return (
        SimpleNamespace(strategy=strategy.CHANNEL),
        SimpleNamespace(strategy=strategy.TOKEN),
    )


def _tensor_fp8_moe_args(module: ModuleType) -> tuple[object, object]:
    strategy = module.QuantizationStrategy
    return (
        SimpleNamespace(strategy=strategy.TENSOR),
        SimpleNamespace(strategy=strategy.TENSOR),
    )


def _fp8_moe_layer() -> SimpleNamespace:
    return SimpleNamespace(
        activation=SimpleNamespace(value="silu"),
        apply_router_weight_on_input=False,
        w13_weight=torch.ones(3, 8, 4),
        w2_weight=torch.ones(3, 4, 4),
        w13_weight_scale=torch.ones(3, 8, 1),
        w2_weight_scale=torch.ones(3, 4, 1),
        w13_input_scale=None,
        w2_input_scale=None,
        global_num_experts=3,
        expert_map=None,
        layer_name="model.layers.0.mlp.experts",
    )


def test_moe_fp8_target_triton_owns_process_and_apply(
    monkeypatch: pytest.MonkeyPatch,
):
    module = _fake_moe_fp8_module()
    method_class = module.CompressedTensorsW8A8Fp8MoEMethod
    target_process = method_class.process_weights_after_loading
    target_apply = method_class.apply
    assert patch_compressed_tensors_moe_w8a8_fp8.apply_to_module(module) is True
    assert patch_compressed_tensors_moe_w8a8_fp8.apply_to_module(module) is False
    assert method_class.process_weights_after_loading is target_process
    assert method_class.apply is target_apply
    assert not hasattr(method_class, "_get_aiter_moe_runtime_config")
    assert not hasattr(method_class, "_get_aiter_weights_for_solution")

    method = method_class(
        *_channel_fp8_moe_args(module),
        SimpleNamespace(moe_backend="triton"),
    )
    layer = _fp8_moe_layer()
    method.process_weights_after_loading(layer)
    assert layer.upstream_processed is True
    x = torch.ones(2, 4)
    weights = torch.ones(2, 2)
    ids = torch.zeros(2, 2, dtype=torch.int64)
    shared = object()
    result = method.apply(layer, x, weights, ids, shared, None)
    assert result == ("upstream", layer, x, weights, ids, shared, None)
    assert method_class.init_calls == [method.moe]


@pytest.mark.parametrize(
    ("target_aiter", "hcu_aiter"),
    [(False, False), (False, True), (True, False), (True, True)],
)
def test_moe_fp8_explicit_aiter_ignores_legacy_environment_half_states(
    monkeypatch: pytest.MonkeyPatch,
    target_aiter: bool,
    hcu_aiter: bool,
):
    module = _fake_moe_fp8_module()
    method_class = module.CompressedTensorsW8A8Fp8MoEMethod
    method_class.selected_backend = "AITER"
    if target_aiter:
        monkeypatch.setenv("VLLM_ROCM_USE_AITER_MOE", "1")
    else:
        monkeypatch.delenv("VLLM_ROCM_USE_AITER_MOE", raising=False)
    if hcu_aiter:
        monkeypatch.setenv("VLLM_HCU_USE_AITER_W8A8_FP8_MOE", "1")
    else:
        monkeypatch.delenv(
            "VLLM_HCU_USE_AITER_W8A8_FP8_MOE",
            raising=False,
        )
    patch_compressed_tensors_moe_w8a8_fp8.apply_to_module(module)
    method = method_class(
        *_channel_fp8_moe_args(module),
        SimpleNamespace(moe_backend="aiter"),
    )
    assert method.fp8_backend.value == "AITER"
    assert len(method_class.init_calls) == 1


def test_moe_fp8_requires_explicit_aiter_or_triton_and_checks_target_selection():
    module = _fake_moe_fp8_module()
    method_class = module.CompressedTensorsW8A8Fp8MoEMethod
    patch_compressed_tensors_moe_w8a8_fp8.apply_to_module(module)
    with pytest.raises(RuntimeError, match="--moe-backend aiter or triton"):
        method_class(
            *_channel_fp8_moe_args(module),
            SimpleNamespace(moe_backend="auto"),
        )
    assert method_class.init_calls == []

    method_class.selected_backend = "AITER"
    with pytest.raises(RuntimeError, match="selected='AITER'"):
        method_class(
            *_channel_fp8_moe_args(module),
            SimpleNamespace(moe_backend="triton"),
        )
    assert len(method_class.init_calls) == 1

    method_class.selected_backend = "TRITON"
    with pytest.raises(RuntimeError, match="selected='TRITON'"):
        method_class(
            *_channel_fp8_moe_args(module),
            SimpleNamespace(moe_backend="aiter"),
        )
    assert len(method_class.init_calls) == 2


def test_moe_fp8_non_channel_routes_delegate_target_without_triton_policy(
):
    module = _fake_moe_fp8_module()
    method_class = module.CompressedTensorsW8A8Fp8MoEMethod
    method_class.selected_backend = "AITER"
    patch_compressed_tensors_moe_w8a8_fp8.apply_to_module(module)
    method = method_class(
        *_tensor_fp8_moe_args(module),
        SimpleNamespace(moe_backend="auto"),
    )
    assert method.fp8_backend.value == "AITER"
    assert len(method_class.init_calls) == 1


def test_moe_fp8_aiter_config_is_layer_aware_and_cached(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[dict[str, object]] = []

    class MoeQuantType:
        FP8_W8A8 = "fp8_w8a8"

    def get_config(**kwargs):
        calls.append(kwargs)
        return True, SimpleNamespace(solution_type="ASM", serial=len(calls))

    monkeypatch.setitem(
        sys.modules,
        "aiter.moe",
        _module(
            "aiter.moe",
            MoeQuantType=MoeQuantType,
            get_aiter_moe_config=get_config,
        ),
    )
    method = SimpleNamespace(_hcu_aiter_moe_config_cache={})
    x = torch.ones(2, 4)
    ids = torch.zeros(2, 2, dtype=torch.int64)
    first_layer = _fp8_moe_layer()
    first = compressed_tensors_moe_runtime.get_aiter_w8a8_runtime_config(
        method, first_layer, x, ids
    )
    again = compressed_tensors_moe_runtime.get_aiter_w8a8_runtime_config(
        method, first_layer, x, ids
    )
    second = compressed_tensors_moe_runtime.get_aiter_w8a8_runtime_config(
        method, _fp8_moe_layer(), x, ids
    )
    assert first is again and first is not second
    assert len(calls) == 2
    assert calls[0]["quant_type"] == "fp8_w8a8"
    assert calls[0]["M"] == 2 and calls[0]["top_k"] == 2


def test_moe_fp8_uses_workspace_aiter_shuffle_and_invalidates_on_weight_change(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[str] = []

    def shuffle_weights(w1, w2, config):
        assert config is moe_config
        calls.append("w1")
        calls.append("w2")
        return w1 + 1, w2 + 2

    moe_config = SimpleNamespace(
        quant_type="fp8_w8a8",
        solution_type="moe_c",
        need_shuffle=True,
    )
    monkeypatch.setitem(
        sys.modules,
        "aiter.moe",
        _module(
            "aiter.moe",
            aiter_moe_shfl_weight=shuffle_weights,
        ),
    )
    layer = _fp8_moe_layer()
    first = compressed_tensors_moe_runtime.get_aiter_weights_for_solution(
        layer, moe_config
    )
    again = compressed_tensors_moe_runtime.get_aiter_weights_for_solution(
        layer, moe_config
    )
    assert first[0] is again[0] and calls == ["w1", "w2"]
    layer.w13_weight.add_(1)
    refreshed = compressed_tensors_moe_runtime.get_aiter_weights_for_solution(
        layer, moe_config
    )
    assert refreshed[0] is not first[0]
    assert calls == ["w1", "w2", "w1", "w2"]


def test_moe_fp8_hcu_aiter_flag_defaults_off(monkeypatch: pytest.MonkeyPatch):
    from vllm_hcu.platforms import envs as henvs

    monkeypatch.delenv("VLLM_HCU_USE_AITER_W8A8_FP8_MOE", raising=False)
    assert henvs.VLLM_HCU_USE_AITER_W8A8_FP8_MOE is False
    monkeypatch.setenv("VLLM_HCU_USE_AITER_W8A8_FP8_MOE", "1")
    assert henvs.VLLM_HCU_USE_AITER_W8A8_FP8_MOE is True


def test_moe_fp8_aiter_path_accepts_v0251_shared_expert_contract(
    monkeypatch: pytest.MonkeyPatch,
):
    kernel_calls: list[dict[str, object]] = []

    class MoeQuantType:
        FP8_W8A8 = "fp8_w8a8"

    def get_config(**kwargs):
        return True, SimpleNamespace(solution_type="ASM")

    def aiter_moe(**kwargs):
        kernel_calls.append(kwargs)
        return torch.full((2, 4), 7.0)

    monkeypatch.setitem(
        sys.modules,
        "aiter.moe",
        _module(
            "aiter.moe",
            MoeQuantType=MoeQuantType,
            get_aiter_moe_config=get_config,
            aiter_moe=aiter_moe,
        ),
    )
    from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig

    assert "disable_inplace" not in FusedMoEConfig.__dataclass_fields__
    target_moe_config = object.__new__(FusedMoEConfig)
    method = SimpleNamespace(
        moe=target_moe_config,
        _hcu_aiter_moe_config_cache={},
    )
    layer = _fp8_moe_layer()
    x = torch.ones(2, 4)
    weights = torch.ones(2, 2)
    ids = torch.zeros(2, 2, dtype=torch.int64)
    output = compressed_tensors_moe_runtime.apply_aiter_w8a8_fp8_moe(
        method, layer, x, weights, ids, None, None
    )
    torch.testing.assert_close(output, torch.full((2, 4), 7.0))
    assert kernel_calls[0]["hidden_states"] is x
    assert kernel_calls[0]["inplace"] is False
    assert kernel_calls[0]["topk_ids"].dtype is torch.int32
    shared = object()
    output_with_shared_contract = (
        compressed_tensors_moe_runtime.apply_aiter_w8a8_fp8_moe(
            method, layer, x, weights, ids, shared, x
        )
    )
    torch.testing.assert_close(
        output_with_shared_contract, torch.full((2, 4), 7.0)
    )
    method.moe = None
    with pytest.raises(
        compressed_tensors_moe_runtime.HcuCompressedTensorsMoeError,
        match="vLLM v0.25.1 MoE configuration",
    ):
        compressed_tensors_moe_runtime.apply_aiter_w8a8_fp8_moe(
            method, layer, x, weights, ids, None, None
        )


@pytest.mark.parametrize(
    ("use_fp8", "use_int8", "expected_quant_type"),
    [
        (True, False, "fp8_w8a8"),
        (False, True, "int8_w8a8"),
    ],
)
def test_quantized_aiter_runtime_selects_exact_quant_type(
    monkeypatch: pytest.MonkeyPatch,
    use_fp8: bool,
    use_int8: bool,
    expected_quant_type: str,
):
    config_calls: list[dict[str, object]] = []
    kernel_calls: list[dict[str, object]] = []
    expected_output = torch.full((2, 4), 9.0)

    class MoeQuantType:
        FP8_W8A8 = "fp8_w8a8"
        W8A8 = "int8_w8a8"

    def get_config(**kwargs):
        config_calls.append(kwargs)
        return True, SimpleNamespace(
            quant_type=kwargs["quant_type"],
            solution_type="asm",
            need_shuffle=False,
        )

    def aiter_moe(**kwargs):
        kernel_calls.append(kwargs)
        return expected_output

    monkeypatch.setitem(
        sys.modules,
        "aiter.moe",
        _module(
            "aiter.moe",
            MoeQuantType=MoeQuantType,
            get_aiter_moe_config=get_config,
            aiter_moe=aiter_moe,
        ),
    )
    hidden_states = torch.ones((2, 4), dtype=torch.bfloat16)
    w1 = torch.zeros((3, 8, 4), dtype=torch.int8)
    w2 = torch.zeros((3, 4, 4), dtype=torch.int8)
    topk_weights = torch.ones((2, 2), dtype=torch.bfloat16)
    topk_ids = torch.zeros((2, 2), dtype=torch.int64)
    w1_scale = torch.ones((3, 8, 1), dtype=torch.float32)
    w2_scale = torch.ones((3, 4, 1), dtype=torch.float32)
    a1q_scale = torch.ones((2, 1), dtype=torch.float32)
    expert_map = torch.tensor([0, 1, 2], dtype=torch.int32)
    quant_config = SimpleNamespace(
        use_fp8_w8a8=use_fp8,
        use_int8_w8a8=use_int8,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        w1_zp=None,
        w2_zp=None,
        a1_scale=None,
        a2_scale=None,
        block_shape=None,
    )

    output = compressed_tensors_moe_runtime.apply_aiter_quantized_moe(
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        vllm_moe_config=SimpleNamespace(num_experts=3),
        activation=SimpleNamespace(value="silu"),
        apply_router_weight_on_input=False,
        expert_map=expert_map,
        quant_config=quant_config,
        a1q_scale=a1q_scale,
        output_dtype=torch.bfloat16,
    )

    assert output is expected_output
    assert config_calls[0]["quant_type"] == expected_quant_type
    assert config_calls[0]["M"] == 2
    assert config_calls[0]["E"] == 3
    assert config_calls[0]["top_k"] == 2
    call = kernel_calls[0]
    assert call["hidden_states"] is hidden_states
    assert call["w1"] is w1 and call["w2"] is w2
    assert call["w1_scale"] is w1_scale and call["w2_scale"] is w2_scale
    assert call["a1_scale"] is a1q_scale
    torch.testing.assert_close(
        call["expert_map"], torch.ones_like(expert_map, dtype=torch.int32)
    )
    assert call["global_num_experts"] == 3
    assert call["inplace"] is False
    assert call["use_weight_shuffle"] is False
    assert call["output_dtype"] is torch.bfloat16
    assert call["topk_weights"].dtype is torch.float32
    assert call["topk_ids"].dtype is torch.int32


@pytest.mark.parametrize(
    ("use_fp8", "use_int8"),
    [(True, False), (False, True)],
)
@pytest.mark.parametrize(
    ("solution_type", "expected_map"),
    [
        ("ASM", [0, 1, 1, 0]),
        ("MOE_C", [-1, 0, 1, -1]),
    ],
)
def test_quantized_aiter_runtime_converts_only_asm_ep_map_to_binary_mask(
    monkeypatch: pytest.MonkeyPatch,
    use_fp8: bool,
    use_int8: bool,
    solution_type: str,
    expected_map: list[int],
):
    calls: list[dict[str, object]] = []

    class MoeQuantType:
        FP8_W8A8 = "fp8_w8a8"
        W8A8 = "int8_w8a8"

    def get_config(**kwargs):
        return True, SimpleNamespace(
            quant_type=kwargs["quant_type"],
            solution_type=solution_type,
            need_shuffle=False,
        )

    def aiter_moe(**kwargs):
        calls.append(kwargs)
        return torch.zeros((2, 4))

    monkeypatch.setitem(
        sys.modules,
        "aiter.moe",
        _module(
            "aiter.moe",
            MoeQuantType=MoeQuantType,
            get_aiter_moe_config=get_config,
            aiter_moe=aiter_moe,
        ),
    )

    def native_fp8_quant(
        x,
        scale=None,
        quant_dtype=torch.int8,
        num_rows=None,
        num_rows_factor=1,
    ):
        del scale, quant_dtype, num_rows, num_rows_factor
        return x, torch.ones((*x.shape[:-1], 1), dtype=torch.float32)

    monkeypatch.setitem(
        sys.modules,
        "aiter.fused_moe_asm_wna16",
        _module(
            "aiter.fused_moe_asm_wna16",
            per_token_quant_int8=lambda x: (x, torch.ones((x.shape[0], 1))),
            per_token_quant_hip=native_fp8_quant,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "boltops.fused_moe.triton.moe_compat",
        _module(
            "boltops.fused_moe.triton.moe_compat",
            per_token_quant_hip=native_fp8_quant,
        ),
    )

    expert_map = torch.tensor([-1, 0, 1, -1], dtype=torch.int64)
    quant_config = SimpleNamespace(
        use_fp8_w8a8=use_fp8,
        use_int8_w8a8=use_int8,
        w1_scale=torch.ones((2, 8, 1)),
        w2_scale=torch.ones((2, 4, 1)),
        w1_zp=None,
        w2_zp=None,
        a1_scale=None,
        a2_scale=None,
        block_shape=None,
    )
    compressed_tensors_moe_runtime.apply_aiter_quantized_moe(
        hidden_states=torch.ones((2, 4), dtype=torch.bfloat16),
        w1=torch.zeros((2, 8, 4), dtype=torch.int8),
        w2=torch.zeros((2, 4, 4), dtype=torch.int8),
        topk_weights=torch.ones((2, 2)),
        topk_ids=torch.zeros((2, 2), dtype=torch.int64),
        vllm_moe_config=SimpleNamespace(num_experts=4),
        activation=SimpleNamespace(value="silu"),
        apply_router_weight_on_input=False,
        expert_map=expert_map,
        quant_config=quant_config,
    )

    passed_map = calls[0]["expert_map"]
    assert isinstance(passed_map, torch.Tensor)
    expected_dtype = torch.int32 if solution_type == "ASM" else torch.int64
    assert passed_map.dtype is expected_dtype
    torch.testing.assert_close(
        passed_map.cpu(),
        torch.tensor(expected_map, dtype=passed_map.dtype),
    )
    if solution_type == "MOE_C":
        assert passed_map is expert_map


@pytest.mark.parametrize(
    ("runtime_kwargs", "message"),
    [
        ({"num_local_tokens": torch.tensor([2], dtype=torch.int32)}, "num_local_tokens"),
        ({"moe_sorting_dispatch_policy": 7}, "moe_sorting_dispatch_policy"),
    ],
)
def test_quantized_aiter_runtime_rejects_unsupported_parallel_metadata(
    runtime_kwargs: dict[str, object],
    message: str,
):
    quant_config = SimpleNamespace(
        use_fp8_w8a8=False,
        use_int8_w8a8=True,
        w1_scale=torch.ones((2, 8, 1)),
        w2_scale=torch.ones((2, 4, 1)),
        block_shape=None,
    )
    with pytest.raises(
        compressed_tensors_moe_runtime.HcuCompressedTensorsMoeError,
        match=message,
    ):
        compressed_tensors_moe_runtime.apply_aiter_quantized_moe(
            hidden_states=torch.ones((2, 4), dtype=torch.bfloat16),
            w1=torch.zeros((2, 8, 4), dtype=torch.int8),
            w2=torch.zeros((2, 4, 4), dtype=torch.int8),
            topk_weights=torch.ones((2, 2)),
            topk_ids=torch.zeros((2, 2), dtype=torch.int64),
            vllm_moe_config=SimpleNamespace(num_experts=2),
            activation=SimpleNamespace(value="silu"),
            apply_router_weight_on_input=False,
            expert_map=None,
            quant_config=quant_config,
            **runtime_kwargs,
        )


@pytest.mark.parametrize(
    ("use_fp8", "use_int8", "solution_type", "expected"),
    [
        (False, True, "asm", "aligned"),
        (True, False, "asm", "native"),
        (False, True, "moe_c", "native"),
    ],
)
def test_quantized_aiter_runtime_scopes_boltops_quant_to_int8_asm(
    monkeypatch: pytest.MonkeyPatch,
    use_fp8: bool,
    use_int8: bool,
    solution_type: str,
    expected: str,
):
    class MoeQuantType:
        FP8_W8A8 = "fp8_w8a8"
        W8A8 = "int8_w8a8"

    def native_quant(x):
        del x
        return "native"

    def boltops_quant(x):
        del x
        return "aligned"

    def native_activation(
        activation,
        is_gated,
        activated_out,
        ffn1_out_2d,
        gemm1_alpha,
        gemm1_limit,
    ):
        del (
            activation,
            is_gated,
            activated_out,
            ffn1_out_2d,
            gemm1_alpha,
            gemm1_limit,
        )

    asm_module = _module(
        "aiter.fused_moe_asm_wna16",
        per_token_quant_int8=native_quant,
        _apply_activation=native_activation,
        per_token_quant_hip=_fp8_quant_abi_stub,
    )
    monkeypatch.setitem(
        sys.modules,
        "aiter.fused_moe_asm_wna16",
        asm_module,
    )
    monkeypatch.setitem(
        sys.modules,
        "boltops.fused_moe.triton.moe_compat",
        _module(
            "boltops.fused_moe.triton.moe_compat",
            per_token_quant_hip=boltops_quant,
        ),
    )

    def get_config(**kwargs):
        return True, SimpleNamespace(
            quant_type=kwargs["quant_type"],
            solution_type=solution_type,
            need_shuffle=False,
        )

    def aiter_moe(**kwargs):
        return asm_module.per_token_quant_int8(kwargs["hidden_states"])

    monkeypatch.setitem(
        sys.modules,
        "aiter.moe",
        _module(
            "aiter.moe",
            MoeQuantType=MoeQuantType,
            get_aiter_moe_config=get_config,
            aiter_moe=aiter_moe,
        ),
    )
    hidden_states = torch.ones((2, 4), dtype=torch.bfloat16)
    quant_config = SimpleNamespace(
        use_fp8_w8a8=use_fp8,
        use_int8_w8a8=use_int8,
        w1_scale=torch.ones((3, 8, 1)),
        w2_scale=torch.ones((3, 4, 1)),
        w1_zp=None,
        w2_zp=None,
        a1_scale=None,
        a2_scale=None,
        block_shape=None,
    )

    output = compressed_tensors_moe_runtime.apply_aiter_quantized_moe(
        hidden_states=hidden_states,
        w1=torch.zeros((3, 8, 4), dtype=torch.int8),
        w2=torch.zeros((3, 4, 4), dtype=torch.int8),
        topk_weights=torch.ones((2, 2)),
        topk_ids=torch.zeros((2, 2), dtype=torch.int64),
        vllm_moe_config=SimpleNamespace(num_experts=3),
        activation=SimpleNamespace(value="silu"),
        apply_router_weight_on_input=False,
        expert_map=None,
        quant_config=quant_config,
    )

    assert output == expected


@pytest.mark.parametrize(
    ("use_fp8", "use_int8", "solution_type", "activation", "expected"),
    [
        (True, False, "asm", "silu", 1.0),
        (False, True, "asm", "silu", 1.0),
        (True, False, "moe_c", "silu", 1.0),
        (True, False, "asm", "gelu", 1.0),
    ],
)
def test_quantized_aiter_runtime_scopes_boltops_quant_to_both_fp8_asm_stages(
    monkeypatch: pytest.MonkeyPatch,
    use_fp8: bool,
    use_int8: bool,
    solution_type: str,
    activation: str,
    expected: float,
):
    calls: list[str] = []

    class MoeQuantType:
        FP8_W8A8 = "fp8_w8a8"
        W8A8 = "int8_w8a8"

    def native_activation(
        activation,
        is_gated,
        activated_out,
        ffn1_out_2d,
        gemm1_alpha,
        gemm1_limit,
    ):
        del activation, is_gated, ffn1_out_2d, gemm1_alpha, gemm1_limit
        calls.append("aiter_activation")
        activated_out.fill_(1)

    def native_fp8_quant(
        x,
        scale=None,
        quant_dtype=torch.int8,
        num_rows=None,
        num_rows_factor=1,
    ):
        del scale, quant_dtype, num_rows, num_rows_factor
        calls.append("aiter_fp8_quant")
        return x, torch.ones((x.shape[0], 1))

    asm_module = _module(
        "aiter.fused_moe_asm_wna16",
        _apply_activation=native_activation,
        per_token_quant_int8=lambda x: (x, torch.ones((x.shape[0], 1))),
        per_token_quant_hip=native_fp8_quant,
    )
    monkeypatch.setitem(sys.modules, "aiter.fused_moe_asm_wna16", asm_module)

    def boltops_fp8_quant(x, scale=None, quant_dtype=torch.int8, **kwargs):
        assert scale is None
        assert quant_dtype == torch.float8_e4m3fn
        assert kwargs == {}
        calls.append("boltops_fp8_quant")
        return x, torch.ones((x.shape[0], 1))

    monkeypatch.setitem(
        sys.modules,
        "boltops.fused_moe.triton.moe_compat",
        _module(
            "boltops.fused_moe.triton.moe_compat",
            per_token_quant_hip=boltops_fp8_quant,
        ),
    )

    def get_config(**kwargs):
        return True, SimpleNamespace(
            quant_type=kwargs["quant_type"],
            solution_type=solution_type,
            need_shuffle=False,
        )

    def aiter_moe(**kwargs):
        if use_fp8:
            asm_module.per_token_quant_hip(
                kwargs["hidden_states"],
                quant_dtype=torch.float8_e4m3fn,
            )
        output = torch.empty((2, 4))
        asm_module._apply_activation(
            activation=kwargs["activation"],
            is_gated=True,
            activated_out=output,
            ffn1_out_2d=torch.empty((2, 8)),
            gemm1_alpha=None,
            gemm1_limit=None,
        )
        if use_fp8:
            asm_module.per_token_quant_hip(
                output,
                quant_dtype=torch.float8_e4m3fn,
            )
        return output

    monkeypatch.setitem(
        sys.modules,
        "aiter.moe",
        _module(
            "aiter.moe",
            MoeQuantType=MoeQuantType,
            get_aiter_moe_config=get_config,
            aiter_moe=aiter_moe,
        ),
    )
    quant_config = SimpleNamespace(
        use_fp8_w8a8=use_fp8,
        use_int8_w8a8=use_int8,
        w1_scale=torch.ones((3, 8, 1)),
        w2_scale=torch.ones((3, 4, 1)),
        w1_zp=None,
        w2_zp=None,
        a1_scale=None,
        a2_scale=None,
        block_shape=None,
    )

    output = compressed_tensors_moe_runtime.apply_aiter_quantized_moe(
        hidden_states=torch.ones((2, 4), dtype=torch.bfloat16),
        w1=torch.zeros((3, 8, 4), dtype=torch.int8),
        w2=torch.zeros((3, 4, 4), dtype=torch.int8),
        topk_weights=torch.ones((2, 2)),
        topk_ids=torch.zeros((2, 2), dtype=torch.int64),
        vllm_moe_config=SimpleNamespace(num_experts=3),
        activation=SimpleNamespace(value=activation),
        apply_router_weight_on_input=False,
        expert_map=None,
        quant_config=quant_config,
    )

    torch.testing.assert_close(output, torch.full_like(output, expected))
    if use_fp8:
        if solution_type == "asm":
            assert calls == [
                "boltops_fp8_quant",
                "aiter_activation",
                "boltops_fp8_quant",
            ]
        else:
            assert calls == [
                "aiter_fp8_quant",
                "aiter_activation",
                "aiter_fp8_quant",
            ]


def test_quantized_aiter_moe_c_preserves_swiglu_limit(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[tuple[str, float | None]] = []

    class MoeQuantType:
        FP8_W8A8 = "fp8_w8a8"
        W8A8 = "int8_w8a8"

    def native_silu(output, input, rows_per_block=1, vec_size=2):
        del input, rows_per_block, vec_size
        calls.append(("native", None))
        output.fill_(1)

    def aligned_activation(
        activation,
        is_gated,
        activated_out,
        ffn1_out_2d,
        gemm1_alpha,
        gemm1_limit,
    ):
        del activation, is_gated, ffn1_out_2d, gemm1_alpha
        calls.append(("aligned", gemm1_limit))
        activated_out.fill_(2)

    moe_c_module = _module(
        "aiter.fused_moe_c",
        moe_c_silu_and_mul=native_silu,
        _apply_activation=aligned_activation,
    )
    monkeypatch.setitem(sys.modules, "aiter.fused_moe_c", moe_c_module)

    def get_config(**kwargs):
        return True, SimpleNamespace(
            quant_type=kwargs["quant_type"],
            solution_type="moe_c",
            need_shuffle=False,
        )

    kernel_calls: list[dict[str, object]] = []

    def aiter_moe(**kwargs):
        kernel_calls.append(kwargs)
        output = torch.empty((2, 4))
        moe_c_module.moe_c_silu_and_mul(output, torch.empty((2, 8)))
        return output

    monkeypatch.setitem(
        sys.modules,
        "aiter.moe",
        _module(
            "aiter.moe",
            MoeQuantType=MoeQuantType,
            get_aiter_moe_config=get_config,
            aiter_moe=aiter_moe,
        ),
    )
    quant_config = SimpleNamespace(
        use_fp8_w8a8=True,
        use_int8_w8a8=False,
        w1_scale=torch.ones((3, 8, 1)),
        w2_scale=torch.ones((3, 4, 1)),
        w1_zp=None,
        w2_zp=None,
        a1_scale=None,
        a2_scale=None,
        block_shape=None,
    )

    output = compressed_tensors_moe_runtime.apply_aiter_quantized_moe(
        hidden_states=torch.ones((2, 4), dtype=torch.bfloat16),
        w1=torch.zeros((3, 8, 4), dtype=torch.int8),
        w2=torch.zeros((3, 4, 4), dtype=torch.int8),
        topk_weights=torch.ones((2, 2)),
        topk_ids=torch.zeros((2, 2), dtype=torch.int64),
        vllm_moe_config=SimpleNamespace(
            num_experts=3,
            swiglu_limit=10.0,
            swiglu_alpha=None,
            swiglu_beta=None,
        ),
        activation=SimpleNamespace(value="silu"),
        apply_router_weight_on_input=False,
        expert_map=None,
        quant_config=quant_config,
    )

    torch.testing.assert_close(output, torch.full_like(output, 2))
    assert calls == [("aligned", 10.0)]
    assert kernel_calls[0]["gemm1_alpha"] is None
    assert kernel_calls[0]["gemm1_limit"] == 10.0


def test_quantized_aiter_runtime_caches_config_and_invalidates_shuffled_weights(
    monkeypatch: pytest.MonkeyPatch,
):
    config_calls: list[dict[str, object]] = []
    shuffle_calls: list[tuple[torch.Tensor, torch.Tensor]] = []
    kernel_calls: list[dict[str, object]] = []

    class MoeQuantType:
        FP8_W8A8 = "fp8_w8a8"
        W8A8 = "int8_w8a8"

    def get_config(**kwargs):
        config_calls.append(kwargs)
        return True, SimpleNamespace(
            quant_type=kwargs["quant_type"],
            solution_type="moe_c",
            need_shuffle=True,
        )

    def shuffle_weights(w1, w2, config):
        assert config.solution_type == "moe_c"
        shuffle_calls.append((w1, w2))
        return w1.clone(), w2.clone()

    def aiter_moe(**kwargs):
        kernel_calls.append(kwargs)
        return kwargs["hidden_states"].clone()

    monkeypatch.setitem(
        sys.modules,
        "aiter.moe",
        _module(
            "aiter.moe",
            MoeQuantType=MoeQuantType,
            get_aiter_moe_config=get_config,
            aiter_moe_shfl_weight=shuffle_weights,
            aiter_moe=aiter_moe,
        ),
    )
    hidden_states = torch.ones((2, 4), dtype=torch.bfloat16)
    w1 = torch.zeros((3, 8, 4), dtype=torch.int8)
    w2 = torch.zeros((3, 4, 4), dtype=torch.int8)
    topk_weights = torch.ones((2, 2))
    topk_ids = torch.zeros((2, 2), dtype=torch.int64)
    quant_config = SimpleNamespace(
        use_fp8_w8a8=False,
        use_int8_w8a8=True,
        w1_scale=torch.ones((3, 8, 1)),
        w2_scale=torch.ones((3, 4, 1)),
        w1_zp=None,
        w2_zp=None,
        a1_scale=None,
        a2_scale=None,
        block_shape=None,
    )

    def run(x=hidden_states, weights=topk_weights, ids=topk_ids):
        return compressed_tensors_moe_runtime.apply_aiter_quantized_moe(
            hidden_states=x,
            w1=w1,
            w2=w2,
            topk_weights=weights,
            topk_ids=ids,
            vllm_moe_config=SimpleNamespace(num_experts=3),
            activation=SimpleNamespace(value="silu"),
            apply_router_weight_on_input=False,
            expert_map=None,
            quant_config=quant_config,
        )

    run()
    run()
    assert len(config_calls) == 1
    assert len(shuffle_calls) == 1
    assert kernel_calls[0]["w1"] is kernel_calls[1]["w1"]
    assert kernel_calls[0]["use_weight_shuffle"] is True

    w1.add_(1)
    run()
    assert len(config_calls) == 1
    assert len(shuffle_calls) == 2
    assert kernel_calls[2]["w1"] is not kernel_calls[1]["w1"]

    larger_x = torch.ones((3, 4), dtype=torch.bfloat16)
    run(larger_x, torch.ones((3, 2)), torch.zeros((3, 2), dtype=torch.int64))
    assert len(config_calls) == 2


def test_quantized_aiter_runtime_pins_destructive_shuffle_solution(
    monkeypatch: pytest.MonkeyPatch,
):
    config_calls: list[dict[str, object]] = []

    def get_config(**kwargs):
        config_calls.append(kwargs)
        requested = kwargs.get("spec_sol_type")
        solution = "moe_c" if requested is None else requested
        return True, SimpleNamespace(
            quant_type=kwargs["quant_type"],
            solution_type=solution,
            need_shuffle=solution == "moe_c",
        )

    monkeypatch.delenv(
        "VLLM_HCU_AITER_MOE_INPLACE_CHUNKED_SHUFFLE", raising=False
    )
    monkeypatch.setitem(
        sys.modules,
        "aiter.moe",
        _module("aiter.moe", get_aiter_moe_config=get_config),
    )
    w1 = torch.zeros((3, 8, 4))
    w2 = torch.zeros((3, 4, 4))

    compressed_tensors_moe_runtime._get_aiter_quantized_runtime_config(
        torch.ones((2, 4), dtype=torch.bfloat16),
        w1,
        w2,
        torch.zeros((2, 2), dtype=torch.int64),
        "fp8_w8a8",
        "silu",
    )
    compressed_tensors_moe_runtime._get_aiter_quantized_runtime_config(
        torch.ones((3, 4), dtype=torch.bfloat16),
        w1,
        w2,
        torch.zeros((3, 2), dtype=torch.int64),
        "fp8_w8a8",
        "silu",
    )

    assert config_calls[0].get("spec_sol_type") is None
    assert config_calls[1]["spec_sol_type"] == "moe_c"


def test_quantized_aiter_runtime_serializes_solution_pin(
    monkeypatch: pytest.MonkeyPatch,
):
    first_config_entered = threading.Event()
    release_first_config = threading.Event()
    config_calls: list[dict[str, object]] = []
    errors: list[BaseException] = []

    def get_config(**kwargs):
        config_calls.append(kwargs)
        if len(config_calls) == 1:
            first_config_entered.set()
            assert release_first_config.wait(timeout=5)
        requested = kwargs.get("spec_sol_type")
        solution = "moe_c" if requested is None else requested
        return True, SimpleNamespace(
            quant_type=kwargs["quant_type"],
            solution_type=solution,
            need_shuffle=solution == "moe_c",
        )

    monkeypatch.delenv(
        "VLLM_HCU_AITER_MOE_INPLACE_CHUNKED_SHUFFLE", raising=False
    )
    monkeypatch.setitem(
        sys.modules,
        "aiter.moe",
        _module("aiter.moe", get_aiter_moe_config=get_config),
    )
    w1 = torch.zeros((3, 8, 4))
    w2 = torch.zeros((3, 4, 4))

    def resolve(tokens):
        try:
            compressed_tensors_moe_runtime._get_aiter_quantized_runtime_config(
                torch.ones((tokens, 4), dtype=torch.bfloat16),
                w1,
                w2,
                torch.zeros((tokens, 2), dtype=torch.int64),
                "fp8_w8a8",
                "silu",
            )
        except BaseException as exc:  # pragma: no cover - assertion aid
            errors.append(exc)

    first = threading.Thread(target=resolve, args=(2,))
    second = threading.Thread(target=resolve, args=(3,))
    first.start()
    assert first_config_entered.wait(timeout=5)
    second.start()
    assert len(config_calls) == 1
    release_first_config.set()
    first.join(timeout=5)
    second.join(timeout=5)

    assert not errors
    assert len(config_calls) == 2
    assert config_calls[1]["spec_sol_type"] == "moe_c"


def test_hcu_model_runner_wraps_complete_reload_transaction():
    source = Path("vllm_hcu/v1/hcu_model_runner.py").read_text()
    tree = ast.parse(source)
    reload_method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "reload_weights"
    )
    called_names = {
        node.func.id
        for node in ast.walk(reload_method)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert {
        "begin_aiter_weight_reload",
        "commit_aiter_weight_reload",
        "abort_aiter_weight_reload",
    } <= called_names


def test_quantized_aiter_runtime_chunked_inplace_shuffle_avoids_full_weight_copy(
    monkeypatch: pytest.MonkeyPatch,
):
    shuffle_batch_sizes: list[int] = []

    def shuffle_weights(w1, w2, config):
        assert config.solution_type == "moe_c"
        shuffle_batch_sizes.append(w1.shape[0])
        return w1 + 1, w2 + 2

    monkeypatch.setenv("VLLM_HCU_AITER_MOE_INPLACE_CHUNKED_SHUFFLE", "1")
    monkeypatch.setitem(
        sys.modules,
        "aiter.moe",
        _module(
            "aiter.moe",
            aiter_moe_shfl_weight=shuffle_weights,
        ),
    )
    config = SimpleNamespace(
        quant_type="fp8_w8a8",
        solution_type="moe_c",
        need_shuffle=True,
    )
    w1 = torch.zeros((3, 8, 4), dtype=torch.int8)
    w2 = torch.zeros((3, 4, 4), dtype=torch.int8)

    prepared_w1, prepared_w2 = (
        compressed_tensors_moe_runtime._get_aiter_quantized_weights(
            w1,
            w2,
            config,
        )
    )

    assert prepared_w1 is w1 and prepared_w2 is w2
    assert shuffle_batch_sizes == [1, 1, 1]
    torch.testing.assert_close(w1, torch.ones_like(w1))
    torch.testing.assert_close(w2, torch.full_like(w2, 2))

    cached_w1, cached_w2 = (
        compressed_tensors_moe_runtime._get_aiter_quantized_weights(
            w1,
            w2,
            config,
        )
    )
    assert cached_w1 is w1 and cached_w2 is w2
    assert shuffle_batch_sizes == [1, 1, 1]


def test_quantized_aiter_runtime_automatically_chunks_fp8_moe_c_shuffle(
    monkeypatch: pytest.MonkeyPatch,
):
    shuffle_batch_sizes: list[int] = []

    def shuffle_weights(w1, w2, config):
        assert config.solution_type == "moe_c"
        shuffle_batch_sizes.append(w1.shape[0])
        return w1 + 1, w2 + 2

    monkeypatch.delenv(
        "VLLM_HCU_AITER_MOE_INPLACE_CHUNKED_SHUFFLE",
        raising=False,
    )
    monkeypatch.delenv(
        "VLLM_HCU_AITER_MOE_SHUFFLE_CHUNK_EXPERTS",
        raising=False,
    )
    monkeypatch.setitem(
        sys.modules,
        "aiter.moe",
        _module(
            "aiter.moe",
            aiter_moe_shfl_weight=shuffle_weights,
        ),
    )
    config = SimpleNamespace(
        quant_type="fp8_w8a8",
        solution_type="moe_c",
        need_shuffle=True,
    )
    w1 = torch.zeros((3, 8, 4), dtype=torch.int8)
    w2 = torch.zeros((3, 4, 4), dtype=torch.int8)

    prepared_w1, prepared_w2 = (
        compressed_tensors_moe_runtime._get_aiter_quantized_weights(
            w1,
            w2,
            config,
        )
    )

    assert prepared_w1 is w1 and prepared_w2 is w2
    assert shuffle_batch_sizes == [1, 1, 1]
    torch.testing.assert_close(w1, torch.ones_like(w1))
    torch.testing.assert_close(w2, torch.full_like(w2, 2))


def test_quantized_aiter_runtime_invalidates_inplace_shuffle_via_weight_loader(
    monkeypatch: pytest.MonkeyPatch,
):
    shuffle_calls = 0

    def shuffle_weights(w1, w2, config):
        nonlocal shuffle_calls
        shuffle_calls += 1
        return w1 + 1, w2 + 2

    def weight_loader(param, loaded_weight):
        param.data.copy_(loaded_weight)

    monkeypatch.setenv("VLLM_HCU_AITER_MOE_INPLACE_CHUNKED_SHUFFLE", "1")
    monkeypatch.setitem(
        sys.modules,
        "aiter.moe",
        _module("aiter.moe", aiter_moe_shfl_weight=shuffle_weights),
    )
    config = SimpleNamespace(
        quant_type="fp8_w8a8",
        solution_type="moe_c",
        need_shuffle=True,
    )
    w1 = torch.nn.Parameter(torch.zeros((1, 8, 4)), requires_grad=False)
    w2 = torch.nn.Parameter(torch.zeros((1, 4, 4)), requires_grad=False)
    w1.weight_loader = weight_loader
    w2.weight_loader = weight_loader

    compressed_tensors_moe_runtime._get_aiter_quantized_weights(w1, w2, config)
    assert shuffle_calls == 1
    assert w1._version == 1

    w1.weight_loader(w1, torch.zeros_like(w1))
    w2.weight_loader(w2, torch.zeros_like(w2))
    assert w1._version == 1
    compressed_tensors_moe_runtime._get_aiter_quantized_weights(w1, w2, config)

    assert shuffle_calls == 2
    torch.testing.assert_close(w1, torch.ones_like(w1))
    torch.testing.assert_close(w2, torch.full_like(w2, 2))


def test_quantized_aiter_runtime_rejects_layout_transition_after_inplace_shuffle(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("VLLM_HCU_AITER_MOE_INPLACE_CHUNKED_SHUFFLE", "1")
    monkeypatch.setitem(
        sys.modules,
        "aiter.moe",
        _module(
            "aiter.moe",
            aiter_moe_shfl_weight=lambda w1, w2, config: (w1 + 1, w2 + 2),
        ),
    )
    shuffled = SimpleNamespace(
        quant_type="fp8_w8a8",
        solution_type="moe_c",
        need_shuffle=True,
    )
    canonical = SimpleNamespace(
        quant_type="fp8_w8a8",
        solution_type="asm",
        need_shuffle=False,
    )
    w1 = torch.zeros((1, 8, 4))
    w2 = torch.zeros((1, 4, 4))

    compressed_tensors_moe_runtime._get_aiter_quantized_weights(
        w1, w2, shuffled
    )
    with pytest.raises(
        compressed_tensors_moe_runtime.HcuCompressedTensorsMoeError,
        match="layout transition",
    ):
        compressed_tensors_moe_runtime._get_aiter_quantized_weights(
            w1, w2, canonical
        )


def test_quantized_aiter_runtime_serializes_inplace_shuffle(
    monkeypatch: pytest.MonkeyPatch,
):
    entered = threading.Event()
    release = threading.Event()
    shuffle_calls = 0
    results: list[tuple[torch.Tensor, torch.Tensor]] = []
    errors: list[BaseException] = []

    def shuffle_weights(w1, w2, config):
        nonlocal shuffle_calls
        shuffle_calls += 1
        if shuffle_calls == 1:
            entered.set()
            assert release.wait(timeout=5)
        return w1 + 1, w2 + 2

    monkeypatch.setenv("VLLM_HCU_AITER_MOE_INPLACE_CHUNKED_SHUFFLE", "1")
    monkeypatch.setitem(
        sys.modules,
        "aiter.moe",
        _module("aiter.moe", aiter_moe_shfl_weight=shuffle_weights),
    )
    config = SimpleNamespace(
        quant_type="fp8_w8a8",
        solution_type="moe_c",
        need_shuffle=True,
    )
    w1 = torch.zeros((2, 8, 4))
    w2 = torch.zeros((2, 4, 4))

    def prepare():
        try:
            results.append(
                compressed_tensors_moe_runtime._get_aiter_quantized_weights(
                    w1, w2, config
                )
            )
        except BaseException as exc:  # pragma: no cover - assertion aid
            errors.append(exc)

    first = threading.Thread(target=prepare)
    second = threading.Thread(target=prepare)
    first.start()
    assert entered.wait(timeout=5)
    second.start()
    release.set()
    first.join(timeout=5)
    second.join(timeout=5)

    assert not errors
    assert len(results) == 2
    assert shuffle_calls == 2
    assert all(pair == (w1, w2) for pair in results)


def test_quantized_aiter_runtime_quarantines_partial_shuffle_failure(
    monkeypatch: pytest.MonkeyPatch,
):
    shuffle_calls = 0

    def shuffle_weights(w1, w2, config):
        nonlocal shuffle_calls
        shuffle_calls += 1
        if shuffle_calls == 2:
            raise RuntimeError("synthetic chunk failure")
        return w1 + 1, w2 + 2

    monkeypatch.setenv("VLLM_HCU_AITER_MOE_INPLACE_CHUNKED_SHUFFLE", "1")
    monkeypatch.setitem(
        sys.modules,
        "aiter.moe",
        _module("aiter.moe", aiter_moe_shfl_weight=shuffle_weights),
    )
    config = SimpleNamespace(
        quant_type="fp8_w8a8",
        solution_type="moe_c",
        need_shuffle=True,
    )
    w1 = torch.zeros((3, 8, 4))
    w2 = torch.zeros((3, 4, 4))

    with pytest.raises(RuntimeError, match="synthetic chunk failure"):
        compressed_tensors_moe_runtime._get_aiter_quantized_weights(
            w1, w2, config
        )
    with pytest.raises(
        compressed_tensors_moe_runtime.HcuCompressedTensorsMoeError,
        match="partially shuffled",
    ):
        compressed_tensors_moe_runtime._get_aiter_quantized_weights(
            w1, w2, config
        )
    assert shuffle_calls == 2


def test_quantized_aiter_runtime_reload_transaction_commits_kernel_weights(
    monkeypatch: pytest.MonkeyPatch,
):
    shuffle_calls = 0

    def shuffle_weights(w1, w2, config):
        nonlocal shuffle_calls
        shuffle_calls += 1
        return w1 + 1, w2 + 2

    monkeypatch.setenv("VLLM_HCU_AITER_MOE_INPLACE_CHUNKED_SHUFFLE", "1")
    monkeypatch.setitem(
        sys.modules,
        "aiter.moe",
        _module("aiter.moe", aiter_moe_shfl_weight=shuffle_weights),
    )
    config = SimpleNamespace(
        quant_type="fp8_w8a8", solution_type="moe_c", need_shuffle=True
    )
    w1 = torch.nn.Parameter(torch.zeros((1, 8, 4)), requires_grad=False)
    w2 = torch.nn.Parameter(torch.zeros((1, 4, 4)), requires_grad=False)

    class Model:
        def named_parameters(self):
            return iter((("experts.w13_weight", w1), ("experts.w2_weight", w2)))

        def get_parameter(self, name):
            return dict(self.named_parameters())[name]

    model = Model()
    compressed_tensors_moe_runtime._get_aiter_quantized_weights(w1, w2, config)
    transaction = compressed_tensors_moe_runtime.begin_aiter_weight_reload(model)
    with torch.no_grad():
        w1.copy_(torch.zeros_like(w1))
        w2.copy_(torch.zeros_like(w2))
    compressed_tensors_moe_runtime.commit_aiter_weight_reload(transaction)

    compressed_tensors_moe_runtime._get_aiter_quantized_weights(w1, w2, config)
    assert shuffle_calls == 2
    torch.testing.assert_close(w1, torch.ones_like(w1))
    torch.testing.assert_close(w2, torch.full_like(w2, 2))


def test_quantized_aiter_runtime_reload_transaction_quarantines_abort(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("VLLM_HCU_AITER_MOE_INPLACE_CHUNKED_SHUFFLE", "1")
    monkeypatch.setitem(
        sys.modules,
        "aiter.moe",
        _module(
            "aiter.moe",
            aiter_moe_shfl_weight=lambda w1, w2, config: (w1 + 1, w2 + 2),
        ),
    )
    config = SimpleNamespace(
        quant_type="fp8_w8a8", solution_type="moe_c", need_shuffle=True
    )
    w1 = torch.nn.Parameter(torch.zeros((1, 8, 4)), requires_grad=False)
    w2 = torch.nn.Parameter(torch.zeros((1, 4, 4)), requires_grad=False)

    class Model:
        def named_parameters(self):
            return iter((("experts.w13_weight", w1), ("experts.w2_weight", w2)))

        def get_parameter(self, name):
            return dict(self.named_parameters())[name]

    model = Model()
    compressed_tensors_moe_runtime._get_aiter_quantized_weights(w1, w2, config)
    transaction = compressed_tensors_moe_runtime.begin_aiter_weight_reload(model)
    with torch.no_grad():
        w1.copy_(torch.zeros_like(w1))
    compressed_tensors_moe_runtime.abort_aiter_weight_reload(transaction)

    with pytest.raises(
        compressed_tensors_moe_runtime.HcuCompressedTensorsMoeError,
        match="partially shuffled|reload",
    ):
        compressed_tensors_moe_runtime._get_aiter_quantized_weights(w1, w2, config)


def test_quantized_aiter_runtime_reload_transaction_blocks_prepare(
    monkeypatch: pytest.MonkeyPatch,
):
    reload_started = threading.Event()
    finish_reload = threading.Event()
    prepare_finished = threading.Event()
    errors: list[BaseException] = []

    monkeypatch.setenv("VLLM_HCU_AITER_MOE_INPLACE_CHUNKED_SHUFFLE", "1")
    monkeypatch.setitem(
        sys.modules,
        "aiter.moe",
        _module(
            "aiter.moe",
            aiter_moe_shfl_weight=lambda w1, w2, config: (w1 + 1, w2 + 2),
        ),
    )
    config = SimpleNamespace(
        quant_type="fp8_w8a8", solution_type="moe_c", need_shuffle=True
    )
    w1 = torch.nn.Parameter(torch.zeros((1, 8, 4)), requires_grad=False)
    w2 = torch.nn.Parameter(torch.zeros((1, 4, 4)), requires_grad=False)

    class Model:
        def named_parameters(self):
            return iter((("experts.w13_weight", w1), ("experts.w2_weight", w2)))

        def get_parameter(self, name):
            return dict(self.named_parameters())[name]

    model = Model()
    compressed_tensors_moe_runtime._get_aiter_quantized_weights(w1, w2, config)

    def reload():
        transaction = compressed_tensors_moe_runtime.begin_aiter_weight_reload(model)
        reload_started.set()
        assert finish_reload.wait(timeout=5)
        with torch.no_grad():
            w1.copy_(torch.zeros_like(w1))
            w2.copy_(torch.zeros_like(w2))
        compressed_tensors_moe_runtime.commit_aiter_weight_reload(transaction)

    def prepare():
        try:
            compressed_tensors_moe_runtime._get_aiter_quantized_weights(
                w1, w2, config
            )
        except BaseException as exc:  # pragma: no cover - assertion aid
            errors.append(exc)
        finally:
            prepare_finished.set()

    reload_thread = threading.Thread(target=reload)
    prepare_thread = threading.Thread(target=prepare)
    reload_thread.start()
    assert reload_started.wait(timeout=5)
    prepare_thread.start()
    assert not prepare_finished.wait(timeout=0.1)
    finish_reload.set()
    reload_thread.join(timeout=5)
    prepare_thread.join(timeout=5)

    assert not errors
    assert prepare_finished.is_set()
    torch.testing.assert_close(w1, torch.ones_like(w1))
    torch.testing.assert_close(w2, torch.full_like(w2, 2))


def test_quantized_aiter_runtime_kernel_lease_blocks_reload(
    monkeypatch: pytest.MonkeyPatch,
):
    kernel_entered = threading.Event()
    release_kernel = threading.Event()
    reload_acquired = threading.Event()
    errors: list[BaseException] = []

    class MoeQuantType:
        FP8_W8A8 = "fp8_w8a8"
        W8A8 = "int8_w8a8"

    def get_config(**kwargs):
        return True, SimpleNamespace(
            quant_type=kwargs["quant_type"],
            solution_type="triton",
            need_shuffle=False,
        )

    def aiter_moe(**kwargs):
        kernel_entered.set()
        assert release_kernel.wait(timeout=5)
        return kwargs["hidden_states"].clone()

    monkeypatch.setitem(
        sys.modules,
        "aiter.moe",
        _module(
            "aiter.moe",
            MoeQuantType=MoeQuantType,
            get_aiter_moe_config=get_config,
            aiter_moe=aiter_moe,
        ),
    )
    w1 = torch.nn.Parameter(torch.ones((1, 8, 4)), requires_grad=False)
    w2 = torch.nn.Parameter(torch.ones((1, 4, 4)), requires_grad=False)
    quant_config = SimpleNamespace(
        use_fp8_w8a8=True,
        use_int8_w8a8=False,
        w1_scale=torch.ones((1, 8, 1)),
        w2_scale=torch.ones((1, 4, 1)),
        w1_zp=None,
        w2_zp=None,
        a1_scale=None,
        a2_scale=None,
        block_shape=None,
    )

    class Model:
        def named_parameters(self):
            return iter((("experts.w13_weight", w1), ("experts.w2_weight", w2)))

        def get_parameter(self, name):
            return dict(self.named_parameters())[name]

    model = Model()

    def infer():
        try:
            compressed_tensors_moe_runtime.apply_aiter_quantized_moe(
                hidden_states=torch.ones((2, 4), dtype=torch.bfloat16),
                w1=w1,
                w2=w2,
                topk_weights=torch.ones((2, 1)),
                topk_ids=torch.zeros((2, 1), dtype=torch.int64),
                vllm_moe_config=SimpleNamespace(num_experts=1),
                activation=SimpleNamespace(value="silu"),
                apply_router_weight_on_input=False,
                expert_map=None,
                quant_config=quant_config,
            )
        except BaseException as exc:  # pragma: no cover - assertion aid
            errors.append(exc)

    def reload():
        transaction = compressed_tensors_moe_runtime.begin_aiter_weight_reload(model)
        reload_acquired.set()
        with torch.no_grad():
            w1.copy_(torch.zeros_like(w1))
            w2.copy_(torch.zeros_like(w2))
        compressed_tensors_moe_runtime.commit_aiter_weight_reload(transaction)

    infer_thread = threading.Thread(target=infer)
    reload_thread = threading.Thread(target=reload)
    infer_thread.start()
    assert kernel_entered.wait(timeout=5)
    reload_thread.start()
    assert not reload_acquired.wait(timeout=0.1)
    release_kernel.set()
    infer_thread.join(timeout=5)
    reload_thread.join(timeout=5)

    assert not errors
    assert reload_acquired.is_set()
    torch.testing.assert_close(w1, torch.zeros_like(w1))
    torch.testing.assert_close(w2, torch.zeros_like(w2))


def test_quantized_aiter_runtime_synchronizes_every_weight_use_stream():
    synchronized: list[int] = []

    class Event:
        def __init__(self, stream_id):
            self.stream_id = stream_id

        def synchronize(self):
            synchronized.append(self.stream_id)

    w1 = torch.zeros((1, 8, 4))
    w2 = torch.zeros((1, 4, 4))
    state = compressed_tensors_moe_runtime._AiterInplaceShuffleState(w1, w2)
    state.use_events = {11: Event(11), 22: Event(22)}

    compressed_tensors_moe_runtime._synchronize_weight_use_events(state)

    assert synchronized == [11, 22]
    assert state.use_events == {}


@pytest.mark.parametrize(
    ("invalid_case", "message"),
    [
        ("topk_shape", "matching rank-2 top-k"),
        ("router_weight", "apply_router_weight_on_input=True"),
        ("block_quant", "channel/token W8A8"),
        ("ambiguous_quant", "exactly one FP8-W8A8 or INT8-W8A8"),
        ("no_solution", "found no backend config"),
    ],
)
def test_quantized_aiter_runtime_rejects_invalid_explicit_contracts(
    monkeypatch: pytest.MonkeyPatch,
    invalid_case: str,
    message: str,
):
    class MoeQuantType:
        FP8_W8A8 = "fp8_w8a8"
        W8A8 = "int8_w8a8"

    def get_config(**kwargs):
        if invalid_case == "no_solution":
            return False, None
        return True, SimpleNamespace(
            quant_type=kwargs["quant_type"],
            solution_type="asm",
            need_shuffle=False,
        )

    monkeypatch.setitem(
        sys.modules,
        "aiter.moe",
        _module(
            "aiter.moe",
            MoeQuantType=MoeQuantType,
            get_aiter_moe_config=get_config,
            aiter_moe=lambda **kwargs: kwargs["hidden_states"].clone(),
        ),
    )
    hidden_states = torch.ones((2, 4), dtype=torch.bfloat16)
    topk_weights = torch.ones((2, 2))
    topk_ids = torch.zeros((2, 2), dtype=torch.int64)
    quant_config = SimpleNamespace(
        use_fp8_w8a8=False,
        use_int8_w8a8=True,
        w1_scale=torch.ones((3, 8, 1)),
        w2_scale=torch.ones((3, 4, 1)),
        w1_zp=None,
        w2_zp=None,
        a1_scale=None,
        a2_scale=None,
        block_shape=None,
    )
    apply_router_weight_on_input = False
    if invalid_case == "topk_shape":
        topk_ids = torch.zeros((2, 1), dtype=torch.int64)
    elif invalid_case == "router_weight":
        apply_router_weight_on_input = True
    elif invalid_case == "block_quant":
        quant_config.block_shape = [128, 128]
    elif invalid_case == "ambiguous_quant":
        quant_config.use_int8_w8a8 = False

    with pytest.raises(
        compressed_tensors_moe_runtime.HcuCompressedTensorsMoeError,
        match=message,
    ):
        compressed_tensors_moe_runtime.apply_aiter_quantized_moe(
            hidden_states=hidden_states,
            w1=torch.zeros((3, 8, 4), dtype=torch.int8),
            w2=torch.zeros((3, 4, 4), dtype=torch.int8),
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            vllm_moe_config=SimpleNamespace(num_experts=3),
            activation=SimpleNamespace(value="silu"),
            apply_router_weight_on_input=apply_router_weight_on_input,
            expert_map=None,
            quant_config=quant_config,
        )


def test_moe_fp8_target_process_has_no_hcu_dpsk_postprocess(
):
    module = _fake_moe_fp8_module()
    patch_compressed_tensors_moe_w8a8_fp8.apply_to_module(module)
    method = module.CompressedTensorsW8A8Fp8MoEMethod(
        *_channel_fp8_moe_args(module),
        SimpleNamespace(moe_backend="triton"),
    )
    processed: list[object] = []
    experts = SimpleNamespace(
        process_weights_after_loading=lambda layer: processed.append(layer)
    )
    method.fp8_backend = SimpleNamespace(value="DPSK_DEEPGEMM")
    method.moe_kernel = SimpleNamespace(
        fused_experts=SimpleNamespace(experts=experts)
    )
    layer = _fp8_moe_layer()
    method.process_weights_after_loading(layer)
    assert layer.upstream_processed is True
    assert processed == []


def _fake_moe_wna16_module():
    attrs_calls: list[tuple[torch.Tensor, dict[str, object]]] = []
    config_calls: list[dict[str, object]] = []

    def set_weight_attrs(weight, attrs):
        attrs_calls.append((weight, dict(attrs)))

    def config_builder(**kwargs):
        config_calls.append(kwargs)
        return SimpleNamespace(**kwargs)

    class CompressedTensorsWNA16MoEMethod:
        def create_weights(
            self,
            layer,
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            params_dtype,
            **extra_weight_attrs,
        ):
            del params_dtype, extra_weight_attrs
            layer.register_parameter(
                "w13_weight_packed",
                torch.nn.Parameter(
                    torch.empty(num_experts, hidden_size, 1), requires_grad=False
                ),
            )
            layer.register_parameter(
                "w2_weight_packed",
                torch.nn.Parameter(
                    torch.empty(
                        num_experts, intermediate_size_per_partition, 1
                    ),
                    requires_grad=False,
                ),
            )
            layer.register_parameter(
                "w13_weight_scale",
                torch.nn.Parameter(torch.ones(1), requires_grad=False),
            )
            layer.register_parameter(
                "w2_weight_scale",
                torch.nn.Parameter(torch.ones(1), requires_grad=False),
            )
            return "upstream-create"

        def get_fused_moe_quant_config(self, layer):
            return ("upstream-config", layer)

    module = _module(
        patch_compressed_tensors_moe_wna16.TARGET_MODULE,
        CompressedTensorsWNA16MoEMethod=CompressedTensorsWNA16MoEMethod,
        set_weight_attrs=set_weight_attrs,
        int4_w4a16_moe_quant_config=config_builder,
    )
    return module, attrs_calls, config_calls


def _wna16_method(module, *, gated: bool, num_bits: int = 4):
    method = module.CompressedTensorsWNA16MoEMethod()
    method.num_bits = num_bits
    method.group_size = 4
    method.strategy = "group"
    method.moe = SimpleNamespace(is_act_and_mul=gated)
    return method


def test_moe_wna16_feature_off_delegates_exactly(
    monkeypatch: pytest.MonkeyPatch,
):
    module, _, config_calls = _fake_moe_wna16_module()
    monkeypatch.setattr(
        patch_compressed_tensors_moe_wna16,
        "_aiter_requested",
        lambda _layer=None: False,
    )
    assert patch_compressed_tensors_moe_wna16.apply_to_module(module) is True
    assert patch_compressed_tensors_moe_wna16.apply_to_module(module) is False
    method = _wna16_method(module, gated=True)
    layer = torch.nn.Module()
    assert method.create_weights(layer, 2, 16, 24, torch.bfloat16) == (
        "upstream-create"
    )
    assert not hasattr(layer, "w13_qzeros") and not hasattr(layer, "w2_qzeros")
    assert method.get_fused_moe_quant_config(layer) == ("upstream-config", layer)
    assert config_calls == []


@pytest.mark.parametrize("gated,shards", [(False, 1), (True, 2)])
def test_moe_wna16_allocates_correct_initialized_qzeros(
    monkeypatch: pytest.MonkeyPatch,
    gated: bool,
    shards: int,
):
    module, attrs_calls, _ = _fake_moe_wna16_module()
    monkeypatch.setattr(
        patch_compressed_tensors_moe_wna16,
        "_aiter_requested",
        lambda _layer=None: True,
    )
    patch_compressed_tensors_moe_wna16.apply_to_module(module)
    method = _wna16_method(module, gated=gated)
    layer = torch.nn.Module()
    method.create_weights(layer, 2, 16, 24, torch.bfloat16, loader="test")
    # AITER/vLLM pack two output-channel zero points per byte, while the
    # K/group axis remains unpacked.
    assert layer.w13_qzeros.shape == (2, shards * 12, 4)
    assert layer.w2_qzeros.shape == (2, 8, 6)
    assert layer.w13_qzeros.dtype is torch.uint8
    assert torch.all(layer.w13_qzeros == 0x88)
    assert torch.all(layer.w2_qzeros == 0x88)
    assert attrs_calls[-2][1] == {
        "loader": "test",
        "is_transposed": True,
        "quant_method": "group",
    }


def test_moe_wna16_quant_config_requires_registered_qzeros(
    monkeypatch: pytest.MonkeyPatch,
):
    module, _, config_calls = _fake_moe_wna16_module()
    monkeypatch.setattr(
        patch_compressed_tensors_moe_wna16,
        "_aiter_requested",
        lambda _layer=None: True,
    )
    patch_compressed_tensors_moe_wna16.apply_to_module(module)
    method = _wna16_method(module, gated=True)
    with pytest.raises(
        compressed_tensors_moe_runtime.HcuCompressedTensorsMoeError,
        match="w13_weight_scale",
    ):
        method.get_fused_moe_quant_config(torch.nn.Module())

    layer = torch.nn.Module()
    method.create_weights(layer, 2, 16, 24, torch.bfloat16)
    config = method.get_fused_moe_quant_config(layer)
    assert config.w1_zp is layer.w13_qzeros
    assert config.w2_zp is layer.w2_qzeros
    assert config.block_shape == [0, 4]
    assert len(config_calls) == 1

    invalid = _wna16_method(module, gated=True, num_bits=8)
    invalid_layer = torch.nn.Module()
    with pytest.raises(
        compressed_tensors_moe_runtime.HcuCompressedTensorsMoeError,
        match="requires 4-bit",
    ):
        invalid.create_weights(invalid_layer, 2, 16, 24, torch.bfloat16)


def _fake_fp8_scheme_module():
    channel = object()

    class CompressedTensorsW8A8Fp8:
        def process_weights_after_loading(self, layer):
            layer.weight = torch.nn.Parameter(
                layer.weight.t(), requires_grad=False
            )
            layer.weight.input_dim = 0
            layer.weight.output_dim = 1

        def apply_weights(self, layer, x, bias=None):
            return self.fp8_linear.apply_weights(layer, x, bias)

    return _module(
        patch_compressed_tensors_w8a8_fp8.TARGET_MODULE,
        CompressedTensorsW8A8Fp8=CompressedTensorsW8A8Fp8,
        QuantizationStrategy=SimpleNamespace(CHANNEL=channel),
    ), channel


def test_fp8_channel_weight_layout_requires_hcu_kernel(monkeypatch: pytest.MonkeyPatch):
    module, channel = _fake_fp8_scheme_module()
    patch_compressed_tensors_w8a8_fp8.apply_to_module(module)
    scheme = module.CompressedTensorsW8A8Fp8()
    scheme.strategy = channel
    scheme.fp8_linear = object()
    with pytest.raises(RuntimeError, match="target Triton scaled-mm adapter"):
        scheme.process_weights_after_loading(SimpleNamespace(weight=torch.ones(2, 3)))


def test_fp8_target_triton_route_is_independent_of_general_custom_gemm_flag(
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm_hcu.platforms import envs as henvs

    monkeypatch.setattr(henvs, "VLLM_HCU_USE_CUSTOM_QUANTIZATION_GEMM", False)
    module, channel = _fake_fp8_scheme_module()
    patch_compressed_tensors_w8a8_fp8.apply_to_module(module)

    class Kernel:
        _hcu_fp8_patch_applied = True
        _hcu_fp8_backend = "target-triton"

    scheme = module.CompressedTensorsW8A8Fp8()
    scheme.strategy = channel
    scheme.fp8_linear = Kernel()
    layer = SimpleNamespace(weight=torch.nn.Parameter(torch.ones(2, 3)))
    scheme.process_weights_after_loading(layer)
    assert layer.weight.shape == (3, 2)
    assert layer.weight.stride() == (1, 3)


def test_fp8_scheme_forwards_prequantized_input(monkeypatch: pytest.MonkeyPatch):
    module, channel = _fake_fp8_scheme_module()
    patch_compressed_tensors_w8a8_fp8.apply_to_module(module)
    calls: list[tuple[object, ...]] = []

    class Kernel:
        _hcu_fp8_patch_applied = True
        _hcu_fp8_backend = "target-triton"

        def supports_quanted_inputs(self):
            return True

        def apply_weights(self, *args, **kwargs):
            calls.append((*args, kwargs))
            return "fp8"

    scheme = module.CompressedTensorsW8A8Fp8()
    scheme.strategy = channel
    scheme.fp8_linear = Kernel()
    layer = SimpleNamespace(weight=torch.nn.Parameter(torch.arange(6.0).view(2, 3)))
    original = layer.weight.detach().clone()
    scheme.process_weights_after_loading(layer)
    torch.testing.assert_close(layer.weight, original.t())
    assert layer.weight.stride() == (1, 3)
    assert layer.weight.input_dim == 0 and layer.weight.output_dim == 1
    assert scheme.supports_quanted_inputs() is True
    pair = (torch.ones(1, 3, dtype=torch.int8), torch.ones(1, 1))
    assert (
        scheme.apply_weights(layer, torch.ones(1, 3), x_and_scale_quanted=pair)
        == "fp8"
    )
    assert calls[0][-1]["x_and_scale_quanted"] is pair


def test_int8_hcu_owned_kernel_validates_and_computes_shapes(
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm_hcu.platforms import envs as henvs

    monkeypatch.setattr(henvs, "VLLM_HCU_USE_CUSTOM_OPS", True)
    monkeypatch.setattr(henvs, "VLLM_HCU_USE_FUSED_SILU_MUL_QUANT", True)
    monkeypatch.setattr(henvs, "VLLM_HCU_USE_FUSED_RMS_QUANT", False)

    def gemm(a, b, scale_a, scale_b, m, n, k, flag, out_dtype):
        assert (m, n, k, flag) == (2, 2, 3, "NT")
        output = (a.float() * scale_a) @ (b.float() * scale_b).t()
        return True, output.to(out_dtype)

    monkeypatch.setitem(
        sys.modules,
        "lmslim",
        _module("lmslim", quant_ops=SimpleNamespace(hipblaslt_w8a8_gemm=gemm)),
    )
    x = torch.ones(1, 2, 3, dtype=torch.bfloat16)
    x_q = torch.tensor([[[1, 2, 3], [4, 5, 6]]], dtype=torch.int8)
    x_scale = torch.tensor([[[0.5], [0.25]]], dtype=torch.float32)
    weight = torch.tensor([[1, 0, -1], [2, 1, 0]], dtype=torch.int8)
    weight_scale = torch.tensor([[0.5], [0.25]], dtype=torch.float32)
    result = int8_runtime.apply_int8_linear(
        x,
        weight,
        weight_scale,
        torch.bfloat16,
        x_and_scale_quanted=(x_q, x_scale),
    )
    expected = (x_q.reshape(2, 3).float() * x_scale.reshape(2, 1)) @ (
        weight.float() * weight_scale
    ).t()
    torch.testing.assert_close(result.reshape(2, 2).float(), expected)
    with pytest.raises(
        int8_runtime.HcuInt8LinearError,
        match="quantized input/scale shapes",
    ):
        int8_runtime.apply_int8_linear(
            torch.ones(1, 2, 6, dtype=torch.bfloat16),
            weight,
            weight_scale,
            torch.bfloat16,
            x_and_scale_quanted=(x_q, x_scale),
        )
    with pytest.raises(int8_runtime.HcuInt8LinearError, match="symmetric"):
        int8_runtime.apply_int8_linear(
            x,
            weight,
            weight_scale,
            torch.bfloat16,
            input_zero_point=torch.ones(1, dtype=torch.int8),
            x_and_scale_quanted=(x_q, x_scale),
        )


def _fake_int8_scheme_module():
    class CompressedTensorsW8A8Int8:
        def process_weights_after_loading(self, layer):
            layer.weight = torch.nn.Parameter(
                layer.weight.t().contiguous(), requires_grad=False
            )

        def apply_weights(self, layer, x, bias):
            return ("upstream", layer, x, bias)

    return _module(
        patch_compressed_tensors_w8a8_int8.TARGET_MODULE,
        CompressedTensorsW8A8Int8=CompressedTensorsW8A8Int8,
    )


def test_int8_scheme_layout_and_feature_off_delegation(monkeypatch: pytest.MonkeyPatch):
    module = _fake_int8_scheme_module()
    monkeypatch.setattr(
        patch_compressed_tensors_w8a8_int8,
        "_custom_quantization_enabled",
        lambda: False,
    )
    patch_compressed_tensors_w8a8_int8.apply_to_module(module)
    scheme = module.CompressedTensorsW8A8Int8()
    layer = SimpleNamespace(weight=torch.nn.Parameter(torch.ones(2, 3)))
    assert scheme.apply_weights(layer, "x", None)[0] == "upstream"
    scheme.process_weights_after_loading(layer)
    assert layer.weight.shape == (3, 2)

    feature_module = _fake_int8_scheme_module()
    monkeypatch.setattr(
        patch_compressed_tensors_w8a8_int8,
        "_custom_quantization_enabled",
        lambda: True,
    )
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        int8_runtime,
        "apply_int8_linear",
        lambda **kwargs: calls.append(kwargs) or "hcu",
    )
    patch_compressed_tensors_w8a8_int8.apply_to_module(feature_module)
    feature_scheme = feature_module.CompressedTensorsW8A8Int8()
    feature_layer = SimpleNamespace(
        weight=torch.nn.Parameter(torch.ones(2, 3)),
        weight_scale=torch.ones(2, 1),
        params_dtype=torch.bfloat16,
        input_scale=None,
        input_zero_point=None,
        azp_adj=None,
    )
    feature_scheme.process_weights_after_loading(feature_layer)
    assert feature_layer.weight.shape == (2, 3)
    assert feature_layer.weight.is_contiguous()
    assert feature_scheme.apply_weights(feature_layer, "x", None) == "hcu"
    assert calls[0]["weight"] is feature_layer.weight


def test_int8_weight_layout_rolls_back_if_upstream_processing_fails(
    monkeypatch: pytest.MonkeyPatch,
):
    class CompressedTensorsW8A8Int8:
        def process_weights_after_loading(self, layer):
            raise RuntimeError("packing failed")

        def apply_weights(self, layer, x, bias):
            return None

    module = _module(
        patch_compressed_tensors_w8a8_int8.TARGET_MODULE,
        CompressedTensorsW8A8Int8=CompressedTensorsW8A8Int8,
    )
    monkeypatch.setattr(
        patch_compressed_tensors_w8a8_int8,
        "_custom_quantization_enabled",
        lambda: True,
    )
    patch_compressed_tensors_w8a8_int8.apply_to_module(module)
    layer = SimpleNamespace(
        weight=torch.nn.Parameter(torch.arange(6.0).reshape(2, 3))
    )
    original_parameter = layer.weight
    original_value = layer.weight.detach().clone()
    with pytest.raises(RuntimeError, match="packing failed"):
        module.CompressedTensorsW8A8Int8().process_weights_after_loading(layer)
    assert layer.weight is original_parameter
    assert layer.weight.shape == (2, 3)
    torch.testing.assert_close(layer.weight, original_value)


def test_lightop_fp8_registration_is_single_owner_and_latched():
    lightop_fp8_runtime._reset_for_tests()
    calls: list[dict[str, object]] = []
    lightop_fp8_runtime.ensure_registered(
        torch.float8_e4m3fn,
        lambda **kwargs: calls.append(kwargs),
    )
    lightop_fp8_runtime.ensure_registered(
        torch.float8_e4m3fn,
        lambda **kwargs: calls.append(kwargs),
    )
    assert len(calls) == 1
    assert calls[0]["op_name"] == "lightop_per_token_quant_fp8"

    lightop_fp8_runtime._reset_for_tests()
    with pytest.raises(lightop_fp8_runtime.HcuLightOpRegistrationError):
        lightop_fp8_runtime.ensure_registered(
            torch.float8_e4m3fn,
            lambda **kwargs: (_ for _ in ()).throw(RuntimeError("duplicate")),
        )
    with pytest.raises(
        lightop_fp8_runtime.HcuLightOpRegistrationError,
        match="previously failed",
    ):
        lightop_fp8_runtime.ensure_registered(
            torch.float8_e4m3fn, lambda **kwargs: None
        )
    lightop_fp8_runtime._reset_for_tests()


def test_lightop_fp8_adapter_has_no_import_time_registration():
    lightop_fp8_runtime._reset_for_tests()
    module, _ = _fake_input_quant_module()
    patch_input_quant_fp8.apply_to_module(module)
    assert lightop_fp8_runtime._REGISTERED is False
    assert lightop_fp8_runtime._REGISTRATION_ERROR is None


def _fake_input_quant_module():
    per_token = object()

    class QuantFP8:
        def forward_cuda(self, x, scale=None, scale_ub=None, use_triton=False):
            return ("cuda", x, scale, scale_ub, use_triton)

        def forward_native(self, x, scale=None, scale_ub=None, use_triton=False):
            return ("native", x, scale, scale_ub, use_triton)

    return _module(
        patch_input_quant_fp8.TARGET_MODULE,
        QuantFP8=QuantFP8,
        GroupShape=SimpleNamespace(PER_TOKEN=per_token),
        _FP8_DTYPE=torch.float8_e4m3fn,
    ), per_token


@pytest.mark.parametrize("method_name", ["forward_cuda", "forward_native"])
def test_quant_fp8_eligibility_and_feature_off(
    monkeypatch: pytest.MonkeyPatch, method_name: str
):
    _install_fake_vllm_torch_utils(monkeypatch)
    module, per_token = _fake_input_quant_module()
    patch_input_quant_fp8.apply_to_module(module)
    instance = module.QuantFP8()
    instance.group_shape = per_token
    instance.num_token_padding = None
    x = torch.ones(2, 4)
    monkeypatch.setattr(patch_input_quant_fp8, "_lightop_requested", lambda: True)
    monkeypatch.setattr(
        lightop_fp8_runtime,
        "quantize",
        lambda value, dtype, register: ("lightop", value, dtype),
    )
    method = getattr(instance, method_name)
    assert method(x)[0] == "lightop"
    assert method(x.t())[0] == method_name.removeprefix("forward_")
    monkeypatch.setattr(patch_input_quant_fp8, "_lightop_requested", lambda: False)
    assert method(x)[0] == "native"


def test_quant_fp8_feature_off_preserves_cuda_for_ineligible_inputs(
    monkeypatch: pytest.MonkeyPatch,
):
    _install_fake_vllm_torch_utils(monkeypatch)
    module, per_token = _fake_input_quant_module()
    patch_input_quant_fp8.apply_to_module(module)
    instance = module.QuantFP8()
    instance.group_shape = per_token
    instance.num_token_padding = None
    monkeypatch.setattr(patch_input_quant_fp8, "_lightop_requested", lambda: False)

    result = instance.forward_cuda(torch.ones(2, 4).t())

    assert result[0] == "cuda"


def test_weight8bit_marlin2_layout_2d_3d_and_validation():
    module = _module(patch_w8a8_utils.TARGET_MODULE)
    patch_w8a8_utils.apply_to_module(module)
    assert (
        module.weight8bit_nt_kpack2_marlin2
        is int8_runtime.weight8bit_nt_kpack2_marlin2
    )
    weight = torch.arange(16 * 64, dtype=torch.int32).to(torch.int8).view(16, 64)
    result = module.weight8bit_nt_kpack2_marlin2(weight)
    reference = (
        weight.reshape(1, 16, 1, 4, 16)
        .permute(2, 0, 3, 1, 4)
        .contiguous()
        .reshape(1, 1024)
    )
    torch.testing.assert_close(result, reference)
    experts = torch.stack((weight, weight + 1))
    result_3d = module.weight8bit_nt_kpack2_marlin2(experts)
    assert result_3d.shape == (2, 1, 1024)
    with pytest.raises(ValueError, match="rank 2 or 3"):
        module.weight8bit_nt_kpack2_marlin2(torch.ones(16, dtype=torch.int8))


def test_slimquant_marlin_module_imports_before_worker_patch():
    repo = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env["VLLM_PLUGINS"] = "__disabled__"
    env["PYTHONPATH"] = os.pathsep.join((str(repo), env.get("PYTHONPATH", "")))
    script = """
from vllm.model_executor.layers.quantization.utils import w8a8_utils
assert not hasattr(w8a8_utils, "weight8bit_nt_kpack2_marlin2")
from vllm_hcu.model_executor.layers.quantization.compressed_tensors import (
    compressed_tensors_moe_marlin,
)
from vllm_hcu.model_executor.layers.quantization import int8_runtime
from vllm.model_executor.layers.fused_moe import config as fused_moe_config
assert (
    compressed_tensors_moe_marlin.weight8bit_nt_kpack2_marlin2
    is int8_runtime.weight8bit_nt_kpack2_marlin2
)
calls = []
def hcu_int8_config(**kwargs):
    calls.append(kwargs)
    return "HCU_INT8_CONFIG"
fused_moe_config.int8_w8a8_moe_quant_config = hcu_int8_config
method = object.__new__(
    compressed_tensors_moe_marlin.CompressedTensorsW8A8Int8MarlinMoEMethod
)
layer = type("Layer", (), {
    "w13_weight_scale": object(),
    "w2_weight_scale": object(),
    "w13_input_scale": object(),
    "w2_input_scale": object(),
})()
assert method.get_fused_moe_quant_config(layer) == "HCU_INT8_CONFIG"
assert calls[0]["block_shape"] is None
print("SLIMQUANT_PREPATCH_IMPORT_OK")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "SLIMQUANT_PREPATCH_IMPORT_OK" in result.stdout


@pytest.mark.parametrize("is_rocm", [False, True])
def test_unquantized_gemm_dispatch_only_changes_rocm(is_rocm: bool):
    default = lambda *args: "default"
    rocm = lambda *args: "rocm"

    def dispatch_unquantized_gemm():
        return rocm if is_rocm else "other"

    module = _module(
        patch_layers_utils.TARGET_MODULE,
        current_platform=SimpleNamespace(is_rocm=lambda: is_rocm),
        default_unquantized_gemm=default,
        dispatch_unquantized_gemm=dispatch_unquantized_gemm,
    )
    patch_layers_utils.apply_to_module(module)
    assert module.dispatch_unquantized_gemm() is (default if is_rocm else "other")


def test_tf32_hc_prenorm_cpu_fallback_and_backend_delegation():
    calls: list[str] = []

    def lazy_init():
        calls.append("lazy")

    def original(x, fn, out, sqrsum, num_split):
        calls.append("backend")
        return "backend"

    module = _module(
        patch_deep_gemm.TARGET_MODULE,
        torch=torch,
        _lazy_init=lazy_init,
        _tf32_hc_prenorm_gemm_impl=None,
        tf32_hc_prenorm_gemm=original,
    )
    patch_deep_gemm.apply_to_module(module)
    x = torch.tensor([[1.0, 2.0], [-1.0, 3.0]])
    fn = torch.tensor([[2.0, 1.0], [0.5, -2.0], [1.0, 1.0]])
    out = torch.empty(2, 2, 3)
    sqrsum = torch.empty(2, 2)
    result = module.tf32_hc_prenorm_gemm(x, fn, out, sqrsum, 2)
    assert result is out
    torch.testing.assert_close(out[0], x @ fn.t())
    torch.testing.assert_close(out[1], torch.zeros_like(out[1]))
    torch.testing.assert_close(sqrsum[0], x.square().sum(-1))
    torch.testing.assert_close(sqrsum[1], torch.zeros_like(sqrsum[1]))

    module._tf32_hc_prenorm_gemm_impl = object()
    assert module.tf32_hc_prenorm_gemm(x, fn, out, sqrsum, 2) == "backend"
    assert calls[-1] == "backend"
