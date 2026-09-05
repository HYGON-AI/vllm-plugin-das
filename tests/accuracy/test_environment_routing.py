# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Environment-variable to runtime-path routing contracts."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from vllm_hcu.patch.worker.core_fix import patch_qwen3_5_mamba_state_dtype
from vllm_hcu.patch.worker.op_opt import (
    patch_compressed_tensors_w8a8_int8,
    patch_fla_chunk_delta_h,
    patch_fla_chunk_o,
    patch_input_quant_fp8,
    patch_mamba_mixer2,
)
from vllm_hcu.model_executor.layers.fused_moe.router_runtime import (
    eplb_map_to_physical_and_record,
)
from vllm_hcu.platforms import envs as hcu_envs


def _module(name: str, **attributes: object) -> ModuleType:
    module = ModuleType(name)
    module.__dict__.update(attributes)
    return module


def _load_topk_topp_sample() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[2]
        / "vllm_hcu"
        / "ops"
        / "topk_topp_sample.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_vllm_hcu_env_topk_topp_sample",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TOPK_TOPP_SAMPLE = _load_topk_topp_sample()


def _clear_materialized_hcu_environment_values() -> None:
    """Keep prior monkeypatch.setattr calls from masking lazy env getters."""
    for name in hcu_envs.hcu_vllm_environment_variables:
        hcu_envs.__dict__.pop(name, None)


@pytest.fixture(autouse=True)
def _preserve_lazy_hcu_environment_contract():
    _clear_materialized_hcu_environment_values()
    yield
    _clear_materialized_hcu_environment_values()


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1", True),
        ("true", True),
        ("TRUE", True),
        ("0", False),
        ("false", False),
        ("unexpected", False),
    ],
)
def test_boolean_environment_values_are_lazily_parsed(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
    expected: bool,
) -> None:
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", value)

    assert hcu_envs.VLLM_HCU_USE_CUSTOM_OPS is expected
    assert hcu_envs.is_set("VLLM_HCU_USE_CUSTOM_OPS") is True


def test_lightop_per_token_quant_defaults_enabled_and_allows_opt_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    name = "VLLM_HCU_USE_LIGHTOP_PER_TOKEN_QUANT_FP8"
    monkeypatch.delenv(name, raising=False)

    assert getattr(hcu_envs, name) is True
    assert hcu_envs.is_set(name) is False

    monkeypatch.setenv(name, "0")

    assert getattr(hcu_envs, name) is False
    assert hcu_envs.is_set(name) is True


def test_lightop_hy_v4_indexer_defaults_enabled_and_allows_opt_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    name = "VLLM_HCU_USE_LIGHTOP_HY_V4_INDEXER"
    monkeypatch.delenv(name, raising=False)

    assert getattr(hcu_envs, name) is True
    assert hcu_envs.is_set(name) is False

    monkeypatch.setenv(name, "0")

    assert getattr(hcu_envs, name) is False
    assert hcu_envs.is_set(name) is True


@pytest.mark.parametrize(
    "name",
    [
        "VLLM_HCU_LIGHTLY_CP_THRESHOLD",
        "VLLM_HCU_DEEPSEEK_V4_MULTI_STREAM_GEMM_TOKEN_THRESHOLD",
        "VLLM_HCU_DEEPSEEK_V4_MULTI_STREAM_COMPRESSOR_TOKEN_THRESHOLD",
        "VLLM_HCU_FLASH_ATTN_BLOCK_ALIGNMENT_SIZE",
        "VLLM_HCU_DEEPEP_NUM_SMS",
    ],
)
def test_integer_environment_values_reach_typed_consumers(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    monkeypatch.setenv(name, "37")

    assert getattr(hcu_envs, name) == 37
    assert isinstance(getattr(hcu_envs, name), int)


@pytest.mark.parametrize(
    "name",
    [
        "VLLM_HCU_FLASH_ATTN_BLOCK_ALIGNMENT_SIZE",
        "VLLM_HCU_DEEPEP_NUM_SMS",
    ],
)
def test_optional_integer_environment_values_preserve_none(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    monkeypatch.delenv(name, raising=False)

    assert getattr(hcu_envs, name) is None
    assert hcu_envs.is_set(name) is False


SELECTORS = (
    (
        "fla_chunk_o",
        patch_fla_chunk_o._enabled,
        "VLLM_HCU_USE_CUSTOM_AITER_FLA",
    ),
    (
        "fla_chunk_delta_h",
        patch_fla_chunk_delta_h._enabled,
        "VLLM_HCU_USE_CUSTOM_AITER_FLA",
    ),
    (
        "lightop_input_fp8",
        patch_input_quant_fp8._lightop_requested,
        "VLLM_HCU_USE_LIGHTOP_PER_TOKEN_QUANT_FP8",
    ),
    (
        "qwen35_mamba_dtype",
        patch_qwen3_5_mamba_state_dtype._auto_dtype_enabled,
        "VLLM_HCU_MAMBA_SSM_CACHE_DTYPE",
    ),
    (
        "topk_topp_sampler",
        TOPK_TOPP_SAMPLE._use_hcu_topk_topp_sampler,
        "VLLM_HCU_USE_CUSTOM_TOPK_TOPP_SAMPLER",
    ),
)


@pytest.mark.parametrize(
    ("selector_name", "selector", "feature_name"),
    SELECTORS,
    ids=[item[0] for item in SELECTORS],
)
@pytest.mark.parametrize(
    ("master_enabled", "feature_enabled"),
    [(False, False), (False, True), (True, False), (True, True)],
)
def test_master_custom_ops_gate_routes_feature_selectors(
    monkeypatch: pytest.MonkeyPatch,
    selector_name: str,
    selector,
    feature_name: str,
    master_enabled: bool,
    feature_enabled: bool,
) -> None:
    del selector_name
    monkeypatch.setenv(
        "VLLM_HCU_USE_CUSTOM_OPS",
        "1" if master_enabled else "0",
    )
    monkeypatch.setenv(feature_name, "true" if feature_enabled else "false")

    assert selector() is (master_enabled and feature_enabled)


@pytest.mark.parametrize("enabled", [False, True])
def test_custom_quantization_environment_reaches_int8_path_selector(
    monkeypatch: pytest.MonkeyPatch,
    enabled: bool,
) -> None:
    monkeypatch.setenv(
        "VLLM_HCU_USE_CUSTOM_QUANTIZATION_GEMM",
        "1" if enabled else "0",
    )

    assert (
        patch_compressed_tensors_w8a8_int8._custom_quantization_enabled()
        is enabled
    )


@pytest.mark.parametrize("enabled", [False, True])
def test_nn_layout_environment_reaches_mamba_path_selector(
    monkeypatch: pytest.MonkeyPatch,
    enabled: bool,
) -> None:
    monkeypatch.setenv("VLLM_USE_NN", "true" if enabled else "false")

    assert patch_mamba_mixer2._nn_enabled() is enabled


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ({}, "varlen"),
        ({"VLLM_HCU_USE_FLASH_ATTN": "1"}, "classic"),
        ({"VLLM_HCU_USE_FLASH_ATTN_VARLEN": "1"}, "varlen"),
        (
            {
                "VLLM_HCU_USE_FLASH_ATTN_UNIFIED": "1",
                "VLLM_HCU_USE_FLASH_ATTN_VARLEN": "1",
            },
            "varlen",
        ),
        (
            {
                "VLLM_HCU_USE_FLASH_ATTN": "1",
                "VLLM_HCU_USE_FLASH_ATTN_UNIFIED": "1",
            },
            "cutlass",
        ),
        (
            {
                "VLLM_HCU_USE_FLASH_ATTN": "1",
                "VLLM_HCU_USE_FLASH_ATTN_UNIFIED": "1",
                "VLLM_HCU_USE_FLASH_ATTN_VARLEN": "1",
                "VLLM_HCU_USE_CUSTOM_FLASH_ATTN": "1",
            },
            "custom",
        ),
    ],
)
def test_flash_attention_environment_priority_routes_expected_backend(
    monkeypatch: pytest.MonkeyPatch,
    values: dict[str, str],
    expected: str,
) -> None:
    names = (
        "VLLM_HCU_USE_FLASH_ATTN",
        "VLLM_HCU_USE_FLASH_ATTN_UNIFIED",
        "VLLM_HCU_USE_FLASH_ATTN_VARLEN",
        "VLLM_HCU_USE_CUSTOM_FLASH_ATTN",
    )
    for name in names:
        monkeypatch.delenv(name, raising=False)
    for name, value in values.items():
        monkeypatch.setenv(name, value)

    assert hcu_envs.resolve_hcu_flash_attn_mode(None) == expected


def test_flash_attention_lazy_environment_defaults_select_varlen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("VLLM_HCU_USE_FLASH_ATTN_UNIFIED", raising=False)
    monkeypatch.delenv("VLLM_HCU_USE_FLASH_ATTN_VARLEN", raising=False)

    assert hcu_envs.VLLM_HCU_USE_FLASH_ATTN_UNIFIED is False
    assert hcu_envs.VLLM_HCU_USE_FLASH_ATTN_VARLEN is True


@pytest.mark.parametrize(
    ("explicit", "expected"),
    [
        ("classic", "classic"),
        ("unified", "cutlass"),
        ("cutlass", "cutlass"),
        ("varlen", "varlen"),
        ("custom", "custom"),
    ],
)
def test_explicit_flash_attention_mode_overrides_environment(
    monkeypatch: pytest.MonkeyPatch,
    explicit: str,
    expected: str,
) -> None:
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_FLASH_ATTN", "1")
    monkeypatch.setenv("VLLM_HCU_USE_FLASH_ATTN_UNIFIED", "1")
    monkeypatch.setenv("VLLM_HCU_USE_FLASH_ATTN", "1")

    assert hcu_envs.resolve_hcu_flash_attn_mode(explicit) == expected


@pytest.mark.parametrize(
    ("master_enabled", "feature_enabled"),
    [(False, False), (False, True), (True, False), (True, True)],
)
def test_input_fp8_environment_selects_custom_or_official_wrapper(
    monkeypatch: pytest.MonkeyPatch,
    master_enabled: bool,
    feature_enabled: bool,
) -> None:
    calls: list[str] = []

    class GroupShape:
        PER_TOKEN = object()

    class QuantFP8:
        group_shape = GroupShape.PER_TOKEN
        num_token_padding = None

        def forward_cuda(
            self,
            x,
            scale=None,
            scale_ub=None,
            use_triton=False,
        ):
            del self, x, scale, scale_ub, use_triton
            calls.append("official-cuda")
            return "official-cuda"

        def forward_native(
            self,
            x,
            scale=None,
            scale_ub=None,
            use_triton=False,
        ):
            del self, x, scale, scale_ub, use_triton
            calls.append("official-native")
            return "official-native"

    target = _module(
        patch_input_quant_fp8.TARGET_MODULE,
        GroupShape=GroupShape,
        QuantFP8=QuantFP8,
        _FP8_DTYPE=torch.float8_e4m3fnuz,
    )
    runtime = _module(
        "vllm_hcu.model_executor.layers.quantization.lightop_fp8_runtime",
        quantize=lambda x, dtype, register: (
            calls.append("custom") or ("custom", x, dtype, register)
        ),
    )
    monkeypatch.setitem(sys.modules, runtime.__name__, runtime)
    import vllm_hcu.model_executor.layers.quantization as quantization_package

    monkeypatch.setattr(
        quantization_package,
        "lightop_fp8_runtime",
        runtime,
        raising=False,
    )
    monkeypatch.setenv(
        "VLLM_HCU_USE_CUSTOM_OPS",
        "1" if master_enabled else "0",
    )
    monkeypatch.setenv(
        "VLLM_HCU_USE_LIGHTOP_PER_TOKEN_QUANT_FP8",
        "1" if feature_enabled else "0",
    )
    patch_input_quant_fp8.apply_to_module(target)
    instance = QuantFP8()
    value = torch.ones((2, 8)).contiguous()

    cuda_result = instance.forward_cuda(value)
    native_result = instance.forward_native(value)

    if master_enabled and feature_enabled:
        assert cuda_result[0] == "custom"
        assert native_result[0] == "custom"
        assert calls == ["custom", "custom"]
    else:
        assert cuda_result == "official-native"
        assert native_result == "official-native"
        assert calls == ["official-native", "official-native"]


@pytest.mark.parametrize(
    ("master_enabled", "feature_enabled"),
    [(False, False), (False, True), (True, False), (True, True)],
)
def test_qwen35_mamba_environment_selects_auto_or_official_dtype(
    monkeypatch: pytest.MonkeyPatch,
    master_enabled: bool,
    feature_enabled: bool,
) -> None:
    class Model:
        @classmethod
        def get_mamba_state_dtype_from_config(cls, vllm_config):
            del cls, vllm_config
            return "official"

    class Calculator:
        @staticmethod
        def gated_delta_net_state_dtype(
            model_dtype,
            mamba_cache_dtype,
            mamba_ssm_cache_dtype="auto",
        ):
            return (
                "custom",
                model_dtype,
                mamba_cache_dtype,
                mamba_ssm_cache_dtype,
            )

    target = _module(
        patch_qwen3_5_mamba_state_dtype.TARGET_MODULE,
        Qwen3_5ForConditionalGeneration=Model,
        MambaStateDtypeCalculator=Calculator,
    )
    monkeypatch.setenv(
        "VLLM_HCU_USE_CUSTOM_OPS",
        "1" if master_enabled else "0",
    )
    monkeypatch.setenv(
        "VLLM_HCU_MAMBA_SSM_CACHE_DTYPE",
        "1" if feature_enabled else "0",
    )
    patch_qwen3_5_mamba_state_dtype.apply_to_module(target)
    config = SimpleNamespace(
        model_config=SimpleNamespace(dtype=torch.bfloat16),
        cache_config=SimpleNamespace(mamba_cache_dtype="float32"),
    )

    result = Model.get_mamba_state_dtype_from_config(config)

    if master_enabled and feature_enabled:
        assert result == ("custom", torch.bfloat16, "float32", "auto")
    else:
        assert result == "official"


@pytest.mark.parametrize("enabled", [False, True])
def test_eplb_environment_selects_official_or_torch_path(
    monkeypatch: pytest.MonkeyPatch,
    enabled: bool,
) -> None:
    monkeypatch.setenv(
        "VLLM_HCU_USE_TORCH_EPLB_MAP_RECORD",
        "1" if enabled else "0",
    )
    ids = torch.tensor([[0, 1]], dtype=torch.int32)
    loads = torch.zeros(2, dtype=torch.int64)
    mapping = torch.tensor([[0], [1]], dtype=torch.int64)
    replicas = torch.tensor([1, 1], dtype=torch.int64)

    result = eplb_map_to_physical_and_record(
        SimpleNamespace(torch=torch),
        lambda *args: args[0] + 10,
        ids,
        loads,
        mapping,
        replicas,
        torch.tensor(True),
    )

    if enabled:
        torch.testing.assert_close(result, ids, rtol=0, atol=0)
        torch.testing.assert_close(
            loads,
            torch.tensor([1, 1]),
            rtol=0,
            atol=0,
        )
    else:
        torch.testing.assert_close(result, ids + 10, rtol=0, atol=0)
        torch.testing.assert_close(loads, torch.zeros_like(loads), rtol=0, atol=0)
