# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Portable numerical-reference tests for HCU operator orchestration."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from vllm_hcu.model_executor.layers.fused_moe import int8_quant_runtime
from vllm_hcu.model_executor.layers.fused_moe.router_runtime import (
    eplb_map_to_physical_and_record,
)
from vllm_hcu.model_executor.layers.kv_cache_utils import split_kv_cache
from vllm_hcu.model_executor.layers.mamba_runtime import (
    mamba_v2_nn_sharded_weight_loader,
)
from vllm_hcu.model_executor.layers.quantization.int8_runtime import (
    apply_int8_linear,
    weight8bit_nt_kpack2_marlin2,
)


def _load_fp8_einsum_fallback() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[2]
        / "vllm_hcu"
        / "ops"
        / "fp8_einsum_fallback.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_vllm_hcu_accuracy_fp8_einsum_fallback",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_decode_e8m0_scales = _load_fp8_einsum_fallback()._decode_e8m0_scales


def _inverse_marlin2_layout(
    packed: torch.Tensor,
    original_shape: tuple[int, ...],
    *,
    k_tile: int,
    k_tile1: int,
    n_tile: int,
) -> torch.Tensor:
    size_n, size_k = original_shape[-2:]
    if len(original_shape) == 2:
        restored = packed.reshape(
            size_k // (k_tile * k_tile1),
            size_n // n_tile,
            k_tile1,
            n_tile,
            k_tile,
        )
        return (
            restored.permute(1, 3, 0, 2, 4)
            .contiguous()
            .reshape(original_shape)
        )
    experts = original_shape[0]
    restored = packed.reshape(
        experts,
        size_k // (k_tile * k_tile1),
        size_n // n_tile,
        k_tile1,
        n_tile,
        k_tile,
    )
    return (
        restored.permute(0, 2, 4, 1, 3, 5)
        .contiguous()
        .reshape(original_shape)
    )


@pytest.mark.parametrize(
    ("shape", "tiles"),
    [
        ((16, 64), (16, 4, 16)),
        ((32, 128), (16, 4, 16)),
        ((2, 16, 64), (16, 4, 16)),
        ((3, 8, 32), (8, 4, 8)),
    ],
)
def test_marlin2_layout_is_lossless_against_inverse_reference(
    shape: tuple[int, ...],
    tiles: tuple[int, int, int],
) -> None:
    weight = (
        torch.arange(torch.tensor(shape).prod().item(), dtype=torch.int64)
        .remainder(255)
        .sub(127)
        .to(torch.int8)
        .reshape(shape)
    )
    k_tile, k_tile1, n_tile = tiles

    packed = weight8bit_nt_kpack2_marlin2(
        weight,
        k_tile=k_tile,
        k_tile1=k_tile1,
        n_tile=n_tile,
    )
    restored = _inverse_marlin2_layout(
        packed,
        shape,
        k_tile=k_tile,
        k_tile1=k_tile1,
        n_tile=n_tile,
    )

    torch.testing.assert_close(restored, weight, rtol=0, atol=0)


def _module(name: str, **attributes: object) -> ModuleType:
    module = ModuleType(name)
    module.__dict__.update(attributes)
    return module


def _install_fake_lmslim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def per_token_quant_int8(
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        absmax = value.float().abs().amax(dim=-1, keepdim=True).clamp_min(1e-10)
        scale = absmax / 127.0
        quantized = torch.round(value.float() / scale).clamp(-127, 127).to(torch.int8)
        return quantized, scale.to(torch.float32)

    def hipblaslt_w8a8_gemm(
        activation: torch.Tensor,
        weight: torch.Tensor,
        activation_scale: torch.Tensor,
        weight_scale: torch.Tensor,
        m: int,
        n: int,
        k: int,
        layout: str,
        output_dtype: torch.dtype,
    ) -> tuple[bool, torch.Tensor]:
        assert activation.shape == (m, k)
        assert weight.shape == (n, k)
        assert layout == "NT"
        output = (activation.float() * activation_scale) @ (
            weight.float() * weight_scale
        ).t()
        return True, output.to(output_dtype)

    lmslim = _module(
        "lmslim",
        quant_ops=SimpleNamespace(
            hipblaslt_w8a8_gemm=hipblaslt_w8a8_gemm,
        ),
    )
    lmslim.__path__ = []
    layers = _module("lmslim.layers")
    layers.__path__ = []
    gemm = _module("lmslim.layers.gemm")
    gemm.__path__ = []
    int8_utils = _module(
        "lmslim.layers.gemm.int8_utils",
        per_token_quant_int8=per_token_quant_int8,
    )
    monkeypatch.setitem(sys.modules, "lmslim", lmslim)
    monkeypatch.setitem(sys.modules, "lmslim.layers", layers)
    monkeypatch.setitem(sys.modules, "lmslim.layers.gemm", gemm)
    monkeypatch.setitem(
        sys.modules,
        "lmslim.layers.gemm.int8_utils",
        int8_utils,
    )


@pytest.mark.parametrize(
    ("input_shape", "output_features", "output_dtype", "with_bias"),
    [
        ((2, 8), 5, torch.bfloat16, False),
        ((3, 2, 7), 4, torch.bfloat16, True),
        ((1, 4, 16), 9, torch.float16, False),
        ((2, 3, 5), 6, torch.float16, True),
    ],
)
def test_w8a8_linear_matches_dequantized_reference(
    monkeypatch: pytest.MonkeyPatch,
    input_shape: tuple[int, ...],
    output_features: int,
    output_dtype: torch.dtype,
    with_bias: bool,
) -> None:
    from vllm_hcu.platforms import envs as henvs

    monkeypatch.setattr(henvs, "VLLM_HCU_USE_CUSTOM_OPS", False)
    monkeypatch.setattr(henvs, "VLLM_HCU_USE_FUSED_SILU_MUL_QUANT", False)
    monkeypatch.setattr(henvs, "VLLM_HCU_USE_FUSED_RMS_QUANT", False)
    _install_fake_lmslim(monkeypatch)

    generator = torch.Generator().manual_seed(20250724)
    activation = torch.randn(input_shape, generator=generator, dtype=torch.float32)
    weight = torch.randint(
        -100,
        101,
        (output_features, input_shape[-1]),
        generator=generator,
        dtype=torch.int8,
    )
    weight_scale = (
        torch.rand((output_features, 1), generator=generator, dtype=torch.float32)
        * 0.02
        + 0.001
    )
    bias = (
        torch.randn(output_features, generator=generator).to(output_dtype)
        if with_bias
        else None
    )

    absmax = activation.abs().amax(dim=-1, keepdim=True).clamp_min(1e-10)
    activation_scale = absmax / 127.0
    activation_q = (
        torch.round(activation / activation_scale)
        .clamp(-127, 127)
        .to(torch.int8)
    )
    reference = (activation_q.float() * activation_scale) @ (
        weight.float() * weight_scale
    ).t()
    if bias is not None:
        reference = reference + bias.float()

    actual = apply_int8_linear(
        activation,
        weight,
        weight_scale,
        output_dtype,
        bias=bias,
    )

    torch.testing.assert_close(
        actual.float(),
        reference.float(),
        rtol=2e-2,
        atol=2e-2,
    )


class _CpuPerTokenQuantLauncher:
    def __getitem__(self, grid: object):
        del grid

        def launch(
            values: torch.Tensor,
            output: torch.Tensor,
            output_scales: torch.Tensor,
            stride_x: int,
            stride_xq: int,
            hidden: int,
            tokens_per_expert: torch.Tensor | None,
            max_tokens: int,
            **kwargs: object,
        ) -> None:
            del stride_x, stride_xq, hidden, kwargs
            output.zero_()
            output_scales.zero_()
            for expert in range(values.shape[0]):
                valid = (
                    max_tokens
                    if tokens_per_expert is None
                    else int(tokens_per_expert[expert])
                )
                rows = values[expert, :valid].float()
                if not rows.numel():
                    continue
                absmax = rows.abs().amax(dim=-1, keepdim=True).clamp_min(1e-10)
                scales = absmax / 127.0
                output[expert, :valid] = (
                    torch.round(rows / scales)
                    .clamp(-127, 127)
                    .to(torch.int8)
                )
                output_scales[expert, :valid] = scales

        return launch


@pytest.mark.parametrize(
    ("shape", "valid_tokens"),
    [
        ((1, 3, 5), None),
        ((2, 4, 8), (4, 2)),
        ((3, 2, 17), (0, 1, 2)),
        ((4, 5, 31), (1, 5, 3, 0)),
    ],
)
def test_expert_int8_quantization_matches_per_token_reference(
    monkeypatch: pytest.MonkeyPatch,
    shape: tuple[int, int, int],
    valid_tokens: tuple[int, ...] | None,
) -> None:
    monkeypatch.setattr(
        int8_quant_runtime,
        "_per_token_quant_int8_one_kernel",
        _CpuPerTokenQuantLauncher(),
    )
    monkeypatch.setattr(
        int8_quant_runtime.triton,
        "next_power_of_2",
        lambda value: 1 << (value - 1).bit_length(),
        raising=False,
    )
    generator = torch.Generator().manual_seed(1219304)
    values = torch.randn(shape, generator=generator, dtype=torch.float32) * 3.0
    counts = (
        None
        if valid_tokens is None
        else torch.tensor(valid_tokens, dtype=torch.int32)
    )

    quantized, scales = int8_quant_runtime.per_token_quant_int8(values, counts)

    for expert in range(shape[0]):
        valid = shape[1] if counts is None else int(counts[expert])
        if valid == 0:
            continue
        rows = values[expert, :valid]
        expected_scale = (
            rows.abs().amax(dim=-1, keepdim=True).clamp_min(1e-10) / 127.0
        )
        expected_quantized = (
            torch.round(rows / expected_scale)
            .clamp(-127, 127)
            .to(torch.int8)
        )
        torch.testing.assert_close(
            scales[expert, :valid],
            expected_scale,
            rtol=1e-6,
            atol=1e-7,
        )
        torch.testing.assert_close(
            quantized[expert, :valid],
            expected_quantized,
            rtol=0,
            atol=0,
        )
        dequantized = quantized[expert, :valid].float() * scales[expert, :valid]
        error = (dequantized - rows).abs()
        assert torch.all(error <= expected_scale / 2 + 1e-6)


def _eplb_reference(
    topk_ids: torch.Tensor,
    mapping: torch.Tensor,
    replica_count: torch.Tensor,
    *,
    load_size: int,
    record_enabled: bool,
    num_unpadded_tokens: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    physical = torch.full_like(topk_ids, -1)
    loads = torch.zeros(load_size, dtype=torch.int64)
    experts_per_token = topk_ids.shape[-1]
    for flat_index, logical_tensor in enumerate(topk_ids.reshape(-1)):
        logical = int(logical_tensor)
        if logical < 0 or logical >= replica_count.numel():
            continue
        token = flat_index // experts_per_token
        count = max(int(replica_count[logical]), 1)
        replica = ((token * 2654435769) % (1 << 32)) % count
        target = int(mapping[logical, replica])
        if target < 0 or target >= load_size:
            continue
        physical.reshape(-1)[flat_index] = target
        if record_enabled and (
            num_unpadded_tokens is None or token < num_unpadded_tokens
        ):
            loads[target] += 1
    return physical, loads


@pytest.mark.parametrize(
    ("topk_ids", "record_enabled", "num_unpadded_tokens"),
    [
        ([[0, 1], [0, 1]], True, None),
        ([[0, 1], [0, 1]], False, None),
        ([[0, 1], [0, 1]], True, 1),
        ([[-1, 2], [1, 9]], True, 2),
    ],
)
def test_eplb_mapping_matches_scalar_reference(
    monkeypatch: pytest.MonkeyPatch,
    topk_ids: list[list[int]],
    record_enabled: bool,
    num_unpadded_tokens: int | None,
) -> None:
    from vllm_hcu.platforms import envs as henvs

    monkeypatch.setattr(henvs, "VLLM_HCU_USE_TORCH_EPLB_MAP_RECORD", True)
    ids = torch.tensor(topk_ids, dtype=torch.int32)
    mapping = torch.tensor([[0, 1], [2, 2], [3, 1]], dtype=torch.int64)
    replica_count = torch.tensor([2, 1, 2], dtype=torch.int64)
    loads = torch.zeros(4, dtype=torch.int64)
    enabled = torch.tensor(record_enabled)
    unpadded = (
        None
        if num_unpadded_tokens is None
        else torch.tensor(num_unpadded_tokens)
    )
    expected_ids, expected_loads = _eplb_reference(
        ids,
        mapping,
        replica_count,
        load_size=loads.numel(),
        record_enabled=record_enabled,
        num_unpadded_tokens=num_unpadded_tokens,
    )

    actual_ids = eplb_map_to_physical_and_record(
        SimpleNamespace(torch=torch),
        lambda *args: args[0],
        ids,
        loads,
        mapping,
        replica_count,
        enabled,
        unpadded,
    )

    torch.testing.assert_close(actual_ids, expected_ids, rtol=0, atol=0)
    torch.testing.assert_close(loads, expected_loads, rtol=0, atol=0)


@pytest.mark.parametrize(
    "case",
    [
        "single_rank_output_last",
        "tp_rank_one_output_first",
        "duplicate_group_uses_rank_zero",
        "trimmed_tail",
    ],
)
def test_mamba_sharded_loader_matches_slice_reference(case: str) -> None:
    if case == "single_rank_output_last":
        spec, tp_size, tp_rank = [(4, 0, False)], 1, 0
        loaded = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        param = torch.zeros(2, 4)
        expected = loaded.clone()
    elif case == "tp_rank_one_output_first":
        spec, tp_size, tp_rank = [(4, 0, False)], 2, 1
        loaded = torch.arange(8, dtype=torch.float32).reshape(4, 2)
        param = torch.zeros(2, 2)
        expected = loaded[2:4].clone()
    elif case == "duplicate_group_uses_rank_zero":
        spec, tp_size, tp_rank = [(4, 0, 1.0)], 2, 1
        loaded = torch.arange(8, dtype=torch.float32).reshape(4, 2)
        param = torch.zeros(2, 2)
        expected = loaded[0:2].clone()
    else:
        spec, tp_size, tp_rank = [(6, 2, False)], 2, 1
        loaded = torch.arange(8, dtype=torch.float32).reshape(4, 2)
        param = torch.zeros(2, 3)
        expected = torch.zeros_like(param)
        expected[:, 0] = loaded[3]

    loader = mamba_v2_nn_sharded_weight_loader(spec, tp_size, tp_rank)
    loader(param, loaded)

    torch.testing.assert_close(param, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("exponent", "expected"),
    [
        (1, math.ldexp(1.0, -126)),
        (64, math.ldexp(1.0, -63)),
        (127, 1.0),
        (128, 2.0),
    ],
)
def test_e8m0_scale_decode_matches_exponent_reference(
    exponent: int,
    expected: float,
) -> None:
    encoded = torch.tensor([exponent], dtype=torch.uint8).view(
        torch.float8_e8m0fnu
    )

    actual = _decode_e8m0_scales(encoded)

    torch.testing.assert_close(
        actual,
        torch.tensor([expected], dtype=torch.float32),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize(
    ("container", "kv_axis"),
    [
        ("tuple", 0),
        ("list", 0),
        ("stacked_custom", 0),
        ("stacked_block_first", 1),
    ],
)
def test_split_kv_cache_is_value_preserving(
    container: str,
    kv_axis: int,
) -> None:
    key = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    value = key.add(100)
    if container == "tuple":
        cache: object = (key, value)
    elif container == "list":
        cache = [key, value]
    else:
        cache = torch.stack((key, value), dim=kv_axis)

    actual_key, actual_value = split_kv_cache(cache, kv_axis=kv_axis)

    torch.testing.assert_close(actual_key, key, rtol=0, atol=0)
    torch.testing.assert_close(actual_value, value, rtol=0, atol=0)
