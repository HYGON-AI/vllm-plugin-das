# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from __future__ import annotations

import importlib
import importlib.util
from types import SimpleNamespace

import pytest
import torch


def _load_autoawq_module():
    spec = importlib.util.find_spec(
        "vllm_hcu.model_executor.layers.quantization.lightop_autoawq"
    )
    assert spec is not None, "LightOp AutoAWQ backend is not implemented"
    return importlib.import_module(
        "vllm_hcu.model_executor.layers.quantization.lightop_autoawq"
    )


def test_awq_conversion_uses_checkpoint_nibble_order_and_raw_zero_points() -> None:
    autoawq = _load_autoawq_module()
    repeated_nibbles = [
        0,
        286331153,
        572662306,
        858993459,
        1145324612,
        1431655765,
        1717986918,
        2004318071,
    ]
    qweight = torch.tensor(repeated_nibbles, dtype=torch.int32).reshape(8, 1)
    qzeros = torch.tensor([-2042464975], dtype=torch.int32).reshape(1, 1)
    scales = torch.tensor(
        [[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]],
        dtype=torch.float16,
    )

    weight_trans, scales_zeros = autoawq.convert_awq_to_lightop_layout(
        qweight, qzeros, scales, group_size=8
    )

    assert weight_trans.shape == (8, 1)
    assert torch.equal(
        weight_trans[:, 0],
        torch.full((8,), 1966171168, dtype=torch.int32),
    )
    metadata = scales_zeros.flatten().view(torch.float16).reshape(8, 1, 2)
    assert torch.equal(
        metadata[:, 0, 0],
        torch.tensor([65, 66, 67, 68, 69, 70, 71, 72], dtype=torch.float16),
    )
    assert torch.equal(metadata[:, 0, 1], scales[0])


@pytest.mark.parametrize(
    ("k", "n", "expected"),
    ((8192, 10240, True), (4096, 8192, True), (4096, 4096, False)),
)
def test_awq_shape_support_is_limited_to_lightop_tuned_pairs(
    k: int, n: int, expected: bool
) -> None:
    autoawq = _load_autoawq_module()
    assert autoawq.is_lightop_awq_shape_supported(k, n) is expected


class _Delegate:
    def __init__(self) -> None:
        self.created = 0
        self.processed = 0
        self.applied = 0

    def create_weights(
        self,
        layer,
        input_size_per_partition,
        output_partition_sizes,
        input_size,
        output_size,
        params_dtype,
        **extra_weight_attrs,
    ) -> None:
        self.created += 1
        output_size_per_partition = sum(output_partition_sizes)
        layer.register_parameter(
            "qweight",
            torch.nn.Parameter(
                torch.zeros(
                    input_size_per_partition,
                    output_size_per_partition // 8,
                    dtype=torch.int32,
                ),
                requires_grad=False,
            ),
        )
        layer.register_parameter(
            "qzeros",
            torch.nn.Parameter(
                torch.zeros(
                    input_size_per_partition // 128,
                    output_size_per_partition // 8,
                    dtype=torch.int32,
                ),
                requires_grad=False,
            ),
        )
        layer.register_parameter(
            "scales",
            torch.nn.Parameter(
                torch.ones(
                    input_size_per_partition // 128,
                    output_size_per_partition,
                    dtype=params_dtype,
                ),
                requires_grad=False,
            ),
        )

    def process_weights_after_loading(self, layer) -> None:
        self.processed += 1

    def apply(self, layer, x, bias=None):
        self.applied += 1
        return x.new_full(x.shape[:-1] + (3,), 17)


def _make_method(autoawq, delegate, *, dtype=torch.float16):
    quant_config = SimpleNamespace(
        weight_bits=4,
        group_size=128,
        zero_point=True,
        pack_factor=8,
    )
    method = autoawq.LightOpAutoAWQLinearMethod(delegate, quant_config)
    layer = torch.nn.Module()
    method.create_weights(layer, 128, [8], 128, 8, dtype)
    return method, layer


def test_hybrid_awq_replaces_standard_weights_once_and_preserves_apply_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    autoawq = _load_autoawq_module()
    delegate = _Delegate()
    method, layer = _make_method(autoawq, delegate)
    gemm_calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def repack(weight_trans, n, k):
        assert (n, k) == (8, 128)
        return weight_trans + 9

    def gemm(inputs, weight, scales_zeros):
        gemm_calls.append((inputs, weight, scales_zeros))
        return torch.arange(
            inputs.shape[0] * scales_zeros.shape[0], dtype=inputs.dtype
        ).reshape(inputs.shape[0], scales_zeros.shape[0])

    monkeypatch.setattr(
        autoawq, "is_lightop_awq_shape_supported", lambda k, n: True
    )
    monkeypatch.setattr(
        autoawq, "_resolve_lightop_awq_ops", lambda: (repack, gemm)
    )
    method.process_weights_after_loading(layer)

    assert delegate.processed == 0
    assert not hasattr(layer, "qzeros")
    assert not hasattr(layer, "scales")
    assert hasattr(layer, "scales_zeros")
    inputs = torch.ones((2, 2, 128), dtype=torch.float16)
    bias = torch.arange(8, dtype=torch.float16)
    output = method.apply(layer, inputs, bias)
    expected = torch.arange(32, dtype=torch.float16).reshape(2, 2, 8) + bias
    assert torch.equal(output, expected)
    assert len(gemm_calls) == 1
    assert gemm_calls[0][0].shape == (4, 128)
    assert delegate.applied == 0


def test_hybrid_awq_delegates_without_mutating_unsupported_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    autoawq = _load_autoawq_module()
    delegate = _Delegate()
    method, layer = _make_method(autoawq, delegate, dtype=torch.bfloat16)
    qweight = layer.qweight
    qzeros = layer.qzeros
    scales = layer.scales
    monkeypatch.setattr(
        autoawq,
        "_resolve_lightop_awq_ops",
        lambda: pytest.fail("LightOp must not be resolved for BF16"),
    )

    method.process_weights_after_loading(layer)
    output = method.apply(layer, torch.ones((2, 128), dtype=torch.bfloat16))

    assert delegate.processed == 1
    assert delegate.applied == 1
    assert layer.qweight is qweight
    assert layer.qzeros is qzeros
    assert layer.scales is scales
    assert torch.equal(output, torch.full((2, 3), 17, dtype=torch.bfloat16))
