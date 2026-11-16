# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.hcu


def _require_hcu() -> None:
    if not torch.cuda.is_available() or torch.version.hip is None:
        pytest.skip("a live HCU/ROCm device is required")
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":", 1)[0]
    if arch not in {"gfx936", "gfx938"}:
        pytest.skip(f"AutoAWQ BF16 regression requires HCU, got {arch}")


@pytest.mark.parametrize(
    ("num_tokens", "expected_kernel"),
    ((1, "gemm"), (255, "gemm"), (256, "dequantize")),
)
@pytest.mark.parametrize(
    "lightop_wrapper",
    (False, True),
    ids=("lightop-disabled", "lightop-bf16-fallback"),
)
def test_autoawq_bf16_executes_real_triton_at_token_boundary(
    monkeypatch: pytest.MonkeyPatch,
    num_tokens: int,
    expected_kernel: str,
    lightop_wrapper: bool,
) -> None:
    """BF16 must compile on HCU and retain AutoAWQ's M=256 boundary."""
    _require_hcu()

    from vllm.model_executor.layers.quantization import awq_triton
    from vllm.model_executor.layers.quantization.auto_awq import (
        AutoAWQConfig,
        AutoAWQLinearMethod,
    )
    from vllm_hcu.model_executor.layers.quantization import (
        lightop_autoawq,
    )
    from vllm_hcu.platforms.hcu import HCUPlatform

    monkeypatch.delenv("VLLM_USE_TRITON_AWQ", raising=False)
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "0")
    monkeypatch.setenv(
        "VLLM_HCU_USE_LIGHTOP_AWQ", "1" if lightop_wrapper else "0"
    )
    HCUPlatform.verify_quantization("auto_awq")

    torch.manual_seed(29)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    k = n = group_size = 128
    qweight = torch.randint(
        -(2**31),
        2**31 - 1,
        (k, n // 8),
        device=device,
        dtype=torch.int32,
    )
    qzeros = torch.randint(
        -(2**31),
        2**31 - 1,
        (k // group_size, n // 8),
        device=device,
        dtype=torch.int32,
    )
    scales = torch.empty(
        (k // group_size, n), device=device, dtype=dtype
    ).uniform_(0.0005, 0.002)
    inputs = torch.randn((num_tokens, k), device=device, dtype=dtype).mul_(0.1)

    original_gemm = awq_triton.awq_gemm_triton
    original_dequantize = awq_triton.awq_dequantize_triton
    dequantized = original_dequantize(qweight, scales, qzeros)
    reference = torch.matmul(inputs.float(), dequantized.float())

    calls: list[str] = []

    def recorded_gemm(*args, **kwargs):
        calls.append("gemm")
        return original_gemm(*args, **kwargs)

    def recorded_dequantize(*args, **kwargs):
        calls.append("dequantize")
        return original_dequantize(*args, **kwargs)

    monkeypatch.setattr(awq_triton, "awq_gemm_triton", recorded_gemm)
    monkeypatch.setattr(
        awq_triton, "awq_dequantize_triton", recorded_dequantize
    )

    config = AutoAWQConfig(4, group_size, True, False)
    delegate = AutoAWQLinearMethod(config)
    method = (
        lightop_autoawq.LightOpAutoAWQLinearMethod(delegate, config)
        if lightop_wrapper
        else delegate
    )
    layer = torch.nn.Module()
    layer.register_parameter(
        "qweight", torch.nn.Parameter(qweight, requires_grad=False)
    )
    layer.register_parameter(
        "qzeros", torch.nn.Parameter(qzeros, requires_grad=False)
    )
    layer.register_parameter(
        "scales", torch.nn.Parameter(scales, requires_grad=False)
    )

    if lightop_wrapper:
        monkeypatch.setattr(
            lightop_autoawq,
            "is_lightop_awq_shape_supported",
            lambda _k, _n: True,
        )
        method._params_dtype = dtype
        method._k = k
        method._n = n
        # With shape and quantization constraints satisfied, BF16 alone must
        # keep the wrapper on the real AutoAWQ Triton delegate.
        assert not method._eligible()
    method.process_weights_after_loading(layer)

    with torch.inference_mode():
        output = method.apply(layer, inputs)
        torch.cuda.synchronize()

    assert calls == [expected_kernel]
    assert output.dtype == dtype
    assert output.shape == (num_tokens, n)
    torch.testing.assert_close(
        output.float(), reference, rtol=0.02, atol=0.0005
    )
