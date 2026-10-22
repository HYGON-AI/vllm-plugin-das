# SPDX-License-Identifier: Apache-2.0
"""ModelOpt W8A16 linear: DeepGEMM marlin weight pre-shuffle.

The layer weight is shuffled once at load time, while DeepGEMM picks
WEIGHT_LAYOUT / KERNEL_TYPE per M bucket, so every M (decode, prefill and
the large-M dequant + rocBLAS path) must still match the raw-layout result.
"""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("requires a GPU", allow_module_level=True)
deepgemm = pytest.importorskip("deepgemm")
if not hasattr(deepgemm, "w8a16_marlin_weight"):
    pytest.skip("deepgemm lacks w8a16 marlin layout", allow_module_level=True)

from vllm_hcu.model_executor.layers.quantization.modelopt_w8a16_hcu import (  # noqa: E402
    ModelOptW8A16LinearMethod,
)
from vllm_hcu.model_executor.layers.quantization.w8a16_gemm import (  # noqa: E402
    W8A16_LAYOUT_MARLIN,
    W8A16_LAYOUT_RAW,
    w8a16_gemm,
    w8a16_should_preshuffle,
    w8a16_unmarlin_weight,
)
from vllm_hcu.model_executor.layers.quantization import (  # noqa: E402
    w8a16_marlin_utils as mu,
)

# Qwen3.5 shapes with tuned WEIGHT_LAYOUT=1 configs in DeepGEMM.
MARLIN_NK = [(12288, 2560), (2560, 4096), (18432, 2560), (2560, 9216), (10240, 2560)]
M_LIST = [1, 2, 7, 16, 17, 32, 33, 64, 200, 256, 511, 512, 1024]


def _inputs(m, n, k, dtype):
    x = torch.randn(m, k, dtype=dtype, device="cuda")
    w = torch.randint(-128, 127, (n, k), dtype=torch.int8, device="cuda")
    scale = ((torch.rand(n, device="cuda") + 0.5) * 0.01).to(dtype)
    return x, w, scale


def _ref(x, w, scale, bias=None):
    w16 = (w.float() * scale.float()[:, None])
    out = x.float() @ w16.t()
    if bias is not None:
        out = out + bias.float()
    return out


@pytest.mark.parametrize("n,k", MARLIN_NK)
def test_unmarlin_roundtrip(n, k):
    w = torch.randint(-128, 127, (n, k), dtype=torch.int8, device="cuda")
    shuffled = deepgemm.w8a16_marlin_weight(w)
    assert not torch.equal(shuffled, w)
    assert torch.equal(w8a16_unmarlin_weight(shuffled), w)


@pytest.mark.parametrize("n,k", MARLIN_NK)
def test_should_preshuffle_tuned_shapes(n, k):
    assert w8a16_should_preshuffle(n, k)


def test_should_not_preshuffle_untuned_or_misaligned():
    assert not w8a16_should_preshuffle(12288 + 8, 2560)
    assert not w8a16_should_preshuffle(12288, 2560 + 64)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("n,k", MARLIN_NK)
@pytest.mark.parametrize("m", M_LIST)
def test_w8a16_gemm_raw_and_marlin(m, n, k, dtype):
    torch.manual_seed(m * 7 + n + k)
    x, w, scale = _inputs(m, n, k, dtype)
    bias = torch.randn(n, dtype=dtype, device="cuda")
    ref = _ref(x, w, scale, bias)
    out_raw = w8a16_gemm(x, w, scale, bias, W8A16_LAYOUT_RAW)
    out_marlin = w8a16_gemm(
        x, deepgemm.w8a16_marlin_weight(w), scale, bias, W8A16_LAYOUT_MARLIN)
    assert out_marlin.shape == (m, n) and out_marlin.dtype == dtype
    tol = dict(atol=0.5, rtol=0.02) if dtype == torch.float16 else dict(atol=1.0, rtol=0.05)
    # Raw weights must not be read as marlin even when DeepGEMM's tuned config
    # for this M bucket says WEIGHT_LAYOUT=1.
    torch.testing.assert_close(out_raw.float(), ref, **tol)
    torch.testing.assert_close(out_marlin.float(), ref, **tol)


def test_marlin_gemm_3d_input():
    n, k = 12288, 2560
    x, w, scale = _inputs(8, n, k, torch.bfloat16)
    x3 = x.view(2, 4, k)
    out = w8a16_gemm(x3, deepgemm.w8a16_marlin_weight(w), scale, None, W8A16_LAYOUT_MARLIN)
    assert out.shape == (2, 4, n)
    torch.testing.assert_close(out.view(8, n).float(), _ref(x, w, scale), atol=1.0, rtol=0.05)


def _make_layer(w, scale):
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(w.clone(), requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(scale.clone(), requires_grad=False)
    return layer


@pytest.mark.parametrize("n,k,layout", [
    (12288, 2560, W8A16_LAYOUT_MARLIN),
    (4096, 4096, W8A16_LAYOUT_RAW),  # no tuned marlin config -> keep raw
])
def test_linear_method_process_and_apply(n, k, layout):
    x, w, scale = _inputs(16, n, k, torch.bfloat16)
    layer = _make_layer(w, scale)
    method = ModelOptW8A16LinearMethod(quant_config=None)
    method.process_weights_after_loading(layer)
    assert layer.weight_layout == layout
    if layout == W8A16_LAYOUT_MARLIN:
        assert torch.equal(layer.weight, deepgemm.w8a16_marlin_weight(w))
    else:
        assert torch.equal(layer.weight, w)
    out = method.apply(layer, x)
    torch.testing.assert_close(out.float(), _ref(x, w, scale), atol=1.0, rtol=0.05)


def test_compressed_tensors_process_and_apply():
    from vllm_hcu.model_executor.layers.quantization import (
        compressed_tensors_w8a16_hcu as ct)

    for n, k, layout in [(12288, 2560, W8A16_LAYOUT_MARLIN),
                         (4096, 4096, W8A16_LAYOUT_RAW)]:
        x, w, scale = _inputs(16, n, k, torch.bfloat16)
        # int8 -> compressed-tensors uint8b128, 4 per int32 word
        packed = (w.view(torch.uint8) ^ 0x80).view(torch.int32)
        layer = torch.nn.Module()
        layer.weight_packed = torch.nn.Parameter(
            packed, requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(
            scale.view(n, 1), requires_grad=False)
        layer.weight_shape = torch.nn.Parameter(
            torch.tensor([n, k]), requires_grad=False)
        method = ct.CompressedTensorsW8A16Hcu()
        method.process_weights_after_loading(layer)
        assert layer.weight_layout == layout
        if layout == W8A16_LAYOUT_MARLIN:
            assert torch.equal(layer.weight,
                               deepgemm.w8a16_marlin_weight(w))
        else:
            assert torch.equal(layer.weight, w)
        out = method.apply_weights(layer, x)
        torch.testing.assert_close(
            out.float(), _ref(x, w, scale), atol=1.0, rtol=0.05)


def test_preshuffle_env_off(monkeypatch):
    monkeypatch.setenv("VLLM_HCU_DEEPGEMM_W8A16_PRESHUFFLE", "0")
    assert not w8a16_should_preshuffle(12288, 2560)


def test_deepgemm_off_marlin_uses_torch_fallback(monkeypatch):
    def boom(*a, **kw):
        raise AssertionError("DeepGEMM must not be called")

    monkeypatch.setenv("VLLM_HCU_USE_DEEPGEMM_W8A16_GEMM", "0")
    monkeypatch.setattr(deepgemm, "w8a16_linear", boom)
    x, w, scale = _inputs(64, 12288, 2560, torch.bfloat16)
    out = w8a16_gemm(x, deepgemm.w8a16_marlin_weight(w), scale, None, W8A16_LAYOUT_MARLIN)
    torch.testing.assert_close(out.float(), _ref(x, w, scale), atol=1.0, rtol=0.05)


# --- large-M (KERNEL_TYPE=2) marlin dequant -----------------------------------


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("n,k", MARLIN_NK)
def test_marlin_dequant_bit_exact(n, k, dtype):
    _, w, scale = _inputs(1, n, k, dtype)
    deq = (deepgemm.op.w8a16_dequant_weight_bf16 if dtype == torch.bfloat16
           else deepgemm.op.w8a16_dequant_weight)
    assert torch.equal(mu.w8a16_marlin_dequant(deepgemm.w8a16_marlin_weight(w), scale, dtype),
                       deq(w, scale))


def test_large_m_marlin_uses_one_pass_dequant(monkeypatch):
    calls = []
    real = mu.w8a16_marlin_dequant
    monkeypatch.setattr(mu, "w8a16_marlin_dequant", lambda *a: calls.append(1) or real(*a))
    n, k, m = 12288, 2560, 2048
    assert int(deepgemm.get_w8a16_linear_config(m, n, k)["KERNEL_TYPE"]) == 2
    x, w, scale = _inputs(m, n, k, torch.bfloat16)
    out = w8a16_gemm(x, deepgemm.w8a16_marlin_weight(w), scale, None, W8A16_LAYOUT_MARLIN)
    torch.testing.assert_close(out.float(), _ref(x, w, scale), atol=1.0, rtol=0.05)
    assert len(calls) == 1
