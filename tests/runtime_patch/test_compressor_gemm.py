# SPDX-License-Identifier: Apache-2.0
import pytest
import torch


def _cuda_usable():
    try:
        torch.empty(1, device="cuda")
        return True
    except RuntimeError:
        return False


pytestmark = pytest.mark.skipif(
    not _cuda_usable(), reason="compressor GEMM tests need a working GPU"
)

from vllm_hcu.patch.worker.core_fix import patch_deepseek_v4_attention as patch


class CompressorLinear(torch.nn.Module):
    def __init__(self, weight, input_size):
        super().__init__()
        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        self.input_size_per_partition = input_size


class CompressorModel(torch.nn.Module):
    def __init__(self, weight, input_size):
        super().__init__()
        self.fused_wkv_wgate = CompressorLinear(weight, input_size)


@pytest.fixture
def blas(monkeypatch):
    calls = []

    def linear(x, w):
        assert w.is_contiguous()
        calls.append(w)
        return x.float() @ w.float().T

    monkeypatch.setattr(patch, "linear_bf16_fp32", linear)
    return calls


def _model(weight, input_size=5):
    return CompressorModel(weight.clone(), input_size).cuda()


@pytest.mark.parametrize("layout", ["nn", "nt"])
@pytest.mark.parametrize("backend", ["torch", "blaslt"])
def test_outputs_and_dispatch(monkeypatch, blas, layout, backend):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.setenv("VLLM_HCU_COMPRESSOR_GEMM_BACKEND", backend)
    x = torch.randn(7, 5, dtype=torch.bfloat16, device="cuda")
    nn = torch.randn(5, 3, dtype=torch.bfloat16, device="cuda")
    w = nn if layout == "nn" else nn.T.contiguous()
    model = _model(w)

    assert patch.bind_compressor_nt_weights(model) == (
        layout == "nn" and backend == "blaslt"
    )
    result = patch._compressor_mm(x, model.fused_wkv_wgate)
    torch.testing.assert_close(result, x.float() @ nn.float(), rtol=1e-2, atol=1e-2)
    assert result.dtype == torch.float32
    assert len(blas) == (layout == "nn" and backend == "blaslt")
    if blas:
        assert blas[0].shape == (3, 5)


def test_master_switch_disables_blaslt(monkeypatch, blas):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "0")
    monkeypatch.setenv("VLLM_HCU_COMPRESSOR_GEMM_BACKEND", "blaslt")
    x = torch.randn(7, 5, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(5, 3, dtype=torch.bfloat16, device="cuda")
    model = _model(w)

    assert patch.bind_compressor_nt_weights(model) == 0
    result = patch._compressor_mm(x, model.fused_wkv_wgate)

    torch.testing.assert_close(result, torch.mm(x, w, out_dtype=torch.float32))
    assert not blas


@pytest.mark.parametrize("case", ["fp32", "square"])
def test_ineligible_weights_keep_torch(monkeypatch, blas, case):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.setenv("VLLM_HCU_COMPRESSOR_GEMM_BACKEND", "blaslt")
    dtype = torch.float32 if case == "fp32" else torch.bfloat16
    k = 5
    w = torch.randn(k, k if case == "square" else 3, dtype=dtype, device="cuda")
    model = _model(w)

    assert patch.bind_compressor_nt_weights(model) == 0
    x = torch.randn(2, k, dtype=dtype, device="cuda")
    torch.testing.assert_close(
        patch._compressor_mm(x, model.fused_wkv_wgate),
        torch.mm(x, w, out_dtype=torch.float32),
    )
    assert not blas


def test_invalid_backend_is_rejected(monkeypatch):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.setenv("VLLM_HCU_COMPRESSOR_GEMM_BACKEND", "invalid")
    model = _model(torch.randn(5, 3, dtype=torch.bfloat16, device="cuda"))

    with pytest.raises(ValueError, match="must be 'blaslt' or 'torch'"):
        patch.bind_compressor_nt_weights(model)


def test_compile_uses_materialized_buffer(monkeypatch):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.setenv("VLLM_HCU_COMPRESSOR_GEMM_BACKEND", "blaslt")
    monkeypatch.setattr(
        patch,
        "linear_bf16_fp32",
        lambda x, w: x.float() @ w.float().T,
    )
    model = _model(torch.randn(5, 3, dtype=torch.bfloat16, device="cuda"))
    assert patch.bind_compressor_nt_weights(model) == 1

    compiled = torch.compile(
        lambda x: patch._compressor_mm(x, model.fused_wkv_wgate),
        fullgraph=True,
        backend="eager",
    )
    x = torch.randn(2, 5, dtype=torch.bfloat16, device="cuda")
    torch.testing.assert_close(compiled(x), x.float() @ model.fused_wkv_wgate.weight.float())


def test_refresh_reuses_buffer_after_hot_reload(monkeypatch):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.setenv("VLLM_HCU_COMPRESSOR_GEMM_BACKEND", "blaslt")
    weight = torch.randn(5, 3, dtype=torch.bfloat16, device="cuda")
    model = _model(weight)
    assert patch.bind_compressor_nt_weights(model) == 1
    linear = model.fused_wkv_wgate
    nt = linear._vllm_hcu_compressor_nt_weight
    pointer = nt.data_ptr()
    version = linear.weight._version

    new_weight = torch.randn_like(linear.weight)
    linear.weight.data.copy_(new_weight)
    assert linear.weight._version == version
    patch.refresh_compressor_nt_weights(model)
    assert nt.data_ptr() == pointer
    torch.testing.assert_close(nt, new_weight.T)

    replacement = torch.nn.Parameter(torch.randn_like(new_weight), requires_grad=False)
    linear.weight = replacement
    patch.refresh_compressor_nt_weights(model)
    assert nt.data_ptr() == pointer
    torch.testing.assert_close(nt, replacement.T)
