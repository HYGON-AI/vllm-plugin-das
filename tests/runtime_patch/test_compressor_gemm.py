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
    not _cuda_usable(), reason="torch.mm(out_dtype=) needs a working GPU"
)

from vllm_hcu.patch.worker.core_fix import patch_deepseek_v4_attention as patch


@pytest.fixture
def blas(monkeypatch):
    calls = []

    def linear(x, w):
        assert w.is_contiguous()
        calls.append(w)
        return x.float() @ w.float().T

    monkeypatch.setattr(patch, "linear_bf16_fp32", linear)
    monkeypatch.setattr(patch, "_compressor_nt_weight",
                        lambda w: None if w.shape[0] == w.shape[1] else w.T.contiguous())
    return calls


@pytest.mark.parametrize("layout", ["nn", "nt"])
@pytest.mark.parametrize("backend", ["torch", "blaslt"])
def test_outputs_and_dispatch(monkeypatch, blas, layout, backend):
    monkeypatch.setenv("VLLM_HCU_COMPRESSOR_GEMM_BACKEND", backend)
    x = torch.randn(7, 5, dtype=torch.bfloat16, device="cuda")
    nn = torch.randn(5, 3, dtype=torch.bfloat16, device="cuda")
    w = nn if layout == "nn" else nn.T.contiguous()
    result = patch._compressor_mm(x, w)
    torch.testing.assert_close(result, x.float() @ nn.float(), rtol=1e-2, atol=1e-2)
    assert result.dtype == torch.float32
    assert len(blas) == (backend == "blaslt")
    if blas:
        assert blas[0].shape == (3, 5)


@pytest.mark.parametrize("case", ["fp32", "square"])
def test_ineligible_weights_keep_torch(monkeypatch, blas, case):
    monkeypatch.setenv("VLLM_HCU_COMPRESSOR_GEMM_BACKEND", "blaslt")
    dtype = torch.float32 if case == "fp32" else torch.bfloat16
    k = 5
    x = torch.randn(2, k, dtype=dtype, device="cuda")
    w = torch.randn(k, k if case == "square" else 3, dtype=dtype, device="cuda")
    torch.testing.assert_close(patch._compressor_mm(x, w),
                               torch.mm(x, w, out_dtype=torch.float32))
    assert not blas


def test_nt_cache_reuse_and_reload():
    w = torch.randn(5, 3, dtype=torch.bfloat16, device="cuda")
    first = patch._compressor_nt_weight(w)
    assert patch._compressor_nt_weight(w) is first
    w.add_(1)
    second = patch._compressor_nt_weight(w)
    assert second is not first
    torch.testing.assert_close(second, w.T)
    assert patch._compressor_nt_weight(torch.ones(4, 4, device="cuda")) is None
