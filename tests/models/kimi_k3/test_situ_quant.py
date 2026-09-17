"""Temporary SiTU quantization: selection and device numerical contracts."""
import pytest
import torch

from vllm_hcu.model_executor.layers.quantization.kimi_k3_situ_quant import (
    resolve_situ_quant, triton_situ_mul_quant,
)


def test_backend_is_explicit(monkeypatch):
    monkeypatch.setenv("VLLM_HCU_KIMI_HT_SITU_BACKEND", "triton")
    assert resolve_situ_quant() == ("triton", triton_situ_mul_quant)
    monkeypatch.setenv("VLLM_HCU_KIMI_HT_SITU_BACKEND", "auto")
    with pytest.raises(ValueError, match="lightop or triton"):
        resolve_situ_quant()


def test_ht_constructor_uses_triton_without_lightop_situ(monkeypatch):
    import sys
    from types import SimpleNamespace
    from vllm_hcu.model_executor.layers.quantization.kimi_k3_ht_runtime import (
        KimiK3HTExperts, DeepEPDeepGemmW4A8ContiguousExperts,
    )
    monkeypatch.setenv("VLLM_HCU_KIMI_HT_SITU_BACKEND", "triton")
    monkeypatch.setattr(DeepEPDeepGemmW4A8ContiguousExperts, "__init__",
                        lambda *args: None)
    monkeypatch.setitem(sys.modules, "lightop", SimpleNamespace())
    config = SimpleNamespace(activation_situ_beta=1., activation_situ_linear_beta=2.)
    experts = KimiK3HTExperts(config, None)
    assert experts._situ_quant is triton_situ_mul_quant
    assert experts._situ_backend == "triton"


@pytest.mark.parametrize("beta", [0, -1, float("nan"), float("inf")])
def test_invalid_beta_fails_before_launch(beta):
    with pytest.raises(ValueError, match="finite and positive"):
        triton_situ_mul_quant(torch.empty(2, 4), beta=beta, linear_beta=2)


@pytest.mark.parametrize("backend", ["triton", "lightop"])
@pytest.mark.parametrize("dim", [33, 384, 3072])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_device_quant_matches_math_reference(dim, dtype, backend, monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("HCU required")
    gen = torch.Generator(device="cuda").manual_seed(42)
    x = torch.randn((17, 2 * dim), generator=gen, device="cuda", dtype=dtype) * 8
    x[0].zero_()
    x[1].fill_(100)
    x[2].fill_(-100)
    monkeypatch.setenv("VLLM_HCU_KIMI_HT_SITU_BACKEND", backend)
    selected_backend, quant = resolve_situ_quant()
    assert selected_backend == backend
    if backend == "lightop" and dim % 32:
        with pytest.raises(ValueError, match="divisible by 32"):
            quant(x, beta=1.25, linear_beta=2.0)
        return
    q, scale = quant(x, beta=1.25, linear_beta=2.0)
    gate, up = x.float().chunk(2, -1)
    ref = 1.25 * torch.tanh(gate / 1.25) * torch.sigmoid(gate)
    ref = ref * 2.0 * torch.tanh(up / 2.0)
    amax = ref.abs().amax(-1, keepdim=True)
    if backend == "lightop":
        # Native LightOp uses scale=1 for zero rows, without Triton's floor.
        expected_scale = torch.where(amax > 0, amax / 127, 1.0)
    else:
        expected_scale = amax.clamp_min(1e-10) / 127
    expected_q = (ref / expected_scale).round().clamp(-127, 127).to(torch.int8)
    assert q.dtype == torch.int8 and scale.dtype == torch.float32
    assert q.shape == (17, dim) and scale.shape == (17, 1)
    torch.testing.assert_close(scale, expected_scale, rtol=2e-5, atol=1e-12)
    # FP32 transcendental approximations can move an exact rounding boundary
    # by one quantization level. Dequantization must stay within one level.
    assert (q.short() - expected_q.short()).abs().max().item() <= 1
    assert torch.all((q.float() * scale - ref).abs() <= expected_scale * 1.01 + 1e-6)
    assert torch.count_nonzero(q[0]) == 0
    assert torch.isfinite(scale).all() and (scale > 0).all()


def test_device_empty_and_noncontiguous():
    if not torch.cuda.is_available():
        pytest.skip("HCU required")
    x = torch.empty((0, 768), device="cuda", dtype=torch.bfloat16)
    q, scale = triton_situ_mul_quant(x, beta=1, linear_beta=2)
    assert q.shape == (0, 384) and scale.shape == (0, 1)
    with pytest.raises(ValueError, match="contiguous"):
        triton_situ_mul_quant(torch.empty((4, 8), device="cuda").T,
                              beta=1, linear_beta=2)
