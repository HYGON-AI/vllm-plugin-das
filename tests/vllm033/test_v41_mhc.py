"""CPU interpreter checks for the HCU V4.1 PCP mHC seams against Core Torch."""
from importlib import import_module

import pytest
import torch
from vllm.model_executor.kernels.mhc.torch import mhc_post_torch
from vllm.model_executor.layers import mhc as native_mhc
from hcu_vllm_das import v41_mhc
from hcu_vllm_das.models import v41_pcp as model

# The package re-exports the triton library name over this submodule attribute.
core_triton = import_module("vllm.model_executor.kernels.mhc.triton")


@pytest.fixture(autouse=True)
def interpret_triton_on_cpu(monkeypatch):
    from triton.runtime.interpreter import InterpretedFunction
    from triton.runtime.jit import JITFunction
    # tl.sum and its nested helpers were imported before this fixture.
    monkeypatch.setattr(JITFunction, "__call__",
        lambda self, *args, **kwargs: InterpretedFunction(self.fn)(*args, **kwargs))
    monkeypatch.setattr(v41_mhc, "_mhc_post_kernel",
                        InterpretedFunction(v41_mhc._mhc_post_kernel.fn))
    monkeypatch.setattr(core_triton, "_hc_head_reduce_store_kernel",
                        InterpretedFunction(core_triton._hc_head_reduce_store_kernel.fn))
    yield
    assert not torch.cuda.is_initialized()


def _post_inputs(tokens, hc, hidden, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(tokens, hidden, generator=g).bfloat16()
    residual = torch.randn(tokens, hc, hidden, generator=g).bfloat16()
    post = torch.sigmoid(torch.randn(tokens, hc, 1, generator=g)) * 2
    comb = torch.softmax(torch.randn(tokens, hc, hc, generator=g), -1)
    return x, residual, post, comb


@pytest.mark.parametrize("tokens,hc,hidden", [(0, 4, 64), (3, 4, 1100), (2, 3, 40)])
def test_post_matches_core_torch_within_bf16_rounding(tokens, hc, hidden):
    args = _post_inputs(tokens, hc, hidden, tokens + hidden)
    actual = model.HcuV41PcpMHCPostOp.forward_hip(None, *args)
    expected = mhc_post_torch(*args)
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    torch.testing.assert_close(actual.float(), expected.float(), rtol=2 ** -7, atol=1e-6)


def test_post_accepts_strided_operands():
    x, residual, post, comb = _post_inputs(2, 4, 80, 7)
    wide = torch.zeros(2, 4, 160, dtype=residual.dtype)
    wide[..., ::2] = residual
    comb_t = comb.transpose(1, 2).contiguous().transpose(1, 2)
    torch.testing.assert_close(
        v41_mhc.mhc_post(x, wide[..., ::2], post, comb_t).float(),
        mhc_post_torch(x, residual, post, comb).float(), rtol=2 ** -7, atol=1e-6)


def test_post_rejects_mismatched_shapes():
    x, residual, post, comb = _post_inputs(2, 4, 16, 1)
    with pytest.raises(ValueError):
        v41_mhc.mhc_post(x[:1], residual, post, comb)


@pytest.mark.parametrize("with_pre_mix", [False, True])
@pytest.mark.parametrize("tokens", [0, 3])
def test_pre_delayed_reuses_core_gates_and_collapse(tokens, with_pre_mix):
    g = torch.Generator().manual_seed(tokens)
    hc, hidden = 4, 64
    residual = torch.randn(tokens, hc, hidden, generator=g).bfloat16()
    fn = torch.randn(2 * hc + hc * hc, hc * hidden, generator=g) * 0.05
    scale, base = torch.rand(3, generator=g), torch.randn(2 * hc + hc * hc, generator=g)
    pre_mix = torch.sigmoid(torch.randn(tokens, hc, generator=g)) if with_pre_mix else None
    args = (residual, fn, scale, base, 1e-6, 1e-6, 1e-6, 2.0, 3)
    op = object.__new__(model.HcuV41PcpMHCPreDelayedOp)
    actual = model.HcuV41PcpMHCPreDelayedOp.forward_hip(op, *args, pre_mix=pre_mix)
    expected = native_mhc.MHCPreDelayedOp.forward_native(op, *args, pre_mix=pre_mix)
    for index in (0, 1, 2, 4):
        assert torch.equal(actual[index], expected[index])
    # Triton and Torch may round the FP32 collapse sum to adjacent BF16 values.
    torch.testing.assert_close(actual[3].float(), expected[3].float(), rtol=2 ** -7, atol=1e-6)
