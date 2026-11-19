"""Admission and BF16 output layout for the opt-in WO-A candidate."""

from contextlib import nullcontext
import os
from types import SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np
import pytest

from hcu_vllm_das import wo_a_token_major as wo


NONE, FULL, PIECEWISE, EAGER = (object() for _ in range(4))


class State:
    model_config = NS(enforce_eager=True)
    vllm_config = NS(compilation_config=NS(mode=EAGER, cudagraph_mode=NONE))


class SWA:
    num_decodes = 2
    num_decode_tokens = 6
    num_prefills = num_prefill_tokens = 0


def _batch():
    return NS(num_tokens=6, num_reqs=2, num_tokens_after_padding=6,
              num_reqs_after_padding=2, pcp=None,
              is_prefilling_np=np.array([False, False]), prefill_len_np=np.array([2, 3]),
              num_computed_prefill_tokens_np=np.array([2, 3]))


@pytest.mark.parametrize(('supported', 'override', 'expected'), [
    (True, None, True), (False, None, False), (True, '0', False), (False, '1', True),
])
def test_startup_policy(monkeypatch, supported, override, expected):
    monkeypatch.delenv(wo._SWITCH, raising=False)
    if override is not None:
        monkeypatch.setenv(wo._SWITCH, override)
    monkeypatch.setattr(wo, '_supported_default', lambda: supported)
    monkeypatch.setattr(wo, '_publication', None)
    publish = Mock()
    monkeypatch.setattr(wo, '_publish', publish)
    spec = wo.extension_spec()
    assert (wo._CAPABILITY in spec.capabilities) is expected
    spec.activate(NS(vllm_version='0.33.0'))
    assert publish.call_count == int(expected)


@pytest.mark.parametrize(('pinned', 'arch', 'expected'), [
    (True, 'gfx938', True), (True, 'gfx936', False), (False, 'gfx938', False),
])
def test_default_stack(monkeypatch, pinned, arch, expected):
    from hcu_adapter.bridges import vllm

    monkeypatch.setattr(vllm, '_installed_vllm_version', lambda: (
        '0.33.0+dtk2610.torch2130.2611101131.g7364da' if pinned else '0.33.0'))
    monkeypatch.setattr(vllm, 'detect_hcu', lambda: NS(detected=True, architecture=arch))
    assert wo._supported_default.__wrapped__() is expected


@pytest.mark.parametrize('scenario', [
    'decode', 'short extend', 'mixed', 'dummy', 'missing phase', 'incomplete', 'padded',
    'padded requests', '48-row verify',
    'full', 'piecewise', 'capture', 'not enforce eager', 'compile',
    'dbatch', 'unsupported state', 'pcp',
])
def test_phase_is_independent_of_swa_counters(scenario):
    batch, state, swa = _batch(), State(), SWA()
    mode, capture, ubatch = NONE, False, 0
    # Every scenario has the same SWA counters, including short extends.
    if scenario == 'short extend':
        batch.is_prefilling_np[:] = True
    elif scenario == 'mixed':
        batch.is_prefilling_np[1] = True
    elif scenario == 'dummy':
        batch.prefill_len_np[:] = 0
    elif scenario == 'missing phase':
        batch.is_prefilling_np = None
    elif scenario == 'incomplete':
        batch.num_computed_prefill_tokens_np[:] = [1, 3]
    elif scenario == 'padded':
        batch.num_tokens_after_padding = 8
    elif scenario == 'padded requests':
        batch.num_reqs_after_padding = 4
    elif scenario == '48-row verify':
        batch.num_tokens = batch.num_tokens_after_padding = swa.num_decode_tokens = 48
        batch.num_reqs = batch.num_reqs_after_padding = swa.num_decodes = 8
        batch.is_prefilling_np = np.zeros(8, dtype=np.bool_)
        batch.prefill_len_np = batch.num_computed_prefill_tokens_np = np.full(8, 4096)
    elif scenario == 'full':
        mode = FULL
    elif scenario == 'piecewise':
        mode = PIECEWISE
    elif scenario == 'capture':
        capture = True
    elif scenario == 'not enforce eager':
        state.model_config = NS(enforce_eager=False)
    elif scenario == 'compile':
        state.vllm_config = NS(compilation_config=NS(mode=object(), cudagraph_mode=NONE))
    elif scenario == 'dbatch':
        ubatch = 1
    elif scenario == 'unsupported state':
        state = type('AnotherState', (State,), {})()
    elif scenario == 'pcp':
        batch.pcp = object()
    mapping = {'swa': swa}
    original = Mock(return_value=mapping)
    prepare = wo._wrap_prepare(original, State, SWA, EAGER, NONE, np)
    assert prepare(state, batch, mode, None, None, None, None, capture, ubatch) is mapping
    assert getattr(swa, wo._PHASE_ROWS, None) == (6 if scenario == 'decode' else None)
    original.assert_called_once()


class Tensor:
    ndim = 3
    is_cuda = True
    dtype = 'bf16'
    device = NS(index=0)

    def __init__(self, shape, contiguous=True):
        self.shape = shape
        self.contiguous = contiguous

    def is_contiguous(self):
        return self.contiguous

    def view(self, *shape):
        assert shape == (6, 8, 4096)
        return self

    def transpose(self, *_):
        return self

    def flatten(self, dim):
        assert dim == 1 and self.shape == (6, 8, 1024)
        return 'token-major WO-B input'


class Attention:
    n_local_groups = 8
    o_lora_rank = 1024
    n_local_heads = 64
    swa_cache_layer = NS(prefix='swa')
    wo_a = object()

    def _wo_b_after_wo_a(self, flattened):
        return flattened


class CommonAttention:
    pass


class Fp4Attention(Attention, CommonAttention):
    """Mirror the owned target's multiple inheritance without importing vLLM."""


def test_fp4_inherits_native_projection_contract(monkeypatch):
    for method in ('forward', '_alloc_attn_out', 'forward_mqa', '_o_proj'):
        monkeypatch.setattr(Attention, method, lambda self: None, raising=False)
    monkeypatch.setattr(Attention, 'get_padded_num_q_heads',
                        classmethod(lambda cls, heads: heads), raising=False)
    wo._check_fp4_inheritance(Attention, Fp4Attention)
    monkeypatch.setattr(Fp4Attention, 'forward_mqa', lambda self: None)
    with pytest.raises(RuntimeError, match='FP4 attention inheritance changed'):
        wo._check_fp4_inheritance(Attention, Fp4Attention)


@pytest.mark.parametrize('scenario', [
    'eligible', 'eligible base', 'foreign subclass', 'fp4 subclass',
    'full', 'piecewise', 'unmarked', 'different rows', 'capture',
    'compile', 'quantized', 'noncontiguous', 'noncontiguous weight', 'other device',
    'too many rows', 'one row', 'missing context',
])
def test_exact_route_and_wo_b_dispatch(scenario):
    out = Tensor((6, 64, 512), contiguous=scenario != 'noncontiguous')
    swa = SWA()
    if scenario != 'unmarked':
        setattr(swa, wo._PHASE_ROWS, 8 if scenario == 'different rows' else 6)
    mode = {'full': FULL, 'piecewise': PIECEWISE}.get(scenario, NONE)
    context = NS(cudagraph_runtime_mode=mode, ubatch_slices=None, attn_metadata={'swa': swa})
    if scenario == 'quantized':
        out = NS(data=out, scale=object())
    elif scenario == 'too many rows':
        out.shape = (33, 64, 512)
    elif scenario == 'one row':
        out.shape = (1, 64, 512)
    weight = Tensor((8, 1024, 4096), contiguous=scenario != 'noncontiguous weight')
    if scenario == 'other device':
        weight.device = NS(index=1)
    torch = NS(Tensor=Tensor, bfloat16='bf16',
               compiler=NS(is_compiling=lambda: scenario == 'compile'),
               cuda=NS(current_device=lambda: 0,
                       is_current_stream_capturing=lambda: scenario == 'capture'),
               empty=lambda shape, **_: Tensor(shape), no_grad=nullcontext, bmm=Mock())
    original = Mock(return_value='stock')
    project = wo._wrap_o_proj(original, (Attention, Fp4Attention), SWA, lambda *_: weight,
                              torch, lambda: context,
                              lambda: scenario != 'missing context', NONE)
    attention_cls = {
        'eligible base': Attention,
        'foreign subclass': type('OtherAttention', (Attention,), {}),
        'fp4 subclass': type('DerivedFp4Attention', (Fp4Attention,), {}),
    }.get(scenario, Fp4Attention)
    o_proj = attention_cls()
    result = project(o_proj, out, 'positions')
    if scenario in ('eligible', 'eligible base'):
        assert result == 'token-major WO-B input'
        original.assert_not_called()
        torch.bmm.assert_called_once()
    else:
        assert result == 'stock'
        original.assert_called_once_with(o_proj, out, 'positions')
        torch.bmm.assert_not_called()


@pytest.mark.skipif(os.environ.get('HCU_TEST_WO_A_DEVICE') != '1', reason='opt-in HCU test')
@pytest.mark.parametrize('rows', [2, 6, 16, 32])
def test_hcu_bitwise(rows):
    import torch
    from hcu_adapter.bridges.vllm import detect_hcu

    assert torch.cuda.is_available() and detect_hcu().detected, 'requires an HCU device'
    torch.manual_seed(625914 + rows)
    x = torch.randn(rows, 8, 4096, device='cuda', dtype=torch.bfloat16)
    w = torch.randn(8, 1024, 4096, device='cuda', dtype=torch.bfloat16) / 64
    result = wo._token_major_bmm(torch, x, w)
    expected = torch.einsum('tgd,grd->tgr', x, w)
    assert result.is_contiguous()
    assert result.flatten(1).data_ptr() == result.data_ptr()
    assert torch.equal(result.view(torch.int16), expected.contiguous().view(torch.int16))
