# SPDX-License-Identifier: Apache-2.0
"""Masked activation qualification and runtime fallback, without a GPU."""
from contextlib import nullcontext
from types import SimpleNamespace as NS

import pytest
import torch

from hcu_vllm_das.moe import clamped_int8_admission as admission


@pytest.fixture
def qualification(monkeypatch):
    state = NS(capturing=False, probes=[], matches=True)
    monkeypatch.setattr(admission, '_QUALIFIED_LAUNCHES', {})
    monkeypatch.setattr(torch.cuda, 'device', lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, 'is_current_stream_capturing',
                        lambda: state.capturing)

    def probe(*args):
        state.probes.append(args)
        return state.matches

    monkeypatch.setattr(admission, '_matches_native', probe)

    def bind(experts=32, device='cuda:0', native=object()):
        monkeypatch.setattr(admission, '_native_provider', lambda *a, **kw: native)
        return admission.masked_qualifier(None, (NS(device=torch.device(device)),),
                                          NS(shape=(experts, 4608, 1)), None, 10)

    state.bind = bind
    return state


def test_each_actual_extent_and_hint_requires_its_own_probe(qualification):
    qualify = qualification.bind()
    for rows, hint in [(1024, 16), (1024, 80), (3072, 80), (1024, 1024)]:
        assert qualify(rows, hint)
        assert qualify(rows, hint)
    assert [(p[2], p[3], p[4], p[5]) for p in qualification.probes] == [
        (32, True, 1024, 16), (32, True, 1024, 80),
        (32, True, 3072, 80), (32, True, 1024, 1024)]


def test_cache_separates_provider_device_and_expert_count(qualification):
    qualifiers = [qualification.bind(), qualification.bind(experts=96),
                  qualification.bind(device='cuda:1'),
                  qualification.bind(native=object())]
    for qualify in qualifiers:
        assert qualify(1024, 80)
        assert qualify(1024, 80)
    assert len(qualification.probes) == 4


def test_failed_geometry_does_not_admit_another_hint(qualification):
    qualify = qualification.bind()
    qualification.matches = False
    assert not qualify(1024, 80)
    qualification.matches = True
    assert not qualify(1024, 80)
    assert qualify(1024, 96)
    assert len(qualification.probes) == 2


def test_capture_miss_can_qualify_in_later_eager_warmup(qualification, monkeypatch):
    qualify = qualification.bind()
    qualification.capturing = True
    assert not qualify(1024, 80)
    assert not qualification.probes
    qualification.capturing = False
    assert qualify(1024, 80)
    qualification.capturing = True
    assert not qualify(1024, 96)
    assert len(qualification.probes) == 1

    def no_cuda_query(*args):
        pytest.fail('cached qualification must not query CUDA during capture')

    monkeypatch.setattr(torch.cuda, 'device', no_cuda_query)
    assert qualify(1024, 80)


@pytest.mark.parametrize('rows,hint', [(None, 80), (0, 80), (-1, 80),
    (True, 80), (1024, None), (1024, 0), (1024, -1), (1024, True)])
def test_unknown_or_invalid_geometry_cannot_be_admitted(qualification, rows, hint):
    assert not qualification.bind()(rows, hint)
    assert not qualification.probes


def test_unsupported_provider_has_no_qualifier(monkeypatch):
    monkeypatch.setattr(admission, '_native_provider', lambda *a, **kw: None)
    assert admission.masked_qualifier(None, (), None, None, 10) is None


@pytest.mark.parametrize('rows,hint', [(7, 3), (1024, 80)])
@pytest.mark.parametrize('corrupt', [None, 'quantized', 'scale'])
def test_probe_checks_actual_stride_hint_and_last_live_row(monkeypatch, rows, hint, corrupt):
    from hcu_vllm_das.moe import clamped_int8_triton

    monkeypatch.setattr(torch.cuda, 'device', lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, 'is_current_stream_capturing', lambda: False)
    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda device: (1 << 40, 1 << 40))
    calls = []

    def outputs(value):
        return (torch.zeros(*value.shape[:-1], 2304, dtype=torch.int8),
                torch.ones(*value.shape[:-1], 1, dtype=torch.float32))

    def native(value, *, limit, mask_m, expect_m):
        assert value.shape == (2, rows, 4608)
        assert limit == 10. and expect_m == hint
        assert mask_m.tolist() == [0, rows]
        assert torch.count_nonzero(value[:, -1]) > 0
        calls.append('native')
        return outputs(value)

    def candidate(value, meta, *, masked, expected_m):
        assert masked and expected_m == hint and meta.tolist() == [0, rows]
        calls.append('candidate')
        quantized, scales = outputs(value)
        if corrupt == 'quantized':
            quantized[-1, -1, -1] = 1
        elif corrupt == 'scale':
            scales[-1, -1, 0] = 2.
        return quantized, scales

    monkeypatch.setattr(clamped_int8_triton, 'clamped_int8_2304', candidate)
    assert admission._matches_native(native, torch.device('cpu'), 2, True,
                                     rows, hint) is (corrupt is None)
    assert calls == ['native', 'candidate']


@pytest.mark.parametrize('rows,small_hint,local_tokens,want_hint', [
    (17, False, 1, 17), (96, True, 700, 80)])
@pytest.mark.parametrize('qualified', [False, True])
def test_actual_expert_apply_qualifies_before_replacing_native(
        monkeypatch, rows, small_hint, local_tokens, want_hint, qualified):
    from hcu_vllm_das.moe import clamped_int8_triton, ct_w4a8_runtime as runtime
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    experts, dispatchers = 32, 4 if small_hint else 1
    counts = torch.full((experts,), rows, dtype=torch.int32)
    calls = []

    def gemm(inputs, weights, output, count_arg, hint, use_hint):
        assert count_arg is counts
        calls.append(('gemm', hint, use_hint))
        output.zero_()

    def qualify(actual_rows, hint):
        calls.append(('qualify', actual_rows, hint))
        return qualified

    def outputs():
        return torch.zeros(experts, rows, 2, dtype=torch.int8), torch.ones(experts, rows, 1)

    def native(value, *, limit, mask_m, expect_m):
        assert mask_m is counts and limit == 10.
        calls.append(('native', value.shape[1], expect_m))
        return outputs()

    def candidate(value, metadata, *, masked, expected_m):
        assert metadata is counts and masked
        calls.append(('optimized', value.shape[1], expected_m))
        return outputs()

    monkeypatch.setattr(clamped_int8_triton, 'clamped_int8_2304', candidate)
    instance = NS(max_num_tokens=rows // dispatchers, num_dispatchers=dispatchers,
        weights=('w1', 'w2'), w1_scale=torch.ones(experts, 4, 1),
        w2_scale=torch.ones(experts, 2, 1), quant_config=NS(gemm1_clamp_limit=10.),
        ops=NS(clamp_quant_ep=native), _small_batch_hint=small_hint,
        _live_row_activation=True, _qualify_live_activation=qualify,
        _check=lambda *a: None, moe_problem_size=lambda *a: (experts, rows, 4, 2, 3),
        expected_m=lambda *a: 13, _masked_gemm=gemm)
    runtime.HcuCTMaskedExperts.apply(instance, torch.empty(experts, rows, 2),
        torch.zeros(experts, rows, 2, dtype=torch.int8), None, None,
        torch.ones(local_tokens, 3), torch.zeros(local_tokens, 3, dtype=torch.int64),
        MoEActivation.SILU, 128, None, torch.ones(experts, rows, 1), None,
        torch.empty(experts, rows, 4), torch.empty(experts, rows, 2),
        NS(expert_num_tokens=counts), False)
    gemm_hint = want_hint if small_hint else 13
    assert calls == [('gemm', gemm_hint, small_hint), ('qualify', rows, want_hint),
                     ('optimized' if qualified else 'native', rows, want_hint),
                     ('gemm', gemm_hint, small_hint)]
