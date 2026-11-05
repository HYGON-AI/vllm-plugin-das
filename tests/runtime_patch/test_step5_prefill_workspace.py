# SPDX-License-Identifier: Apache-2.0
"""Score storage is preaccounted, bounded, contiguous and stream ordered."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).parents[2]
spec = importlib.util.spec_from_file_location(
    'tested_prefill_workspace', ROOT / 'vllm_hcu/ops/step5/prefill_workspace.py')
workspace = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workspace)


def test_shapes_reuse_one_contiguous_allocation_without_resizing():
    arena = workspace.PrefillScoreWorkspace(16, 12, 'cpu')
    ptr = arena.buffer.data_ptr()
    for rows, regions in [(16, 12), (8, 7), (3, 11), (16, 1)]:
        view = arena.get(rows, regions, device='cpu')
        assert view.shape == (rows, regions) and view.is_contiguous()
        assert view.stride(0) == regions and view.data_ptr() == ptr
    assert arena.buffer.numel() == 192
    assert arena.get(0, 4).shape == (0, 4)


def test_dense_equivalence_zeroes_only_the_requested_view():
    arena = workspace.PrefillScoreWorkspace(8, 8, 'cpu')
    arena.buffer.fill_(37)
    assert torch.equal(arena.get(3, 5, zero=True), torch.zeros(3, 5))
    assert torch.equal(arena.buffer[15:], torch.full((49,), 37.0))


@pytest.mark.parametrize('shape', [(17, 1), (1, 13), (-1, 3), (1, 0)])
def test_out_of_bounds_cannot_grow_the_reservation(shape):
    arena = workspace.PrefillScoreWorkspace(16, 12, 'cpu')
    with pytest.raises(ValueError, match='startup reservation'):
        arena.get(*shape)


@pytest.mark.parametrize('shape', [(0, 4), (4, 0), (-1, 4)])
def test_invalid_capacity_is_rejected(shape):
    with pytest.raises(ValueError, match='positive'):
        workspace.PrefillScoreWorkspace(*shape, 'cpu')


def test_cross_device_use_is_rejected_before_access():
    arena = workspace.PrefillScoreWorkspace(4, 4, 'cpu')
    with pytest.raises(ValueError, match='another device'):
        arena.get(1, 1, device='cuda:0')


def test_stream_change_waits_for_prior_readers_and_capture_is_rejected(monkeypatch):
    arena = workspace.PrefillScoreWorkspace(4, 4, 'cpu')
    cpu = arena.buffer

    class Buffer:
        device = torch.device('cuda:0')
        is_cuda = True

        def __getitem__(self, key):
            return cpu[key]

    calls = []
    first = SimpleNamespace()
    second = SimpleNamespace(wait_event=lambda event: calls.append(('wait', event)))
    active = [first]

    class Event:
        def record(self, stream):
            calls.append(('record', stream))

    arena.buffer = Buffer()
    monkeypatch.setattr(torch.cuda, 'is_current_stream_capturing', lambda: False)
    monkeypatch.setattr(torch.cuda, 'current_stream', lambda device: active[0])
    monkeypatch.setattr(torch.cuda, 'Event', Event)
    arena.get(2, 3);arena.get(3, 2)
    assert calls == []
    active[0] = second;arena.get(2, 2)
    assert calls == [('record', first), ('wait', arena._fence)]
    monkeypatch.setattr(torch.cuda, 'is_current_stream_capturing', lambda: True)
    with pytest.raises(RuntimeError, match='eager'):
        arena.get(1, 1)


@pytest.mark.parametrize('case', ['producer', 'consumer', 'small', 'compiled', 'other_model', 'no_groups', 'cpu'])
def test_reservation_scope_is_only_large_eager_pd_producers(monkeypatch, case):
    monkeypatch.setattr(workspace, 'is_step5_pd_config', lambda config: case != 'other_model')
    state = SimpleNamespace(num_kv_groups=0 if case == 'no_groups' else 4,
                            geometry=SimpleNamespace(region_size=8))
    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(kv_role='kv_consumer' if case == 'consumer' else 'kv_producer'),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8192 if case == 'small' else 16384),
        model_config=SimpleNamespace(enforce_eager=case != 'compiled', max_model_len=140000),
        device_config=SimpleNamespace(device='cpu' if case == 'cpu' else 'cuda:0'),
    )
    calls = [];marker = object()
    monkeypatch.setattr(workspace, 'PrefillScoreWorkspace', lambda *args: calls.append(args) or marker)
    result = workspace.create_pd_prefill_workspace(state, config)
    assert result is (marker if case == 'producer' else None)
    assert calls == ([(65536, 17500, 'cuda:0')] if case == 'producer' else [])


@pytest.mark.parametrize('dense_equiv', [False, True])
def test_real_scorer_uses_reserved_storage_and_preserves_zero_fill(dense_equiv):
    path = ROOT / 'vllm_hcu/model_executor/layers/step4_dsa.py'
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Step4Indexer')
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_score')
    scope = {'torch': torch, 'PREFILL_FP8_INDEXER': True, 'DENSE_EQUIV': dense_equiv, 'INDEXER_Q_CHUNK': 0}
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, fn], type_ignores=[])), str(path), 'exec'), scope)
    arena = workspace.PrefillScoreWorkspace(64, 16, 'cpu');arena.buffer.fill_(37)
    state = SimpleNamespace(summary=torch.empty(1, 8, 4, 128), _pd_prefill_workspace=arena)
    indexer = SimpleNamespace(state=state, num_kv_groups=4, slot=0, _prescaled=lambda x:x)
    md = SimpleNamespace(num_decodes=0, num_decode_tokens=0, num_actual_tokens=8, num_prefills=1)
    sched = SimpleNamespace(spec=False, decode=False, requests=[])
    result, decode, width = scope['_score'](indexer, md, sched, torch.empty(8, 4, 128), torch.empty(8, 4))
    assert result.data_ptr() == arena.buffer.data_ptr() and result.shape == (32, 1)
    assert decode is None and width == 0
    assert torch.equal(result, torch.full((32, 1), 0.0 if dense_equiv else 37.0))
