# SPDX-License-Identifier: Apache-2.0
"""Plan creation stays in an adequately sized unsplit startup profile."""
import ast
from pathlib import Path
import sys
import threading
from types import ModuleType, SimpleNamespace

import pytest


class Tensor:
    def __init__(self, rows, cols):
        self.shape = (rows, cols)
        self.device = SimpleNamespace(index=0)

    def __getitem__(self, item):
        assert isinstance(item, slice) and item.start is None
        assert item.stop <= self.shape[0]
        return Tensor(item.stop, self.shape[1])


@pytest.fixture
def runtime(monkeypatch):
    path = Path(__file__).parents[2] / 'vllm_hcu/ops/step5/fp8_dense_tuned.py'
    tree = ast.parse(path.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'tuned_fp8_gemm')
    phase = SimpleNamespace(dbo=False, capture=False, architecture='gfx938', custom=True)
    module = ModuleType('vllm.v1.worker.ubatching')
    module.dbo_enabled = lambda: phase.dbo
    monkeypatch.setitem(sys.modules, module.__name__, module)
    calls = []

    def op(x, xs, w, ws, selected, create):
        assert xs.shape[0] == x.shape[0]
        calls.append((x.shape[0], selected, create))
        return 'output'

    scope = {
        'henvs': SimpleNamespace(custom_ops_enabled=lambda: phase.custom),
        'torch': SimpleNamespace(cuda=SimpleNamespace(
            get_device_properties=lambda _: SimpleNamespace(gcnArchName=phase.architecture),
            is_current_stream_capturing=lambda: phase.capture)),
        '_BY_NK': {(14080, 4096): [(1536, 1), (4096, 2), (8192, 3)]},
        '_CHOICES': {(m, 14080, 4096): i for m, i in [(1536, 1), (4096, 2), (8192, 3)]},
        '_UNSPLIT_PROFILE_ROWS': 8192, '_warmed': set(), '_unsupported_devices': set(),
        '_lock': threading.RLock(), '_native_gemm': lambda: op,
        'logger': SimpleNamespace(info=lambda *a, **k: None),
    }
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), str(path), 'exec'), scope)

    def run(rows):
        return scope['tuned_fp8_gemm'](Tensor(rows, 4096), Tensor(rows, 1), Tensor(14080, 4096), Tensor(14080, 1))

    return run, scope, phase, calls


@pytest.mark.parametrize('rows', [8192, 16384, 32768])
def test_larger_startup_prepares_every_tuned_shape_before_serving(runtime, rows):
    run, scope, phase, calls = runtime
    result = run(rows)
    assert calls[:3] == [(1536, 1, True), (4096, 2, True), (8192, 3, True)]
    assert scope['_warmed'] == {(0, 14080, 4096)}
    assert result == ('output' if rows == 8192 else None)
    calls.clear();phase.dbo = True
    assert run(4096) == 'output'
    assert calls == [(4096, 2, False)]


@pytest.mark.parametrize('rows', [1536, 4096])
def test_serving_cannot_lazily_allocate_unprepared_plans(runtime, rows):
    run, _, _, calls = runtime
    with pytest.raises(RuntimeError, match='at least 8192'):
        run(rows)
    assert not calls


@pytest.mark.parametrize('mode', ['dbo', 'capture'])
def test_even_a_large_batch_cannot_initialize_in_dbo_or_graphs(runtime, mode):
    run, _, phase, calls = runtime
    setattr(phase, mode, True)
    with pytest.raises(RuntimeError, match='before DBO/graphs'):
        run(16384)
    assert not calls


def test_profile_must_cover_the_largest_external_recipe(runtime):
    run, scope, _, calls = runtime
    scope['_BY_NK'][(14080, 4096)].append((16384, 4))
    scope['_CHOICES'][(16384, 14080, 4096)] = 4
    with pytest.raises(RuntimeError, match='at least 16384'):
        run(8192)
    assert not calls
    assert run(32768) is None
    assert calls == [(1536, 1, True), (4096, 2, True), (8192, 3, True), (16384, 4, True)]


def test_small_untuned_batch_keeps_the_original_fallback(runtime):
    run, scope, _, calls = runtime
    assert run(64) is None
    assert not calls and not scope['_warmed']


@pytest.mark.parametrize('change', ['architecture', 'custom'])
def test_unsupported_devices_and_disabled_custom_ops_keep_the_fallback(runtime, change):
    run, _, phase, calls = runtime
    setattr(phase, change, 'gfx942' if change == 'architecture' else False)
    assert run(16384) is None
    assert not calls
