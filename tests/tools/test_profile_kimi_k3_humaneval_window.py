import importlib.util
import json
import sys
from types import SimpleNamespace
from pathlib import Path


def test_stop_profile_failure_returns_nonzero(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[2] / 'tools/profile_kimi_k3_humaneval_window.py'
    spec = importlib.util.spec_from_file_location('profile_window', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setitem(sys.modules, 'datasets', SimpleNamespace(load_from_disk=lambda _: [dict(task_id='x/0', prompt='x')]))
    monkeypatch.setattr(sys, 'argv', ['profile', '--api-base', 'http://test', '--dataset', 'x', '--output', str(tmp_path), '--concurrency', '1', '--delay-seconds', '0'])
    def request(base, path, payload=None, timeout=900):
        if path == '/v1/models':
            return {'data': [{'id': 'model'}]}
        if path == '/stop_profile':
            raise RuntimeError('export failed')
        return {'choices': [{'finish_reason': 'stop'}]}
    monkeypatch.setattr(module, 'request', request)
    assert module.main() == 1
    assert 'export failed' in json.loads((tmp_path / 'trace_requests.json').read_text())['profile_stop']['error']
