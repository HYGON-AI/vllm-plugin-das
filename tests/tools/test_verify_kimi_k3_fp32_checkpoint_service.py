import json
import runpy
import sys
import urllib.request
from pathlib import Path
import pytest

@pytest.mark.parametrize('case', ['ok', 'no_hit', 'missing', 'divergent'])
def test_checkpoint_cli(case, tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[2] / 'tools/verify_kimi_k3_fp32_checkpoint_service.py'
    monkeypatch.syspath_prepend(str(script.parent))
    monkeypatch.setattr(sys, 'argv', [str(script), 'http://test', str(tmp_path / 'result.json')])
    calls = {'completion': 0, 'metrics': 0}
    class Response:
        def __init__(self, obj): self.obj = obj
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self): return self.obj
    class Opener:
        def open(self, req, timeout):
            path = req.full_url.split('http://test')[1]
            if path == '/metrics':
                calls['metrics'] += 1
                n = 3072 if calls['metrics'] >= 3 and case != 'no_hit' else 0
                return Response(f'vllm:prefix_cache_hits_total{{model="x"}} {n}\n'.encode())
            if path == '/v1/models': return Response(b'{"data":[{"id":"m"}]}')
            if path == '/v1/completions':
                calls['completion'] += 1
                lp = -9.0 if case == 'divergent' and calls['completion'] == 3 else -0.1
                choice = {'token_ids': [1], 'logprobs': {'tokens': ['x'], 'token_logprobs': [lp], 'top_logprobs': [{'x': lp}]}}
                if case == 'missing': choice = {}
                return Response(json.dumps({'choices': [choice]}).encode())
            return Response(b'{}')
    monkeypatch.setattr(urllib.request, 'build_opener', lambda *a: Opener())
    if case == 'ok': runpy.run_path(str(script), run_name='__main__')
    else:
        with pytest.raises(SystemExit) as exc: runpy.run_path(str(script), run_name='__main__')
        assert exc.value.code == 1
