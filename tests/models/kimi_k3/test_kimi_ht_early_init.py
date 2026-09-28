"""The diagnostic early HT initialization is opt-in and Kimi-specific."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS

import pytest


@pytest.mark.parametrize('master,leaf,quant,backend,expected', [
    (True, '1', 'kimi_k3_w4a8', 'deepep_high_throughput', 1),
    (False, '1', 'kimi_k3_w4a8', 'deepep_high_throughput', 0),
    (True, '0', 'kimi_k3_w4a8', 'deepep_high_throughput', 0),
    (True, '1', 'other', 'deepep_high_throughput', 0),
    (True, '1', 'kimi_k3_w4a8', 'deepep_low_latency', 0),
])
def test_early_ht_init_gate(monkeypatch, master, leaf, quant, backend, expected):
    monkeypatch.setenv('VLLM_HCU_KIMI_EARLY_HT_INIT', leaf)
    path = Path(__file__).resolve().parents[3] / 'vllm_hcu/v1/hcu_model_runner.py'
    tree = ast.parse(path.read_text())
    assert any(isinstance(n, ast.ImportFrom) and n.module == 'vllm.distributed'
               and any(a.name == 'get_ep_group' for a in n.names) for n in tree.body)
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                  and n.name == '_preload_kimi_ht_communication')
    calls = []
    manager = NS(get_handle=lambda kwargs: calls.append(kwargs))
    context = dict(henvs=NS(VLLM_HCU_USE_CUSTOM_OPS=master),
        get_ep_group=lambda: NS(device_communicator=NS(all2all_manager=manager)),
        logger=NS(info=lambda *a: None))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
                 str(path), 'exec'), context)
    runner = NS(model_config=NS(quantization=quant), parallel_config=NS(
        enable_expert_parallel=True, all2all_backend=backend))
    context['_preload_kimi_ht_communication'](runner)
    assert calls == ([{}] if expected else [])
