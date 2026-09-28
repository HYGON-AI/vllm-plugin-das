"""Execute the production preload method without importing the GPU runner."""
import ast
from pathlib import Path
from types import SimpleNamespace

import torch


def test_preload_skips_only_explicit_weight_only_modules(monkeypatch):
    path = Path(__file__).resolve().parents[3] / 'vllm_hcu/v1/hcu_model_runner.py'
    tree = ast.parse(path.read_text())
    method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                  and node.name == '_preload_unquantized_gemm_kernels')
    code = ast.Module(body=[method], type_ignores=[])
    context = dict(torch=torch, henvs=SimpleNamespace(VLLM_USE_NN=False),
                   logger=SimpleNamespace(info=lambda *a: None))
    exec(compile(ast.fix_missing_locations(code), str(path), 'exec'), context)
    model = torch.nn.ModuleList([torch.nn.Linear(7168, 1, bias=False),
                                 torch.nn.Linear(32, 8, bias=False),
                                 torch.nn.Linear(64, 1, bias=False)]).to(torch.bfloat16)
    for module in model:
        module.quant_method = object()
    model[0]._hcu_weight_only_linear = True
    called = []
    monkeypatch.setattr(torch.nn.functional, 'linear',
                        lambda x, w: called.append(tuple(w.shape)))
    monkeypatch.setattr(torch.accelerator, 'synchronize', lambda: None)
    monkeypatch.setattr(torch.accelerator, 'empty_cache', lambda: None)
    context['_preload_unquantized_gemm_kernels'](SimpleNamespace(
        model=model, max_num_tokens=1, max_num_reqs=1, device='cpu'))
    assert called == [(8, 32), (1, 64)]
