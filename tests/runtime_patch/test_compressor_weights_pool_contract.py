# SPDX-License-Identifier: Apache-2.0
"""CPU control-flow contracts for the initial worker weights pool."""

import ast
from contextlib import contextmanager
import os
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import pytest
import torch


ROOT = Path(__file__).parents[2]


def load_method(path, name, base, namespace):
    tree = ast.parse((ROOT / path).read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "load_model")
    wrapper = ast.ClassDef(
        name=name, bases=[ast.Name(id="Base", ctx=ast.Load())],
        keywords=[], body=[method], decorator_list=[],
    )
    namespace = {**namespace, "Base": base}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[])), str(ROOT / path), "exec"), namespace)
    return namespace[name]


@pytest.mark.parametrize("with_draft", [False, True])
@pytest.mark.parametrize("dummy", [False, True])
def test_worker_binds_target_and_draft_in_initial_pool(monkeypatch, with_draft, dummy):
    # The extracted method body reads this backend switch through the exec
    # namespace. Keep this non-Kimi contract on the default path regardless of
    # ambient developer or CI settings.
    monkeypatch.setenv("VLLM_KIMI_KDA_PREFILL_BACKEND", "legacy")
    active_pool = None
    pool_entries = []
    allocations = []
    bound = set()
    vllm_config = SimpleNamespace()

    def is_deepseek_v4(config):
        assert config is vllm_config
        return False

    @contextmanager
    def weights_pool():
        nonlocal active_pool
        assert not pool_entries, "must not reopen weights pool after loading"
        pool_entries.append("weights")
        active_pool = "weights"
        try:
            yield
        finally:
            active_pool = None

    def bind(model):
        if id(model) in bound:
            return 0
        assert active_pool == "weights"
        allocations.append((model, active_pool))
        bound.add(id(model))
        return 1

    modules = {
        "vllm_hcu.patch.worker": SimpleNamespace(validate_worker_patches=lambda **kw: None),
        "vllm_hcu.patch.worker.core_fix.patch_deepseek_v4_attention": SimpleNamespace(bind_compressor_nt_weights=bind),
        "vllm_hcu.model_executor.layers.fused_moe.glm53_router_gemv": SimpleNamespace(bind_glm53_router_gates=lambda model: 0),
    }
    for name, contents in modules.items():
        module = ModuleType(name)
        vars(module).update(vars(contents))
        monkeypatch.setitem(sys.modules, name, module)

    class RunnerBase:
        def load_model(self, load_dummy_weights=False):
            assert active_pool == "weights"
            assert load_dummy_weights is dummy
            self.model = torch.nn.Module()
            self.draft = torch.nn.Module() if with_draft else None

        def get_draft_model(self):
            return self.draft

    RunnerBase.vllm_config = vllm_config
    runner_cls = load_method(
        "vllm_hcu/v1/hcu_model_runner_v2.py",
        "HcuGPUModelRunnerV2",
        RunnerBase,
        {"is_deepseek_v4": is_deepseek_v4},
    )

    class WorkerBase:
        def load_model(self, *, load_dummy_weights=False):
            with weights_pool():
                self.model_runner.load_model(load_dummy_weights=load_dummy_weights)
            # Match upstream's return after the pool context exits.
            assert active_pool is None

    worker_cls = load_method(
        "vllm_hcu/v1/worker.py", "HcuGPUWorker", WorkerBase,
        # The extracted body does not inherit module-level imports, so provide
        # every module-level dependency it reads directly.
        {"runner_logger": SimpleNamespace(info=lambda *args: None), "os": os},
    )
    worker = worker_cls()
    worker.model_runner = runner_cls()
    worker.load_model(load_dummy_weights=dummy)
    expected = [worker.model_runner.model]
    if with_draft:
        expected.append(worker.model_runner.draft)
    assert allocations == [(model, "weights") for model in expected]
    assert worker.model_runner._vllm_hcu_num_compressor_weights == len(expected)
    assert pool_entries == ["weights"]
