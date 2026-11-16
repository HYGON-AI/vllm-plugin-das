# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for post-load compressor binding and prefetch lifecycles."""

import ast
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


@pytest.fixture
def binding(monkeypatch):
    # Execute the production helpers without importing GPU-dependent adapters.
    path = (
        Path(__file__).parents[2]
        / "vllm_hcu/patch/worker/core_fix/patch_deepseek_v4_attention.py"
    )
    names = {
        "_offloaded_param_ids", "_validate_compressor_backend",
        "bind_compressor_nt_weights",
    }
    tree = ast.parse(path.read_text())
    namespace = {
        "torch": torch,
        "henvs": SimpleNamespace(
            custom_ops_enabled=lambda: True,
            VLLM_HCU_COMPRESSOR_GEMM_BACKEND="blaslt",
        ),
        "_COMPRESSOR_NT_BUFFER": "_vllm_hcu_compressor_nt_weight",
    }
    functions = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"),
        namespace,
    )

    class PrefetchOffloader:
        module_offloaders = []

    offloader = PrefetchOffloader()
    for name in (
        "vllm.model_executor.offloader.base",
        "vllm.model_executor.offloader.prefetch",
    ):
        module = ModuleType(name)
        module.get_offloader = lambda: offloader
        module.PrefetchOffloader = PrefetchOffloader
        monkeypatch.setitem(sys.modules, name, module)
    return SimpleNamespace(**namespace), offloader


@pytest.mark.parametrize("deleted_kind", ["marked", "missing", "property"])
def test_binding_skips_deleted_entries_and_keeps_live_offloads(binding, deleted_kind):
    patch, offloader = binding
    model = torch.nn.Module()
    linear = torch.nn.Linear(5, 3, bias=False, dtype=torch.bfloat16)
    model.fused_wkv_wgate = linear

    class DeletedParam:
        @property
        def _param(self):
            if deleted_kind == "marked":
                pytest.fail("marked deleted parameters must not be accessed")
            raise AttributeError("transient parameter removed after loading")

    deleted = (
        SimpleNamespace() if deleted_kind == "missing" else DeletedParam()
    )
    if deleted_kind == "marked":
        deleted._param_deleted = True
    offloader.module_offloaders = [
        SimpleNamespace(),
        SimpleNamespace(_param_offloaders={
            "unrelated.transient": deleted,
            "fused_wkv_wgate.weight": SimpleNamespace(_param=linear.weight),
        }),
    ]

    assert patch._offloaded_param_ids() == {id(linear.weight)}
    assert patch.bind_compressor_nt_weights(model) == 0
    assert not hasattr(linear, patch._COMPRESSOR_NT_BUFFER)
