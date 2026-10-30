# SPDX-License-Identifier: Apache-2.0
"""CPU-only regressions for the ownership of Step5 runner DBO hooks.

Execute the actual worker-enable and runner-install bodies without importing
vLLM, torch, or HCU extensions. Only patch dependencies and imported runtime
modules are doubled; model and master-switch predicates use their real bodies.
"""
from __future__ import annotations

import ast
import contextlib
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest


ROOT = Path(__file__).resolve().parents[2]
MISSING = object()


def _load_functions(path, names, namespace, *, assignments=()):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    nodes = [
        node for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name in names)
        or (isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in assignments
            for target in node.targets
        ))
    ]
    assert {node.name for node in nodes if isinstance(node, ast.FunctionDef)} == set(names)
    module = ast.Module(body=[
        ast.ImportFrom("__future__", [ast.alias("annotations")], 0), *nodes,
    ], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), path, "exec"), namespace)
    return namespace


def _runtime(monkeypatch, *, threshold=4096, master=True):
    monkeypatch.setenv("HCU_DBO_MIN_TOKENS", str(threshold))
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", str(int(master)))
    env = _load_functions("vllm_hcu/platforms/envs.py", {
        "_environment_flag", "custom_ops_enabled", "optional_custom_op_enabled",
    }, {"os": os})
    model = _load_functions("vllm_hcu/runtime_compat/step5.py", {
        "is_step5_config",
    }, {})

    # Both successful installs and regressions reach valid import boundaries,
    # so a rejected runner fails on unwanted mutation, never on missing vLLM.
    modules = {}
    for name in (
        "vllm", "vllm.v1", "vllm.v1.worker", "vllm.v1.worker.gpu",
        "vllm.v1.worker.gpu.ubatch_utils", "vllm.v1.worker.workspace",
        "vllm.v1.worker.ubatching", "vllm.utils", "vllm.utils.torch_utils",
    ):
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
        modules[name] = module
        if "." in name:
            parent, child = name.rsplit(".", 1)
            setattr(modules[parent], child, module)
    uu = modules["vllm.v1.worker.gpu.ubatch_utils"]
    ws = modules["vllm.v1.worker.workspace"]
    uu.create_sm_control_context = Mock(name="native_sm_control")
    ws.dbo_current_ubatch_id = Mock(name="native_ubatch_id")
    modules["vllm.utils.torch_utils"].current_stream = Mock(name="current_stream")
    modules["vllm.v1.worker.ubatching"].dbo_enabled = lambda: False
    modules["vllm.v1.worker.ubatching"].dbo_current_ubatch_id = lambda: 0

    dsa = ModuleType("vllm_hcu.model_executor.layers.step4_dsa")

    class DSAState:
        def schedule(self, metadata):
            return metadata

    dsa.Step4DSAState = DSAState
    monkeypatch.setitem(sys.modules, dsa.__name__, dsa)

    patches = {
        name: Mock(name=name) for name in (
            "_patch_compute_hash", "_patch_revalidation", "_patch_dp_threshold",
            "_patch_slicing", "_patch_thread_device",
        )
    }
    namespace = _load_functions("vllm_hcu/v1/hcu_dbo.py", {
        "enable_in_worker", "install_runner", "_patch_dsa", "_noop",
    }, {
        "os": os, "sys": sys, "contextlib": contextlib,
        "henvs": SimpleNamespace(optional_custom_op_enabled=env["optional_custom_op_enabled"]),
        "is_step5_config": model["is_step5_config"],
        "logger": Mock(name="logger"), **patches,
    }, assignments={"DBO_MIN_TOKENS"})
    patches["_patch_dsa"] = Mock(name="_patch_dsa", wraps=namespace["_patch_dsa"])
    namespace["_patch_dsa"] = patches["_patch_dsa"]
    return SimpleNamespace(
        enable=namespace["enable_in_worker"], install=namespace["install_runner"],
        patches=patches, uu=uu, ws=ws, dsa_state=DSAState,
    )


def _config(*, model="step5", enabled=True, owner=MISSING):
    pc = SimpleNamespace(data_parallel_size=4, enable_dbo=enabled)
    if owner is not MISSING:
        pc._hcu_step5_dbo = owner
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type=model)),
        parallel_config=pc,
    )


def _runner(config):
    return SimpleNamespace(
        vllm_config=config,
        ubatch_runner=SimpleNamespace(prepare=Mock(name="native_prepare"),
                                      run=Mock(name="native_run")),
    )


def _assert_no_runner_hooks(runtime, runner):
    ur = runner.ubatch_runner
    original = vars(ur).copy()
    native_schedule = runtime.dsa_state.schedule
    runtime.install(runner)
    assert ur.prepare is original["prepare"]
    assert ur.run is original["run"]
    assert vars(ur) == original
    assert runtime.dsa_state.schedule is native_schedule
    for patch in runtime.patches.values():
        patch.assert_not_called()


@pytest.mark.parametrize("threshold", [4096, 0], ids=["hcu-threshold", "env-disabled"])
@pytest.mark.parametrize("owner", [MISSING, False, None, 1, "true"],
                         ids=["unmarked", "false", "none", "one", "string"])
def test_native_step5_dbo_never_acquires_hcu_runner_hooks(monkeypatch, threshold, owner):
    runtime = _runtime(monkeypatch, threshold=threshold)
    config = _config(owner=owner)
    original = vars(config.parallel_config).copy()

    assert runtime.enable(config) is False
    assert vars(config.parallel_config) == original
    for patch in runtime.patches.values():
        patch.assert_not_called()
    _assert_no_runner_hooks(runtime, _runner(config))


@pytest.mark.parametrize("options", [
    {"threshold": 0}, {"master": False},
], ids=["env-disabled", "master-disabled"])
def test_owner_marker_does_not_override_disabled_hcu_switches(monkeypatch, options):
    runtime = _runtime(monkeypatch, **options)
    config = _config(owner=True)
    assert runtime.enable(config) is False
    _assert_no_runner_hooks(runtime, _runner(config))


@pytest.mark.parametrize("model", ["qwen3", "deepseek_v3", "glm", "unknown"])
def test_other_models_with_inherited_owner_marker_keep_native_runner(monkeypatch, model):
    runtime = _runtime(monkeypatch)
    config = _config(model=model, owner=True)
    assert runtime.enable(config) is False
    _assert_no_runner_hooks(runtime, _runner(config))


def test_owner_marker_requires_dbo_to_remain_enabled(monkeypatch):
    runtime = _runtime(monkeypatch)
    _assert_no_runner_hooks(runtime, _runner(_config(enabled=False, owner=True)))


@pytest.mark.parametrize("parallel_config", [MISSING, None, SimpleNamespace()],
                         ids=["missing", "none", "missing-flags"])
def test_runner_without_parallel_dbo_config_keeps_native_hooks(monkeypatch, parallel_config):
    runtime = _runtime(monkeypatch)
    config = _config()
    if parallel_config is MISSING:
        del config.parallel_config
    else:
        config.parallel_config = parallel_config
    _assert_no_runner_hooks(runtime, _runner(config))


@pytest.mark.parametrize("config", [MISSING, None], ids=["missing", "none"])
def test_runner_without_vllm_config_keeps_native_hooks(monkeypatch, config):
    runtime = _runtime(monkeypatch)
    runner = _runner(config)
    if config is MISSING:
        del runner.vllm_config
    _assert_no_runner_hooks(runtime, runner)


@pytest.mark.parametrize("ubatch_runner", [MISSING, None], ids=["missing", "none"])
def test_owned_runner_without_microbatch_runner_is_unchanged(monkeypatch, ubatch_runner):
    runtime = _runtime(monkeypatch)
    runner = _runner(_config(owner=True))
    if ubatch_runner is MISSING:
        del runner.ubatch_runner
    else:
        runner.ubatch_runner = ubatch_runner
    original = vars(runner).copy()
    runtime.install(runner)
    assert vars(runner) == original
    runtime.patches["_patch_dsa"].assert_not_called()


@pytest.mark.parametrize("model", ["step5", "step5_mtp"])
def test_hcu_auto_enable_marks_owner_and_installs_hooks_once(monkeypatch, model):
    runtime = _runtime(monkeypatch)
    config = _config(model=model, enabled=False)
    runner = _runner(config)
    native_prepare = runner.ubatch_runner.prepare
    native_run = runner.ubatch_runner.run
    native_sm_control = runtime.uu.create_sm_control_context
    native_ubatch_id = runtime.ws.dbo_current_ubatch_id
    native_schedule = runtime.dsa_state.schedule

    assert runtime.enable(config) is True
    assert config.parallel_config.enable_dbo is True
    assert config.parallel_config._hcu_step5_dbo is True
    for name, patch in runtime.patches.items():
        if name != "_patch_dsa":
            patch.assert_called_once_with()
    runtime.patches["_patch_dsa"].assert_not_called()
    assert runtime.uu.create_sm_control_context is not native_sm_control
    with runtime.uu.create_sm_control_context(config.parallel_config) as value:
        assert value is None
    assert runtime.ws.dbo_current_ubatch_id is not native_ubatch_id
    assert runtime.ws.dbo_current_ubatch_id() == 0

    runtime.install(runner)
    wrapped_prepare = runner.ubatch_runner.prepare
    wrapped_run = runner.ubatch_runner.run
    assert wrapped_prepare is not native_prepare
    assert wrapped_run is not native_run
    assert runner.ubatch_runner._hcu_dbo is True
    runtime.patches["_patch_dsa"].assert_called_once_with()
    wrapped_schedule = runtime.dsa_state.schedule
    assert wrapped_schedule is not native_schedule
    assert wrapped_schedule._hcu_dbo is True
    metadata = object()
    assert runtime.dsa_state().schedule(metadata) is metadata

    runner._hsl_enabled = False
    batch = SimpleNamespace(num_tokens=4096)
    block_tables, slot_mappings = object(), object()
    assert wrapped_prepare(batch, block_tables, slot_mappings) is native_prepare.return_value
    native_prepare.assert_called_once_with(batch, block_tables, slot_mappings)

    assert runtime.enable(config) is False
    runtime.install(runner)
    assert runner.ubatch_runner.prepare is wrapped_prepare
    assert runner.ubatch_runner.run is wrapped_run
    assert runtime.dsa_state.schedule is wrapped_schedule
    for patch in runtime.patches.values():
        patch.assert_called_once_with()
