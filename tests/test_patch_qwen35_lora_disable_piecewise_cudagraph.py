from __future__ import annotations

import ast
import importlib.util
import os
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_patch_module():
    patch_path = (
        ROOT / "vllm_hcu/patches/patch_qwen35_lora_disable_piecewise_cudagraph.py"
    )
    spec = importlib.util.spec_from_file_location(
        "patch_qwen35_lora_piecewise_test",
        patch_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _install_fake_modules():
    injected_module_names = [
        "vllm",
        "vllm.config",
        "vllm.v1",
        "vllm.v1.cudagraph_dispatcher",
    ]
    original_modules = {name: sys.modules.get(name) for name in injected_module_names}

    class _Mode:
        def __init__(self, name: str):
            self.name = name

        def __repr__(self) -> str:
            return self.name

        def __hash__(self) -> int:
            return hash(self.name)

        def __eq__(self, other) -> bool:
            return isinstance(other, _Mode) and self.name == other.name

    class _CUDAGraphMode:
        NONE = _Mode("NONE")
        PIECEWISE = _Mode("PIECEWISE")
        FULL = _Mode("FULL")

    class _Logger:
        def __init__(self):
            self.messages: list[str] = []

        def warning(self, message, *args):
            self.messages.append(message % args if args else message)

    class CudagraphDispatcher:
        def __init__(self, vllm_config):
            self.vllm_config = vllm_config
            self.dispatch_calls: list[dict[str, object]] = []

        def dispatch(
            self,
            num_tokens,
            uniform_decode=False,
            has_lora=False,
            num_active_loras=0,
            valid_modes=None,
            invalid_modes=None,
        ):
            self.dispatch_calls.append(
                {
                    "num_tokens": num_tokens,
                    "uniform_decode": uniform_decode,
                    "has_lora": has_lora,
                    "num_active_loras": num_active_loras,
                    "valid_modes": valid_modes,
                    "invalid_modes": invalid_modes,
                }
            )
            return invalid_modes

        def get_capture_descs(self):
            return [
                (
                    _CUDAGraphMode.PIECEWISE,
                    [
                        types.SimpleNamespace(num_active_loras=0),
                        types.SimpleNamespace(num_active_loras=2),
                    ],
                ),
                (
                    _CUDAGraphMode.FULL,
                    [
                        types.SimpleNamespace(num_active_loras=2),
                    ],
                ),
            ]

    config_module = types.ModuleType("vllm.config")
    config_module.CUDAGraphMode = _CUDAGraphMode

    dispatcher_module = types.ModuleType("vllm.v1.cudagraph_dispatcher")
    dispatcher_module.CudagraphDispatcher = CudagraphDispatcher
    dispatcher_module.logger = _Logger()

    vllm_module = types.ModuleType("vllm")
    vllm_v1_module = types.ModuleType("vllm.v1")
    vllm_v1_module.cudagraph_dispatcher = dispatcher_module

    sys.modules["vllm"] = vllm_module
    sys.modules["vllm.config"] = config_module
    sys.modules["vllm.v1"] = vllm_v1_module
    sys.modules["vllm.v1.cudagraph_dispatcher"] = dispatcher_module

    return {
        "restore": original_modules,
        "injected": injected_module_names,
        "config_module": config_module,
        "dispatcher_module": dispatcher_module,
    }


def test_patch_filters_all_piecewise_capture_for_qwen35() -> None:
    state = _install_fake_modules()
    try:
        patch_module = _load_patch_module()
        patch_module.patch_qwen35_lora_disable_piecewise_cudagraph()

        class _ModelConfig:
            architecture = "Qwen3_5ForConditionalGeneration"

        config = types.SimpleNamespace(
            model_config=_ModelConfig(),
            lora_config=object(),
        )
        dispatcher = state["dispatcher_module"].CudagraphDispatcher(config)

        capture_descs = dispatcher.get_capture_descs()
        assert len(capture_descs) == 1
        assert capture_descs[0][0] == state["config_module"].CUDAGraphMode.FULL
        assert [desc.num_active_loras for desc in capture_descs[0][1]] == [2]
        assert state["dispatcher_module"].logger.messages
    finally:
        for name in state["injected"]:
            original_module = state["restore"][name]
            if original_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original_module


def test_patch_disables_piecewise_dispatch_for_qwen35() -> None:
    state = _install_fake_modules()
    try:
        patch_module = _load_patch_module()
        patch_module.patch_qwen35_lora_disable_piecewise_cudagraph()

        class _ModelConfig:
            architecture = "Qwen3_5ForConditionalGeneration"

        config = types.SimpleNamespace(
            model_config=_ModelConfig(),
            lora_config=object(),
        )
        dispatcher = state["dispatcher_module"].CudagraphDispatcher(config)

        invalid_modes = dispatcher.dispatch(
            num_tokens=8,
            uniform_decode=False,
            has_lora=False,
            num_active_loras=0,
            invalid_modes={state["config_module"].CUDAGraphMode.FULL},
        )
        assert invalid_modes == {
            state["config_module"].CUDAGraphMode.FULL,
            state["config_module"].CUDAGraphMode.PIECEWISE,
        }
    finally:
        for name in state["injected"]:
            original_module = state["restore"][name]
            if original_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original_module


def test_patch_allows_env_opt_out_for_qwen35() -> None:
    state = _install_fake_modules()
    original_env = os.environ.get("VLLM_HCU_QWEN35_LORA_ALLOW_UNSAFE_COMPILE")
    os.environ["VLLM_HCU_QWEN35_LORA_ALLOW_UNSAFE_COMPILE"] = "1"
    try:
        patch_module = _load_patch_module()
        patch_module.patch_qwen35_lora_disable_piecewise_cudagraph()

        class _ModelConfig:
            architecture = "Qwen3_5ForConditionalGeneration"

        config = types.SimpleNamespace(
            model_config=_ModelConfig(),
            lora_config=object(),
        )
        dispatcher = state["dispatcher_module"].CudagraphDispatcher(config)

        capture_descs = dispatcher.get_capture_descs()
        assert [desc.num_active_loras for desc in capture_descs[0][1]] == [0, 2]
        invalid_modes = dispatcher.dispatch(
            num_tokens=8,
            uniform_decode=False,
            has_lora=True,
            num_active_loras=2,
            invalid_modes=None,
        )
        assert invalid_modes is None
    finally:
        if original_env is None:
            os.environ.pop("VLLM_HCU_QWEN35_LORA_ALLOW_UNSAFE_COMPILE", None)
        else:
            os.environ["VLLM_HCU_QWEN35_LORA_ALLOW_UNSAFE_COMPILE"] = original_env
        for name in state["injected"]:
            original_module = state["restore"][name]
            if original_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original_module


def test_patch_leaves_other_models_unchanged() -> None:
    state = _install_fake_modules()
    try:
        patch_module = _load_patch_module()
        patch_module.patch_qwen35_lora_disable_piecewise_cudagraph()

        class _ModelConfig:
            architecture = "LlamaForCausalLM"

        config = types.SimpleNamespace(
            model_config=_ModelConfig(),
            lora_config=object(),
        )
        dispatcher = state["dispatcher_module"].CudagraphDispatcher(config)

        capture_descs = dispatcher.get_capture_descs()
        assert [desc.num_active_loras for desc in capture_descs[0][1]] == [0, 2]
        invalid_modes = dispatcher.dispatch(
            num_tokens=8,
            uniform_decode=False,
            has_lora=True,
            num_active_loras=2,
            invalid_modes=None,
        )
        assert invalid_modes is None
    finally:
        for name in state["injected"]:
            original_module = state["restore"][name]
            if original_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original_module


def test_patch_utils_registers_qwen35_piecewise_patch() -> None:
    source = (ROOT / "vllm_hcu/patch_utils.py").read_text(encoding="utf-8")
    module = ast.parse(source)

    function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "patch_module_class_function"
    )

    names = {
        node.id
        for node in ast.walk(function)
        if isinstance(node, ast.Name)
    }

    assert "patch_qwen35_lora_disable_piecewise_cudagraph" in names
