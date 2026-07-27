# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from __future__ import annotations

import builtins
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

import vllm_hcu.patch.runtime_callbacks as runtime_callbacks
from vllm_hcu.patch._stage3_common import Stage3CompatibilityError
from vllm_hcu.patch.import_coordinator import ExactImportCoordinator
from vllm_hcu.patch.runtime_callbacks import (
    apply_base_linear_parameter,
    apply_fp8_scaled_mm,
    apply_hcu_lora_column_parallel,
    apply_kimi_k25_vision_prompt,
    apply_qwen35_lora_cudagraph,
    apply_weight_debug_skip,
    register_runtime_method_callbacks,
    runtime_callback_names,
)
from vllm_hcu.patch.runtime_state import PatchRegistry, PatchStatus


EXPECTED_ORDER = (
    (
        "runtime_method.base_linear_parameter",
        "vllm.model_executor.parameter",
    ),
    (
        "runtime_method.fp8_scaled_mm",
        "vllm.model_executor.kernels.linear.scaled_mm.pytorch",
    ),
    (
        "runtime_method.hcu_lora_column_parallel",
        "vllm.lora.layers.column_parallel_linear",
    ),
    (
        "runtime_method.qwen35_lora_cudagraph",
        "vllm.v1.cudagraph_dispatcher",
    ),
    (
        "runtime_method.weight_debug_skip",
        "vllm.model_executor.model_loader.default_loader",
    ),
    (
        "runtime_method.kimi_k25_vision_prompt",
        "vllm.model_executor.models.kimi_k25",
    ),
)

EXPECTED_IMPLEMENTATIONS = {
    "parameter": (
        "vllm_hcu.runtime_compat.base_linear_parameter",
        "install_base_linear_parameter_compat",
    ),
    "fp8": (
        "vllm_hcu.runtime_compat.scaled_mm",
        "install_fp8_scaled_mm_compat",
    ),
    "lora": (
        "vllm_hcu.runtime_compat.lora_column_parallel",
        "install_hcu_lora_column_parallel_compat",
    ),
    "qwen35": (
        "vllm_hcu.runtime_compat.qwen35_lora_cudagraph",
        "install_qwen35_lora_cudagraph_compat",
    ),
    "weight": (
        "vllm_hcu.runtime_compat.weight_loading",
        "install_weight_debug_skip_compat",
    ),
    "kimi": (
        "vllm_hcu.runtime_compat.kimi_k25_vision_prompt",
        "install_kimi_k25_vision_prompt_compat",
    ),
}


def test_runtime_callbacks_register_in_old_explicit_order_without_importing_targets(
    monkeypatch: pytest.MonkeyPatch,
):
    for _, module_name in EXPECTED_ORDER:
        monkeypatch.delitem(sys.modules, module_name, raising=False)
    original_import = builtins.__import__
    registry = PatchRegistry()
    coordinator = ExactImportCoordinator(registry=registry)

    first = register_runtime_method_callbacks(coordinator)
    second = register_runtime_method_callbacks(coordinator)

    assert runtime_callback_names() == EXPECTED_ORDER
    assert [item.patch_id for item in first] == [item[0] for item in EXPECTED_ORDER]
    assert [item.status for item in first] == [PatchStatus.ARMED.value] * len(
        EXPECTED_ORDER
    )
    assert [item.status for item in second] == [PatchStatus.ARMED.value] * len(
        EXPECTED_ORDER
    )
    assert len(coordinator.registrations()) == len(EXPECTED_ORDER)
    assert not any(name in sys.modules for _, name in EXPECTED_ORDER)
    assert builtins.__import__ is original_import


def test_fp8_callback_invokes_implementation_only_after_exact_target(
    monkeypatch: pytest.MonkeyPatch,
):
    module = ModuleType(EXPECTED_ORDER[1][1])

    class ChannelWise:
        pass

    class TorchKernel:
        pass

    module.ChannelWiseTorchFP8ScaledMMLinearKernel = ChannelWise
    module.TorchFP8ScaledMMLinearKernel = TorchKernel
    calls: list[str] = []

    def patch(target):
        assert target is module
        calls.append("fp8")
        ChannelWise._hcu_fp8_patch_applied = True

    loaded: list[tuple[str, str]] = []

    def load(module_name: str, function_name: str):
        loaded.append((module_name, function_name))
        return patch

    monkeypatch.setattr(
        runtime_callbacks,
        "_load_runtime_callable",
        load,
    )
    apply_fp8_scaled_mm(module)
    assert calls == ["fp8"]
    assert loaded == [EXPECTED_IMPLEMENTATIONS["fp8"]]


def test_base_linear_parameter_callback_requires_classes_and_postcondition(
    monkeypatch: pytest.MonkeyPatch,
):
    module = ModuleType(EXPECTED_ORDER[0][1])
    for class_name in (
        "BasevLLMParameter",
        "_ColumnvLLMParameter",
        "RowvLLMParameter",
        "ModelWeightParameter",
        "PackedColumnParameter",
        "PackedvLLMParameter",
    ):
        setattr(module, class_name, type(class_name, (), {}))
    passed: list[ModuleType] = []
    loaded: list[tuple[str, str]] = []

    def patch(target: ModuleType):
        passed.append(target)
        target._hcu_base_linear_parameter_patch_applied = True

    def load(module_name: str, function_name: str):
        loaded.append((module_name, function_name))
        return patch

    monkeypatch.setattr(runtime_callbacks, "_load_runtime_callable", load)
    apply_base_linear_parameter(module)

    assert passed == [module]
    assert loaded == [EXPECTED_IMPLEMENTATIONS["parameter"]]


def test_lora_qwen_and_kimi_callbacks_require_postcondition_markers(
    monkeypatch: pytest.MonkeyPatch,
):
    lora = ModuleType(EXPECTED_ORDER[2][1])
    lora.ColumnParallelLinearWithLoRA = type("ColumnParallelLinearWithLoRA", (), {})
    qwen = ModuleType(EXPECTED_ORDER[3][1])
    qwen.CudagraphDispatcher = type("CudagraphDispatcher", (), {})
    kimi = ModuleType(EXPECTED_ORDER[5][1])
    kimi.KimiK25MultiModalProcessor = type("KimiK25MultiModalProcessor", (), {})
    passed_lora_modules: list[ModuleType] = []

    def patch_lora(module: ModuleType):
        passed_lora_modules.append(module)
        setattr(module, "_hcu_lora_column_parallel_linear_patch_applied", True)

    callbacks = iter(
        [
            patch_lora,
            lambda: setattr(
                qwen.CudagraphDispatcher,
                "_hcu_qwen35_lora_piecewise_cudagraph_patch_applied",
                True,
            ),
            lambda: setattr(
                kimi.KimiK25MultiModalProcessor,
                "_hcu_kimi_k25_prompt_patch_applied",
                True,
            ),
        ]
    )
    loaded: list[tuple[str, str]] = []

    def load(module_name: str, function_name: str):
        loaded.append((module_name, function_name))
        return next(callbacks)

    monkeypatch.setattr(
        runtime_callbacks,
        "_load_runtime_callable",
        load,
    )

    apply_hcu_lora_column_parallel(lora)
    apply_qwen35_lora_cudagraph(qwen)
    apply_kimi_k25_vision_prompt(kimi)
    assert loaded == [
        EXPECTED_IMPLEMENTATIONS["lora"],
        EXPECTED_IMPLEMENTATIONS["qwen35"],
        EXPECTED_IMPLEMENTATIONS["kimi"],
    ]
    assert passed_lora_modules == [lora]


def test_weight_callback_updates_both_import_bindings(
    monkeypatch: pytest.MonkeyPatch,
):
    default_loader = ModuleType(EXPECTED_ORDER[4][1])
    weight_utils_name = "vllm.model_executor.model_loader.weight_utils"
    weight_utils = ModuleType(weight_utils_name)

    def original_iterator():
        return None

    weight_utils.safetensors_weights_iterator = original_iterator
    default_loader.safetensors_weights_iterator = original_iterator
    monkeypatch.setitem(sys.modules, weight_utils_name, weight_utils)
    passed_modules: list[tuple[ModuleType, ModuleType]] = []

    def patch(weight_utils_arg: ModuleType, default_loader_arg: ModuleType):
        passed_modules.append((weight_utils_arg, default_loader_arg))

        def patched_iterator():
            return None

        weight_utils_arg.safetensors_weights_iterator = patched_iterator
        default_loader_arg.safetensors_weights_iterator = patched_iterator
        weight_utils_arg._hcu_skip_weight_patch_applied = True

    loaded: list[tuple[str, str]] = []

    def load(module_name: str, function_name: str):
        loaded.append((module_name, function_name))
        return patch

    monkeypatch.setattr(
        runtime_callbacks,
        "_load_runtime_callable",
        load,
    )
    apply_weight_debug_skip(default_loader)
    assert (
        weight_utils.safetensors_weights_iterator
        is default_loader.safetensors_weights_iterator
    )
    assert passed_modules == [(weight_utils, default_loader)]
    assert loaded == [EXPECTED_IMPLEMENTATIONS["weight"]]


def test_weight_installer_is_direct_idempotent_and_keeps_bindings_coherent(
    monkeypatch: pytest.MonkeyPatch,
):
    import vllm_hcu.runtime_compat.weight_loading as weight_loading

    weight_utils = ModuleType("vllm.model_executor.model_loader.weight_utils")
    default_loader = ModuleType("vllm.model_executor.model_loader.default_loader")

    def original_iterator(*args, **kwargs):
        yield from ()

    weight_utils.safetensors_weights_iterator = original_iterator
    default_loader.safetensors_weights_iterator = original_iterator
    monkeypatch.setitem(sys.modules, weight_utils.__name__, weight_utils)
    imported: list[str] = []

    def import_module(name: str):
        imported.append(name)
        assert name == default_loader.__name__
        return default_loader

    monkeypatch.setattr(weight_loading.importlib, "import_module", import_module)

    weight_loading.install_weight_debug_skip_compat()
    first_wrapper = weight_utils.safetensors_weights_iterator
    weight_loading.install_weight_debug_skip_compat()

    assert first_wrapper is not original_iterator
    assert weight_utils.safetensors_weights_iterator is first_wrapper
    assert default_loader.safetensors_weights_iterator is first_wrapper
    assert weight_utils._hcu_skip_weight_patch_applied is True
    assert imported == [default_loader.__name__, default_loader.__name__]


def test_clean_v0251_model_loader_import_order_has_no_weight_debug_cycle():
    repo = Path(__file__).resolve().parents[2]
    target_vllm = Path(
        os.environ.get("VLLM_V0251_SOURCE_ROOT", repo.parent / "vllm_0251")
    ).resolve()
    if not (target_vllm / "vllm" / "__init__.py").is_file():
        raise RuntimeError(
            f"VLLM_V0251_SOURCE_ROOT does not contain vllm: {target_vllm}"
        )
    env = dict(os.environ)
    env["VLLM_PLUGINS"] = "__disabled__"
    env["VLLM_V0251_SOURCE_ROOT"] = str(target_vllm)
    env["PYTHONPATH"] = os.pathsep.join((str(target_vllm), str(repo)))
    code = r'''
import importlib.abc
import json
import os
import sys
from pathlib import Path

import vllm

target_root = Path(os.environ["VLLM_V0251_SOURCE_ROOT"]).resolve()
target_file = Path(vllm.__file__).resolve()
assert target_file.is_relative_to(target_root), (
    f"vllm resolved outside target root: {target_file} not under {target_root}"
)

from vllm_hcu.patch import apply_platform_patches, patch_report

apply_platform_patches()

prefix = "vllm.model_executor.model_loader"
events = []

class ImportRecorder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == prefix or fullname.startswith(prefix + "."):
            events.append(fullname)
        return None

recorder = ImportRecorder()
sys.meta_path.insert(0, recorder)
try:
    from vllm.model_executor.model_loader import DefaultModelLoader
finally:
    sys.meta_path.remove(recorder)

weight_name = prefix + ".weight_utils"
default_name = prefix + ".default_loader"
weight_utils = sys.modules[weight_name]
default_loader = sys.modules[default_name]
required = [
    prefix + ".base_loader",
    prefix + ".reload.layerwise",
    weight_name,
    default_name,
]
positions = [events.index(name) for name in required]
record = patch_report()["patches"]["runtime_method.weight_debug_skip"]
print(json.dumps({
    "class_module": DefaultModelLoader.__module__,
    "required": required,
    "positions": positions,
    "status": record["status"],
    "marker": getattr(weight_utils, "_hcu_skip_weight_patch_applied", False),
    "same_iterator": (
        weight_utils.safetensors_weights_iterator
        is default_loader.safetensors_weights_iterator
    ),
}))
'''
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload == {
        "class_module": "vllm.model_executor.model_loader.default_loader",
        "required": [
            "vllm.model_executor.model_loader.base_loader",
            "vllm.model_executor.model_loader.reload.layerwise",
            "vllm.model_executor.model_loader.weight_utils",
            "vllm.model_executor.model_loader.default_loader",
        ],
        "positions": sorted(payload["positions"]),
        "status": "applied",
        "marker": True,
        "same_iterator": True,
    }


def test_runtime_callback_missing_marker_fails_instead_of_silent_skip(
    monkeypatch: pytest.MonkeyPatch,
):
    module = ModuleType(EXPECTED_ORDER[5][1])
    module.KimiK25MultiModalProcessor = type("KimiK25MultiModalProcessor", (), {})
    monkeypatch.setattr(
        runtime_callbacks, "_load_runtime_callable", lambda *args: lambda: None
    )
    with pytest.raises(Stage3CompatibilityError, match="did not apply"):
        apply_kimi_k25_vision_prompt(module)


def test_loaded_target_callback_is_applied_and_reported(
    monkeypatch: pytest.MonkeyPatch,
):
    module = ModuleType(EXPECTED_ORDER[1][1])

    class ChannelWise:
        pass

    module.ChannelWiseTorchFP8ScaledMMLinearKernel = ChannelWise
    module.TorchFP8ScaledMMLinearKernel = type("TorchKernel", (), {})

    def patch(target):
        assert target is module
        ChannelWise._hcu_fp8_patch_applied = True

    monkeypatch.setattr(
        runtime_callbacks,
        "_load_runtime_callable",
        lambda *args: patch,
    )
    monkeypatch.setitem(sys.modules, module.__name__, module)
    registry = PatchRegistry()
    coordinator = ExactImportCoordinator(registry=registry)

    registration = coordinator.register_callback(
        EXPECTED_ORDER[1][0],
        EXPECTED_ORDER[1][1],
        apply_fp8_scaled_mm,
    )
    assert registration.status == PatchStatus.APPLIED.value
    assert registry.get(EXPECTED_ORDER[1][0]).status is PatchStatus.APPLIED
