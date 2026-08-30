# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from __future__ import annotations

import importlib.machinery
import importlib.util
import inspect
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import cloudpickle
import pytest
import torch

from vllm_hcu.patch._stage3_common import Stage3CompatibilityError
from vllm_hcu.patch.import_coordinator import ExactImportCoordinator
from vllm_hcu.patch.runtime_callbacks import apply_base_linear_parameter
from vllm_hcu.patch.runtime_state import LatchedPatchError, PatchRegistry
from vllm_hcu.runtime_compat.base_linear_parameter import (
    install_base_linear_parameter_compat,
)


TARGET = "vllm.model_executor.parameter"
REMOVED_HCU_TARGET = "vllm_hcu.model_executor.parameter"
REPO_ROOT = Path(__file__).resolve().parents[2]
_target_source_root = os.environ.get("VLLM_TARGET_SOURCE_ROOT")
if _target_source_root is None:
    _target_spec = importlib.util.find_spec("vllm")
    if _target_spec is None or _target_spec.origin is None:
        raise RuntimeError("cannot locate target vLLM source root")
    TARGET_VLLM_ROOT = Path(_target_spec.origin).resolve().parents[1]
else:
    TARGET_VLLM_ROOT = Path(_target_source_root).resolve()
if not (TARGET_VLLM_ROOT / "vllm" / "__init__.py").is_file():
    raise RuntimeError(
        f"VLLM_TARGET_SOURCE_ROOT does not contain vllm: {TARGET_VLLM_ROOT}"
    )


def _fake_parameter_module() -> ModuleType:
    module = ModuleType(TARGET)
    source = r'''
class BasevLLMParameter:
    def _assert_and_load(self, loaded_weight):
        assert self.data.shape == loaded_weight.shape
        self.data.copy_(loaded_weight)

    def load_column_parallel_weight(self, loaded_weight):
        self._assert_and_load(loaded_weight)

    def load_row_parallel_weight(self, loaded_weight):
        self._assert_and_load(loaded_weight)


class _ColumnvLLMParameter(BasevLLMParameter):
    def load_column_parallel_weight(self, loaded_weight):
        shard_size = self.data.shape[self.output_dim]
        loaded_weight = loaded_weight.narrow(
            self.output_dim, self.tp_rank * shard_size, shard_size
        )
        assert self.data.shape == loaded_weight.shape
        self.data.copy_(loaded_weight)

    def load_merged_column_weight(self, loaded_weight, **kwargs):
        shard_offset = kwargs["shard_offset"]
        shard_size = kwargs["shard_size"]
        if (
            isinstance(self, (PackedColumnParameter, PackedvLLMParameter))
            and self.packed_dim == self.output_dim
        ):
            shard_size, shard_offset = self.adjust_shard_indexes_for_packing(
                shard_offset=shard_offset, shard_size=shard_size
            )
        param_data = self.data.narrow(
            self.output_dim, shard_offset, shard_size
        )
        loaded_weight = loaded_weight.narrow(
            self.output_dim, self.tp_rank * shard_size, shard_size
        )
        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)

    def load_qkv_weight(self, loaded_weight, **kwargs):
        shard_offset = kwargs["shard_offset"]
        shard_size = kwargs["shard_size"]
        shard_id = kwargs["shard_id"]
        num_heads = kwargs["num_heads"]
        if (
            isinstance(self, (PackedColumnParameter, PackedvLLMParameter))
            and self.output_dim == self.packed_dim
        ):
            shard_size, shard_offset = self.adjust_shard_indexes_for_packing(
                shard_offset=shard_offset, shard_size=shard_size
            )
        shard_rank = self.tp_rank if shard_id == "q" else self.tp_rank // num_heads
        param_data = self.data.narrow(
            self.output_dim, shard_offset, shard_size
        )
        loaded_weight = loaded_weight.narrow(
            self.output_dim, shard_rank * shard_size, shard_size
        )
        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)


class RowvLLMParameter(BasevLLMParameter):
    def load_row_parallel_weight(self, loaded_weight):
        shard_size = self.data.shape[self.input_dim]
        loaded_weight = loaded_weight.narrow(
            self.input_dim, self.tp_rank * shard_size, shard_size
        )
        if len(loaded_weight.shape) == 0:
            loaded_weight = loaded_weight.reshape(1)
        assert self.data.shape == loaded_weight.shape
        self.data.copy_(loaded_weight)


class ModelWeightParameter(_ColumnvLLMParameter, RowvLLMParameter):
    pass


class PackedColumnParameter(_ColumnvLLMParameter):
    def adjust_shard_indexes_for_packing(self, shard_size, shard_offset):
        return shard_size, shard_offset


class PackedvLLMParameter(ModelWeightParameter):
    def adjust_shard_indexes_for_packing(self, shard_size, shard_offset):
        return shard_size, shard_offset
'''
    exec(compile(source, "<fake-vllm-parameter>", "exec"), module.__dict__)
    return module


@pytest.fixture
def isolated_parameter_modules(monkeypatch: pytest.MonkeyPatch):
    module = _fake_parameter_module()
    parent = ModuleType("vllm.model_executor")
    monkeypatch.setitem(sys.modules, TARGET, module)
    monkeypatch.setitem(sys.modules, "vllm.model_executor", parent)
    monkeypatch.delitem(sys.modules, REMOVED_HCU_TARGET, raising=False)
    return module, parent


def _instance(owner: type, **attributes):
    value = object.__new__(owner)
    for name, item in attributes.items():
        setattr(value, name, item)
    return value


def test_adapter_is_idempotent_and_preserves_unique_parent_identity(
    isolated_parameter_modules,
):
    module, parent = isolated_parameter_modules
    parent.BasevLLMParameter = module.BasevLLMParameter
    parent.PackedvLLMParameter = module.PackedvLLMParameter

    install_base_linear_parameter_compat(module)
    first = module._ColumnvLLMParameter.load_column_parallel_weight
    install_base_linear_parameter_compat(module)

    assert module._ColumnvLLMParameter.load_column_parallel_weight is first
    assert module._hcu_base_linear_parameter_patch_applied is True
    assert parent.BasevLLMParameter is module.BasevLLMParameter
    assert parent.PackedvLLMParameter is module.PackedvLLMParameter
    assert REMOVED_HCU_TARGET not in sys.modules
    assert tuple(inspect.signature(first).parameters) == (
        "self",
        "loaded_weight",
        "is_quantization",
    )


def test_nn_layout_loaders_transpose_and_use_physical_dimensions(
    isolated_parameter_modules,
    monkeypatch: pytest.MonkeyPatch,
):
    module, _ = isolated_parameter_modules
    from vllm_hcu.platforms import envs as henvs

    monkeypatch.setattr(henvs, "VLLM_USE_NN", True)
    install_base_linear_parameter_compat(module)

    full_column = torch.arange(12).reshape(4, 3)
    base_column = _instance(
        module.BasevLLMParameter,
        data=torch.zeros(3, 2, dtype=full_column.dtype),
        tp_rank=1,
    )
    base_column.load_column_parallel_weight(full_column)
    torch.testing.assert_close(base_column.data, full_column[2:4].t())

    column = _instance(
        module._ColumnvLLMParameter,
        data=torch.zeros(3, 2, dtype=full_column.dtype),
        output_dim=0,
        tp_rank=1,
    )
    column.load_column_parallel_weight(full_column)
    torch.testing.assert_close(column.data, full_column[2:4].t())

    # vLLM's v2 loader may already provide the local logical shard.  The
    # NN-layout adapter must not narrow that shard a second time.
    local_column = _instance(
        module._ColumnvLLMParameter,
        data=torch.zeros(3, 2, dtype=full_column.dtype),
        output_dim=0,
        tp_rank=1,
    )
    local_column.load_column_parallel_weight(full_column[2:4])
    torch.testing.assert_close(local_column.data, full_column[2:4].t())

    # Equal logical and physical shapes do not imply equal layouts.  Square
    # checkpoint matrices still require the NN-layout transpose.
    square_weight = torch.arange(9).reshape(3, 3)
    square_column = _instance(
        module._ColumnvLLMParameter,
        data=torch.zeros_like(square_weight),
        output_dim=0,
        tp_rank=0,
    )
    square_column.load_column_parallel_weight(square_weight)
    torch.testing.assert_close(square_column.data, square_weight.t())

    merged = _instance(
        module._ColumnvLLMParameter,
        data=torch.zeros(3, 4, dtype=full_column.dtype),
        output_dim=0,
        tp_rank=1,
    )
    merged.load_merged_column_weight(
        full_column,
        shard_offset=2,
        shard_size=2,
    )
    torch.testing.assert_close(merged.data[:, :2], torch.zeros(3, 2, dtype=torch.int64))
    torch.testing.assert_close(merged.data[:, 2:4], full_column[2:4].t())

    qkv = _instance(
        module._ColumnvLLMParameter,
        data=torch.zeros(3, 4, dtype=full_column.dtype),
        output_dim=0,
        tp_rank=1,
    )
    qkv.load_qkv_weight(
        full_column,
        shard_offset=0,
        shard_size=2,
        shard_id="q",
        num_heads=1,
    )
    torch.testing.assert_close(qkv.data[:, :2], full_column[2:4].t())
    torch.testing.assert_close(qkv.data[:, 2:4], torch.zeros(3, 2, dtype=torch.int64))

    full_row = torch.arange(12).reshape(3, 4)
    base_row = _instance(
        module.BasevLLMParameter,
        data=torch.zeros(2, 3, dtype=full_row.dtype),
        tp_rank=1,
    )
    base_row.load_row_parallel_weight(full_row)
    torch.testing.assert_close(base_row.data, full_row[:, 2:4].t())

    row = _instance(
        module.RowvLLMParameter,
        data=torch.zeros(2, 3, dtype=full_row.dtype),
        input_dim=1,
        tp_rank=1,
    )
    row.load_row_parallel_weight(full_row)
    torch.testing.assert_close(row.data, full_row[:, 2:4].t())


@pytest.mark.parametrize("use_nn,is_quantization", [(False, False), (True, True)])
def test_feature_off_and_quantized_paths_delegate_to_official_layout(
    isolated_parameter_modules,
    monkeypatch: pytest.MonkeyPatch,
    use_nn: bool,
    is_quantization: bool,
):
    module, _ = isolated_parameter_modules
    from vllm_hcu.platforms import envs as henvs

    monkeypatch.setattr(henvs, "VLLM_USE_NN", use_nn)
    install_base_linear_parameter_compat(module)
    loaded = torch.arange(12).reshape(4, 3)
    column = _instance(
        module._ColumnvLLMParameter,
        data=torch.zeros(2, 3, dtype=loaded.dtype),
        output_dim=0,
        tp_rank=1,
    )

    column.load_column_parallel_weight(
        loaded,
        is_quantization=is_quantization,
    )
    torch.testing.assert_close(column.data, loaded[2:4])


def test_signature_drift_fails_before_mutation(isolated_parameter_modules):
    module, _ = isolated_parameter_modules
    original_column = module._ColumnvLLMParameter.load_column_parallel_weight

    def drifted(self, loaded_weight, unexpected=False):
        return None

    drifted.__module__ = TARGET
    module.BasevLLMParameter.load_row_parallel_weight = drifted

    with pytest.raises(Stage3CompatibilityError, match="signature drifted"):
        install_base_linear_parameter_compat(module)

    assert module._ColumnvLLMParameter.load_column_parallel_weight is original_column
    assert not getattr(module, "_hcu_base_linear_parameter_patch_applied", False)


def test_final_identity_failure_rolls_back_every_binding_and_marker(
    isolated_parameter_modules,
):
    module, parent = isolated_parameter_modules
    bindings = {
        (module.BasevLLMParameter, "load_column_parallel_weight"),
        (module.BasevLLMParameter, "load_row_parallel_weight"),
        (module._ColumnvLLMParameter, "load_column_parallel_weight"),
        (module._ColumnvLLMParameter, "load_merged_column_weight"),
        (module._ColumnvLLMParameter, "load_qkv_weight"),
        (module.RowvLLMParameter, "load_row_parallel_weight"),
    }
    originals = {
        (owner, attribute): vars(owner)[attribute]
        for owner, attribute in bindings
    }
    parent.BasevLLMParameter = type("SplitBasevLLMParameter", (), {})

    with pytest.raises(Stage3CompatibilityError, match="parent export"):
        install_base_linear_parameter_compat(module)

    assert not hasattr(module, "_hcu_base_linear_parameter_patch_applied")
    for binding, original in originals.items():
        owner, attribute = binding
        assert vars(owner)[attribute] is original


def test_partial_import_waits_for_drain_and_failure_is_latched(
    isolated_parameter_modules,
):
    module, _ = isolated_parameter_modules
    spec = importlib.machinery.ModuleSpec(TARGET, loader=None)
    spec._initializing = True
    module.__spec__ = spec
    registry = PatchRegistry()
    coordinator = ExactImportCoordinator(registry=registry)

    registration = coordinator.register_callback(
        "runtime_method.base_linear_parameter",
        TARGET,
        apply_base_linear_parameter,
    )
    assert registration.status == "armed"
    assert not getattr(module, "_hcu_base_linear_parameter_patch_applied", False)

    spec._initializing = False
    coordinator.drain_ready_callbacks()
    assert registry.get("runtime_method.base_linear_parameter").status.value == "applied"
    assert module._hcu_base_linear_parameter_patch_applied is True

    module._ColumnvLLMParameter.load_column_parallel_weight = lambda self: None
    with pytest.raises(Stage3CompatibilityError, match="signature drifted"):
        install_base_linear_parameter_compat(module)


def test_invalid_loaded_target_is_failed_once_and_never_retried(
    isolated_parameter_modules,
):
    module, _ = isolated_parameter_modules
    module.BasevLLMParameter.load_row_parallel_weight = lambda self: None
    registry = PatchRegistry()
    coordinator = ExactImportCoordinator(registry=registry)

    with pytest.raises(Stage3CompatibilityError, match="signature drifted"):
        coordinator.register_callback(
            "runtime_method.base_linear_parameter",
            TARGET,
            apply_base_linear_parameter,
        )
    assert registry.get("runtime_method.base_linear_parameter").status.value == "failed"
    with pytest.raises(LatchedPatchError, match="previously failed"):
        coordinator.register_callback(
            "runtime_method.base_linear_parameter",
            TARGET,
            apply_base_linear_parameter,
        )


def _clean_target_environment(cache_root: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["VLLM_TARGET_SOURCE_ROOT"] = str(TARGET_VLLM_ROOT)
    env["PYTHONPATH"] = os.pathsep.join(
        (str(TARGET_VLLM_ROOT), str(REPO_ROOT))
    )
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["VLLM_CACHE_ROOT"] = str(cache_root)
    env["VLLM_PLUGINS"] = "hcu,hcu_model,hcu_ops"
    return env


@pytest.mark.hcu
def test_official_registry_stdin_protocol_applies_adapter_and_one_linear(
    tmp_path: Path,
):
    def inspection_probe() -> dict[str, object]:
        import importlib
        import os
        import sys
        from pathlib import Path

        import vllm
        import vllm.model_executor as model_executor
        from vllm_hcu.patch import patch_report

        target_root = Path(os.environ["VLLM_TARGET_SOURCE_ROOT"]).resolve()
        target_file = Path(vllm.__file__).resolve()
        assert target_file.is_relative_to(target_root), (
            f"vllm resolved outside target root: {target_file} not under {target_root}"
        )

        parameter = importlib.import_module("vllm.model_executor.parameter")
        linear = importlib.import_module("vllm.model_executor.layers.linear")
        from vllm.model_executor.custom_op import op_registry

        return {
            "parameter_origin": parameter.__file__,
            "parameter_status": patch_report()["patches"][
                "runtime_method.base_linear_parameter"
            ]["status"],
            "parameter_marker": getattr(
                parameter, "_hcu_base_linear_parameter_patch_applied", False
            ),
            "base_identity": (
                model_executor.BasevLLMParameter is parameter.BasevLLMParameter
            ),
            "packed_identity": (
                model_executor.PackedvLLMParameter is parameter.PackedvLLMParameter
            ),
            "hcu_parameter_loaded": (
                "vllm_hcu.model_executor.parameter" in sys.modules
            ),
            "linear_origin": linear.__file__,
            "linear_alias": (
                linear is sys.modules.get("vllm_hcu.model_executor.layers.linear")
            ),
            "linear_registry": {
                "replicated_linear": (
                    op_registry["replicated_linear"] is linear.ReplicatedLinear
                ),
                "column_parallel_linear": (
                    op_registry["column_parallel_linear"]
                    is linear.ColumnParallelLinear
                ),
                "row_parallel_linear": (
                    op_registry["row_parallel_linear"] is linear.RowParallelLinear
                ),
            },
        }

    output_path = tmp_path / "registry-output.pkl"
    payload = cloudpickle.dumps((inspection_probe, str(output_path)))
    env = _clean_target_environment(tmp_path / "cache")
    result = subprocess.run(
        [sys.executable, "-m", "vllm.model_executor.models.registry"],
        input=payload,
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        timeout=240,
    )

    assert result.returncode == 0, result.stderr.decode(errors="replace")
    assert b"LateModuleReplacementError" not in result.stderr
    with output_path.open("rb") as output_file:
        probe = pickle.load(output_file)
    assert probe["parameter_status"] == "applied"
    assert probe["parameter_marker"] is True
    assert probe["base_identity"] is True
    assert probe["packed_identity"] is True
    assert probe["hcu_parameter_loaded"] is False
    assert probe["parameter_origin"].replace("\\", "/").endswith(
        "vllm/model_executor/parameter.py"
    )
    assert "vllm_hcu/" not in probe["parameter_origin"].replace("\\", "/")
    assert probe["linear_origin"].replace("\\", "/").endswith(
        "vllm_hcu/model_executor/layers/linear.py"
    )
    assert probe["linear_alias"] is True
    assert probe["linear_registry"] == {
        "replicated_linear": True,
        "column_parallel_linear": True,
        "row_parallel_linear": True,
    }


@pytest.mark.hcu
def test_qwen35_real_inspect_cache_miss_uses_official_registry_command(
    tmp_path: Path,
):
    env = _clean_target_environment(tmp_path / "cache")
    code = r'''
import importlib
import json
import os
import sys
from pathlib import Path

import vllm
registry = importlib.import_module("vllm.model_executor.models.registry")
from vllm.plugins import load_general_plugins
load_general_plugins()
load_general_plugins()

target_root = Path(os.environ["VLLM_TARGET_SOURCE_ROOT"]).resolve()
target_file = Path(vllm.__file__).resolve()
assert target_file.is_relative_to(target_root), (
    f"vllm resolved outside target root: {target_file} not under {target_root}"
)

registered = registry.ModelRegistry.models["Qwen3_5ForConditionalGeneration"]
model_info = registered.inspect_model_cls()
parameter = importlib.import_module("vllm.model_executor.parameter")
from vllm_hcu.patch import patch_report

print(json.dumps({
    "command": registry._SUBPROCESS_COMMAND,
    "expected_command": [
        sys.executable, "-m", "vllm.model_executor.models.registry"
    ],
    "is_hybrid": model_info.is_hybrid,
    "has_inner_state": model_info.has_inner_state,
    "parameter_status": patch_report()["patches"][
        "runtime_method.base_linear_parameter"
    ]["status"],
    "parameter_marker": getattr(
        parameter, "_hcu_base_linear_parameter_patch_applied", False
    ),
    "hcu_parameter_loaded": "vllm_hcu.model_executor.parameter" in sys.modules,
}))
'''
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert result.returncode == 0, result.stderr
    assert "LateModuleReplacementError" not in result.stderr
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload == {
        "command": payload["expected_command"],
        "expected_command": payload["expected_command"],
        "is_hybrid": True,
        "has_inner_state": False,
        "parameter_status": "applied",
        "parameter_marker": True,
        "hcu_parameter_loaded": False,
    }
