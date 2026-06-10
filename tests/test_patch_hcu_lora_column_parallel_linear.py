from __future__ import annotations

import ast
import importlib.util
import sys
import types
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


class _LoRAConfig:
    def __init__(self, fully_sharded_loras: bool = False) -> None:
        self.fully_sharded_loras = fully_sharded_loras


def _load_patch_module():
    patch_path = (
        ROOT
        / "vllm_hcu/patches/patch_hcu_lora_column_parallel_linear.py"
    )
    spec = importlib.util.spec_from_file_location("patch_hcu_lora_test", patch_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _install_fake_modules():
    injected_module_names = [
        "vllm",
        "vllm.lora",
        "vllm.lora.layers",
        "vllm.lora.layers.column_parallel_linear",
        "vllm_hcu",
        "vllm_hcu.model_executor",
        "vllm_hcu.model_executor.layers",
        "vllm_hcu.model_executor.layers.linear",
    ]
    original_modules = {
        name: sys.modules.get(name)
        for name in injected_module_names
    }

    class OfficialColumnParallelLinear:
        pass

    class OfficialMergedColumnParallelLinear(OfficialColumnParallelLinear):
        def __init__(self, output_sizes, prefix="test", tp_size=1, tp_rank=0):
            self.output_sizes = output_sizes
            self.prefix = prefix
            self.tp_size = tp_size
            self.tp_rank = tp_rank

    class OfficialQKVParallelLinear(OfficialColumnParallelLinear):
        pass

    HCUColumnParallelLinear = type(
        "ColumnParallelLinear",
        (),
        {"__module__": "vllm_hcu.model_executor.layers.linear"},
    )

    def _init_hcu_merged(self, output_sizes, prefix="test", tp_size=1, tp_rank=0):
        self.output_sizes = output_sizes
        self.prefix = prefix
        self.tp_size = tp_size
        self.tp_rank = tp_rank

    HCUMergedColumnParallelLinear = type(
        "MergedColumnParallelLinear",
        (HCUColumnParallelLinear,),
        {
            "__module__": "vllm_hcu.model_executor.layers.linear",
            "__init__": _init_hcu_merged,
        },
    )
    HCUQKVParallelLinear = type(
        "QKVParallelLinear",
        (HCUColumnParallelLinear,),
        {"__module__": "vllm_hcu.model_executor.layers.linear"},
    )

    class ColumnParallelLinearWithLoRA:
        def __init__(self, base_layer):
            self.base_layer = base_layer
            self.is_merged_col_linear = False
            self.tp_size = getattr(base_layer, "tp_size", 1)
            self.tp_rank = getattr(base_layer, "tp_rank", 0)

        @classmethod
        def can_replace_layer(
            cls,
            source_layer,
            lora_config,
            packed_modules_list,
            model_config=None,
            *,
            decorate: bool = True,
        ):
            return False

    class MergedColumnParallelLinearWithLoRA(ColumnParallelLinearWithLoRA):
        def __init__(self, base_layer):
            super().__init__(base_layer)
            self.output_slices = tuple(base_layer.output_sizes)
            self.n_slices = len(self.output_slices)
            self.output_ids = (self.tp_rank,) * self.n_slices

        def slice_lora_b(self, lora_b):
            return lora_b

        @classmethod
        def can_replace_layer(
            cls,
            source_layer,
            lora_config,
            packed_modules_list,
            model_config=None,
            *,
            decorate: bool = True,
        ):
            return False

    class QKVParallelLinearWithLoRA(ColumnParallelLinearWithLoRA):
        @classmethod
        def can_replace_layer(
            cls,
            source_layer,
            lora_config,
            packed_modules_list,
            model_config=None,
            *,
            decorate: bool = True,
        ):
            return False

    class MergedQKVParallelLinearWithLoRA(MergedColumnParallelLinearWithLoRA):
        @classmethod
        def can_replace_layer(
            cls,
            source_layer,
            lora_config,
            packed_modules_list,
            model_config=None,
            *,
            decorate: bool = True,
        ):
            return False

    class MergedColumnParallelLinearVariableSliceWithLoRA(
        MergedColumnParallelLinearWithLoRA
    ):
        @classmethod
        def can_replace_layer(
            cls,
            source_layer,
            lora_config,
            packed_modules_list,
            model_config=None,
            *,
            decorate: bool = True,
        ):
            return False

    lora_module = types.ModuleType("vllm.lora.layers.column_parallel_linear")
    lora_module.ColumnParallelLinear = OfficialColumnParallelLinear
    lora_module.MergedColumnParallelLinear = OfficialMergedColumnParallelLinear
    lora_module.QKVParallelLinear = OfficialQKVParallelLinear
    lora_module.ColumnParallelLinearWithLoRA = ColumnParallelLinearWithLoRA
    lora_module.MergedColumnParallelLinearWithLoRA = MergedColumnParallelLinearWithLoRA
    lora_module.QKVParallelLinearWithLoRA = QKVParallelLinearWithLoRA
    lora_module.MergedQKVParallelLinearWithLoRA = MergedQKVParallelLinearWithLoRA
    lora_module.MergedColumnParallelLinearVariableSliceWithLoRA = (
        MergedColumnParallelLinearVariableSliceWithLoRA
    )

    hcu_linear_module = types.ModuleType("vllm_hcu.model_executor.layers.linear")
    hcu_linear_module.ColumnParallelLinear = HCUColumnParallelLinear
    hcu_linear_module.MergedColumnParallelLinear = HCUMergedColumnParallelLinear
    hcu_linear_module.QKVParallelLinear = HCUQKVParallelLinear

    vllm_module = types.ModuleType("vllm")
    vllm_lora_module = types.ModuleType("vllm.lora")
    vllm_lora_layers_module = types.ModuleType("vllm.lora.layers")
    vllm_lora_layers_module.column_parallel_linear = lora_module

    vllm_hcu_module = types.ModuleType("vllm_hcu")
    vllm_hcu_model_executor_module = types.ModuleType("vllm_hcu.model_executor")
    vllm_hcu_layers_module = types.ModuleType("vllm_hcu.model_executor.layers")
    vllm_hcu_layers_module.linear = hcu_linear_module

    sys.modules["vllm"] = vllm_module
    sys.modules["vllm.lora"] = vllm_lora_module
    sys.modules["vllm.lora.layers"] = vllm_lora_layers_module
    sys.modules["vllm.lora.layers.column_parallel_linear"] = lora_module
    sys.modules["vllm_hcu"] = vllm_hcu_module
    sys.modules["vllm_hcu.model_executor"] = vllm_hcu_model_executor_module
    sys.modules["vllm_hcu.model_executor.layers"] = vllm_hcu_layers_module
    sys.modules["vllm_hcu.model_executor.layers.linear"] = hcu_linear_module

    return {
        "restore": original_modules,
        "injected": injected_module_names,
        "lora_module": lora_module,
        "hcu_column": HCUColumnParallelLinear,
        "hcu_merged": HCUMergedColumnParallelLinear,
        "hcu_qkv": HCUQKVParallelLinear,
    }


def test_patch_accepts_hcu_linear_types() -> None:
    state = _install_fake_modules()
    try:
        patch_module = _load_patch_module()
        patch_module.patch_hcu_lora_column_parallel_linear()

        lora_module = state["lora_module"]
        hcu_column = state["hcu_column"]
        hcu_merged = state["hcu_merged"]
        hcu_qkv = state["hcu_qkv"]
        lora_config = _LoRAConfig()

        merged_wrapper = lora_module.ColumnParallelLinearWithLoRA(
            hcu_merged([8, 8])
        )
        assert merged_wrapper.is_merged_col_linear is True

        assert lora_module.ColumnParallelLinearWithLoRA.can_replace_layer(
            source_layer=hcu_column(),
            lora_config=lora_config,
            packed_modules_list=[],
            model_config=None,
        )
        assert lora_module.MergedColumnParallelLinearWithLoRA.can_replace_layer(
            source_layer=hcu_merged([8, 8]),
            lora_config=lora_config,
            packed_modules_list=["gate_proj", "up_proj"],
            model_config=None,
        )
        assert lora_module.QKVParallelLinearWithLoRA.can_replace_layer(
            source_layer=hcu_qkv(),
            lora_config=lora_config,
            packed_modules_list=["qkv_proj"],
            model_config=None,
        )
        assert lora_module.MergedColumnParallelLinearVariableSliceWithLoRA.can_replace_layer(
            source_layer=hcu_merged([4, 4, 4]),
            lora_config=lora_config,
            packed_modules_list=["a"],
            model_config=None,
        )
    finally:
        for name in state["injected"]:
            original_module = state["restore"][name]
            if original_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original_module


def test_patch_utils_registers_hcu_lora_patch() -> None:
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

    assert "patch_hcu_lora_column_parallel_linear" in names


def test_patch_file_does_not_import_hcu_linear_module() -> None:
    source = (
        ROOT / "vllm_hcu/patches/patch_hcu_lora_column_parallel_linear.py"
    ).read_text(encoding="utf-8")
    module = ast.parse(source)

    import_froms = [
        node.module
        for node in ast.walk(module)
        if isinstance(node, ast.ImportFrom)
    ]

    assert "vllm_hcu.model_executor.layers" not in import_froms


def test_patch_groups_qwen35_in_proj_qkvz_into_two_loras() -> None:
    state = _install_fake_modules()
    try:
        patch_module = _load_patch_module()
        patch_module.patch_hcu_lora_column_parallel_linear()

        lora_module = state["lora_module"]
        layer = state["hcu_merged"](
            [6, 4, 2, 8],
            prefix="model.layers.0.linear_attn.in_proj_qkvz",
            tp_size=2,
            tp_rank=1,
        )
        wrapped = lora_module.MergedColumnParallelLinearWithLoRA(layer)

        assert wrapped._hcu_grouped_qkvz_lora is True
        assert wrapped.n_slices == 2
        assert wrapped.output_slices == (6, 4)

        qkv_lora_b = torch.arange(12 * 3, dtype=torch.float32).reshape(12, 3)
        z_lora_b = torch.arange(8 * 3, dtype=torch.float32).reshape(8, 3)
        sliced_qkv, sliced_z = wrapped.slice_lora_b([qkv_lora_b, z_lora_b])

        expected_q = qkv_lora_b[3:6]
        expected_k = qkv_lora_b[8:10]
        expected_v = qkv_lora_b[11:12]
        assert torch.equal(
            sliced_qkv,
            torch.cat([expected_q, expected_k, expected_v], dim=0),
        )
        assert torch.equal(sliced_z, z_lora_b[4:8])
    finally:
        for name in state["injected"]:
            original_module = state["restore"][name]
            if original_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original_module
