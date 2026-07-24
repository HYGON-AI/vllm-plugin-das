# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import builtins
import importlib.abc
import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

import pytest

import vllm_hcu.patch.import_coordinator as coordinator_module
from vllm_hcu.patch.import_coordinator import (
    ExactImportCoordinator,
    LateModuleReplacementError,
)
from vllm_hcu.patch.module_exchange import (
    _validate_hcu_replacement_path,
    module_exchange_names,
    register_all_module_exchanges,
    register_modular_kernel_exchange,
)
from vllm_hcu.patch.runtime_state import PatchRegistry, PatchStatus


REPO_ROOT = Path(__file__).resolve().parents[2]
TARGET_VLLM_ROOT = Path(
    os.environ.get("VLLM_V0251_SOURCE_ROOT", REPO_ROOT.parent / "vllm_0251")
).resolve()


def _remove_exchange_targets(monkeypatch: pytest.MonkeyPatch) -> None:
    for canonical, replacement in module_exchange_names():
        monkeypatch.delitem(sys.modules, canonical, raising=False)
        monkeypatch.delitem(sys.modules, replacement, raising=False)


def test_all_exchange_registration_is_exact_lazy_and_idempotent(
    monkeypatch: pytest.MonkeyPatch,
):
    _remove_exchange_targets(monkeypatch)
    original_import = builtins.__import__
    registry = PatchRegistry()
    coordinator = ExactImportCoordinator(registry=registry)
    replacements_before = {
        replacement
        for _, replacement in module_exchange_names()
        if replacement in sys.modules
    }

    first = register_all_module_exchanges(coordinator)
    second = register_all_module_exchanges(coordinator)

    assert len(first) == len(second) == 7
    assert len(coordinator.registrations()) == 7
    assert all(item.status == PatchStatus.ARMED.value for item in first)
    assert builtins.__import__ is original_import
    assert replacements_before == set()
    assert not any(
        replacement in sys.modules for _, replacement in module_exchange_names()
    )

    # Exact lookup only: neither ancestors, descendants, nor unrelated names
    # produce a coordinator spec or resolve any replacement.
    assert coordinator.find_spec("vllm.model_executor.layers", None) is None
    assert coordinator.find_spec(
        "vllm.model_executor.layers.linear.unrelated_child", None
    ) is None
    assert coordinator.find_spec("unrelated_hcu_module", None) is None


def test_exchange_inventory_arms_dependencies_before_canonical_consumers():
    order = {
        canonical: index
        for index, (canonical, _) in enumerate(module_exchange_names())
    }
    dependencies = {
        "vllm.model_executor.layers.fused_moe.deep_gemm_utils": {
            "vllm.model_executor.layers.fused_moe.modular_kernel",
        },
        "vllm.model_executor.layers.fused_moe.experts.deep_gemm_moe": {
            "vllm.model_executor.layers.fused_moe.modular_kernel",
            "vllm.model_executor.layers.fused_moe.deep_gemm_utils",
        },
    }
    for consumer, required in dependencies.items():
        assert all(order[dependency] < order[consumer] for dependency in required)

    assert "vllm.model_executor.parameter" not in order


def test_deep_gemm_replacement_preserves_v0251_warmup_helper():
    repo = Path(__file__).resolve().parents[2]
    replacement = repo / (
        "vllm_hcu/model_executor/layers/fused_moe/deep_gemm_utils.py"
    )
    tree = ast.parse(replacement.read_text(encoding="utf-8"))
    functions = {
        node.name for node in tree.body if isinstance(node, ast.FunctionDef)
    }

    assert "compute_aligned_M_and_alignment" in functions


@pytest.mark.parametrize(
    ("target_relative", "replacement_relative"),
    (
        (
            "vllm/model_executor/layers/fused_moe/deep_gemm_utils.py",
            "vllm_hcu/model_executor/layers/fused_moe/deep_gemm_utils.py",
        ),
        (
            "vllm/model_executor/layers/fused_moe/experts/deep_gemm_moe.py",
            "vllm_hcu/model_executor/layers/fused_moe/experts/deep_gemm_moe.py",
        ),
        (
            "vllm/model_executor/layers/fused_moe/experts/"
            "batched_deep_gemm_moe.py",
            "vllm_hcu/model_executor/layers/fused_moe/experts/"
            "batched_deep_gemm_moe.py",
        ),
    ),
)
def test_deep_gemm_replacements_preserve_v0251_surface_and_signatures(
    target_relative: str,
    replacement_relative: str,
):
    """Always-on whole-module replacements must keep the exact v0.25.1 API."""

    target_tree = ast.parse(
        (TARGET_VLLM_ROOT / target_relative).read_text(encoding="utf-8")
    )
    replacement_tree = ast.parse(
        (REPO_ROOT / replacement_relative).read_text(encoding="utf-8")
    )

    def definitions(tree: ast.Module) -> list[ast.AST]:
        return [
            node
            for node in tree.body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        ]

    target_definitions = definitions(target_tree)
    replacement_definitions = definitions(replacement_tree)
    assert [node.name for node in replacement_definitions] == [
        node.name for node in target_definitions
    ]

    for target_node, replacement_node in zip(
        target_definitions, replacement_definitions, strict=True
    ):
        assert type(replacement_node) is type(target_node), target_node.name
        if isinstance(target_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            assert isinstance(
                replacement_node, (ast.FunctionDef, ast.AsyncFunctionDef)
            )
            assert ast.dump(
                replacement_node.args, include_attributes=False
            ) == ast.dump(target_node.args, include_attributes=False), target_node.name
            continue

        assert isinstance(target_node, ast.ClassDef)
        assert isinstance(replacement_node, ast.ClassDef)
        target_methods = [
            node
            for node in target_node.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        replacement_methods = [
            node
            for node in replacement_node.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        assert [node.name for node in replacement_methods] == [
            node.name for node in target_methods
        ], target_node.name
        for target_method, replacement_method in zip(
            target_methods, replacement_methods, strict=True
        ):
            assert ast.dump(
                replacement_method.args, include_attributes=False
            ) == ast.dump(
                target_method.args, include_attributes=False
            ), f"{target_node.name}.{target_method.name}"


def test_deep_gemm_replacements_keep_target_features_and_scoped_hcu_deltas():
    experts_path = REPO_ROOT / (
        "vllm_hcu/model_executor/layers/fused_moe/experts/deep_gemm_moe.py"
    )
    batched_path = REPO_ROOT / (
        "vllm_hcu/model_executor/layers/fused_moe/experts/"
        "batched_deep_gemm_moe.py"
    )
    utils_path = REPO_ROOT / (
        "vllm_hcu/model_executor/layers/fused_moe/deep_gemm_utils.py"
    )
    call_sites_path = REPO_ROOT / (
        "vllm_hcu/model_executor/layers/quantization/compressed_tensors/"
        "compressed_tensors_moe_marlin.py"
    )

    experts_source = experts_path.read_text(encoding="utf-8")
    batched_source = batched_path.read_text(encoding="utf-8")
    utils_source = utils_path.read_text(encoding="utf-8")
    for target_feature in (
        "kMxfp8Static",
        "SWIGLUOAI_UNINTERLEAVE",
        "mk_alignment_scope",
        "is_device_capability_family(120)",
    ):
        assert target_feature in experts_source
    assert "is_device_capability_family(120)" in batched_source

    # DCU deltas are private/conditional and do not widen public constructors
    # or apply signatures.
    assert {"_hcu_logical_n", "_hcu_logical_k"} <= {
        node.attr
        for tree in (ast.parse(experts_source), ast.parse(batched_source))
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
    }
    assert "use_nn_moe" not in experts_source
    assert "use_nn_moe" not in batched_source
    assert "m_grouped_w8a8_gemm_nt_contig_asm" in experts_source
    assert "m_grouped_w8a8_gemm_nt_masked" in batched_source
    assert "VLLM_HCU_USE_LIGHTOP_EP_SCATTER" in utils_source
    assert "_HCU_TOKEN_ALIGNMENT = 256" in utils_source

    call_sites = ast.parse(call_sites_path.read_text(encoding="utf-8"))
    deep_gemm_calls = [
        node
        for node in ast.walk(call_sites)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"DeepGemmExperts", "BatchedDeepGemmExperts"}
    ]
    assert deep_gemm_calls
    assert all(
        {keyword.arg for keyword in call.keywords}.isdisjoint({"N", "K"})
        for call in deep_gemm_calls
    )
    assert all(
        keyword.arg != "use_nn_moe"
        for node in ast.walk(call_sites)
        if isinstance(node, ast.Call)
        for keyword in node.keywords
    )


def test_linear_replacement_preserves_v0251_stacked_weight_loaders():
    repo = Path(__file__).resolve().parents[2]
    replacement = repo / "vllm_hcu/model_executor/layers/linear.py"
    tree = ast.parse(replacement.read_text(encoding="utf-8"))
    classes = {
        node.name: node for node in tree.body if isinstance(node, ast.ClassDef)
    }

    for class_name in ("MergedColumnParallelLinear", "QKVParallelLinear"):
        method = next(
            (
                node
                for node in classes[class_name].body
                if isinstance(node, ast.FunctionDef) and node.name == "load_weights"
            ),
            None,
        )
        assert method is not None
        assert any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value == "shard_id"
            for node in ast.walk(method)
        )
        assert any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "weight_loader"
            and any(
                isinstance(argument, ast.Name) and argument.id == "shard_id"
                for argument in node.args
            )
            for node in ast.walk(method)
        )


def test_linear_replacement_preserves_v0251_minimax_indexer_contract():
    repo = Path(__file__).resolve().parents[2]
    replacement = repo / "vllm_hcu/model_executor/layers/linear.py"
    tree = ast.parse(replacement.read_text(encoding="utf-8"))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "MinimaxM3QKVParallelLinearWithIndexer"
    )

    bases = {
        node.id for node in cls.bases if isinstance(node, ast.Name)
    }
    methods = {
        node.name for node in cls.body if isinstance(node, ast.FunctionDef)
    }
    assert "QKVParallelLinear" in bases
    assert {
        "validate_shard_id",
        "_get_shard_offset_mapping",
        "_get_shard_size_mapping",
        "weight_loader_v2",
        "weight_loader",
    } <= methods


def test_replacement_consumers_use_canonical_aliases():
    repo = Path(__file__).resolve().parents[2]
    contracts = {
        "vllm_hcu/model_executor/layers/fused_moe/experts/deep_gemm_moe.py": {
            "vllm.model_executor.layers.fused_moe.deep_gemm_utils",
        },
        "vllm_hcu/model_executor/layers/fused_moe/experts/"
        "dpsk_v4_deep_gemm_moe.py": {
            "vllm.model_executor.layers.fused_moe.deep_gemm_utils",
        },
        "vllm_hcu/models/deepseek_v2.py": {
            "vllm.model_executor.layers.sparse_attn_indexer",
        },
    }
    replacement_by_canonical = dict(module_exchange_names())
    for relative, expected in contracts.items():
        tree = ast.parse((repo / relative).read_text(encoding="utf-8"))
        imported = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        imported.update(
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        )
        assert expected <= imported
        assert not {
            replacement_by_canonical[canonical]
            for canonical in expected
            if canonical in replacement_by_canonical
        } & imported


def test_deepep_auto_does_not_restore_removed_expected_m_interface():
    path = REPO_ROOT / (
        "vllm_hcu/model_executor/layers/fused_moe/experts/"
        "dpsk_v4_deep_gemm_moe.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "DeepEPAutoDeepGemmExperts"
    )
    methods = {
        node.name for node in cls.body if isinstance(node, ast.FunctionDef)
    }
    attributes = {
        node.attr for node in ast.walk(cls) if isinstance(node, ast.Attribute)
    }
    assert {"set_expected_m", "get_expected_m"}.isdisjoint(methods)
    assert "expected_m" not in attributes


def test_v0251_native_mhc_contract_is_not_replaced():
    exchanges = dict(module_exchange_names())
    assert "vllm.model_executor.layers.mhc" not in exchanges

    tree = ast.parse(
        (TARGET_VLLM_ROOT / "vllm/model_executor/layers/mhc.py").read_text(
            encoding="utf-8"
        )
    )
    exports = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
    }
    exports.update(
        target.id
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    )
    assert {
        "HAS_AITER_MHC",
        "HAS_TILELANG_MHC",
        "HCHeadOp",
        "MHCFusedPostPreOp",
        "MHCPostOp",
        "MHCPreOp",
    } <= exports


def test_v0251_native_amd_deepseek_v4_owns_model_and_mtp_contracts():
    exchanges = dict(module_exchange_names())
    assert "vllm.v1.attention.backends.mla.sparse_swa" in exchanges
    assert {
        "vllm.model_executor.layers.deepseek_compressor",
        "vllm.model_executor.layers.deepseek_v4_attention",
        "vllm.v1.attention.ops.deepseek_v4_ops.fused_compress_quant_cache",
    }.isdisjoint(exchanges)

    hcu_registry = (
        REPO_ROOT / "vllm_hcu/models/__init__.py"
    ).read_text(encoding="utf-8")
    assert '"DeepseekV4ForCausalLM"' not in hcu_registry
    assert '"DeepSeekV4MTPModel"' not in hcu_registry

    target_registry_tree = ast.parse(
        (
            TARGET_VLLM_ROOT / "vllm/model_executor/models/registry.py"
        ).read_text(encoding="utf-8")
    )

    def registry_value(table_name: str, architecture: str) -> tuple[str, str]:
        assignment = next(
            node
            for node in target_registry_tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == table_name
                for target in node.targets
            )
        )
        assert isinstance(assignment.value, ast.Dict)
        for key, value in zip(
            assignment.value.keys, assignment.value.values, strict=True
        ):
            if isinstance(key, ast.Constant) and key.value == architecture:
                result = ast.literal_eval(value)
                assert isinstance(result, tuple)
                return result
        raise AssertionError(f"missing {architecture!r} in {table_name}")

    assert registry_value("_TEXT_GENERATION_MODELS", "DeepseekV4ForCausalLM") == (
        "vllm.models.deepseek_v4",
        "DeepseekV4ForCausalLM",
    )
    assert registry_value("_SPECULATIVE_DECODING_MODELS", "DeepSeekV4MTPModel") == (
        "vllm.models.deepseek_v4",
        "DeepSeekV4MTP",
    )

    native_entry = (
        TARGET_VLLM_ROOT / "vllm/models/deepseek_v4/__init__.py"
    ).read_text(encoding="utf-8")
    assert "if current_platform.is_rocm():" in native_entry
    assert "from .amd.model import DeepseekV4ForCausalLM" in native_entry
    assert "from .amd.mtp import DeepSeekV4MTP" in native_entry


def test_aiter_replacement_preserves_v0251_public_method_surface_and_ar_lifecycle():
    target_path = TARGET_VLLM_ROOT / "vllm/_aiter_ops.py"
    replacement_path = REPO_ROOT / (
        "vllm_hcu/model_executor/layers/fused_moe/aiter_ops.py"
    )

    def aiter_class(path: Path) -> ast.ClassDef:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        return next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "rocm_aiter_ops"
        )

    def public_methods(cls: ast.ClassDef) -> dict[str, ast.FunctionDef]:
        return {
            node.name: node
            for node in cls.body
            if isinstance(node, ast.FunctionDef)
            and not node.name.startswith("_")
        }

    target_class = aiter_class(target_path)
    replacement_class = aiter_class(replacement_path)
    target_methods = public_methods(target_class)
    replacement_methods = public_methods(replacement_class)

    assert replacement_methods.keys() == target_methods.keys()
    for name, target_method in target_methods.items():
        replacement_method = replacement_methods[name]
        assert ast.dump(replacement_method.args, include_attributes=False) == ast.dump(
            target_method.args, include_attributes=False
        ), name
        assert [ast.unparse(item) for item in replacement_method.decorator_list] == [
            ast.unparse(item) for item in target_method.decorator_list
        ], name

    restored_v0251_methods = {
        "fused_moe_supports_gate_mode",
        "get_fused_allreduce_rmsnorm_quant_per_group_op",
        "get_fused_allreduce_rmsnorm_quant_per_group_with_bf16_norm_op",
        "get_fused_mla_dual_rms_norm_per_token_quant_op",
        "get_fused_rms_gated_fp8_group_quant_op",
        "get_moe_dispatch_policy",
        "hc_head",
        "hipb_mm_fp8",
        "is_custom_all_reduce_enabled",
        "is_linear_hipbmm_enabled",
        "mhc_post",
        "mhc_pre",
        "shuffle_mxfp8_moe_weights",
    }
    assert restored_v0251_methods <= replacement_methods.keys()

    # v0.25.1 owns the communicator instance on the TP device communicator.
    # The legacy class-owned initialize/destroy lifecycle must not survive.
    assert {
        "initialize_aiter_allreduce",
        "destroy_aiter_allreduce",
        "get_aiter_allreduce_max_size",
    }.isdisjoint(replacement_methods)
    for method_name in (
        "refresh_env_variables",
        "is_custom_all_reduce_enabled",
        "get_aiter_allreduce",
    ):
        assert ast.dump(
            replacement_methods[method_name], include_attributes=False
        ) == ast.dump(target_methods[method_name], include_attributes=False)


def test_sparse_indexer_replacement_keeps_v0251_q_rope_quant_contract():
    repo = Path(__file__).resolve().parents[2]
    path = repo / "vllm_hcu/model_executor/layers/sparse_attn_indexer.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "fused_indexer_q_rope_quant"
    )
    assert [argument.arg for argument in function.args.args] == [
        "positions",
        "q",
        "cos_sin_cache",
        "weights",
        "softmax_scale",
        "head_scale",
        "is_neox",
    ]


@pytest.mark.parametrize(
    ("target_relative", "replacement_relative"),
    (
        (
            "vllm/model_executor/layers/sparse_attn_indexer.py",
            "vllm_hcu/model_executor/layers/sparse_attn_indexer.py",
        ),
        (
            "vllm/v1/attention/backends/mla/sparse_swa.py",
            "vllm_hcu/v1/attention/backends/mla/sparse_swa.py",
        ),
    ),
)
def test_sparse_replacements_preserve_v0251_definition_surface_and_signatures(
    target_relative: str,
    replacement_relative: str,
):
    """Whole-module replacements must carry the complete target definition API."""

    target_tree = ast.parse(
        (TARGET_VLLM_ROOT / target_relative).read_text(encoding="utf-8")
    )
    replacement_tree = ast.parse(
        (REPO_ROOT / replacement_relative).read_text(encoding="utf-8")
    )

    def definitions(tree: ast.Module) -> dict[str, ast.AST]:
        return {
            node.name: node
            for node in tree.body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        }

    target_definitions = definitions(target_tree)
    replacement_definitions = definitions(replacement_tree)
    assert target_definitions.keys() <= replacement_definitions.keys()
    hcu_owned_definitions = set(replacement_definitions).difference(
        target_definitions
    )
    if replacement_relative.endswith("sparse_attn_indexer.py"):
        assert hcu_owned_definitions == {"V32SparseAttnIndexer"}
    else:
        assert hcu_owned_definitions == set()

    for name, target_node in target_definitions.items():
        replacement_node = replacement_definitions[name]
        assert type(replacement_node) is type(target_node), name
        if isinstance(target_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            assert isinstance(
                replacement_node, (ast.FunctionDef, ast.AsyncFunctionDef)
            )
            assert ast.dump(
                replacement_node.args, include_attributes=False
            ) == ast.dump(target_node.args, include_attributes=False), name
            continue

        assert isinstance(target_node, ast.ClassDef)
        assert isinstance(replacement_node, ast.ClassDef)
        target_methods = {
            node.name: node
            for node in target_node.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        replacement_methods = {
            node.name: node
            for node in replacement_node.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        assert replacement_methods.keys() == target_methods.keys(), name
        for method_name, target_method in target_methods.items():
            replacement_method = replacement_methods[method_name]
            assert ast.dump(
                replacement_method.args, include_attributes=False
            ) == ast.dump(target_method.args, include_attributes=False), (
                f"{name}.{method_name}"
            )


def test_sparse_replacements_keep_reviewed_dcu_deltas():
    indexer_source = (
        REPO_ROOT / "vllm_hcu/model_executor/layers/sparse_attn_indexer.py"
    ).read_text(encoding="utf-8")
    assert "assert current_platform.is_cuda_alike()" in indexer_source
    assert (
        "from vllm_hcu.v1.attention.ops.rocm_aiter_mla_sparse import ("
        in indexer_source
    )
    assert "torch.ops.vllm.rocm_aiter_sparse_attn_indexer(" in indexer_source

    sparse_swa_source = (
        REPO_ROOT / "vllm_hcu/v1/attention/backends/mla/sparse_swa.py"
    ).read_text(encoding="utf-8")
    assert (
        "from vllm_hcu.v1.attention.ops.flashmla import "
        "FlashMLASchedMeta, get_mla_metadata" in sparse_swa_source
    )
    build_tile_scheduler = sparse_swa_source.split(
        "    def build_tile_scheduler(", 1
    )[1].split("    def _build_deepseek_v4_metadata(", 1)[0]
    assert "current_platform.is_rocm()" not in build_tile_scheduler
    assert "current_platform.is_xpu()" in build_tile_scheduler


def test_replacement_paths_are_validated_without_importing():
    for _, replacement in module_exchange_names():
        _validate_hcu_replacement_path(replacement)
    with pytest.raises(ModuleNotFoundError, match="has no source"):
        _validate_hcu_replacement_path("vllm_hcu.missing.stage3_module")
    with pytest.raises(ValueError, match="absolute vllm_hcu"):
        _validate_hcu_replacement_path("other_backend.module")


class _ExplodingLoader(importlib.abc.Loader):
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def create_module(self, spec):
        self.events.append("official-create")
        raise AssertionError("official modular_kernel must never execute")

    def exec_module(self, module):
        self.events.append("official-exec")
        raise AssertionError("official modular_kernel must never execute")


class _OfficialFinder(importlib.abc.MetaPathFinder):
    def __init__(self, target: str, events: list[str]) -> None:
        self.target = target
        self.events = events

    def find_spec(self, fullname, path=None, target=None):
        if fullname != self.target:
            return None
        self.events.append("official-find")
        return importlib.util.spec_from_loader(
            fullname, _ExplodingLoader(self.events), origin="official-test-module"
        )


def _fake_package(name: str) -> ModuleType:
    package = ModuleType(name)
    package.__package__ = name
    package.__path__ = []
    package.__spec__ = importlib.util.spec_from_loader(
        name, loader=None, is_package=True
    )
    return package


def test_modular_kernel_canonical_import_never_executes_official_module(
    monkeypatch: pytest.MonkeyPatch,
):
    canonical, replacement_name = module_exchange_names()[0]
    assert canonical == "vllm.model_executor.layers.fused_moe.modular_kernel"
    for package_name in (
        "vllm",
        "vllm.model_executor",
        "vllm.model_executor.layers",
        "vllm.model_executor.layers.fused_moe",
    ):
        monkeypatch.setitem(sys.modules, package_name, _fake_package(package_name))
    monkeypatch.delitem(sys.modules, canonical, raising=False)
    monkeypatch.delitem(sys.modules, replacement_name, raising=False)

    registry = PatchRegistry()
    coordinator = ExactImportCoordinator(registry=registry)
    register_modular_kernel_exchange(coordinator)
    coordinator.install()
    events: list[str] = []
    official_finder = _OfficialFinder(canonical, events)
    sys.meta_path.insert(1, official_finder)

    replacement = ModuleType(replacement_name)
    replacement_imports: list[str] = []
    original_import_module = coordinator_module.importlib.import_module

    def import_replacement(name: str, package: str | None = None):
        if name == replacement_name:
            replacement_imports.append(name)
            return replacement
        return original_import_module(name, package)

    monkeypatch.setattr(
        coordinator_module.importlib, "import_module", import_replacement
    )
    original_builtin_import = builtins.__import__
    try:
        loaded = builtins.__import__(canonical, fromlist=["*"])
    finally:
        while official_finder in sys.meta_path:
            sys.meta_path.remove(official_finder)
        coordinator.reset_for_tests()

    assert loaded is replacement
    assert replacement_imports == [replacement_name]
    assert events == []
    assert builtins.__import__ is original_builtin_import


def test_modular_kernel_strict_late_policy_keeps_official_and_hcu_exclusive(
    monkeypatch: pytest.MonkeyPatch,
):
    canonical, replacement = module_exchange_names()[0]
    official_module = ModuleType(canonical)
    monkeypatch.setitem(sys.modules, canonical, official_module)
    monkeypatch.delitem(sys.modules, replacement, raising=False)
    registry = PatchRegistry()
    coordinator = ExactImportCoordinator(registry=registry)

    with pytest.raises(LateModuleReplacementError, match="already imported"):
        register_modular_kernel_exchange(coordinator)

    assert sys.modules[canonical] is official_module
    assert replacement not in sys.modules
    record = registry.get("module_exchange.modular_kernel")
    assert record is not None and record.status is PatchStatus.FAILED
