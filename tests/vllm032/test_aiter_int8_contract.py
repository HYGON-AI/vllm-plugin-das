from __future__ import annotations

import ast
import os
from pathlib import Path


def target_root() -> Path:
    value = os.environ.get("VLLM_TARGET_ROOT")
    if not value:
        raise AssertionError("VLLM_TARGET_ROOT must name official vLLM 0.32 source")
    root = Path(value).resolve()
    if not (root / "vllm" / "__init__.py").is_file():
        raise AssertionError(f"VLLM_TARGET_ROOT does not contain vLLM: {root}")
    return root


def function_parameters(node: ast.FunctionDef) -> tuple[str, ...]:
    return tuple(argument.arg for argument in (*node.args.args, *node.args.kwonlyargs))


def top_level_functions(tree: ast.Module) -> dict[str, ast.FunctionDef]:
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }


def test_official_int8_oracle_exposes_the_guarded_v032_contract() -> None:
    source = (
        target_root()
        / "vllm/model_executor/layers/fused_moe/oracle/int8.py"
    )
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=os.fspath(source))
    functions = top_level_functions(tree)

    assert function_parameters(functions["backend_to_kernel_cls"]) == ("backend",)
    assert function_parameters(functions["map_int8_backend"]) == (
        "runner_backend",
    )
    assert function_parameters(functions["select_int8_moe_backend"]) == (
        "config",
        "weight_key",
        "activation_key",
    )
    assert function_parameters(functions["convert_to_int8_moe_kernel_format"]) == (
        "int8_backend",
        "w13",
        "w2",
        "layer",
        "w13_scale",
    )
    assert function_parameters(functions["make_int8_moe_quant_config"]) == (
        "int8_backend",
        "w1_scale",
        "w2_scale",
        "a1_scale",
        "a2_scale",
        "w1_bias",
        "w2_bias",
        "per_act_token_quant",
        "layer",
    )
    assert function_parameters(functions["make_int8_moe_kernel"]) == (
        "int8_backend",
        "moe_quant_config",
        "moe_config",
        "experts_cls",
        "routing_tables",
    )


def test_official_aiter_expert_exposes_the_v032_apply_contract() -> None:
    source = (
        target_root()
        / "vllm/model_executor/layers/fused_moe/experts/rocm_aiter_moe.py"
    )
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=os.fspath(source))
    experts = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "AiterExperts"
    )
    methods = {
        node.name: node
        for node in experts.body
        if isinstance(node, ast.FunctionDef)
    }

    assert function_parameters(methods["_supports_quant_scheme"]) == (
        "weight_key",
        "activation_key",
    )
    assert function_parameters(methods["apply"]) == (
        "self",
        "output",
        "hidden_states",
        "w1",
        "w2",
        "topk_weights",
        "topk_ids",
        "activation",
        "global_num_experts",
        "expert_map",
        "a1q_scale",
        "a2_scale",
        "workspace13",
        "workspace2",
        "expert_tokens_meta",
        "apply_router_weight_on_input",
    )


def test_v032_int8_consumers_freeze_oracle_aliases_at_import_time() -> None:
    relative_paths = (
        "vllm/model_executor/layers/quantization/compressed_tensors/"
        "compressed_tensors_moe/compressed_tensors_moe_w8a8_int8.py",
        "vllm/model_executor/layers/quantization/online/int8.py",
        "vllm/model_executor/layers/quantization/quark/quark_moe.py",
    )
    required = {
        "convert_to_int8_moe_kernel_format",
        "make_int8_moe_kernel",
        "make_int8_moe_quant_config",
        "select_int8_moe_backend",
    }

    for relative in relative_paths:
        source = target_root() / relative
        tree = ast.parse(
            source.read_text(encoding="utf-8"),
            filename=os.fspath(source),
        )
        imported = {
            alias.name
            for node in tree.body
            if isinstance(node, ast.ImportFrom)
            and node.module
            == "vllm.model_executor.layers.fused_moe.oracle.int8"
            for alias in node.names
        }
        assert required.issubset(imported), relative
