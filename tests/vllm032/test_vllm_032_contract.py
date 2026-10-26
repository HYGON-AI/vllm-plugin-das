from __future__ import annotations

import ast
import os
from pathlib import Path


def target_root() -> Path:
    value = os.environ.get("VLLM_TARGET_ROOT")
    if not value:
        raise AssertionError("VLLM_TARGET_ROOT must name the official vLLM 0.32 source")
    root = Path(value).resolve()
    if not (root / "vllm" / "__init__.py").is_file():
        raise AssertionError(f"VLLM_TARGET_ROOT does not contain vLLM: {root}")
    return root


def find_class(tree: ast.Module, name: str) -> ast.ClassDef:
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == name
    )


def method_names(node: ast.ClassDef) -> set[str]:
    return {
        item.name
        for item in node.body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def test_official_vllm_032_exposes_the_silu_oot_contract() -> None:
    root = target_root()
    source = root / "vllm" / "model_executor" / "layers" / "activation.py"
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=os.fspath(source))
    silu = find_class(tree, "SiluAndMul")

    decorators = [ast.unparse(value) for value in silu.decorator_list]
    assert decorators == ["CustomOp.register('silu_and_mul')"]
    assert {"forward_native", "forward_cuda"}.issubset(method_names(silu))


def test_official_vllm_032_exposes_the_rmsnorm_oot_contract() -> None:
    root = target_root()
    source = root / "vllm" / "model_executor" / "layers" / "layernorm.py"
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=os.fspath(source))
    rms_norm = find_class(tree, "RMSNorm")

    decorators = [ast.unparse(value) for value in rms_norm.decorator_list]
    assert decorators == ["CustomOp.register('rms_norm')"]
    assert {"forward_native", "forward_cuda"}.issubset(method_names(rms_norm))
    initializer = next(
        item
        for item in rms_norm.body
        if isinstance(item, ast.FunctionDef) and item.name == "__init__"
    )
    parameter_names = [item.arg for item in initializer.args.args]
    parameter_names.extend(item.arg for item in initializer.args.kwonlyargs)
    assert parameter_names == [
        "self",
        "hidden_size",
        "eps",
        "var_hidden_size",
        "has_weight",
        "dtype",
    ]
    assigned_attributes = {
        target.attr
        for node in ast.walk(initializer)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        for target in (
            node.targets if isinstance(node, ast.Assign) else [node.target]
        )
        if isinstance(target, ast.Attribute)
        and isinstance(target.value, ast.Name)
        and target.value.id == "self"
    }
    assert {
        "has_weight",
        "hidden_size",
        "pass_weight",
        "pass_weight_add",
        "variance_epsilon",
        "variance_size_override",
        "weight",
    }.issubset(assigned_attributes)


def test_official_vllm_032_exposes_registration_interfaces() -> None:
    root = target_root()
    custom_op_path = root / "vllm" / "model_executor" / "custom_op.py"
    custom_op_tree = ast.parse(
        custom_op_path.read_text(encoding="utf-8"),
        filename=os.fspath(custom_op_path),
    )
    custom_op = find_class(custom_op_tree, "CustomOp")
    register_oot = next(
        item
        for item in custom_op.body
        if isinstance(item, ast.FunctionDef) and item.name == "register_oot"
    )
    assert any(
        isinstance(decorator, ast.Name) and decorator.id == "classmethod"
        for decorator in register_oot.decorator_list
    )

    torch_utils_path = root / "vllm" / "utils" / "torch_utils.py"
    torch_utils_tree = ast.parse(
        torch_utils_path.read_text(encoding="utf-8"),
        filename=os.fspath(torch_utils_path),
    )
    register = next(
        node
        for node in torch_utils_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "direct_register_custom_op"
    )
    parameter_names = [item.arg for item in register.args.args]
    assert parameter_names[:4] == [
        "op_name",
        "op_func",
        "mutates_args",
        "fake_impl",
    ]
