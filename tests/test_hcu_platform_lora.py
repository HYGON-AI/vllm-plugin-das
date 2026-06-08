import ast
from pathlib import Path


def test_hcu_platform_uses_gpu_punica_wrapper() -> None:
    source = Path("vllm_hcu/platforms/hcu.py").read_text(encoding="utf-8")
    module = ast.parse(source)

    hcu_platform = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "HCUPlatform"
    )
    method = next(
        node
        for node in hcu_platform.body
        if isinstance(node, ast.FunctionDef) and node.name == "get_punica_wrapper"
    )
    return_stmt = next(
        node for node in method.body if isinstance(node, ast.Return)
    )

    assert isinstance(return_stmt.value, ast.Constant)
    assert (
        return_stmt.value.value
        == "vllm.lora.punica_wrapper.punica_gpu.PunicaWrapperGPU"
    )
