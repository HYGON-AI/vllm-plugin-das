from __future__ import annotations

import ast
import importlib.util
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_patch_module():
    patch_path = (
        ROOT
        / "vllm_hcu/patches/patch_kimi_k25_tokenizer_regex.py"
    )
    spec = importlib.util.spec_from_file_location("patch_kimi_k25_tokenizer_test",
                                                  patch_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _install_fake_modules():
    injected_module_names = [
        "vllm",
        "vllm.tokenizers",
        "vllm.tokenizers.hf",
    ]
    original_modules = {
        name: sys.modules.get(name)
        for name in injected_module_names
    }

    calls: list[dict[str, object]] = []

    class CachedHfTokenizer:
        @classmethod
        def from_pretrained(cls, path_or_repo_id, *args, **kwargs):
            calls.append(dict(kwargs))
            if kwargs.get("fix_mistral_regex") is True:
                raise AttributeError(
                    "'tokenizers.Tokenizer' object has no attribute "
                    "'backend_tokenizer'"
                )
            return {"path": path_or_repo_id, "kwargs": kwargs}

    vllm_module = types.ModuleType("vllm")
    tokenizers_module = types.ModuleType("vllm.tokenizers")
    hf_module = types.ModuleType("vllm.tokenizers.hf")
    hf_module.CachedHfTokenizer = CachedHfTokenizer
    tokenizers_module.hf = hf_module
    vllm_module.tokenizers = tokenizers_module

    sys.modules["vllm"] = vllm_module
    sys.modules["vllm.tokenizers"] = tokenizers_module
    sys.modules["vllm.tokenizers.hf"] = hf_module

    return {
        "restore": original_modules,
        "injected": injected_module_names,
        "calls": calls,
        "tokenizer_cls": CachedHfTokenizer,
    }


def test_patch_retries_without_fix_mistral_regex_for_backend_tokenizer_error() -> None:
    state = _install_fake_modules()
    try:
        patch_module = _load_patch_module()
        patch_module.patch_kimi_k25_tokenizer_regex()

        result = state["tokenizer_cls"].from_pretrained("/models/Kimi-K2.5")

        assert state["calls"] == [
            {"fix_mistral_regex": True},
            {},
        ]
        assert result["path"] == "/models/Kimi-K2.5"
        assert result["kwargs"] == {}
    finally:
        for name in state["injected"]:
            original_module = state["restore"][name]
            if original_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original_module


def test_patch_leaves_non_kimi_tokenizers_unchanged() -> None:
    state = _install_fake_modules()
    try:
        patch_module = _load_patch_module()
        patch_module.patch_kimi_k25_tokenizer_regex()

        result = state["tokenizer_cls"].from_pretrained("/models/OtherModel")

        assert state["calls"] == [{}]
        assert result["path"] == "/models/OtherModel"
        assert result["kwargs"] == {}
    finally:
        for name in state["injected"]:
            original_module = state["restore"][name]
            if original_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original_module


def test_patch_utils_registers_kimi_tokenizer_patch() -> None:
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

    assert "patch_kimi_k25_tokenizer_regex" in names
