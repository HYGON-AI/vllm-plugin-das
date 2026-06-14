from __future__ import annotations

import ast
import importlib.util
import json
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_patch_module():
    patch_path = (
        ROOT
        / "vllm_hcu/patches/patch_deepseek_r1_distill_llama_70b_tokenizer.py"
    )
    spec = importlib.util.spec_from_file_location(
        "patch_deepseek_r1_distill_llama_70b_tokenizer_test",
        patch_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _install_fake_modules(tokenizer_ctor_calls: list[dict[str, object]]):
    injected_module_names = [
        "transformers",
        "transformers.models",
        "transformers.models.llama",
        "transformers.models.llama.tokenization_llama",
        "vllm",
        "vllm.tokenizers",
        "vllm.tokenizers.hf",
    ]
    original_modules = {
        name: sys.modules.get(name)
        for name in injected_module_names
    }

    original_calls: list[dict[str, object]] = []

    class CachedHfTokenizer:
        @classmethod
        def from_pretrained(cls, path_or_repo_id, *args, **kwargs):
            original_calls.append({
                "path": path_or_repo_id,
                "kwargs": dict(kwargs),
            })
            return {"path": path_or_repo_id, "kwargs": dict(kwargs)}

    class LlamaTokenizerFast:
        def __init__(self, **kwargs):
            tokenizer_ctor_calls.append(dict(kwargs))
            self.init_kwargs = dict(kwargs)
            self.chat_template = None
            self.name_or_path = None

    def get_cached_tokenizer(tokenizer):
        return {
            "cached": tokenizer,
            "chat_template": tokenizer.chat_template,
            "name_or_path": tokenizer.name_or_path,
            "init_kwargs": tokenizer.init_kwargs,
        }

    transformers_module = types.ModuleType("transformers")
    transformers_models_module = types.ModuleType("transformers.models")
    transformers_llama_module = types.ModuleType("transformers.models.llama")
    tokenization_llama_module = types.ModuleType(
        "transformers.models.llama.tokenization_llama")
    tokenization_llama_module.LlamaTokenizerFast = LlamaTokenizerFast

    transformers_module.models = transformers_models_module
    transformers_models_module.llama = transformers_llama_module
    transformers_llama_module.tokenization_llama = tokenization_llama_module

    vllm_module = types.ModuleType("vllm")
    tokenizers_module = types.ModuleType("vllm.tokenizers")
    hf_module = types.ModuleType("vllm.tokenizers.hf")
    hf_module.CachedHfTokenizer = CachedHfTokenizer
    hf_module.get_cached_tokenizer = get_cached_tokenizer
    tokenizers_module.hf = hf_module
    vllm_module.tokenizers = tokenizers_module

    sys.modules["transformers"] = transformers_module
    sys.modules["transformers.models"] = transformers_models_module
    sys.modules["transformers.models.llama"] = transformers_llama_module
    sys.modules[
        "transformers.models.llama.tokenization_llama"
    ] = tokenization_llama_module
    sys.modules["vllm"] = vllm_module
    sys.modules["vllm.tokenizers"] = tokenizers_module
    sys.modules["vllm.tokenizers.hf"] = hf_module

    return {
        "restore": original_modules,
        "injected": injected_module_names,
        "original_calls": original_calls,
        "tokenizer_cls": CachedHfTokenizer,
    }


def test_patch_uses_tokenizer_json_for_target_model(tmp_path: Path) -> None:
    tokenizer_ctor_calls: list[dict[str, object]] = []
    state = _install_fake_modules(tokenizer_ctor_calls)
    model_dir = tmp_path / "DeepSeek-R1-Distill-Llama-70B"
    model_dir.mkdir()
    (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (model_dir / "tokenizer_config.json").write_text(
        json.dumps({
            "legacy": True,
            "model_max_length": 16384,
            "clean_up_tokenization_spaces": False,
            "add_bos_token": True,
            "add_eos_token": False,
            "bos_token": {
                "content": "<bos>",
            },
            "eos_token": {
                "content": "<eos>",
            },
            "pad_token": {
                "content": "<pad>",
            },
            "chat_template": "tmpl",
        }),
        encoding="utf-8",
    )

    try:
        patch_module = _load_patch_module()
        patch_module.patch_deepseek_r1_distill_llama_70b_tokenizer()

        result = state["tokenizer_cls"].from_pretrained(str(model_dir))

        assert state["original_calls"] == []
        assert tokenizer_ctor_calls == [{
            "tokenizer_file": str(model_dir / "tokenizer.json"),
            "legacy": True,
            "model_max_length": 16384,
            "clean_up_tokenization_spaces": False,
            "add_bos_token": True,
            "add_eos_token": False,
            "bos_token": "<bos>",
            "eos_token": "<eos>",
            "pad_token": "<pad>",
        }]
        assert result["chat_template"] == "tmpl"
        assert result["name_or_path"] == str(model_dir)
        assert result["init_kwargs"]["tokenizer_file"] == str(
            model_dir / "tokenizer.json")
    finally:
        for name in state["injected"]:
            original_module = state["restore"][name]
            if original_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original_module


def test_patch_leaves_non_target_models_unchanged(tmp_path: Path) -> None:
    tokenizer_ctor_calls: list[dict[str, object]] = []
    state = _install_fake_modules(tokenizer_ctor_calls)
    model_dir = tmp_path / "OtherModel"
    model_dir.mkdir()

    try:
        patch_module = _load_patch_module()
        patch_module.patch_deepseek_r1_distill_llama_70b_tokenizer()

        result = state["tokenizer_cls"].from_pretrained(
            str(model_dir),
            trust_remote_code=True,
        )

        assert tokenizer_ctor_calls == []
        assert state["original_calls"] == [{
            "path": str(model_dir),
            "kwargs": {
                "trust_remote_code": True,
                "revision": None,
                "download_dir": None,
            },
        }]
        assert result["path"] == str(model_dir)
    finally:
        for name in state["injected"]:
            original_module = state["restore"][name]
            if original_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original_module


def test_patch_utils_registers_deepseek_tokenizer_patch() -> None:
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

    assert "patch_deepseek_r1_distill_llama_70b_tokenizer" in names
