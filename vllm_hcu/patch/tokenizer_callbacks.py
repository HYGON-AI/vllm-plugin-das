# SPDX-License-Identifier: Apache-2.0
"""Strict post-import tokenizer compatibility callbacks."""

from __future__ import annotations

import functools
import inspect
import json
from pathlib import Path
from types import ModuleType
from typing import Any

from ._stage3_common import (
    Stage3CompatibilityError,
    require_callable,
    require_exact_module,
    require_type,
)
from .import_coordinator import (
    IMPORT_COORDINATOR,
    ExactImportCoordinator,
    ImportRegistration,
)

TARGET_MODULE = "vllm.tokenizers.hf"
KIMI_PATCH_ID = "post_import.tokenizer.kimi_k25_regex"
DEEPSEEK_PATCH_ID = "post_import.tokenizer.deepseek_r1_distill_llama_70b"
_KIMI_MARKER = "_hcu_kimi_k25_regex_patch_applied"
_DEEPSEEK_MARKER = "_hcu_deepseek_r1_distill_llama_70b_patch_applied"


def _require_from_pretrained(tokenizer_class: type) -> tuple[classmethod, Any]:
    descriptor = inspect.getattr_static(tokenizer_class, "from_pretrained", None)
    if not isinstance(descriptor, classmethod):
        raise Stage3CompatibilityError(
            "required runtime target "
            "vllm.tokenizers.hf.CachedHfTokenizer.from_pretrained "
            "is not a classmethod"
        )
    function = descriptor.__func__
    try:
        parameters = tuple(inspect.signature(function).parameters.values())
    except (TypeError, ValueError) as exc:
        raise Stage3CompatibilityError(
            "cannot inspect CachedHfTokenizer.from_pretrained"
        ) from exc

    expected = (
        ("cls", inspect.Parameter.POSITIONAL_OR_KEYWORD),
        ("path_or_repo_id", inspect.Parameter.POSITIONAL_OR_KEYWORD),
        ("args", inspect.Parameter.VAR_POSITIONAL),
        ("trust_remote_code", inspect.Parameter.KEYWORD_ONLY),
        ("revision", inspect.Parameter.KEYWORD_ONLY),
        ("download_dir", inspect.Parameter.KEYWORD_ONLY),
        ("kwargs", inspect.Parameter.VAR_KEYWORD),
    )
    actual = tuple((parameter.name, parameter.kind) for parameter in parameters)
    if actual != expected:
        raise Stage3CompatibilityError(
            "required runtime target CachedHfTokenizer.from_pretrained has "
            f"incompatible signature {inspect.signature(function)}"
        )
    # Keep the unbound function.  Capturing ``tokenizer_class.from_pretrained``
    # would bind the base class permanently and break subclass dispatch when
    # two wrappers are composed.
    return descriptor, function


def _is_kimi_k25(path_or_repo_id: object) -> bool:
    model_id = str(path_or_repo_id).lower()
    return "kimi-k2.5" in model_id or "kimi_k25" in model_id


def _is_backend_tokenizer_attr_error(error: BaseException) -> bool:
    return (
        isinstance(error, AttributeError)
        and "'tokenizers.Tokenizer' object has no attribute 'backend_tokenizer'"
        in str(error)
    )


def apply_kimi_k25_tokenizer(module: ModuleType) -> bool:
    """Install the Kimi K2.5 Mistral-regex compatibility wrapper."""

    tokenizer_module = require_exact_module(module, TARGET_MODULE)
    tokenizer_class = require_type(
        tokenizer_module,
        "CachedHfTokenizer",
        "vllm.tokenizers.hf.CachedHfTokenizer",
    )
    if getattr(tokenizer_class, _KIMI_MARKER, False):
        return False
    descriptor, original = _require_from_pretrained(tokenizer_class)

    @classmethod
    @functools.wraps(descriptor.__func__)
    def from_pretrained(cls, path_or_repo_id, *args, **kwargs):
        if not _is_kimi_k25(path_or_repo_id):
            return original(cls, path_or_repo_id, *args, **kwargs)

        patched_kwargs = dict(kwargs)
        patched_kwargs.setdefault("fix_mistral_regex", True)
        try:
            return original(cls, path_or_repo_id, *args, **patched_kwargs)
        except Exception as exc:
            if (
                patched_kwargs.get("fix_mistral_regex") is True
                and _is_backend_tokenizer_attr_error(exc)
            ):
                patched_kwargs.pop("fix_mistral_regex", None)
                return original(cls, path_or_repo_id, *args, **patched_kwargs)
            raise

    setattr(tokenizer_class, "_hcu_kimi_k25_original_from_pretrained", original)
    setattr(tokenizer_class, "from_pretrained", from_pretrained)
    setattr(tokenizer_class, _KIMI_MARKER, True)
    return True


def _is_deepseek_distill_llama_70b(path_or_repo_id: object) -> bool:
    return "deepseek-r1-distill-llama-70b" in str(path_or_repo_id).lower()


def _unwrap_added_token(token: object) -> object:
    if isinstance(token, dict):
        return token.get("content")
    return token


def _build_deepseek_tokenizer_kwargs(
    config: dict[str, object], tokenizer_file: Path
) -> dict[str, object]:
    kwargs: dict[str, object] = {"tokenizer_file": str(tokenizer_file)}
    for key in (
        "legacy",
        "model_max_length",
        "clean_up_tokenization_spaces",
        "add_bos_token",
        "add_eos_token",
        "use_default_system_prompt",
    ):
        value = config.get(key)
        if value is not None:
            kwargs[key] = value
    for key in ("bos_token", "eos_token", "pad_token", "unk_token"):
        value = _unwrap_added_token(config.get(key))
        if value is not None:
            kwargs[key] = value
    return kwargs


def _load_llama_tokenizer_fast():
    from transformers.models.llama.tokenization_llama import LlamaTokenizerFast

    return LlamaTokenizerFast


def apply_deepseek_distill_tokenizer(module: ModuleType) -> bool:
    """Install the local tokenizer.json path for DeepSeek R1 Llama 70B."""

    tokenizer_module = require_exact_module(module, TARGET_MODULE)
    tokenizer_class = require_type(
        tokenizer_module,
        "CachedHfTokenizer",
        "vllm.tokenizers.hf.CachedHfTokenizer",
    )
    get_cached_tokenizer = require_callable(
        tokenizer_module,
        "get_cached_tokenizer",
        "vllm.tokenizers.hf.get_cached_tokenizer",
    )
    if getattr(tokenizer_class, _DEEPSEEK_MARKER, False):
        return False
    descriptor, original = _require_from_pretrained(tokenizer_class)

    @classmethod
    @functools.wraps(descriptor.__func__)
    def from_pretrained(
        cls,
        path_or_repo_id,
        *args,
        trust_remote_code: bool = False,
        revision: str | None = None,
        download_dir: str | None = None,
        **kwargs,
    ):
        if not _is_deepseek_distill_llama_70b(path_or_repo_id):
            return original(
                cls,
                path_or_repo_id,
                *args,
                trust_remote_code=trust_remote_code,
                revision=revision,
                download_dir=download_dir,
                **kwargs,
            )

        model_path = Path(path_or_repo_id)
        tokenizer_file = model_path / "tokenizer.json"
        tokenizer_config_file = model_path / "tokenizer_config.json"
        if not tokenizer_file.is_file() or not tokenizer_config_file.is_file():
            return original(
                cls,
                path_or_repo_id,
                *args,
                trust_remote_code=trust_remote_code,
                revision=revision,
                download_dir=download_dir,
                **kwargs,
            )

        with tokenizer_config_file.open("r", encoding="utf-8") as file:
            tokenizer_config = json.load(file)
        if not isinstance(tokenizer_config, dict):
            raise Stage3CompatibilityError(
                f"{tokenizer_config_file} must contain a JSON object"
            )

        tokenizer_factory = _load_llama_tokenizer_fast()
        tokenizer = tokenizer_factory(
            **_build_deepseek_tokenizer_kwargs(tokenizer_config, tokenizer_file)
        )
        if "chat_template" in tokenizer_config:
            tokenizer.chat_template = tokenizer_config["chat_template"]
        tokenizer.name_or_path = str(model_path)
        return get_cached_tokenizer(tokenizer)

    setattr(
        tokenizer_class,
        "_hcu_deepseek_r1_distill_llama_70b_original_from_pretrained",
        original,
    )
    setattr(tokenizer_class, "from_pretrained", from_pretrained)
    setattr(tokenizer_class, _DEEPSEEK_MARKER, True)
    return True


def register_tokenizer_callbacks(
    coordinator: ExactImportCoordinator = IMPORT_COORDINATOR,
) -> tuple[ImportRegistration, ...]:
    """Register both wrappers in deterministic legacy-compatible order."""

    kimi = coordinator.register_callback(
        KIMI_PATCH_ID,
        TARGET_MODULE,
        apply_kimi_k25_tokenizer,
        targets="vllm.tokenizers.hf.CachedHfTokenizer.from_pretrained",
    )
    deepseek = coordinator.register_callback(
        DEEPSEEK_PATCH_ID,
        TARGET_MODULE,
        apply_deepseek_distill_tokenizer,
        targets=(
            "vllm.tokenizers.hf.CachedHfTokenizer.from_pretrained",
            "vllm.tokenizers.hf.get_cached_tokenizer",
        ),
    )
    return kimi, deepseek


__all__ = [
    "apply_deepseek_distill_tokenizer",
    "apply_kimi_k25_tokenizer",
    "register_tokenizer_callbacks",
]
