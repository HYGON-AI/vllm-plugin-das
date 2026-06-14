from __future__ import annotations

import json
from pathlib import Path


def _is_target_model(path_or_repo_id: object) -> bool:
    return "deepseek-r1-distill-llama-70b" in str(path_or_repo_id).lower()


def _unwrap_added_token(token: object) -> object:
    if isinstance(token, dict):
        return token.get("content")
    return token


def _build_tokenizer_kwargs(config: dict[str, object],
                            tokenizer_file: Path) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "tokenizer_file": str(tokenizer_file),
    }

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


def patch_deepseek_r1_distill_llama_70b_tokenizer() -> None:
    try:
        from vllm.tokenizers import hf as vllm_tokenizer_hf
        from transformers.models.llama.tokenization_llama import (
            LlamaTokenizerFast,
        )
    except Exception:
        return

    tokenizer_cls = getattr(vllm_tokenizer_hf, "CachedHfTokenizer", None)
    get_cached_tokenizer = getattr(vllm_tokenizer_hf, "get_cached_tokenizer",
                                   None)
    if tokenizer_cls is None:
        return
    if get_cached_tokenizer is None:
        return
    if getattr(tokenizer_cls, "_hcu_deepseek_r1_distill_llama_70b_patch_applied",
               False):
        return

    original_from_pretrained = tokenizer_cls.from_pretrained

    def from_pretrained(cls,
                        path_or_repo_id,
                        *args,
                        trust_remote_code: bool = False,
                        revision: str | None = None,
                        download_dir: str | None = None,
                        **kwargs):
        if not _is_target_model(path_or_repo_id):
            return original_from_pretrained(
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
            return original_from_pretrained(
                path_or_repo_id,
                *args,
                trust_remote_code=trust_remote_code,
                revision=revision,
                download_dir=download_dir,
                **kwargs,
            )

        with tokenizer_config_file.open("r", encoding="utf-8") as fh:
            tokenizer_config = json.load(fh)

        tokenizer = LlamaTokenizerFast(
            **_build_tokenizer_kwargs(tokenizer_config, tokenizer_file))
        if "chat_template" in tokenizer_config:
            tokenizer.chat_template = tokenizer_config["chat_template"]
        tokenizer.name_or_path = str(model_path)
        return get_cached_tokenizer(tokenizer)

    tokenizer_cls.from_pretrained = classmethod(from_pretrained)
    tokenizer_cls._hcu_deepseek_r1_distill_llama_70b_patch_applied = True
