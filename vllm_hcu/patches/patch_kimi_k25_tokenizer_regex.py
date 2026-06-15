from __future__ import annotations


def _should_fix_mistral_regex(path_or_repo_id: object) -> bool:
    model_id = str(path_or_repo_id).lower()
    return "kimi-k2.5" in model_id or "kimi_k25" in model_id


def _is_backend_tokenizer_attr_error(exc: BaseException) -> bool:
    return (
        isinstance(exc, AttributeError)
        and "'tokenizers.Tokenizer' object has no attribute 'backend_tokenizer'"
        in str(exc)
    )


def patch_kimi_k25_tokenizer_regex() -> None:
    try:
        from vllm.tokenizers import hf as vllm_tokenizer_hf
    except Exception:
        return

    tokenizer_cls = getattr(vllm_tokenizer_hf, "CachedHfTokenizer", None)
    if tokenizer_cls is None:
        return
    if getattr(tokenizer_cls, "_hcu_kimi_k25_regex_patch_applied", False):
        return

    original_from_pretrained = tokenizer_cls.from_pretrained

    def from_pretrained(cls, path_or_repo_id, *args, **kwargs):
        if not _should_fix_mistral_regex(path_or_repo_id):
            return original_from_pretrained(path_or_repo_id, *args, **kwargs)

        patched_kwargs = dict(kwargs)
        patched_kwargs.setdefault("fix_mistral_regex", True)

        try:
            return original_from_pretrained(path_or_repo_id, *args, **patched_kwargs)
        except Exception as exc:
            if (
                patched_kwargs.get("fix_mistral_regex") is True
                and _is_backend_tokenizer_attr_error(exc)
            ):
                patched_kwargs.pop("fix_mistral_regex", None)
                return original_from_pretrained(
                    path_or_repo_id,
                    *args,
                    **patched_kwargs,
                )
            raise

    tokenizer_cls.from_pretrained = classmethod(from_pretrained)
    tokenizer_cls._hcu_kimi_k25_regex_patch_applied = True
