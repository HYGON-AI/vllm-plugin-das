# SPDX-License-Identifier: Apache-2.0
"""Runtime migration of HYV3 schema unions and token suffix support."""

from __future__ import annotations

import functools
from types import ModuleType

from ._common import (
    PatchCompatibilityError,
    apply_once,
    load_exact_module,
    require_callable,
    require_positional_signature,
)

TARGET_MODULE = "vllm.tool_parsers.hy_v3_tool_parser"
PATCH_ID = "platform.core_fix.hy_v3_tool_parser"
TARGETS = (
    f"{TARGET_MODULE}.HYV3ToolParser._get_schema_options",
    f"{TARGET_MODULE}.HYV3ToolParser.__init__",
)
_MARKER = "_vllm_hcu_schema_suffix_patch_applied"

_TOKENS = (
    "<tool_calls>",
    "</tool_calls>",
    "<tool_call>",
    "</tool_call>",
    "<tool_sep>",
    "<arg_key>",
    "</arg_key>",
    "<arg_value>",
    "</arg_value>",
)


def _with_suffix(token: str, suffix: str) -> str:
    return f"{token[:-1]}{suffix}>"


class _TokenizerWithLegacyAliases:
    """Expose suffixed token IDs under old names during upstream init only."""

    def __init__(self, tokenizer: object, suffix: str) -> None:
        self._tokenizer = tokenizer
        self._suffix = suffix

    def get_vocab(self) -> dict[str, int]:
        get_vocab = getattr(self._tokenizer, "get_vocab", None)
        if not callable(get_vocab):
            raise PatchCompatibilityError("HYV3 tokenizer does not expose get_vocab()")
        vocabulary = dict(get_vocab())
        for token in _TOKENS:
            suffixed = _with_suffix(token, self._suffix)
            if suffixed in vocabulary:
                vocabulary[token] = vocabulary[suffixed]
        return vocabulary

    def __getattr__(self, name: str):
        return getattr(self._tokenizer, name)


def apply_to_module(module: ModuleType) -> bool:
    """Apply to an exact module from the import coordinator, without reporting."""

    parser_module = load_exact_module(TARGET_MODULE, module)
    parser_class = getattr(parser_module, "HYV3ToolParser", None)
    if not isinstance(parser_class, type):
        raise PatchCompatibilityError(
            "required HCU patch target "
            "vllm.tool_parsers.hy_v3_tool_parser.HYV3ToolParser is missing"
        )
    if getattr(parser_class, _MARKER, False):
        return False

    def install() -> None:
        original_schema_options = require_callable(
            parser_class,
            "_get_schema_options",
            "vllm.tool_parsers.hy_v3_tool_parser."
            "HYV3ToolParser._get_schema_options",
        )
        require_positional_signature(
            original_schema_options,
            "vllm.tool_parsers.hy_v3_tool_parser."
            "HYV3ToolParser._get_schema_options",
            ("arg_schema",),
        )
        original_init = require_callable(
            parser_class,
            "__init__",
            "vllm.tool_parsers.hy_v3_tool_parser.HYV3ToolParser.__init__",
        )
        require_positional_signature(
            original_init,
            "vllm.tool_parsers.hy_v3_tool_parser.HYV3ToolParser.__init__",
            ("self", "tokenizer", "tools"),
        )
        regex_module = getattr(parser_module, "re", None)
        if not callable(getattr(regex_module, "compile", None)) or not hasattr(
            regex_module, "DOTALL"
        ):
            raise PatchCompatibilityError(
                "required HCU patch target vllm.tool_parsers.hy_v3_tool_parser.re "
                "is incompatible"
            )

        @staticmethod
        def hcu_get_schema_options(arg_schema: dict) -> list[dict]:
            if "type" in arg_schema and isinstance(arg_schema["type"], list):
                return [{"type": value} for value in arg_schema["type"]]
            return original_schema_options(arg_schema)

        @functools.wraps(original_init)
        def hcu_init(self, tokenizer, tools=None):
            init_kwargs = getattr(tokenizer, "init_kwargs", None) or {}
            suffix = init_kwargs.get("token_suffix") or ""
            if not suffix:
                original_init(self, tokenizer, tools)
                self.suffix = ""
                return

            proxy = _TokenizerWithLegacyAliases(tokenizer, suffix)
            original_init(self, proxy, tools)
            # Restore the actual tokenizer and invalidate the cached proxy
            # vocabulary before assigning the suffixed parser vocabulary.
            self.model_tokenizer = tokenizer
            getattr(self, "__dict__", {}).pop("vocab", None)
            self.suffix = suffix

            self.tool_calls_start_token = _with_suffix("<tool_calls>", suffix)
            self.tool_calls_end_token = _with_suffix("</tool_calls>", suffix)
            self.tool_call_start_token = _with_suffix("<tool_call>", suffix)
            self.tool_call_end_token = _with_suffix("</tool_call>", suffix)
            self.tool_sep_token = _with_suffix("<tool_sep>", suffix)
            self.arg_key_start_token = _with_suffix("<arg_key>", suffix)
            self.arg_key_end_token = _with_suffix("</arg_key>", suffix)
            self.arg_value_start_token = _with_suffix("<arg_value>", suffix)
            self.arg_value_end_token = _with_suffix("</arg_value>", suffix)

            self.tool_call_regex = regex_module.compile(
                rf"{self.tool_call_start_token}(.*?){self.tool_sep_token}"
                rf"(.*?){self.tool_call_end_token}",
                regex_module.DOTALL,
            )
            self.tool_call_portion_regex = regex_module.compile(
                rf"{self.tool_call_start_token}(.*?){self.tool_sep_token}(.*)",
                regex_module.DOTALL,
            )
            self.func_args_regex = regex_module.compile(
                rf"{self.arg_key_start_token}(.*?){self.arg_key_end_token}\s*"
                rf"{self.arg_value_start_token}(.*?){self.arg_value_end_token}",
                regex_module.DOTALL,
            )

            self.tool_calls_start_token_id = self.vocab.get(
                self.tool_calls_start_token
            )
            self.tool_calls_end_token_id = self.vocab.get(self.tool_calls_end_token)
            self.tool_call_start_token_id = self.vocab.get(self.tool_call_start_token)
            self.tool_call_end_token_id = self.vocab.get(self.tool_call_end_token)
            if (
                self.tool_calls_start_token_id is None
                or self.tool_calls_end_token_id is None
            ):
                raise RuntimeError(
                    "HYV3 Tool parser could not locate tool call start/end "
                    "tokens in the tokenizer!"
                )

        setattr(
            parser_class,
            "_vllm_hcu_original_get_schema_options",
            original_schema_options,
        )
        setattr(parser_class, "_vllm_hcu_original_init", original_init)
        setattr(parser_class, "_get_schema_options", hcu_get_schema_options)
        setattr(parser_class, "__init__", hcu_init)
        setattr(parser_class, _MARKER, True)

    install()
    return True


def apply(module: ModuleType | None = None) -> bool:
    """Support JSON-schema type arrays and tokenizer-defined HYV3 suffixes."""

    parser_module = load_exact_module(TARGET_MODULE, module)
    parser_class = getattr(parser_module, "HYV3ToolParser", None)
    if not isinstance(parser_class, type):
        raise PatchCompatibilityError(
            "required HCU patch target "
            "vllm.tool_parsers.hy_v3_tool_parser.HYV3ToolParser is missing"
        )
    return apply_once(
        patch_id=PATCH_ID,
        targets=TARGETS,
        marker_owner=parser_class,
        marker=_MARKER,
        callback=lambda: apply_to_module(parser_module),
    )


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
