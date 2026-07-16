# SPDX-License-Identifier: Apache-2.0
"""Runtime migration of HYV3 reasoning-token suffix support."""

from __future__ import annotations

import functools
import inspect
from types import ModuleType

from ._common import (
    PatchCompatibilityError,
    apply_once,
    load_exact_module,
    require_callable,
    require_positional_signature,
)

TARGET_MODULE = "vllm.reasoning.hy_v3_reasoning_parser"
PATCH_ID = "platform.core_fix.hy_v3_reasoning_parser"
TARGETS = (
    f"{TARGET_MODULE}.HYV3ReasoningParser.__init__",
    f"{TARGET_MODULE}.HYV3ReasoningParser.start_token",
    f"{TARGET_MODULE}.HYV3ReasoningParser.end_token",
)
_MARKER = "_vllm_hcu_token_suffix_patch_applied"


def _get_parser_class(parser_module: ModuleType) -> type:
    parser_class = getattr(parser_module, "HYV3ReasoningParser", None)
    if not isinstance(parser_class, type):
        raise PatchCompatibilityError(
            "required HCU patch target "
            "vllm.reasoning.hy_v3_reasoning_parser.HYV3ReasoningParser is missing"
        )
    return parser_class


def apply_to_module(module: ModuleType) -> bool:
    """Apply to an exact module from the import coordinator, without reporting."""

    parser_module = load_exact_module(TARGET_MODULE, module)
    parser_class = _get_parser_class(parser_module)
    if getattr(parser_class, _MARKER, False):
        return False

    original_init = require_callable(
        parser_class,
        "__init__",
        "vllm.reasoning.hy_v3_reasoning_parser.HYV3ReasoningParser.__init__",
    )
    require_positional_signature(
        original_init,
        "vllm.reasoning.hy_v3_reasoning_parser.HYV3ReasoningParser.__init__",
        ("self", "tokenizer"),
        var_positional="args",
        var_keyword="kwargs",
    )
    for property_name in ("start_token", "end_token"):
        descriptor = inspect.getattr_static(parser_class, property_name, None)
        if not isinstance(descriptor, property):
            raise PatchCompatibilityError(
                "required HCU patch target "
                "vllm.reasoning.hy_v3_reasoning_parser."
                f"HYV3ReasoningParser.{property_name} is not a property"
            )

    @functools.wraps(original_init)
    def hcu_init(self, tokenizer, *args, **kwargs):
        init_kwargs = getattr(tokenizer, "init_kwargs", None) or {}
        self.suffix = init_kwargs.get("token_suffix") or ""
        original_init(self, tokenizer, *args, **kwargs)

    def hcu_start_token(self) -> str:
        return f"<think{getattr(self, 'suffix', '')}>"

    def hcu_end_token(self) -> str:
        return f"</think{getattr(self, 'suffix', '')}>"

    setattr(parser_class, "_vllm_hcu_original_init", original_init)
    setattr(parser_class, "__init__", hcu_init)
    setattr(parser_class, "start_token", property(hcu_start_token))
    setattr(parser_class, "end_token", property(hcu_end_token))
    setattr(parser_class, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    """Make HYV3 use ``<think{token_suffix}>`` delimiter tokens."""

    parser_module = load_exact_module(TARGET_MODULE, module)
    parser_class = _get_parser_class(parser_module)

    return apply_once(
        patch_id=PATCH_ID,
        targets=TARGETS,
        marker_owner=parser_class,
        marker=_MARKER,
        callback=lambda: apply_to_module(parser_module),
    )


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
