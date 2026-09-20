# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Exact runtime registrations required for Kimi K3 chat rendering."""

from __future__ import annotations

import functools
import inspect
from types import ModuleType

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


MODEL_CONFIG_TARGET = "vllm.config.model"
TOKENIZER_REGISTRY_TARGET = "vllm.tokenizers.registry"
RENDERER_REGISTRY_TARGET = "vllm.renderers.registry"
REASONING_REGISTRY_TARGET = "vllm.reasoning.abs_reasoning_parsers"
DELEGATING_PARSER_TARGET = "vllm.parser.abstract_parser"
DELEGATING_PARSER_PATCH_ID = "post_import.kimi_k3.streaming_content"
MTP_ARCH_TARGET = "vllm.transformers_utils.model_arch_config_convertor"
MTP_ARCH_PATCH_ID = "post_import.kimi_k3.mtp_arch_config"
MODEL_CONFIG_PATCH_ID = "post_import.kimi_k3.model_config"
TOKENIZER_REGISTRY_PATCH_ID = "post_import.kimi_k3.tokenizer_registry"
RENDERER_REGISTRY_PATCH_ID = "post_import.kimi_k3.renderer_registry"
REASONING_REGISTRY_PATCH_ID = "post_import.kimi_k3.reasoning_registry"
_MODEL_CONFIG_MARKER = "_hcu_kimi_k3_tokenizer_mode_patch_applied"
_TOKENIZER_REGISTRY_MARKER = "_hcu_kimi_k3_tokenizer_registry_patch_applied"
_RENDERER_REGISTRY_MARKER = "_hcu_kimi_k3_renderer_registry_patch_applied"
_REASONING_REGISTRY_MARKER = "_hcu_kimi_k3_reasoning_registry_patch_applied"
_KIMI_K3_ARCH = "KimiK3ForConditionalGeneration"


def apply_kimi_k3_model_config(module: ModuleType) -> bool:
    """Select the K3 tokenizer mode only when the user left it as ``auto``."""

    target = require_exact_module(module, MODEL_CONFIG_TARGET)
    model_config = require_type(target, "ModelConfig", f"{MODEL_CONFIG_TARGET}.ModelConfig")
    if getattr(model_config, _MODEL_CONFIG_MARKER, False):
        return False
    original = require_callable(
        model_config, "__post_init__", f"{MODEL_CONFIG_TARGET}.ModelConfig.__post_init__"
    )
    try:
        signature = inspect.signature(original)
    except (TypeError, ValueError) as exc:
        raise Stage3CompatibilityError("cannot inspect ModelConfig.__post_init__") from exc
    parameters = tuple(signature.parameters.values())
    if not parameters or parameters[0].name != "self":
        raise Stage3CompatibilityError(
            "required runtime target ModelConfig.__post_init__ has incompatible "
            f"signature {signature}"
        )

    @functools.wraps(original)
    def post_init(self, *args, **kwargs) -> None:
        requested_auto = self.tokenizer_mode == "auto"
        original(self, *args, **kwargs)
        if requested_auto and getattr(self, "_architecture", None) == _KIMI_K3_ARCH:
            self.tokenizer_mode = "kimi_k3"

    setattr(model_config, "_hcu_kimi_k3_original_post_init", original)
    setattr(model_config, "__post_init__", post_init)
    setattr(model_config, _MODEL_CONFIG_MARKER, True)
    return True


def apply_kimi_k3_tokenizer_registry(module: ModuleType) -> bool:
    """Register K3 mode with the ordinary cached HF tokenizer."""

    target = require_exact_module(module, TOKENIZER_REGISTRY_TARGET)
    registry = require_type(
        target, "_TokenizerRegistry", f"{TOKENIZER_REGISTRY_TARGET}._TokenizerRegistry"
    )
    global_registry = getattr(target, "TokenizerRegistry", None)
    if not isinstance(global_registry, registry):
        raise Stage3CompatibilityError(
            f"required runtime target {TOKENIZER_REGISTRY_TARGET}.TokenizerRegistry is missing"
        )
    register = require_callable(registry, "register", f"{TOKENIZER_REGISTRY_TARGET}._TokenizerRegistry.register")
    if getattr(global_registry, _TOKENIZER_REGISTRY_MARKER, False):
        return False
    register(global_registry, "kimi_k3", "vllm.tokenizers.hf", "CachedHfTokenizer")
    setattr(global_registry, _TOKENIZER_REGISTRY_MARKER, True)
    return True


def apply_kimi_k3_renderer_registry(module: ModuleType) -> bool:
    """Register the HCU-owned native K3 token-id renderer."""

    target = require_exact_module(module, RENDERER_REGISTRY_TARGET)
    registry = require_type(
        target, "RendererRegistry", f"{RENDERER_REGISTRY_TARGET}.RendererRegistry"
    )
    global_registry = getattr(target, "RENDERER_REGISTRY", None)
    if not isinstance(global_registry, registry):
        raise Stage3CompatibilityError(
            f"required runtime target {RENDERER_REGISTRY_TARGET}.RENDERER_REGISTRY is missing"
        )
    register = require_callable(registry, "register", f"{RENDERER_REGISTRY_TARGET}.RendererRegistry.register")
    if getattr(global_registry, _RENDERER_REGISTRY_MARKER, False):
        return False
    register(
        global_registry,
        "kimi_k3",
        "vllm_hcu.runtime_compat.kimi_k3_renderer",
        "KimiK3Renderer",
    )
    setattr(global_registry, _RENDERER_REGISTRY_MARKER, True)
    return True


def apply_kimi_k3_reasoning_registry(module: ModuleType) -> bool:
    """Register the HCU-owned K3 XTML parser without modifying vLLM source."""

    target = require_exact_module(module, REASONING_REGISTRY_TARGET)
    manager = require_type(
        target,
        "ReasoningParserManager",
        f"{REASONING_REGISTRY_TARGET}.ReasoningParserManager",
    )
    if getattr(manager, _REASONING_REGISTRY_MARKER, False):
        return False
    register = require_callable(
        manager,
        "register_lazy_module",
        f"{REASONING_REGISTRY_TARGET}.ReasoningParserManager.register_lazy_module",
    )
    register(
        "kimi_k3",
        "vllm_hcu.runtime_compat.kimi_k3_reasoning_parser",
        "KimiK3ReasoningParser",
    )
    setattr(manager, _REASONING_REGISTRY_MARKER, True)
    return True


def apply_kimi_k3_mtp_arch_config(module: ModuleType) -> bool:
    target = require_exact_module(module, MTP_ARCH_TARGET)
    marker = "_hcu_kimi_k3_mtp_arch_config_applied"
    if getattr(target, marker, False):
        return False
    mapping = getattr(target, "MODEL_ARCH_CONFIG_CONVERTORS", None)
    if not isinstance(mapping, dict) or "kimi_k3_mtp" in mapping:
        raise Stage3CompatibilityError("Kimi-K3 MTP architecture converter cannot be registered")
    base = require_type(target, "ModelArchConfigConvertorBase", MTP_ARCH_TARGET)

    class KimiK3MTPModelArchConfigConvertor(base):
        def get_num_hidden_layers(self):
            count = getattr(self.hf_text_config, "num_nextn_predict_layers", None)
            if type(count) is not int or count < 1:
                raise ValueError("Kimi-K3 MTP requires positive text_config.num_nextn_predict_layers")
            return count

    mapping["kimi_k3_mtp"] = KimiK3MTPModelArchConfigConvertor
    setattr(target, marker, True)
    return True


def apply_kimi_k3_streaming_content(module: ModuleType) -> bool:
    target = require_exact_module(module, DELEGATING_PARSER_TARGET)
    owner = f"{DELEGATING_PARSER_TARGET}.DelegatingParser"
    parser_class = require_type(target, "DelegatingParser", owner)
    marker = "_hcu_kimi_k3_streaming_content_applied"
    if getattr(parser_class, marker, False):
        return False
    original = require_callable(parser_class, "parse_delta", f"{owner}.parse_delta")
    signature = inspect.signature(original)
    if tuple(signature.parameters) != (
        "self", "delta_text", "delta_token_ids", "request",
        "prompt_token_ids", "finished",
    ):
        raise Stage3CompatibilityError(
            f"DelegatingParser.parse_delta has incompatible signature {signature}"
        )

    @functools.wraps(original)
    def parse_delta(
        self, delta_text, delta_token_ids, request, prompt_token_ids=None,
        *, finished,
    ):
        delta = original(
            self, delta_text, delta_token_ids, request, prompt_token_ids,
            finished=finished,
        )
        from vllm_hcu.runtime_compat.kimi_k3_reasoning_parser import (
            KimiK3ReasoningParser,
        )
        from vllm.entrypoints.openai.engine.protocol import DeltaMessage

        reasoning_parser = self._reasoning_parser
        if not isinstance(reasoning_parser, KimiK3ReasoningParser):
            return delta
        if reasoning_parser._preserve_tool_channels(request):
            return delta
        reasoning_parser.initialize_content_from_prompt(prompt_token_ids)
        content = reasoning_parser.filter_content_delta(
            (delta.content or "") if delta else "", finished=finished
        )
        if delta is None:
            return DeltaMessage(content=content) if content else None
        delta.content = content or None
        return delta if delta.model_dump(exclude_none=True) else None

    parser_class.parse_delta = parse_delta
    setattr(parser_class, marker, True)
    return True


def register_kimi_k3_callbacks(
    coordinator: ExactImportCoordinator = IMPORT_COORDINATOR,
) -> tuple[ImportRegistration, ...]:
    """Arm K3 prompt-rendering callbacks before server configuration is built."""

    return (
        coordinator.register_callback(
            MODEL_CONFIG_PATCH_ID,
            MODEL_CONFIG_TARGET,
            apply_kimi_k3_model_config,
            targets=f"{MODEL_CONFIG_TARGET}.ModelConfig.__post_init__",
        ),
        coordinator.register_callback(
            MTP_ARCH_PATCH_ID,
            MTP_ARCH_TARGET,
            apply_kimi_k3_mtp_arch_config,
            targets=f"{MTP_ARCH_TARGET}.MODEL_ARCH_CONFIG_CONVERTORS",
        ),
        coordinator.register_callback(
            DELEGATING_PARSER_PATCH_ID,
            DELEGATING_PARSER_TARGET,
            apply_kimi_k3_streaming_content,
            targets=f"{DELEGATING_PARSER_TARGET}.DelegatingParser.parse_delta",
        ),
        coordinator.register_callback(
            TOKENIZER_REGISTRY_PATCH_ID,
            TOKENIZER_REGISTRY_TARGET,
            apply_kimi_k3_tokenizer_registry,
            targets=f"{TOKENIZER_REGISTRY_TARGET}.TokenizerRegistry.register",
        ),
        coordinator.register_callback(
            RENDERER_REGISTRY_PATCH_ID,
            RENDERER_REGISTRY_TARGET,
            apply_kimi_k3_renderer_registry,
            targets=f"{RENDERER_REGISTRY_TARGET}.RENDERER_REGISTRY.register",
        ),
        coordinator.register_callback(
            REASONING_REGISTRY_PATCH_ID,
            REASONING_REGISTRY_TARGET,
            apply_kimi_k3_reasoning_registry,
            targets=(
                f"{REASONING_REGISTRY_TARGET}."
                "ReasoningParserManager.register_lazy_module"
            ),
        ),
    )


__all__ = [
    "apply_kimi_k3_model_config",
    "apply_kimi_k3_mtp_arch_config",
    "apply_kimi_k3_streaming_content",
    "apply_kimi_k3_reasoning_registry",
    "apply_kimi_k3_renderer_registry",
    "apply_kimi_k3_tokenizer_registry",
    "register_kimi_k3_callbacks",
]
