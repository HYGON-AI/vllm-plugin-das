# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class _PromptReplacement:
    def __init__(self, modality, target, replacement):
        self.modality = modality
        self.target = target
        self.replacement = replacement


class _VisionChunkProcessorItems:
    pass


def _load_runtime_compat_module():
    compat_path = ROOT / "vllm_hcu/runtime_compat/kimi_k25_vision_prompt.py"
    spec = importlib.util.spec_from_file_location(
        "kimi_k25_vision_prompt_compat_test",
        compat_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _install_fake_modules():
    injected_module_names = [
        "vllm",
        "vllm.multimodal",
        "vllm.multimodal.processing",
        "vllm.multimodal.parse",
        "vllm.model_executor",
        "vllm.model_executor.models",
        "vllm.model_executor.models.kimi_k25",
    ]
    original_modules = {name: sys.modules.get(name) for name in injected_module_names}

    processing_module = types.ModuleType("vllm.multimodal.processing")
    processing_module.PromptReplacement = _PromptReplacement

    parse_module = types.ModuleType("vllm.multimodal.parse")
    parse_module.VisionChunkProcessorItems = _VisionChunkProcessorItems

    class _Processor:
        def __init__(self) -> None:
            self.info = types.SimpleNamespace(
                get_hf_config=lambda: types.SimpleNamespace(
                    media_placeholder_token_id=163605,
                ),
                media_tokens_calculator=lambda item: 8,
            )

        def _get_prompt_updates(self, mm_items, hf_processor_mm_kwargs, out_mm_kwargs):
            return [
                _PromptReplacement(
                    modality="vision_chunk",
                    target=[163605],
                    replacement=lambda _: [163605] * 8,
                )
            ]

    class _DummyInputsBuilder:
        def get_dummy_text(self, mm_counts):
            return "<|media_pad|>" * mm_counts.get("vision_chunk", 0)

    kimi_module = types.ModuleType("vllm.model_executor.models.kimi_k25")
    kimi_module.KimiK25MultiModalProcessor = _Processor
    kimi_module.KimiK25DummyInputsBuilder = _DummyInputsBuilder

    vllm_module = types.ModuleType("vllm")
    multimodal_module = types.ModuleType("vllm.multimodal")
    model_executor_module = types.ModuleType("vllm.model_executor")
    model_executor_models_module = types.ModuleType("vllm.model_executor.models")

    sys.modules["vllm"] = vllm_module
    sys.modules["vllm.multimodal"] = multimodal_module
    sys.modules["vllm.multimodal.processing"] = processing_module
    sys.modules["vllm.multimodal.parse"] = parse_module
    sys.modules["vllm.model_executor"] = model_executor_module
    sys.modules["vllm.model_executor.models"] = model_executor_models_module
    sys.modules["vllm.model_executor.models.kimi_k25"] = kimi_module

    return {
        "restore": original_modules,
        "injected": injected_module_names,
        "kimi_module": kimi_module,
    }


def test_runtime_compat_adds_media_pad_fallback_and_dummy_prompt_shell() -> None:
    state = _install_fake_modules()
    try:
        compat_module = _load_runtime_compat_module()
        compat_module.install_kimi_k25_vision_prompt_compat()

        processor = state["kimi_module"].KimiK25MultiModalProcessor()
        updates = processor._get_prompt_updates(None, {}, {})
        targets = [update.target for update in updates]

        assert [163605] in targets
        assert "<|media_pad|>" in targets
        assert (
            "<|media_begin|>image<|media_content|><|media_pad|><|media_end|>"
            in targets
        )
        assert (
            "<|media_begin|>video<|media_content|><|media_pad|><|media_end|>"
            in targets
        )

        builder = state["kimi_module"].KimiK25DummyInputsBuilder()
        assert builder.get_dummy_text({"vision_chunk": 2}) == (
            "<|media_begin|>image<|media_content|><|media_pad|><|media_end|>"
            "<|media_begin|>image<|media_content|><|media_pad|><|media_end|>"
        )
    finally:
        for name in state["injected"]:
            original_module = state["restore"][name]
            if original_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original_module
