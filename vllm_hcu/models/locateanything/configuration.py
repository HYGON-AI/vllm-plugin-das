# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Checkpoint-preserving configuration for native LocateAnything inference."""

from transformers import PretrainedConfig, Qwen2Config
from vllm.transformers_utils.configs.moonvit import MoonViTConfig


def normalize_hf_attention_metadata(config):
    """Remove the checkpoint's private HF-only Magi backend marker.

    Transformers 5 validates this metadata in MoonViT's PreTrainedModel
    constructor. The native vLLM implementations never call HF attention:
    their actual kernels are selected by vLLM's decoder/encoder backend flags.
    SDPA here is a supported constructor marker, not an attention dispatch.
    """
    components = (config, config.vision_config, config.text_config)
    markers = [getattr(component, "_attn_implementation", None)
               for component in components]
    if "magi" not in markers:
        return
    # HF's composite-config setter can propagate the parent's value to its
    # children. Snapshot first, then restore each valid child marker after
    # normalizing the parent, so e.g. an explicit flash_attention_2 is retained.
    for component, marker in zip(components, markers):
        component._attn_implementation = "sdpa" if marker == "magi" else marker


class LocateAnythingConfig(PretrainedConfig):
    model_type = "locateanything"
    is_composition = True
    sub_configs = {"vision_config": MoonViTConfig, "text_config": Qwen2Config}

    def __init__(self, vision_config=None, text_config=None,
                 image_token_index=151665, **kwargs):
        if isinstance(vision_config, dict) or vision_config is None:
            vision_config = MoonViTConfig(**(vision_config or {}))
        if isinstance(text_config, dict) or text_config is None:
            text_config = Qwen2Config(**(text_config or {}))
        if vision_config.model_type != "moonvit":
            raise ValueError("LocateAnything native adapter requires MoonViT")
        if text_config.model_type != "qwen2":
            raise ValueError("LocateAnything native adapter currently requires Qwen2")
        self.vision_config = vision_config
        self.text_config = text_config
        self.image_token_index = int(image_token_index)
        # KimiVL's reusable embedding/encoder graph infrastructure uses this name.
        self.media_placeholder_token_id = self.image_token_index
        from vllm_hcu.patch.locateanything import locateanything_mode
        if locateanything_mode() == "fast":
            kwargs["architectures"] = ["LocateAnythingFastForConditionalGeneration"]
        else:
            kwargs.setdefault("architectures", ["LocateAnythingForConditionalGeneration"])
        kwargs["tie_word_embeddings"] = text_config.tie_word_embeddings
        super().__init__(**kwargs)
        normalize_hf_attention_metadata(self)
