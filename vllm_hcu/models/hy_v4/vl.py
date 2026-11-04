# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The HunYuan team and the vLLM team.
# Adapted vision projection and M-RoPE contracts from Tencent-Hunyuan/HY-Embodied
# commit 711d267dff2c6811c81c108ad5e9c96dd829b5f0, hunyuan_vision.py.
"""HYV4 image tower and vLLM multimodal integration."""

from functools import partial

import torch
from torch import nn
from transformers import BatchFeature
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.models.hunyuan_vision import (
    HunYuanVisionBlock,
    HunYuanVisionPatchEmbed,
    HunYuanVisionTransformer,
    HunYuanVLDummyInputsBuilder,
    HunYuanVLMultiModalProcessor,
    HunYuanVLProcessingInfo,
)
from vllm.model_executor.models.interfaces import SupportsMRoPE, SupportsMultiModal
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.processing import PromptReplacement
from vllm.transformers_utils.processors.hunyuan_vl import HunYuanVLProcessor
from vllm.transformers_utils.processors.hunyuan_vl_image import HunYuanVLImageProcessor

from .model import HYV4ForCausalLM


class HYV4VLProcessor(HunYuanVLProcessor):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        from transformers import AutoTokenizer

        kwargs.pop("typ", None)
        kwargs.pop("use_fast", None)
        tokenizer = kwargs.pop("tokenizer", None)
        return cls(
            image_processor=HunYuanVLImageProcessor.from_pretrained(
                pretrained_model_name_or_path, **kwargs
            ),
            tokenizer=tokenizer
            or AutoTokenizer.from_pretrained(pretrained_model_name_or_path),
        )

    def __call__(self, text=None, images=None, videos=None, **kwargs):
        if videos is not None:
            raise ValueError("HYV4VL video input has not been implemented")
        return_tensors = kwargs.pop("return_tensors", None)
        kwargs.pop("add_special_tokens", None)
        image_keys = (
            "min_pixels",
            "max_pixels",
            "size",
            "do_resize",
            "resample",
            "do_rescale",
            "rescale_factor",
            "do_normalize",
            "image_mean",
            "image_std",
            "do_convert_rgb",
            "patch_size",
            "merge_size",
            "temporal_patch_size",
            "input_data_format",
            "data_format",
        )
        image_kwargs = {key: kwargs.pop(key) for key in image_keys if key in kwargs}
        image_inputs = {}
        if images is not None:
            image_inputs = self.image_processor(
                images=images, return_tensors="pt", **image_kwargs
            )
        texts = [text] if isinstance(text, str) else list(text)
        # Cached multimodal inputs are tokenized without images by vLLM.
        if images is not None:
            grids = iter(image_inputs.get("image_grid_thw", []))
            for i, prompt in enumerate(texts):
                pieces = prompt.split(self.image_token)
                expanded = pieces[0]
                for tail in pieces[1:]:
                    grid = next(grids, None)
                    if grid is None:
                        raise ValueError("More image placeholders than images")
                    h, w = (int(x) // self.image_processor.merge_size for x in grid[-2:])
                    expanded += self.image_token * (h * (w + 1)) + tail
                texts[i] = expanded
            if next(grids, None) is not None:
                raise ValueError("More images than image placeholders")
        tokens = self.tokenizer(
            texts, add_special_tokens=False, return_tensors=return_tensors, **kwargs
        )
        return BatchFeature({**tokens, **image_inputs}, tensor_type=return_tensors)


class HYV4VLProcessingInfo(HunYuanVLProcessingInfo):
    def get_hf_config(self):
        return self.ctx.model_config.hf_config

    def get_hf_processor(self, **kwargs):
        kwargs.pop("typ", None)
        return self.ctx.get_hf_processor(HYV4VLProcessor, **kwargs)

    def _get_vision_info(self, **kwargs):
        size, tokens = super()._get_vision_info(**kwargs)
        return size, tokens - 2


class HYV4VLMultiModalProcessor(HunYuanVLMultiModalProcessor):
    def _get_prompt_updates(self, mm_items, hf_processor_mm_kwargs, out_mm_kwargs):
        proc = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        merge = proc.image_processor.merge_size

        def replacement(i):
            grid = out_mm_kwargs["image"][i]["image_grid_thw"].data
            h, w = (int(x) // merge for x in grid[-2:])
            return [proc.image_token_id] * (h * (w + 1))

        return [
            PromptReplacement(
                modality="image", target=[proc.image_token_id], replacement=replacement
            )
        ]


class HYV4VisionMerger(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.proj = nn.Sequential(
            nn.Conv2d(
                config.hidden_size,
                config.out_hidden_size,
                config.spatial_merge_size,
                config.spatial_merge_size,
            ),
            nn.GELU(),
            nn.Conv2d(config.out_hidden_size, config.out_hidden_size, 1),
        )
        self.image_newline = nn.Parameter(torch.empty(config.out_hidden_size))

    def forward(self, x, size):
        b, _, d = x.shape
        x = self.norm(x).transpose(1, 2).reshape(b, d, *size)
        x = self.proj(x)
        b, d, h, _ = x.shape
        newline = self.image_newline.reshape(1, d, 1, 1).expand(b, d, h, 1)
        return torch.cat([x, newline], dim=-1).flatten(2).transpose(1, 2)


class HYV4VisionTransformer(HunYuanVisionTransformer):
    def __init__(self, config, prefix):
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.spatial_merge_size = config.spatial_merge_size
        self.embeddings = HunYuanVisionPatchEmbed(config)
        self.layers = nn.ModuleList(
            [
                HunYuanVisionBlock(
                    dim=config.hidden_size,
                    num_heads=config.num_attention_heads,
                    mlp_hidden_dim=config.intermediate_size,
                    act_fn=get_act_fn(config.hidden_act),
                    norm_layer=partial(nn.LayerNorm, eps=config.rms_norm_eps),
                    quant_config=None,
                    prefix=f"{prefix}.layers.{i}",
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        self.perceive = HYV4VisionMerger(config)


@MULTIMODAL_REGISTRY.register_processor(
    HYV4VLMultiModalProcessor,
    info=HYV4VLProcessingInfo,
    dummy_inputs=HunYuanVLDummyInputsBuilder,
)
class HYV4VLForConditionalGeneration(
    HYV4ForCausalLM, SupportsMultiModal, SupportsMRoPE
):
    def __init__(self, *, vllm_config, prefix=""):
        vision = vllm_config.model_config.hf_config.vision_config
        if (
            getattr(vision, "projector_type", None) != "rmsnorm_mlp_adapter"
            or bool(getattr(vision, "cat_extra_token", False))
            or bool(getattr(vision, "perceive_post_norm", False))
        ):
            raise ValueError(
                "HYV4VL requires rmsnorm_mlp_adapter without extra tokens or post norm"
            )
        if vllm_config.speculative_config is not None:
            raise ValueError("HYV4VL speculative decoding is not validated")
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        with self._mark_tower_model(vllm_config, {"image"}):
            self.vit = HYV4VisionTransformer(self.config.vision_config, prefix="vit")
        # HCU's fused 1-D RoPE kernel does not implement the three M-RoPE axes.
        for layer in self.model.layers:
            for name in ("rotary_emb", "indexer_rope_emb"):
                rope = getattr(getattr(layer, "self_attn", None), name, None)
                if rope is not None:
                    rope._forward_method = rope.forward_native

    @classmethod
    def get_placeholder_str(cls, modality, i):
        if modality == "image":
            return "<｜hy_image▁start:opensource｜><｜hy_image▁pad:opensource｜><｜hy_image▁end:opensource｜>"
        raise ValueError(f"Unsupported modality: {modality}")

    def embed_input_ids(
        self, input_ids, multimodal_embeddings=None, *, is_multimodal=None
    ):
        from vllm.model_executor.models.utils import _merge_multimodal_embeddings

        embeds = super().embed_input_ids(input_ids)
        if multimodal_embeddings is not None:
            embeds = _merge_multimodal_embeddings(
                embeds, multimodal_embeddings, is_multimodal=is_multimodal
            )
        return embeds

    def embed_multimodal(self, **kwargs):
        if kwargs.get("image_embeds") is not None:
            raise ValueError("HYV4VL precomputed image embeddings are not implemented")
        pixels = kwargs.get("pixel_values")
        if pixels is None:
            return []
        if isinstance(pixels, list):
            pixels = torch.cat(pixels)
        grids = kwargs["image_grid_thw"].reshape(-1, 3)
        return tuple(self.vit(pixels.reshape(-1, pixels.shape[-1]), grids.tolist()))

    def get_mrope_input_positions(self, input_tokens, mm_features):
        parts = []
        cursor = start = 0
        for feature in sorted(mm_features, key=lambda f: f.mm_position.offset):
            grid = feature.data["image_grid_thw"].data.reshape(-1, 3)[0]
            h, w = (
                int(x) // self.config.vision_config.spatial_merge_size
                for x in grid[-2:]
            )
            offset = feature.mm_position.offset
            ntext = offset - cursor
            parts.append(torch.arange(ntext).repeat(3, 1) + start)
            start += ntext
            visual = (
                torch.stack(
                    [
                        torch.zeros(h * (w + 1), dtype=torch.long),
                        torch.arange(h).repeat_interleave(w + 1),
                        torch.arange(w + 1).repeat(h),
                    ]
                )
                + start
            )
            parts.append(visual)
            start = int(visual.max()) + 1
            cursor = offset + h * (w + 1)
        parts.append(torch.arange(len(input_tokens) - cursor).repeat(3, 1) + start)
        positions = torch.cat(parts, dim=1)
        return positions, int(positions.max()) + 1 - len(input_tokens)

    def load_weights(self, weights):
        visual = []

        def language_weights():
            for name, weight in weights:
                if name.startswith("vit."):
                    visual.append((name[4:], weight))
                else:
                    yield name, weight

        # The language loader enforces completeness; the tower is loaded separately.
        vit = self.vit
        del self.vit
        try:
            loaded = super().load_weights(language_weights())
        finally:
            self.vit = vit
        vision_loaded = vit.load_weights(visual)
        missing = set(dict(vit.named_parameters())) - vision_loaded
        if missing:
            raise RuntimeError(f"Missing HYV4VL vision weights: {sorted(missing)}")
        return loaded | {"vit." + n for n in vision_loaded}

    def get_mm_mapping(self):
        return MultiModelKeys.from_string_field(
            language_model="model", connector="vit.perceive", tower_model="vit"
        )
