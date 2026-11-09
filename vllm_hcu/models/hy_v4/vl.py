# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The HunYuan team and the vLLM team.
# Adapted vision projection and M-RoPE contracts from Tencent-Hunyuan/HY-Embodied
# commit 711d267dff2c6811c81c108ad5e9c96dd829b5f0, hunyuan_vision.py.
"""HYV4 image tower and vLLM multimodal integration."""

from functools import partial
from itertools import accumulate
from types import MethodType

import torch
from torch import nn
from torch.nn import functional as F
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
from .mrope import forward_hcu_mrope


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
                    h, w = (
                        int(x) // self.image_processor.merge_size for x in grid[-2:]
                    )
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


class HYV4PatchProjection(nn.Conv2d):
    """Project extracted patches as spatial positions in one 1x1 convolution."""

    def forward(self, patches):
        flat = patches.flatten(1)
        weight = self.weight.flatten(1)
        if not (
            patches.is_cuda
            and patches.dtype == torch.bfloat16
            and self.weight.shape == (1536, 3, 16, 16)
            and 2048 <= flat.shape[0] <= 8192
        ):
            # Small spatial convolutions select a different MIOpen reduction
            # order. Retain the original path outside the validated range.
            return F.conv2d(flat[:, :, None, None], weight[:, :, None, None], self.bias)
        # Canonical channels-last views include strides for singleton axes,
        # without the allocations introduced by .to(memory_format=...).
        # Keep MIOpen's accumulation/bias rounding and the checkpoint layout.
        projected = F.conv2d(
            flat[None, None, :, :].permute(0, 3, 1, 2),
            weight[:, None, None, :].permute(0, 3, 1, 2),
            self.bias,
        )
        return projected.permute(3, 1, 0, 2).contiguous()


class HYV4VisionPatchEmbed(HunYuanVisionPatchEmbed):
    def __init__(self, config):
        super().__init__(config)
        self.patch_embedding = HYV4PatchProjection(
            config.num_channels,
            config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=True,
        )

    def forward(self, pixel_values, grid_thw):
        if self.patch_pos_embed is not None:
            # Native caches an FP32 copy of BF16/FP16 positions. Kernel-format
            # reload bypasses model.load_weights, so refresh from the parameter
            # at the consumer. Keep storage stable for already captured work;
            # copy_ casts directly without allocating another FP32 position table.
            positions = (
                self.position_embedding.weight[1:]
                .reshape(1, self.position_edge, self.position_edge, self.embed_dim)
                .permute(0, 3, 1, 2)
            )
            self.patch_pos_embed.copy_(positions)
        return super().forward(pixel_values, grid_thw)


class HYV4VisionBlock(HunYuanVisionBlock):
    def forward(self, x, cu_seqlens, max_seqlen):
        attention = self.self_attn
        qkv, _ = attention.qkv(self.input_layernorm(x))
        q, k, v = qkv.chunk(3, dim=-1)
        out = attention.attn(q, k, v, cu_seqlens, max_seqlen)
        out, _ = attention.o_proj(out)
        x = x + out
        return x + self.mlp(self.post_attention_layernorm(x))


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
        self.embeddings = HYV4VisionPatchEmbed(config)
        self.layers = nn.ModuleList(
            [
                HYV4VisionBlock(
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

    def forward(self, x, grid_thw):
        lengths = [int(h) * int(w) for _, h, w in grid_thw]
        hidden_states = self.embeddings(
            x.to(device=self.device, dtype=self.dtype), grid_thw
        )
        if self.layers and self.layers[0].self_attn.tp_size > 1:
            # Changing GEMM batch shapes under TP changes BF16 accumulation;
            # small differences amplify through this 40-layer vision tower.
            # Keep per-image GEMMs/collectives, with attention metadata prepared
            # once rather than allocating boundaries in every attention layer.
            metadata = [
                (
                    torch.tensor([0, length], dtype=torch.int32, device=self.device),
                    torch.tensor(length, dtype=torch.int32),
                )
                for length in lengths
            ]
            image_states = hidden_states.split(lengths, dim=1)
            del hidden_states
            for layer in self.layers:
                # Keep image tensors separate between layers. Repacking them
                # copies the whole activation without changing any computation.
                image_states = [
                    layer(item, *meta) for item, meta in zip(image_states, metadata)
                ]
        else:
            cu_seqlens = torch.tensor(
                [0, *accumulate(lengths)], dtype=torch.int32, device=self.device
            )
            # The attention wrapper reads .item(); keep this scalar on the CPU.
            max_seqlen = torch.tensor(max(lengths), dtype=torch.int32)
            for layer in self.layers:
                hidden_states = layer(hidden_states, cu_seqlens, max_seqlen)
            image_states = hidden_states.split(lengths, dim=1)
        return [
            self.perceive(item.contiguous(), size=grid[1:]).squeeze(0)
            for grid, item in zip(grid_thw, image_states)
        ]


@MULTIMODAL_REGISTRY.register_processor(
    HYV4VLMultiModalProcessor,
    info=HYV4VLProcessingInfo,
    dummy_inputs=HunYuanVLDummyInputsBuilder,
)
class HYV4VLForConditionalGeneration(
    HYV4ForCausalLM, SupportsMultiModal, SupportsMRoPE
):
    def __init__(self, *, vllm_config, prefix=""):
        mm_config = vllm_config.model_config.multimodal_config
        if mm_config is not None and mm_config.mm_encoder_only:
            raise ValueError("HYV4VL encoder-only mode is not implemented")
        lora_config = vllm_config.lora_config
        if lora_config is not None and lora_config.enable_tower_connector_lora:
            raise ValueError(
                "HYV4VL vision tower/connector LoRA is not implemented; "
                "disable enable_tower_connector_lora to use language-only LoRA"
            )
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
        # The generic HCU RoPE kernel only supports a single position axis.
        for layer in self.model.layers:
            for name in ("rotary_emb", "indexer_rope_emb"):
                rope = getattr(getattr(layer, "self_attn", None), name, None)
                if rope is not None:
                    rope._forward_method = MethodType(forward_hcu_mrope, rope)

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
