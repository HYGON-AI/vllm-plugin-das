# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Native multimodal LocateAnything Slow implementation for vLLM 0.28.1.

MoonViT, paged Qwen2 attention and encoder/decoder graph execution use the
installed vLLM implementations. The checkpoint-specific connector and processor
remain exact; no HF generate loop or HTTP prompt-embedding transport is used.
"""

from collections.abc import Mapping
from functools import cached_property

import torch
from torch import nn

from vllm.model_executor.models.kimi_vl import (
    KimiVLDummyInputsBuilder,
    KimiVLForConditionalGeneration,
    KimiVLMultiModalProcessor,
    KimiVLProcessingInfo,
)
from vllm.model_executor.models.moonvit import MoonVitPretrainedModel
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    init_vllm_registered_model,
    maybe_prefix,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.processing import PromptReplacement

from .configuration import LocateAnythingConfig, normalize_hf_attention_metadata
from .processing import (
    IMAGE_TOKEN,
    LocateAnythingImageProcessor,
    LocateAnythingProcessor,
    image_placeholder,
    max_image_grid,
    rescale_size,
)


class LocateAnythingProcessingInfo(KimiVLProcessingInfo):
    def get_hf_config(self):
        return self.ctx.get_hf_config(LocateAnythingConfig)

    @cached_property
    def _native_processor(self):
        image_processor = LocateAnythingImageProcessor.from_pretrained(
            self.model_id, **self.ctx.get_merged_mm_kwargs({}, modality="image")
        )
        config = self.get_hf_config()
        if image_processor.patch_size != config.vision_config.patch_size:
            raise ValueError("Image processor patch size does not match vision weights")
        if tuple(image_processor.merge_kernel_size) != tuple(config.vision_config.merge_kernel_size):
            raise ValueError("Image processor merge order/size does not match vision weights")
        return LocateAnythingProcessor(self.get_tokenizer(), image_processor)

    def get_hf_processor(self, **kwargs):
        # The original HF processor cannot construct under Transformers 5.
        # InputProcessingContext explicitly accepts plain callable processors.
        merged = self.ctx.get_merged_mm_kwargs(kwargs, modality="image")
        shape_keys = {"patch_size", "in_token_limit", "merge_kernel_size", "image_mean", "image_std"}
        if shape_keys.intersection(merged):
            image_processor = LocateAnythingImageProcessor.from_pretrained(self.model_id, **merged)
            config = self.get_hf_config().vision_config
            if image_processor.patch_size != config.patch_size or tuple(image_processor.merge_kernel_size) != tuple(config.merge_kernel_size):
                raise ValueError("Cannot override checkpoint patch or merge size")
            return LocateAnythingProcessor(self.get_tokenizer(), image_processor)
        return self._native_processor

    def get_num_image_tokens(self, *, image_width, image_height):
        p = self.get_hf_processor().image_processor
        width, height = rescale_size(
            image_width, image_height, p.patch_size,
            p.in_token_limit, p.merge_kernel_size,
        )
        return (height // p.patch_size) * (width // p.patch_size) // (
            p.merge_kernel_size[0] * p.merge_kernel_size[1]
        )

    def get_mm_max_tokens_per_item(self, seq_len, mm_counts):
        p = self.get_hf_processor().image_processor
        height, width, _, _ = max_image_grid(p.in_token_limit, p.merge_kernel_size, p.patch_size)
        return {"image": height * width // (p.merge_kernel_size[0] * p.merge_kernel_size[1])}


class LocateAnythingDummyInputsBuilder(KimiVLDummyInputsBuilder):
    def get_dummy_text(self, mm_counts: Mapping[str, int]):
        return "".join(image_placeholder(i + 1) for i in range(mm_counts.get("image", 0)))

    def get_dummy_mm_data(self, seq_len, mm_counts, mm_options):
        p = self.info.get_hf_processor().image_processor
        _, _, source_h, source_w = max_image_grid(p.in_token_limit, p.merge_kernel_size, p.patch_size)
        return {
            "image": self._get_dummy_images(
                width=source_w,
                height=source_h,
                num_images=mm_counts.get("image", 0),
                overrides=mm_options.get("image"),
            )
        }


class LocateAnythingMultiModalProcessor(KimiVLMultiModalProcessor):
    def _get_prompt_updates(self, mm_items, hf_processor_mm_kwargs, out_mm_kwargs):
        image_token_id = self.info.image_token_id
        kh, kw = self.info.get_hf_config().vision_config.merge_kernel_size

        def get_replacement(item_idx):
            # Use the actual processed grid, including per-request overrides,
            # instead of recomputing from a default processor configuration.
            data = out_mm_kwargs["image"][item_idx].get_data()
            if "image_grid_hws" not in data:
                raise ValueError("LocateAnything supports image pixels, not external image embeddings")
            grid = data["image_grid_hws"].reshape(-1, 2)
            if grid.shape[0] != 1:
                raise ValueError("Each multimodal image item must contain exactly one grid")
            height, width = grid[0].tolist()
            return [image_token_id] * (int(height) * int(width) // (kh * kw))

        return [PromptReplacement(modality="image", target=[image_token_id], replacement=get_replacement)]


class LocateAnythingProjector(nn.Sequential):
    """Original merged-patch LayerNorm and two-layer connector, unchanged."""

    def __init__(self, config, prefix=""):
        merged_size = (config.vision_config.hidden_size
                       * config.vision_config.merge_kernel_size[0]
                       * config.vision_config.merge_kernel_size[1])
        hidden_size = config.text_config.hidden_size
        super().__init__(
            nn.LayerNorm(merged_size),
            nn.Linear(merged_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.merged_size = merged_size

    def forward(self, image_features):
        # vLLM MoonViT emits [tokens, 4, 1152]; LocateAnything uses [tokens, 4608].
        # Their merge permutations are identical. Normalize AFTER flattening.
        return super().forward(image_features.reshape(-1, self.merged_size))


@MULTIMODAL_REGISTRY.register_processor(
    LocateAnythingMultiModalProcessor,
    info=LocateAnythingProcessingInfo,
    dummy_inputs=LocateAnythingDummyInputsBuilder,
)
class LocateAnythingForConditionalGeneration(KimiVLForConditionalGeneration):
    """Full original checkpoint; native causal Qwen2 decoding (Slow only)."""

    hf_to_vllm_mapper = WeightsMapper(orig_to_new_prefix={
        "vision_model.": "vision_tower.",
        "mlp1.": "multi_modal_projector.",
    })

    @classmethod
    def get_placeholder_str(cls, modality, i):
        if modality.startswith("image"):
            return image_placeholder(i)
        raise ValueError("LocateAnything native adapter currently supports images only")

    def __init__(self, vllm_config, prefix=""):
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config
        if not isinstance(config, LocateAnythingConfig):
            raise TypeError("Load the hcu_model plugin before engine construction")
        if vllm_config.quant_config is not None:
            raise ValueError("Initial LocateAnything accuracy baseline requires unquantized BF16 weights")
        if getattr(config, "use_backbone_lora", 0) or getattr(config, "use_llm_lora", 0):
            raise ValueError("Checkpoint-side LoRA is not supported by this adapter")
        # AutoConfig/from_pretrained may reapply parent attention metadata after
        # our constructor; normalize again immediately before HF MoonViT init.
        normalize_hf_attention_metadata(config)
        self.config = config
        self.model_config = vllm_config.model_config
        self.quant_config = vllm_config.quant_config
        self.hidden_size = config.text_config.hidden_size
        self.use_data_parallel = self.model_config.multimodal_config.mm_encoder_tp_mode == "data"
        self.media_placeholder = config.image_token_index

        with self._mark_tower_model(vllm_config, "image"):
            self.vision_tower = MoonVitPretrainedModel(
                config.vision_config, prefix=maybe_prefix(prefix, "vision_tower")
            )
            self.multi_modal_projector = LocateAnythingProjector(
                config, prefix=maybe_prefix(prefix, "multi_modal_projector")
            )
        with self._mark_language_model(vllm_config):
            self.language_model = init_vllm_registered_model(
                vllm_config=vllm_config,
                hf_config=config.text_config,
                prefix=maybe_prefix(prefix, "language_model"),
                architectures=["Qwen2ForCausalLM"],
            )
        self.make_empty_intermediate_tensors = self.language_model.make_empty_intermediate_tensors
        # Report actual installed origins once per engine construction. This is
        # useful for verifying wheel-only deployment after removing old overlays.
        import json
        import sys
        from vllm_hcu.patch.locateanything import locateanything_mode
        observed = {}
        names = [__name__, "vllm.v1.core.sched.scheduler",
                 "vllm.v1.worker.gpu.model_runner", "vllm.v1.worker.gpu.input_batch",
                 "vllm.v1.worker.gpu.warmup",
                 "vllm.v1.attention.ops.triton_unified_attention"]
        for name in names:
            module = sys.modules.get(name)
            if module is not None:
                observed[name] = getattr(module, "__file__", None)
        print("LOCATEANYTHING_HCU_RUNTIME " + json.dumps(
            {"mode": locateanything_mode(), "modules": observed,
             "torch": torch.__file__, "model_class": type(self).__name__}), flush=True)

    def prepare_encoder_cudagraph_capture_inputs(self, token_budget, max_batch_size,
                                                 max_frames_per_batch, device, dtype,
                                                 path="default"):
        captured = super().prepare_encoder_cudagraph_capture_inputs(
            token_budget, max_batch_size, max_frames_per_batch, device, dtype, path
        )
        # KimiVL passes the output-token budget as the FA sequence bound,
        # although MoonViT runs on four times as many input patches. Use the
        # complete input buffer capacity, also covering synthetic-grid rounding.
        # The scalar stays on CPU as required by the installed MoonViT backend.
        captured.values["max_seqlen"] = torch.tensor(
            captured.values["pixel_values"].shape[0], dtype=torch.int32
        )
        return captured

    def encoder_cudagraph_forward(self, values, path="default"):
        metadata = {key: value for key, value in values.items() if key != "pixel_values"}
        image_features = self.vision_tower(
            values["pixel_values"], grid_hw=None, encoder_metadata=metadata
        )
        return self.multi_modal_projector(image_features)

    def load_weights(self, weights):
        # Delegates Qwen2 QKV/gate-up packing and tied embeddings to its native
        # loader. MoonViT keeps identical module/parameter suffixes.
        return AutoWeightsLoader(self).load_weights(weights, mapper=self.hf_to_vllm_mapper)


class LocateAnythingFastForConditionalGeneration(LocateAnythingForConditionalGeneration):
    """Original six-position Fast decoder, using the HCU-owned runtime."""

    @classmethod
    def get_model_state_cls(cls):
        from vllm_hcu.patch.locateanything import locateanything_mode
        if locateanything_mode() != "fast":
            raise ValueError("LocateAnything Fast requires VLLM_HCU_LOCATEANYTHING_MODE=fast before importing vLLM")
        from vllm_hcu.v1.locateanything.core import FastModelState
        return FastModelState
