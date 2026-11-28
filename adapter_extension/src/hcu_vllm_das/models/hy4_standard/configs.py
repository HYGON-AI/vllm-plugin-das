# SPDX-License-Identifier: Apache-2.0
"""Standard HY4 VL metadata composed with the unchanged Core text config."""

from copy import deepcopy
from typing import ClassVar

from transformers.configuration_utils import PreTrainedConfig
from vllm.transformers_utils.configs.hy_v4 import HYV4Config

from ...model_runtime_activation import ModelRuntimeConfig


class HYV4VLTextConfig(HYV4Config):
    """Owned text identity without changing the standard HY4 architecture."""

    ignore_keys_at_rope_validation: ClassVar = {"mrope_section", "mrope_interleaved"}


class HYV4VLVisionConfig(PreTrainedConfig):
    """Vision tower and projector configuration for HYV4 VL."""

    model_type = "hy_v4_vl_vision"
    base_config_key = "vision_config"

    def __init__(
        self,
        hidden_size=1152,
        intermediate_size=4304,
        num_hidden_layers=27,
        num_attention_heads=16,
        num_channels=3,
        patch_size=16,
        spatial_merge_size=2,
        spatial_patch_size=1,
        temporal_patch_size=1,
        out_hidden_size=None,
        hidden_act="gelu",
        rms_norm_eps=1e-5,
        max_image_size=2048,
        interpolate_mode="bilinear",
        cat_extra_token=0,
        perceive_pre_norm=True,
        perceive_post_norm=True,
        projector_type=None,
        perceive_intermediate_size=None,
        use_image_newline_token=True,
        use_2d_rope=False,
        rope_2d_theta=10000.0,
        rope_2d_percent=1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.spatial_patch_size = spatial_patch_size
        self.temporal_patch_size = temporal_patch_size
        self.out_hidden_size = out_hidden_size
        self.hidden_act = hidden_act
        self.rms_norm_eps = rms_norm_eps
        self.max_image_size = max_image_size
        self.interpolate_mode = interpolate_mode
        self.cat_extra_token = cat_extra_token
        self.perceive_pre_norm = perceive_pre_norm
        self.perceive_post_norm = perceive_post_norm
        self.projector_type = projector_type
        self.perceive_intermediate_size = perceive_intermediate_size
        self.use_image_newline_token = use_image_newline_token
        self.use_2d_rope = use_2d_rope
        self.rope_2d_theta = rope_2d_theta
        self.rope_2d_percent = rope_2d_percent


class HYV4VLConfig(ModelRuntimeConfig, PreTrainedConfig):
    """Local HYV4 vision/text composition, including flat checkpoint support."""

    model_type = "hy_v4_vl"
    architectures: list[str] | None = None
    sub_configs: ClassVar = {
        "text_config": HYV4VLTextConfig,
        "vision_config": HYV4VLVisionConfig,
    }
    keys_to_ignore_at_inference: ClassVar = ["past_key_values"]
    ignore_keys_at_rope_validation: ClassVar = {"mrope_section", "mrope_interleaved"}

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        image_start_token_id=120118,
        image_end_token_id=120119,
        image_token_id=120120,
        image_newline_token_id=120121,
        video_start_token_id=120122,
        video_end_token_id=120123,
        video_token_id=120124,
        **kwargs,
    ):
        metadata_keys = {"model_type", "architectures", "auto_map", "processor_class"}
        outer_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key in metadata_keys or key.startswith("dfly_")
        }
        text_config_cls = self.sub_configs["text_config"]
        nested = (
            text_config.to_dict()
            if isinstance(text_config, HYV4Config)
            else deepcopy(text_config or {})
        )
        # Nested text fields take precedence; never annotate the caller's object.
        text_kwargs = {**kwargs, **nested}
        for key in metadata_keys:
            text_kwargs.pop(key, None)
        self._prepare_text_config(text_kwargs)
        explicit_rope = dict(text_kwargs.get("rope_parameters") or {})
        text = text_config_cls(**text_kwargs)
        if not any(isinstance(value, dict) for value in explicit_rope.values()):
            rope = dict(text.rope_parameters or {})
            rope.update(explicit_rope)
            text.rope_parameters = rope
        text.hy4_standard_vl = True
        super().__init__(**outer_kwargs)
        self.text_config = text
        self.vision_config = (
            vision_config
            if isinstance(vision_config, HYV4VLVisionConfig)
            else HYV4VLVisionConfig(**(vision_config or {}))
        )
        if self.vision_config.out_hidden_size is None:
            self.vision_config.out_hidden_size = text.hidden_size
        if self.vision_config.out_hidden_size != text.hidden_size:
            raise ValueError("HYV4 VL projector output must match text hidden_size")
        self.vision_config.text_hidden_size = text.hidden_size
        self.image_start_token_id = image_start_token_id
        self.image_end_token_id = image_end_token_id
        self.image_token_id = image_token_id
        self.image_newline_token_id = image_newline_token_id
        self.video_start_token_id = video_start_token_id
        self.video_end_token_id = video_end_token_id
        self.video_token_id = video_token_id
        for name in (
            "dtype",
            "tie_word_embeddings",
            "bos_token_id",
            "eos_token_id",
            "pad_token_id",
        ):
            setattr(self, name, getattr(text, name))
        if getattr(text, "head_dtype", None) is not None:
            self.head_dtype = text.head_dtype
        if self.architectures is None:
            self.architectures = [self.default_architecture]

    default_architecture = "HYV4VLForConditionalGeneration"

    @classmethod
    def request_runtime(cls):
        from ...hy4_vl_activation import request_model_runtime

        request_model_runtime()

    @staticmethod
    def _prepare_text_config(text_kwargs):
        rope = dict(text_kwargs.pop("rope_scaling", None) or {})
        rope.update(text_kwargs.get("rope_parameters") or {})
        rope.setdefault("rope_type", "default")
        rope.setdefault("mrope_interleaved", True)
        rope.setdefault("mrope_section", [12, 10, 10])
        text_kwargs["rope_parameters"] = rope

    def __getattr__(self, name):
        text = self.__dict__.get("text_config")
        if text is not None:
            return getattr(text, name)
        raise AttributeError(name)


__all__ = ["HYV4Config", "HYV4VLConfig", "HYV4VLTextConfig", "HYV4VLVisionConfig"]
