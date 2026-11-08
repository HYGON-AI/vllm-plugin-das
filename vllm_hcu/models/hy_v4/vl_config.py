# SPDX-License-Identifier: Apache-2.0
"""Configuration registration for HYV4 image checkpoints."""

from typing import ClassVar

from transformers import AutoConfig
from vllm.transformers_utils.configs.hunyuan_vl import HunYuanVLVisionConfig

from .config import HYV4Config


class HYV4VLConfig(HYV4Config):
    model_type = "hy_v4_vl"
    ignore_keys_at_rope_validation: ClassVar[set[str]] = {
        "mrope_section",
        "mrope_interleaved",
    }
    attribute_map: ClassVar[dict[str, str]] = {"num_local_experts": "n_routed_experts"}

    def __init__(self, vision_config=None, **kwargs):
        super().__init__(**kwargs)
        self.vision_config = (
            HunYuanVLVisionConfig(**(vision_config or {}))
            if isinstance(vision_config, (dict, type(None)))
            else vision_config
        )


def register_hy_v4_vl_config():
    from vllm.transformers_utils import config

    existing = config._CONFIG_REGISTRY.get(HYV4VLConfig.model_type)
    if existing is not None and existing is not HYV4VLConfig:
        raise RuntimeError(f"HYV4VL config already has a different owner: {existing}")
    config._CONFIG_REGISTRY[HYV4VLConfig.model_type] = HYV4VLConfig
    AutoConfig.register(HYV4VLConfig.model_type, HYV4VLConfig, exist_ok=True)
