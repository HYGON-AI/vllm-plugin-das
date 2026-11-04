# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""LocateAnything registration, loaded by the existing hcu_model entry point."""


def register():
    from transformers import AutoConfig
    from vllm import ModelRegistry
    from vllm.transformers_utils.config import _CONFIG_REGISTRY
    from .configuration import LocateAnythingConfig

    AutoConfig.register("locateanything", LocateAnythingConfig, exist_ok=True)
    _CONFIG_REGISTRY["locateanything"] = LocateAnythingConfig
    for name in ("LocateAnythingForConditionalGeneration", "LocateAnythingFastForConditionalGeneration"):
        ModelRegistry.register_model(name, f"vllm_hcu.models.locateanything.model:{name}")
