# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from __future__ import annotations

import os
from types import ModuleType, SimpleNamespace

from vllm.config import vllm as vllm_config
from vllm.transformers_utils.model_arch_config_convertor import (
    ModelArchConfigConvertorBase,
)

from vllm_hcu.models.hy_v4.config import HYV4Config
from vllm_hcu.patch.platform.core_fix import (
    patch_hy_v4_model_arch_config,
    patch_hy_v4_vllm_config,
)


def _convertor(config) -> ModelArchConfigConvertorBase:
    return ModelArchConfigConvertorBase(config, config)


def test_hyv4_is_a_default_model_runner_v2_architecture() -> None:
    patch_hy_v4_vllm_config.apply_to_module(vllm_config)

    assert "HYV4ForCausalLM" in vllm_config.DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES


def test_hyv4_target_and_mtp_are_classified_as_mla() -> None:
    import vllm.transformers_utils.model_arch_config_convertor as convertor_module

    patch_hy_v4_model_arch_config.apply_to_module(convertor_module)
    target = HYV4Config(architectures=["HYV4ForCausalLM"])
    draft = SimpleNamespace(
        model_type="hy_v4_mtp",
        kv_lora_rank=512,
    )
    unrelated = SimpleNamespace(model_type="unrelated")

    assert _convertor(target).is_deepseek_mla() is True
    assert _convertor(draft).is_deepseek_mla() is True
    assert _convertor(unrelated).is_deepseek_mla() is False


def _fake_vllm_config_module() -> ModuleType:
    module = ModuleType("vllm.config.vllm")
    module.DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES = frozenset({"ExistingModel"})

    class VllmConfig:
        def __init__(self, architecture: str):
            self.model_config = SimpleNamespace(architectures=[architecture])
            self.original_post_init_called = False

        def __post_init__(self) -> None:
            self.original_post_init_called = True

    module.VllmConfig = VllmConfig
    return module


def test_hyv4_defaults_to_breakable_cudagraph_without_overriding_user_env(
    monkeypatch,
) -> None:
    module = _fake_vllm_config_module()
    patch_hy_v4_vllm_config.apply_to_module(module)

    monkeypatch.delenv("VLLM_USE_BREAKABLE_CUDAGRAPH", raising=False)
    target = module.VllmConfig("HYV4ForCausalLM")
    target.__post_init__()
    assert target.original_post_init_called is True
    assert os.environ["VLLM_USE_BREAKABLE_CUDAGRAPH"] == "1"

    monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", "0")
    draft = module.VllmConfig("HYV4MTPModel")
    draft.__post_init__()
    assert draft.original_post_init_called is True
    assert os.environ["VLLM_USE_BREAKABLE_CUDAGRAPH"] == "0"
