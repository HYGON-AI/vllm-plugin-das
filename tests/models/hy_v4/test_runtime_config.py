# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from __future__ import annotations

import os
from types import ModuleType, SimpleNamespace

import torch
import pytest
from vllm.config.model import ModelConfig
from vllm.config import vllm as vllm_config
from vllm.transformers_utils.model_arch_config_convertor import (
    ModelArchConfigConvertorBase,
)

from vllm_hcu.models.hy_v4.config import HYV4Config
from vllm_hcu.patch.platform.core_fix import (
    patch_hy_v4_model_arch_config,
    patch_hy_v4_model_head_dtype,
    patch_hy_v4_vllm_config,
    patch_logits_processor_head_dtype,
)


def _convertor(config) -> ModelArchConfigConvertorBase:
    return ModelArchConfigConvertorBase(config, config)


def test_hyv4_is_a_default_model_runner_v2_architecture() -> None:
    patch_hy_v4_vllm_config.apply_to_module(vllm_config)

    assert "HYV4ForCausalLM" in vllm_config.DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES


def test_hyv4_target_are_classified_as_mla() -> None:
    import vllm.transformers_utils.model_arch_config_convertor as convertor_module

    patch_hy_v4_model_arch_config.apply_to_module(convertor_module)
    target = HYV4Config(architectures=["HYV4ForCausalLM"])
    draft = SimpleNamespace(
        model_type="hy_v4_mtp",
        kv_lora_rank=512,
    )
    unrelated = SimpleNamespace(model_type="unrelated")

    assert _convertor(target).is_deepseek_mla() is True
    assert _convertor(draft).is_deepseek_mla() is False
    assert _convertor(unrelated).is_deepseek_mla() is False


def test_hyv4_generation_model_config_honors_explicit_fp32_head_dtype() -> None:
    import vllm.config.model as model_module

    patch_hy_v4_model_head_dtype.apply_to_module(model_module)
    config = object.__new__(ModelConfig)
    config.hf_config = SimpleNamespace(
        model_type="hy_v4",
        head_dtype="float32",
    )
    config.dtype = torch.bfloat16
    config.runner_type = "generate"

    assert config.head_dtype is torch.float32


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


@pytest.mark.parametrize("adapter", [patch_hy_v4_model_arch_config,
    patch_hy_v4_vllm_config, patch_hy_v4_model_head_dtype,
    patch_logits_processor_head_dtype])
def test_adapters_reject_wrong_owner_and_incompatible_api(adapter):
    from vllm_hcu.patch.platform.core_fix._common import PatchCompatibilityError
    with pytest.raises(PatchCompatibilityError, match="expected module"):
        adapter.apply_to_module(ModuleType("wrong.owner"))
    with pytest.raises(PatchCompatibilityError):
        adapter.apply_to_module(ModuleType(adapter.TARGET_MODULE))


def test_runtime_adapter_is_marked_and_applies_once():
    module = _fake_vllm_config_module()
    assert patch_hy_v4_vllm_config.apply_to_module(module) is True
    callback = module.VllmConfig.__post_init__
    assert getattr(module, patch_hy_v4_vllm_config._MARKER)
    assert patch_hy_v4_vllm_config.apply_to_module(module) is False
    assert module.VllmConfig.__post_init__ is callback


def test_target_adapters_registered_in_dependency_order():
    from vllm_hcu.patch.platform.core_fix import platform_core_callback_names
    names = [name for name, _ in platform_core_callback_names()]
    expected = [patch_hy_v4_model_arch_config.PATCH_ID,
                patch_hy_v4_vllm_config.PATCH_ID,
                patch_hy_v4_model_head_dtype.PATCH_ID,
                patch_logits_processor_head_dtype.PATCH_ID]
    assert [name for name in names if name in expected] == expected


@pytest.mark.parametrize("adapter", [patch_hy_v4_model_arch_config,
    patch_hy_v4_vllm_config, patch_hy_v4_model_head_dtype,
    patch_logits_processor_head_dtype])
def test_pinned_adapters_expose_marker_and_are_idempotent(adapter):
    import importlib
    module = importlib.import_module(adapter.TARGET_MODULE)
    adapter.apply_to_module(module)
    assert getattr(module, adapter._MARKER) is True
    assert adapter.apply_to_module(module) is False


@pytest.mark.parametrize("head_dtype,expected", [
    ("model", torch.bfloat16), ("bfloat16", torch.bfloat16),
    ("float32", torch.float32)])
def test_target_head_dtype_preserves_explicit_choice(head_dtype, expected):
    import vllm.config.model as model_module
    patch_hy_v4_model_head_dtype.apply_to_module(model_module)
    config = object.__new__(ModelConfig)
    config.hf_config = SimpleNamespace(model_type="hy_v4", head_dtype=head_dtype)
    config.dtype = torch.bfloat16
    config.runner_type = "generate"
    assert config.head_dtype is expected
