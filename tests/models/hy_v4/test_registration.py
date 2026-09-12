# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace

import pytest
from transformers import AutoConfig
from transformers.configuration_utils import PreTrainedConfig
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

from vllm_hcu.models.hy_v4.config import HYV4Config, register_hy_v4_config


def test_register_hy_v4_config_is_idempotent(monkeypatch) -> None:
    from vllm.transformers_utils import config as vllm_config

    registry: dict[str, object] = {}
    monkeypatch.setattr(vllm_config, "_CONFIG_REGISTRY", registry)

    register_hy_v4_config()
    register_hy_v4_config()

    assert registry == {"hy_v4": HYV4Config}
    assert isinstance(AutoConfig.for_model("hy_v4"), HYV4Config)


def test_register_hy_v4_config_rejects_different_vllm_owner(monkeypatch) -> None:
    from vllm.transformers_utils import config as vllm_config

    class ForeignConfig(PreTrainedConfig):
        model_type = "hy_v4"

    registry: dict[str, object] = {"hy_v4": ForeignConfig}
    monkeypatch.setattr(vllm_config, "_CONFIG_REGISTRY", registry)

    with pytest.raises(RuntimeError, match="vLLM.*different owner"):
        register_hy_v4_config()

    assert registry == {"hy_v4": ForeignConfig}


def test_register_hy_v4_config_rejects_different_transformers_owner(
    monkeypatch,
) -> None:
    from vllm.transformers_utils import config as vllm_config

    class ForeignConfig(PreTrainedConfig):
        model_type = "hy_v4"

    registry: dict[str, object] = {}
    monkeypatch.setattr(vllm_config, "_CONFIG_REGISTRY", registry)
    monkeypatch.setitem(CONFIG_MAPPING._extra_content, "hy_v4", ForeignConfig)

    with pytest.raises(RuntimeError, match="Transformers.*different owner"):
        register_hy_v4_config()

    assert registry == {}


def test_hy_v4_config_derives_architecture_defaults() -> None:
    config = HYV4Config(
        num_hidden_layers=6,
        qk_nope_head_dim=192,
        qk_rope_head_dim=64,
    )

    assert config.qk_head_dim == 256
    assert config.head_dim == 64
    assert config.mlp_layer_types == [
        "dense",
        "sparse",
        "sparse",
        "sparse",
        "sparse",
        "sparse",
    ]
    assert config.indexer_types == [
        "full",
        "full",
        "shared",
        "shared",
        "shared",
        "full",
    ]


def test_hy_v4_config_normalizes_sparse_layer_spelling() -> None:
    config = HYV4Config(
        num_hidden_layers=3,
        layer_types=["sparse", "sparse_attention", "full_attention"],
    )

    assert config.layer_types == [
        "deepseek_sparse_attention",
        "deepseek_sparse_attention",
        "full_attention",
    ]


def test_hy_v4_fp32_lm_head_uses_logits_processor_head_dtype() -> None:
    config = HYV4Config(enable_lm_head_fp32=True)

    assert config.head_dtype == "float32"


def test_hy_v4_explicit_head_dtype_is_preserved() -> None:
    config = HYV4Config(enable_lm_head_fp32=True, head_dtype="model")

    assert config.head_dtype == "model"


def test_hy_v4_disabled_fp32_lm_head_does_not_invent_head_dtype() -> None:
    config = HYV4Config(enable_lm_head_fp32=False)

    assert getattr(config, "head_dtype", None) is None


def test_hy_v4_registry_includes_target_and_mtp_architectures(monkeypatch) -> None:
    import vllm_hcu.models as models

    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        models.ModelRegistry,
        "register_model",
        lambda name, path: calls.append((name, path)),
    )
    monkeypatch.setattr(models, "register_hy_v4_config", lambda: None)

    models.register_model()

    assert (
        "HYV4ForCausalLM",
        "vllm_hcu.models.hy_v4:HYV4ForCausalLM",
    ) in calls
    assert ("HYV4MTPModel", "vllm_hcu.models.hy_v4:HYV4MTP") in calls


def test_config_registration_does_not_eagerly_import_model() -> None:
    code = """
import sys
import vllm_hcu.models.hy_v4.config
assert 'vllm_hcu.models.hy_v4.model' not in sys.modules
assert 'vllm_hcu.models.hy_v4.mtp' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_hy_v4_mtp_registry_resolves_lazy_native_class():
    import vllm_hcu.models as models
    from vllm_hcu.models.hy_v4 import HYV4MTP

    models.register_model()
    cls, architecture = models.ModelRegistry.resolve_model_cls(
        ["HYV4MTPModel"], SimpleNamespace(model_impl="auto"),
    )
    assert cls is HYV4MTP
    assert architecture == "HYV4MTPModel"
