# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from __future__ import annotations

import subprocess
import sys

from transformers import AutoConfig

from vllm_hcu.models.hy_v4.config import HYV4Config, register_hy_v4_config


def test_register_hy_v4_config_is_idempotent(monkeypatch) -> None:
    from vllm.transformers_utils import config as vllm_config

    registry: dict[str, object] = {}
    monkeypatch.setattr(vllm_config, "_CONFIG_REGISTRY", registry)

    register_hy_v4_config()
    register_hy_v4_config()

    assert registry == {"hy_v4": HYV4Config}
    assert isinstance(AutoConfig.for_model("hy_v4"), HYV4Config)


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


def test_hy_v4_registry_is_backbone_only(monkeypatch) -> None:
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
    assert all(name != "HYV4MTPModel" for name, _ in calls)


def test_config_registration_does_not_eagerly_import_model() -> None:
    code = """
import sys
import vllm_hcu.models.hy_v4.config
assert 'vllm_hcu.models.hy_v4.model' not in sys.modules
from vllm_hcu.models.hy_v4 import HYV4ForCausalLM
assert HYV4ForCausalLM.__name__ == 'HYV4ForCausalLM'
"""
    subprocess.run([sys.executable, "-c", code], check=True)
