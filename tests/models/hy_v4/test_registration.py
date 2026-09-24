# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from vllm import ModelRegistry
from vllm.transformers_utils.configs.hy_v4 import HYV4Config

from vllm_hcu.models import register_model


def test_hcu_hyv4_registration_uses_plugin_models(monkeypatch):
    registrations: list[tuple[str, str]] = []
    monkeypatch.setattr(
        ModelRegistry,
        "register_model",
        lambda architecture, implementation: registrations.append(
            (architecture, implementation)
        ),
    )

    register_model()

    assert (
        "HYV4ForCausalLM",
        "vllm_hcu.models.hy_v4:HYV4ForCausalLM",
    ) in registrations
    assert (
        "HYV4MTPModel",
        "vllm_hcu.models.hy_v4:HYV4MTP",
    ) in registrations


def test_hcu_hyv4_reuses_target_config_owner():
    from vllm_hcu.models.hy_v4 import HYV4Config as PluginHYV4Config

    assert PluginHYV4Config is HYV4Config
