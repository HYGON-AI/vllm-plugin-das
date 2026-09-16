# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Downstream adapters must remain optional and preserve split-P/D contracts."""

from types import SimpleNamespace
import sys

import pytest

from vllm_hcu import scheduler_registry as registry
from vllm_hcu.patch.platform.framework_opt import patch_scheduler
from vllm_hcu.platforms import envs as henvs


@pytest.fixture(autouse=True)
def isolated_registry(monkeypatch):
    monkeypatch.setattr(registry, "_SCHEDULER_ADAPTERS", {})
    monkeypatch.setattr(henvs, "VLLM_HCU_USE_PD_SPLIT", True)
    monkeypatch.setattr(henvs, "VLLM_HCU_USE_CUSTOM_OPS", True)


def config(path, async_mode):
    return SimpleNamespace(
        additional_config={"hcu": {}},
        cache_config=SimpleNamespace(enable_prefix_caching=False),
        scheduler_config=SimpleNamespace(
            scheduler_cls=path, async_scheduling=async_mode
        ),
    )


@pytest.mark.parametrize("async_mode", [False, True])
def test_lazy_idempotent_registration_preserves_explicit_selection(async_mode):
    path = "uninstalled_downstream.scheduler.CustomScheduler"
    for _ in range(2):
        registry.register_scheduler_adapter(path, async_scheduling=async_mode)
    assert "uninstalled_downstream" not in sys.modules
    value = config(path, async_mode)
    assert patch_scheduler.select_hcu_scheduler(value) is False
    assert value.scheduler_config.scheduler_cls == path
    value.scheduler_config.async_scheduling = not async_mode
    with pytest.raises(RuntimeError):
        patch_scheduler.select_hcu_scheduler(value)


def test_unregistered_custom_scheduler_is_not_implicitly_trusted():
    with pytest.raises(RuntimeError, match="another scheduler_cls"):
        patch_scheduler.select_hcu_scheduler(config("downstream.Custom", True))


@pytest.mark.parametrize("first", [False, True])
def test_conflicting_registration_does_not_change_existing_contract(first):
    registry.register_scheduler_adapter("downstream.Custom", async_scheduling=first)
    with pytest.raises(ValueError, match="Conflicting"):
        registry.register_scheduler_adapter(
            "downstream.Custom", async_scheduling=not first
        )
    assert registry.get_scheduler_adapter_mode("downstream.Custom") is first


@pytest.mark.parametrize("path", ["", "NoModule", "a..Class", "a.invalid-name", None])
def test_invalid_path_is_rejected(path):
    with pytest.raises(ValueError):
        registry.register_scheduler_adapter(path, async_scheduling=True)


def test_builtin_scheduler_contract_cannot_be_overridden():
    with pytest.raises(ValueError, match="Built-in"):
        registry.register_scheduler_adapter(
            patch_scheduler.HCU_SCHEDULER_PATH, async_scheduling=True
        )
    with pytest.raises(TypeError, match="bool"):
        registry.register_scheduler_adapter("downstream.Custom", async_scheduling=1)


def test_native_async_selection_without_downstream_packages():
    value = config(patch_scheduler.HCU_ASYNC_SCHEDULER_PATH, True)
    assert patch_scheduler.select_hcu_scheduler(value) is False
    value.scheduler_config.async_scheduling = False
    with pytest.raises(RuntimeError):
        patch_scheduler.select_hcu_scheduler(value)


def test_registered_sync_adapter_pins_unspecified_async_mode():
    registry.register_scheduler_adapter("downstream.Sync", async_scheduling=False)
    value = config("downstream.Sync", None)
    assert patch_scheduler.select_hcu_scheduler(value) is False
    assert value.scheduler_config.async_scheduling is False
