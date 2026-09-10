# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Contracts for handing HCU PCP+MTP to the plugin-owned PCP manager."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest


TARGET_MODULE = "vllm.v1.worker.gpu.pcp_manager"
ADAPTER_MODULE = (
    "vllm_hcu.patch.worker.framework_opt.patch_pcp_spec_validation"
)


def _load_adapter(monkeypatch: pytest.MonkeyPatch):
    calls: list[tuple[object, bool]] = []
    target = ModuleType(TARGET_MODULE)

    class PCPManager:
        @staticmethod
        def validate_config(vllm_config, supports_mm_inputs):
            calls.append((vllm_config, supports_mm_inputs))
            if vllm_config.speculative_config is not None:
                raise NotImplementedError(
                    "MRV2 PCP does not support speculative decoding yet."
                )
            if supports_mm_inputs:
                raise RuntimeError("official validation after speculative guard")

    target.PCPManager = PCPManager
    monkeypatch.setitem(sys.modules, TARGET_MODULE, target)
    monkeypatch.delitem(sys.modules, ADAPTER_MODULE, raising=False)
    adapter = importlib.import_module(ADAPTER_MODULE)
    return adapter, target, PCPManager, calls


def test_supported_hcu_pcp_mtp_preserves_official_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, target, manager, calls = _load_adapter(monkeypatch)
    monkeypatch.setattr(adapter, "_validate_hcu_pcp_scope", lambda config: True)
    speculative_config = object()
    config = SimpleNamespace(speculative_config=speculative_config)

    assert adapter.apply_to_module(target) is True
    manager.validate_config(config, False)

    assert len(calls) == 1
    validated_config, supports_mm_inputs = calls[0]
    assert validated_config is not config
    assert validated_config.speculative_config is None
    assert supports_mm_inputs is False
    assert config.speculative_config is speculative_config

    with pytest.raises(RuntimeError, match="official validation after"):
        manager.validate_config(config, True)


def test_non_speculative_pcp_delegates_without_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, target, manager, calls = _load_adapter(monkeypatch)
    config = SimpleNamespace(speculative_config=None)

    adapter.apply_to_module(target)
    manager.validate_config(config, False)

    assert calls == [(config, False)]


def test_unsupported_hcu_pcp_spec_fails_before_official_bypass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, target, manager, calls = _load_adapter(monkeypatch)

    def reject(config):
        raise ValueError("unsupported HCU PCP scope")

    monkeypatch.setattr(adapter, "_validate_hcu_pcp_scope", reject)
    config = SimpleNamespace(speculative_config=object())

    adapter.apply_to_module(target)
    with pytest.raises(ValueError, match="unsupported HCU PCP scope"):
        manager.validate_config(config, False)

    assert calls == []
