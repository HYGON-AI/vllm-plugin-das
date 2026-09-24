# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Fail-closed contracts for the DeepSeek-V4/V4.1 PCP capability gate."""

from __future__ import annotations

import os
from types import ModuleType

import pytest

from vllm_hcu.patch.worker.core_fix import patch_deepseek_v4_pcp


def _backend_module() -> ModuleType:
    """Build a stand-in for ``vllm.models.deepseek_v41.sparse_mla``."""

    module = ModuleType(patch_deepseek_v4_pcp.TARGET_MODULE)

    class DeepseekV4SparseMLABackend:
        @staticmethod
        def get_impl_cls():
            raise NotImplementedError("no separate impl class")

    class DeepseekV4FlashMLABackend(DeepseekV4SparseMLABackend):
        pass

    class FlashMLAMegaAttnBackend(DeepseekV4FlashMLABackend):
        pass

    module.DeepseekV4SparseMLABackend = DeepseekV4SparseMLABackend
    module.DeepseekV4FlashMLABackend = DeepseekV4FlashMLABackend
    module.FlashMLAMegaAttnBackend = FlashMLAMegaAttnBackend
    return module


def _stub_auxiliary_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the lazily imported cache-owner modules with stubs."""

    def make(name: str, class_name: str) -> ModuleType:
        module = ModuleType(name)
        setattr(module, class_name, type(class_name, (), {}))
        return module

    for module_name, class_names in patch_deepseek_v4_pcp._AUXILIARY_BACKENDS:
        stub = make(module_name, class_names[0])
        monkeypatch.setitem(
            __import__("sys").modules, module_name, stub
        )


def test_capability_stays_disabled_without_the_experimental_switch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unvalidated V4/V4.1 PCP path must not arm by default."""

    monkeypatch.delenv(patch_deepseek_v4_pcp._ENV_FLAG, raising=False)
    module = _backend_module()

    assert patch_deepseek_v4_pcp.apply_to_module(module) is False
    assert not hasattr(module.DeepseekV4SparseMLABackend, "supports_pcp")
    assert not hasattr(module, "_vllm_hcu_dsv4_pcp_capability_applied")


def test_capability_is_inert_for_other_flag_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only an explicit truthy switch may open the gate."""

    monkeypatch.setenv(patch_deepseek_v4_pcp._ENV_FLAG, "0")
    module = _backend_module()

    assert patch_deepseek_v4_pcp.apply_to_module(module) is False
    assert not hasattr(module.DeepseekV4SparseMLABackend, "supports_pcp")


def test_capability_is_published_only_under_the_switch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bring-up switch must publish PCP on every validated backend."""

    monkeypatch.setenv(patch_deepseek_v4_pcp._ENV_FLAG, "1")
    _stub_auxiliary_modules(monkeypatch)
    module = _backend_module()

    assert patch_deepseek_v4_pcp.apply_to_module(module) is True
    for backend in (
        module.DeepseekV4SparseMLABackend,
        module.DeepseekV4FlashMLABackend,
        module.FlashMLAMegaAttnBackend,
    ):
        assert backend.supports_pcp() is True
    # The missing-impl contract is preserved: attention still owns the forward.
    with pytest.raises(NotImplementedError):
        module.DeepseekV4SparseMLABackend.get_impl_cls()


def test_capability_application_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A repeated callback must not re-publish or double-wrap the capability."""

    monkeypatch.setenv(patch_deepseek_v4_pcp._ENV_FLAG, "1")
    _stub_auxiliary_modules(monkeypatch)
    module = _backend_module()

    assert patch_deepseek_v4_pcp.apply_to_module(module) is True
    assert patch_deepseek_v4_pcp.apply_to_module(module) is False


def test_existing_capability_is_never_masked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Refuse to overwrite a real upstream capability decision."""

    monkeypatch.setenv(patch_deepseek_v4_pcp._ENV_FLAG, "1")
    _stub_auxiliary_modules(monkeypatch)
    module = _backend_module()
    module.DeepseekV4SparseMLABackend.supports_pcp = classmethod(lambda cls: False)

    from vllm_hcu.patch.worker.core_fix._common import PatchCompatibilityError

    with pytest.raises(PatchCompatibilityError, match="already defines"):
        patch_deepseek_v4_pcp.apply_to_module(module)


def test_wrong_target_module_is_rejected() -> None:
    """Pointing the callback at another module must fail loudly."""

    from vllm_hcu.patch.worker.core_fix._common import PatchCompatibilityError

    other = ModuleType("vllm.models.deepseek_v41.attention")
    with pytest.raises(PatchCompatibilityError, match="expected module"):
        patch_deepseek_v4_pcp.apply_to_module(other)


def test_flag_default_matches_the_config_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    """The adapter and the platform scope gate must share one switch name."""

    from vllm_hcu.patch.platform.core_fix import patch_vllm_config

    assert (
        patch_deepseek_v4_pcp._ENV_FLAG
        == patch_vllm_config._DSV4_PCP_EXPERIMENTAL_ENV
    )
    monkeypatch.setenv(patch_deepseek_v4_pcp._ENV_FLAG, "1")
    assert patch_deepseek_v4_pcp._experimental_enabled() is True
    assert patch_vllm_config._dsv4_pcp_experimental() is True
