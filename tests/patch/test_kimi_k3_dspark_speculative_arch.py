# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Mechanism tests for the K3 DSpark speculative-architecture adapter.

The fake wheel module mirrors the audited real lifecycle: ``__init__`` leaves
``draft_model_config`` as ``None`` and ``__post_init__`` creates it and
rewrites the DSv4-bound string fields.  The K3 draft is represented by a
genuinely named ``K3DSparkConfig`` class because the adapter identifies the
draft by its surviving config class, not by the rewritten strings.
"""

from __future__ import annotations

import types

import pytest

from vllm_hcu.patch.platform.core_fix import patch_kimi_k3_dspark_speculative_arch
from vllm_hcu.patch.platform.core_fix._common import PatchCompatibilityError

REWRITE_ARCHS = ["DSparkDraftModel"]


def _make_fake_module():
    class K3DSparkConfig:
        model_type = "k3_dspark"

        def __init__(self, architectures):
            self.model_type = "k3_dspark"
            self.architectures = list(architectures)

    class DeepseekV4HFConfig:
        model_type = "deepseek_v4"

        def __init__(self, architectures):
            self.model_type = "deepseek_v4"
            self.architectures = list(architectures)

    module = types.ModuleType(patch_kimi_k3_dspark_speculative_arch.TARGET_MODULE)

    class SpeculativeConfig:
        def __init__(self, method, hf_config):
            self.method = method
            # Wheel truth: the dataclass default is None and the first
            # assignment happens inside __post_init__.
            self.draft_model_config = None
            self._pending_hf_config = hf_config
            self.update_calls = 0
            self.final_architectures = None
            self.final_model_type = None

        def update_arch_(self):
            self.update_calls += 1
            hf_config = self.draft_model_config.hf_config
            self.final_architectures = list(hf_config.architectures)
            self.final_model_type = hf_config.model_type

        def __post_init__(self):
            hf_config = self._pending_hf_config
            self.draft_model_config = types.SimpleNamespace(hf_config=hf_config)
            if self.method == "dspark" and (
                "Qwen3DSparkModel" not in hf_config.architectures
            ):
                hf_config.model_type = "deepseek_v4"
                hf_config.architectures = list(REWRITE_ARCHS)

    module.SpeculativeConfig = SpeculativeConfig
    module.K3DSparkConfig = K3DSparkConfig
    module.DeepseekV4HFConfig = DeepseekV4HFConfig
    return module


def _k3_instance(module):
    instance = module.SpeculativeConfig(
        "dspark", module.K3DSparkConfig(["K3DSparkModel"])
    )
    instance.__post_init__()
    return instance


def _read_back(instance):
    hf_config = instance.draft_model_config.hf_config
    return (
        hf_config.model_type,
        list(hf_config.architectures),
        instance.update_calls,
        instance.final_architectures,
        instance.final_model_type,
    )


def test_k3_dspark_identity_is_restored_and_resynchronized():
    module = _make_fake_module()
    assert patch_kimi_k3_dspark_speculative_arch.apply_to_module(module) is True

    instance = _k3_instance(module)

    assert _read_back(instance) == (
        "k3_dspark",
        ["K3DSparkModel"],
        1,
        ["K3DSparkModel"],
        "k3_dspark",
    )


def test_restoration_fires_even_though_draft_is_none_before_post_init():
    """Regression for the deployed no-op: no state is captured pre-call."""

    module = _make_fake_module()
    patch_kimi_k3_dspark_speculative_arch.apply_to_module(module)

    instance = module.SpeculativeConfig(
        "dspark", module.K3DSparkConfig(["K3DSparkModel"])
    )
    assert instance.draft_model_config is None

    instance.__post_init__()

    assert _read_back(instance)[:2] == ("k3_dspark", ["K3DSparkModel"])


def test_restoration_mutates_the_wheel_config_object_in_place():
    module = _make_fake_module()
    patch_kimi_k3_dspark_speculative_arch.apply_to_module(module)

    instance = module.SpeculativeConfig(
        "dspark", module.K3DSparkConfig(["K3DSparkModel"])
    )
    hf_config = instance._pending_hf_config
    instance.__post_init__()

    assert instance.draft_model_config.hf_config is hf_config
    assert hf_config.model_type == "k3_dspark"


def test_dsv4_dspark_normalization_stays_untouched():
    module = _make_fake_module()
    patch_kimi_k3_dspark_speculative_arch.apply_to_module(module)

    instance = module.SpeculativeConfig(
        "dspark", module.DeepseekV4HFConfig(REWRITE_ARCHS)
    )
    instance.__post_init__()

    assert _read_back(instance) == (
        "deepseek_v4",
        REWRITE_ARCHS,
        0,
        None,
        None,
    )


def test_upstream_qwen_exemption_is_not_second_guessed():
    module = _make_fake_module()
    patch_kimi_k3_dspark_speculative_arch.apply_to_module(module)

    instance = module.SpeculativeConfig(
        "dspark", module.DeepseekV4HFConfig(["Qwen3DSparkModel"])
    )
    instance.__post_init__()

    hf_config = instance.draft_model_config.hf_config
    assert hf_config.architectures == ["Qwen3DSparkModel"]
    assert instance.update_calls == 0


def test_non_dspark_methods_are_never_rewritten():
    module = _make_fake_module()
    patch_kimi_k3_dspark_speculative_arch.apply_to_module(module)

    instance = module.SpeculativeConfig(
        "eagle3", module.K3DSparkConfig(["K3DSparkModel"])
    )
    instance.__post_init__()

    assert instance.update_calls == 0
    assert _read_back(instance)[:2] == ("k3_dspark", ["K3DSparkModel"])


def test_wrapped_post_init_keeps_name_and_sets_class_marker():
    module = _make_fake_module()
    patch_kimi_k3_dspark_speculative_arch.apply_to_module(module)

    wrapped = module.SpeculativeConfig.__post_init__
    assert wrapped.__name__ == "__post_init__"
    assert getattr(
        module.SpeculativeConfig,
        "_vllm_hcu_kimi_k3_dspark_speculative_arch_applied",
    )


def test_direct_apply_is_one_shot_per_module():
    module = _make_fake_module()
    assert patch_kimi_k3_dspark_speculative_arch.apply(module) is True
    assert patch_kimi_k3_dspark_speculative_arch.apply(module) is False


def test_rejects_incompatible_post_init_signature():
    module = _make_fake_module()

    def __post_init__(self, unexpected):
        raise AssertionError("never called")

    module.SpeculativeConfig.__post_init__ = __post_init__

    with pytest.raises(PatchCompatibilityError, match="incompatible signature"):
        patch_kimi_k3_dspark_speculative_arch.apply_to_module(module)


def test_rejects_missing_update_arch_resync_entry_point():
    module = _make_fake_module()
    del module.SpeculativeConfig.update_arch_

    with pytest.raises(PatchCompatibilityError, match="update_arch_"):
        patch_kimi_k3_dspark_speculative_arch.apply_to_module(module)


def test_installs_against_the_real_wheel_module():
    real_module = pytest.importorskip("vllm.config.speculative")

    assert (
        patch_kimi_k3_dspark_speculative_arch.apply_to_module(real_module) is True
    )
    assert getattr(
        real_module.SpeculativeConfig,
        "_vllm_hcu_kimi_k3_dspark_speculative_arch_applied",
    )
    # The wrapper stays signature-faithful to the wrapped entry point.
    wrapped = real_module.SpeculativeConfig.__post_init__
    assert callable(wrapped)
    assert wrapped.__name__ == "__post_init__"
