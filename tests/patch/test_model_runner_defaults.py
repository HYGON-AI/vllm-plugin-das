# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""HCU runner defaults, explicit overrides, and upstream validation contracts."""

from __future__ import annotations

import functools
import logging
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from vllm_hcu.patch.platform.core_fix import patch_envs
from vllm_hcu.patch.platform.core_fix._common import PatchCompatibilityError

RUNNER_ENV = "VLLM_USE_V2_MODEL_RUNNER"
REPO = Path(__file__).resolve().parents[2]


def _env_module(runner_getter) -> ModuleType:
    module = ModuleType("vllm.envs")
    module.environment_variables = {
        RUNNER_ENV: runner_getter,
        "VLLM_ROCM_USE_AITER_MOE": lambda: True,
    }
    module.validate_environ = lambda hard_fail: None
    module.logger = logging.getLogger(__name__)

    @functools.cache
    def get_value(name):
        if name not in module.environment_variables:
            raise AttributeError(name)
        return module.environment_variables[name]()

    module.__getattr__ = get_value
    return module


def test_unset_runner_defaults_to_v2_without_exporting_process_environment(monkeypatch):
    monkeypatch.delenv(RUNNER_ENV, raising=False)
    monkeypatch.setenv("VLLM_KV_CACHE_LAYOUT", "NHD")
    module = _env_module(lambda: None)

    patch_envs.apply_to_module(module)

    assert module.VLLM_USE_V2_MODEL_RUNNER is True
    assert RUNNER_ENV not in os.environ
    assert os.environ["VLLM_KV_CACHE_LAYOUT"] == "NHD"


@pytest.mark.parametrize("explicit", ["1", "true", "True"])
def test_explicit_v2_is_preserved(monkeypatch, explicit):
    monkeypatch.setenv(RUNNER_ENV, explicit)
    module = _env_module(lambda: True)

    patch_envs.apply_to_module(module)

    assert module.VLLM_USE_V2_MODEL_RUNNER is True
    assert os.environ[RUNNER_ENV] == explicit


@pytest.mark.parametrize("explicit", ["0", "false", "False"])
def test_explicit_v1_is_rejected_without_overwriting(monkeypatch, explicit):
    monkeypatch.setenv(RUNNER_ENV, explicit)
    module = _env_module(lambda: False)
    patch_envs.apply_to_module(module)

    with pytest.raises(ValueError, match="HCU supports only Model Runner V2"):
        _ = module.VLLM_USE_V2_MODEL_RUNNER

    assert os.environ[RUNNER_ENV] == explicit


def test_cached_upstream_none_is_invalidated():
    module = _env_module(lambda: None)
    assert module.VLLM_USE_V2_MODEL_RUNNER is None

    patch_envs.apply_to_module(module)

    assert module.VLLM_USE_V2_MODEL_RUNNER is True


def test_original_parser_errors_are_preserved():
    def reject_invalid_value():
        raise ValueError("invalid upstream boolean")

    module = _env_module(reject_invalid_value)
    patch_envs.apply_to_module(module)

    with pytest.raises(ValueError, match="^invalid upstream boolean$"):
        _ = module.VLLM_USE_V2_MODEL_RUNNER


def test_missing_upstream_getter_fails_before_mutating_other_defaults():
    module = _env_module(lambda: None)
    del module.environment_variables[RUNNER_ENV]
    original_aiter = module.environment_variables["VLLM_ROCM_USE_AITER_MOE"]

    with pytest.raises(PatchCompatibilityError, match="VLLM_USE_V2_MODEL_RUNNER"):
        patch_envs.apply_to_module(module)

    assert module.environment_variables["VLLM_ROCM_USE_AITER_MOE"] is original_aiter


def test_adapter_is_idempotent_and_preserves_original_getter():
    def original():
        return None

    module = _env_module(original)
    assert patch_envs.apply_to_module(module) is True
    installed = module.environment_variables[RUNNER_ENV]

    assert patch_envs.apply_to_module(module) is False
    assert module.environment_variables[RUNNER_ENV] is installed
    assert module._vllm_hcu_original_model_runner_v2_getter is original


def test_unadapted_environment_keeps_upstream_default():
    hcu_module = _env_module(lambda: None)
    upstream_module = _env_module(lambda: None)

    patch_envs.apply_to_module(hcu_module)

    assert hcu_module.VLLM_USE_V2_MODEL_RUNNER is True
    assert upstream_module.VLLM_USE_V2_MODEL_RUNNER is None


def test_fresh_child_process_applies_hcu_default_without_export(monkeypatch):
    monkeypatch.delenv(RUNNER_ENV, raising=False)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(REPO)
    environment["VLLM_PLUGINS"] = "__disabled__"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import logging, os; from types import ModuleType; "
                "from vllm_hcu.patch.platform.core_fix import patch_envs; "
                "m = ModuleType('vllm.envs'); "
                "m.environment_variables = {'VLLM_USE_V2_MODEL_RUNNER': lambda: None, "
                "'VLLM_ROCM_USE_AITER_MOE': lambda: True}; "
                "m.validate_environ = lambda hard_fail: None; "
                "m.logger = logging.getLogger('child'); "
                "patch_envs.apply_to_module(m); "
                "assert m.environment_variables['VLLM_USE_V2_MODEL_RUNNER']() is True; "
                "assert 'VLLM_USE_V2_MODEL_RUNNER' not in os.environ; "
                "print('hcu-v2-default-ok')"
            ),
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "hcu-v2-default-ok" in result.stdout


@pytest.fixture
def target_vllm(monkeypatch):
    """Use the real vLLM parser/property/validator when the runtime is installed."""
    envs = pytest.importorskip("vllm.envs")
    config_module = pytest.importorskip("vllm.config.vllm")
    adapter = _env_module(envs.environment_variables[RUNNER_ENV])
    patch_envs.apply_to_module(adapter)
    cache_clear = getattr(envs.__getattr__, "cache_clear", lambda: None)
    cache_clear()
    with monkeypatch.context() as context:
        context.setitem(
            envs.environment_variables,
            RUNNER_ENV,
            adapter.environment_variables[RUNNER_ENV],
        )
        context.setattr(config_module, "HAS_TRITON", True)
        try:
            yield envs, config_module.VllmConfig
        finally:
            cache_clear()
    cache_clear()


@pytest.mark.parametrize(
    "architecture",
    [
        "Qwen3ForCausalLM",
        "DeepseekV3ForCausalLM",
        "DeepseekV32ForCausalLM",
        "DeepseekV4ForCausalLM",
        "GlmMoeDsaForCausalLM",
    ],
)
@pytest.mark.parametrize("setting", [None, "1", "0"])
def test_real_vllm_runner_selection(target_vllm, monkeypatch, architecture, setting):
    envs, config_class = target_vllm
    if setting is None:
        monkeypatch.delenv(RUNNER_ENV, raising=False)
    else:
        monkeypatch.setenv(RUNNER_ENV, setting)
    getattr(envs.__getattr__, "cache_clear", lambda: None)()
    config = SimpleNamespace(
        model_config=SimpleNamespace(architectures=[architecture]),
        _get_v2_model_runner_unsupported_features=list,
    )
    if setting == "0":
        with pytest.raises(ValueError, match="HCU supports only Model Runner V2"):
            config_class.use_v2_model_runner.fget(config)
        assert os.environ[RUNNER_ENV] == "0"
    else:
        assert config_class.use_v2_model_runner.fget(config) is True


def test_real_vllm_still_rejects_unsupported_v2_features(target_vllm, monkeypatch):
    envs, config_class = target_vllm
    monkeypatch.delenv(RUNNER_ENV, raising=False)
    getattr(envs.__getattr__, "cache_clear", lambda: None)()
    config = SimpleNamespace(
        _get_v2_model_runner_unsupported_features=lambda: [
            "ngram/ngram_gpu speculative decoding"
        ],
    )
    assert config_class.use_v2_model_runner.fget(config) is True

    with pytest.raises(ValueError, match="ngram/ngram_gpu speculative decoding"):
        config_class._validate_v2_model_runner(config)
