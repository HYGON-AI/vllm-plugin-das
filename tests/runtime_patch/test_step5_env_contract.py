# SPDX-License-Identifier: Apache-2.0
"""Step5 environment values must agree across config and model construction."""

import subprocess
import sys
import warnings

import pytest

from vllm_hcu.platforms import step5_envs


@pytest.mark.parametrize(
    "value, disabled",
    [
        (None, False),
        ("", False),
        ("0", False),
        ("false", False),
        (" FALSE ", False),
        ("1", True),
        ("true", True),
        ("TRUE", True),
        (" TrUe ", True),
        (" 1 ", True),
        ("yes", False),
    ],
)
def test_dsa_switch_controls_compilation_registration(monkeypatch, value, disabled):
    from vllm.config.compilation import CompilationConfig
    from vllm_hcu.models.step5_config import Step5Config

    for name in step5_envs.IGNORED_ENVIRONMENT_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    if value is None:
        monkeypatch.delenv("VLLM_STEP5_DISABLE_DSA", raising=False)
    else:
        monkeypatch.setenv("VLLM_STEP5_DISABLE_DSA", value)
    monkeypatch.setattr(CompilationConfig, "_attention_ops", [])

    Step5Config(sparse_config={"enabled": True})
    registered = "vllm::step4_dsa_index" in CompilationConfig._attention_ops
    getter = step5_envs.ENVIRONMENT_VARIABLES["VLLM_STEP5_DISABLE_DSA"]
    assert getter() is disabled
    assert registered is not disabled


def test_custom_ops_master_does_not_disable_dsa_semantics(monkeypatch):
    from vllm.config.compilation import CompilationConfig
    from vllm_hcu.models.step5_config import Step5Config

    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "0")
    monkeypatch.delenv("VLLM_STEP5_DISABLE_DSA", raising=False)
    monkeypatch.setattr(CompilationConfig, "_attention_ops", [])

    assert step5_envs.is_dsa_disabled() is False
    Step5Config(sparse_config={"enabled": True})
    assert "vllm::step4_dsa_index" in CompilationConfig._attention_ops


@pytest.mark.parametrize("name", step5_envs.IGNORED_ENVIRONMENT_VARIABLES)
def test_obsolete_settings_warn_only_when_explicitly_set(monkeypatch, name):
    from vllm_hcu.models.step5_config import Step5Config

    for key in step5_envs.IGNORED_ENVIRONMENT_VARIABLES:
        monkeypatch.delenv(key, raising=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Step5Config()
    assert not [warning for warning in caught
                if "is obsolete and has no effect" in str(warning.message)]
    monkeypatch.setenv(name, "0")
    with pytest.warns(UserWarning, match=f"{name} is obsolete and has no effect"):
        Step5Config()
    assert name not in step5_envs.ENVIRONMENT_VARIABLES


def test_env_helpers_remain_dependency_light():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import vllm_hcu.platforms.step5_envs; "
            "assert 'torch' not in sys.modules; "
            "assert 'vllm' not in sys.modules",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
