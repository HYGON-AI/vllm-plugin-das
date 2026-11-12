# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""The Gemma4 BW1000 switch: what it reads, and where a retired name may fail.

The second half is the point of this file. ``warn_retired_switches`` runs from
model registration, which is a ``vllm.general_plugins`` entry point that vLLM
invokes without wrapping the call, so raising there aborts startup for every
model this plugin serves -- DeepSeek, GLM, Kimi -- over a variable that only
ever concerned Gemma4. Only ``check_retired_switches`` may refuse, and only
for the one combination that would otherwise serve Gemma4 unoptimized in
silence.
"""

from types import SimpleNamespace

import pytest

from vllm_hcu.utils.gemma4_switches import (
    SWITCH,
    check_retired_switches,
    gemma4_bw1000_enabled,
    warn_retired_switches,
)

RETIRED = ("VLLM_HCU_GEMMA4_USE_FA", "VLLM_HCU_GEMMA4_BW1000_OPT")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in (SWITCH, *RETIRED):
        monkeypatch.delenv(name, raising=False)


def config(model_type):
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type=model_type))
    )


@pytest.mark.parametrize(
    "raw, expected",
    [("1", True), ("true", True), ("TRUE", True), (" 1 ", True),
     ("0", False), ("false", False), ("", False), ("   ", False)],
)
def test_switch_spellings(monkeypatch, raw, expected):
    monkeypatch.setenv(SWITCH, raw)
    assert gemma4_bw1000_enabled() is expected


def test_switch_unset_is_off():
    assert gemma4_bw1000_enabled() is False


def test_typo_raises_rather_than_reading_as_off(monkeypatch):
    """A typo that read as "off" would be an unlogged no-op."""
    monkeypatch.setenv(SWITCH, "yes")
    with pytest.raises(ValueError, match=SWITCH):
        gemma4_bw1000_enabled()


@pytest.mark.parametrize("retired", RETIRED)
def test_retired_name_does_not_affect_the_switch(monkeypatch, retired):
    monkeypatch.setenv(retired, "1")
    assert gemma4_bw1000_enabled() is False


@pytest.mark.parametrize("retired", RETIRED)
def test_warn_never_raises(monkeypatch, retired):
    """Registration runs for every model; it must not be able to abort."""
    monkeypatch.setenv(retired, "1")
    warn_retired_switches()  # must not raise


def test_warn_without_retired_names_is_quiet():
    warn_retired_switches()


@pytest.mark.parametrize("retired", RETIRED)
def test_retired_name_refuses_only_for_gemma4(monkeypatch, retired):
    monkeypatch.setenv(retired, "1")
    with pytest.raises(ValueError, match=SWITCH):
        check_retired_switches(config("gemma4"))


@pytest.mark.parametrize("model_type", ["deepseek_v3", "glm4_moe", "kimi_k3", None])
@pytest.mark.parametrize("retired", RETIRED)
def test_retired_name_is_harmless_for_other_models(monkeypatch, retired, model_type):
    monkeypatch.setenv(retired, "1")
    check_retired_switches(config(model_type))  # must not raise


@pytest.mark.parametrize("retired", RETIRED)
def test_retired_name_is_harmless_when_the_new_switch_is_set(monkeypatch, retired):
    """A leftover variable changes nothing once the real switch is on."""
    monkeypatch.setenv(retired, "1")
    monkeypatch.setenv(SWITCH, "1")
    check_retired_switches(config("gemma4"))  # must not raise


def test_check_tolerates_a_config_without_model_config():
    check_retired_switches(SimpleNamespace())


@pytest.mark.parametrize("retired", RETIRED)
def test_check_tolerates_a_bare_config_with_retired_name(monkeypatch, retired):
    monkeypatch.setenv(retired, "1")
    check_retired_switches(SimpleNamespace())  # model unknown -> do not refuse
