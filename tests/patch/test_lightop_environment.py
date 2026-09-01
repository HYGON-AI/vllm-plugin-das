# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Contract tests for the dependency-light LightOp environment bridge."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from vllm_hcu.lightop_env import (
    LightOpEnvironmentError,
    configure_lightop_environment,
)


REPOSITORY = Path(__file__).resolve().parents[2]


def test_new_hcu_names_populate_lightop_supported_aliases():
    """The plugin-owned names configure the aliases LightOp reads."""

    env = {
        "VLLM_HCU_FUSED_MOE_CHUNK_SIZE": "8192",
        "VLLM_HCU_USE_GLOBAL_MOE_CACHE": "true",
        "VLLM_HCU_USE_FUSED_RMS_QUANT": "1",
        "VLLM_HCU_USE_FUSE_SILU_AND_MUL": "yes",
    }

    configure_lightop_environment(env)

    assert env["VLLM_FUSED_MOE_CHUNK_SIZE"] == "8192"
    assert env["VLLM_USE_GLOBAL_CACHE13"] == "1"
    assert env["USE_FUSED_RMS_QUANT"] == "1"
    assert env["VLLM_USE_FUSE_SILU_AND_MUL"] == "1"


@pytest.mark.parametrize(
    ("hcu_name", "alias_name", "configured", "canonical"),
    (
        (
            "VLLM_HCU_FUSED_MOE_CHUNK_SIZE",
            "VLLM_FUSED_MOE_CHUNK_SIZE",
            "08192",
            "8192",
        ),
        (
            "VLLM_HCU_USE_GLOBAL_MOE_CACHE",
            "VLLM_USE_GLOBAL_CACHE13",
            "true",
            "1",
        ),
        (
            "VLLM_HCU_USE_FUSED_RMS_QUANT",
            "USE_FUSED_RMS_QUANT",
            "yes",
            "1",
        ),
        (
            "VLLM_HCU_USE_FUSE_SILU_AND_MUL",
            "VLLM_USE_FUSE_SILU_AND_MUL",
            "off",
            "0",
        ),
    ),
)
def test_hcu_names_replace_empty_lightop_aliases_with_canonical_values(
    hcu_name: str,
    alias_name: str,
    configured: str,
    canonical: str,
) -> None:
    """An empty dependency alias is unconfigured, not an override."""

    env = {hcu_name: configured, alias_name: ""}

    configure_lightop_environment(env)

    assert env[hcu_name] == configured
    assert env[alias_name] == canonical


def test_conflicting_hcu_and_dependency_values_fail_closed():
    """A plugin value cannot silently override a conflicting LightOp alias."""

    env = {
        "VLLM_HCU_USE_GLOBAL_MOE_CACHE": "1",
        "VLLM_USE_GLOBAL_CACHE13": "0",
    }

    with pytest.raises(LightOpEnvironmentError, match="conflicting"):
        configure_lightop_environment(env)


def test_legacy_lmslim_value_warns_once_and_bridges(caplog):
    """Legacy input is migrated once per process without retaining LMSlim use."""

    env = {"LMSLIM_USE_GLOBAL_MOE_CACHE": "true"}

    configure_lightop_environment(env)
    configure_lightop_environment(env)

    assert env["VLLM_HCU_USE_GLOBAL_MOE_CACHE"] == "1"
    assert env["VLLM_USE_GLOBAL_CACHE13"] == "1"
    assert caplog.text.count("LMSLIM_USE_GLOBAL_MOE_CACHE is deprecated") == 1


def test_package_bootstrap_configures_lightop_before_lightop_import():
    """Package import sets LightOp's neutral alias before its env module reads it."""

    env = dict(os.environ)
    for name in (
        "VLLM_HCU_USE_GLOBAL_MOE_CACHE",
        "VLLM_USE_GLOBAL_CACHE13",
        "LMSLIM_USE_GLOBAL_MOE_CACHE",
    ):
        env.pop(name, None)
    env["VLLM_HCU_USE_GLOBAL_MOE_CACHE"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(REPOSITORY), env.get("PYTHONPATH")) if part
    )

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import vllm_hcu; import lightop.envs; "
            "assert lightop.envs.LMSLIM_USE_GLOBAL_MOE_CACHE is True",
        ],
        cwd=REPOSITORY,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr
