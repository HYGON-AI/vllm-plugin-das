# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Contract tests for the AITER-owned tuned MoE configs.

AITER owns the concrete vLLM-schema tuning data in ``aiter/configs/vllm_moe``.
The plugin keeps only the discovery hook that points
``VLLM_TUNED_CONFIG_FOLDER`` at that package directory.  These tests pin the
ownership boundary and the target lookup; the tile values themselves are
validated in AITER's own static test
(``op_tests/test_vllm_moe_config_delivery.py``).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from vllm_hcu.moe_config_env import (
    TUNED_CONFIG_FOLDER_ENV,
    configure_tuned_moe_config_folder,
    get_tuned_moe_config_folder,
)


REPOSITORY = Path(__file__).resolve().parents[2]
PLUGIN_MOE_CONFIG_FOLDER = (
    REPOSITORY
    / "vllm_hcu"
    / "model_executor"
    / "layers"
    / "fused_moe"
    / "moe_configs"
)

# The target-model bf16 MoE layers 88-90 of Step5: 352 routed experts over
# EP8, so E=44 is the local expert count, N is `moe_intermediate_size` and
# K is `hidden_size`.  vLLM resolves one file per (shape, dtype, device).
DRAFT_MOE_E = 44
DRAFT_MOE_N = 1536
DRAFT_MOE_K = 4096
DRAFT_MOE_TOPK = 8
TUNED_DEVICE_NAME = "BW1100"
ALIAS_DEVICE_NAMES = ("BW1100", "BW1101")
TUNED_MAX_M = 256
TUNED_TILE = {
    "BLOCK_SIZE_M": 16,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 128,
    "GROUP_SIZE_M": 1,
    "num_warps": 2,
    "num_stages": 1,
}


def _aiter_moe_config():
    return pytest.importorskip("aiter.configs.vllm_moe")


def _aiter_folder() -> Path:
    folder = get_tuned_moe_config_folder()
    assert folder is not None, "AITER's vllm_moe config folder is not installed"
    return Path(folder)


def _config_table() -> dict[int, dict[str, int]]:
    module = _aiter_moe_config()
    table = module.load_config(
        DRAFT_MOE_E, DRAFT_MOE_N, device_name=TUNED_DEVICE_NAME
    )
    assert table is not None
    return table


def test_plugin_no_longer_ships_authoritative_tuned_config():
    """The concrete regular-MoE tuning row is AITER's, not the plugin's."""

    if PLUGIN_MOE_CONFIG_FOLDER.exists():
        assert not list(PLUGIN_MOE_CONFIG_FOLDER.glob("*.json"))
    assert not (
        PLUGIN_MOE_CONFIG_FOLDER
        / f"E={DRAFT_MOE_E},N={DRAFT_MOE_N},device_name={TUNED_DEVICE_NAME}.json"
    ).exists()


def test_aiter_owns_the_target_config_and_both_aliases():
    folder = _aiter_folder()

    for device_name in ALIAS_DEVICE_NAMES:
        config_file = folder / (
            f"E={DRAFT_MOE_E},N={DRAFT_MOE_N},device_name={device_name}.json"
        )
        assert config_file.is_file(), config_file

    primary = folder / (
        f"E={DRAFT_MOE_E},N={DRAFT_MOE_N},device_name={ALIAS_DEVICE_NAMES[0]}.json"
    )
    alias = folder / (
        f"E={DRAFT_MOE_E},N={DRAFT_MOE_N},device_name={ALIAS_DEVICE_NAMES[1]}.json"
    )
    assert primary.read_bytes() == alias.read_bytes()


def test_target_lookup_returns_the_measured_tile():
    module = _aiter_moe_config()

    for device_name in ALIAS_DEVICE_NAMES:
        assert (
            module.get_config(
                DRAFT_MOE_E,
                DRAFT_MOE_N,
                128,
                device_name=device_name,
            )
            == TUNED_TILE
        )


def test_decode_range_carries_the_tuned_tile():
    table = _config_table()

    assert {m for m, config in table.items() if config == TUNED_TILE} == {
        m for m in table if m <= TUNED_MAX_M
    }
    for m, config in table.items():
        if m <= TUNED_MAX_M:
            assert config == TUNED_TILE, m
        else:
            assert config["BLOCK_SIZE_M"] >= 64, m


def test_non_target_lookups_keep_vllm_fallback():
    module = _aiter_moe_config()

    assert module.get_config(352, DRAFT_MOE_N, 128) is None
    assert module.get_config(DRAFT_MOE_E, 3072, 128) is None
    assert (
        module.get_config(
            DRAFT_MOE_E, DRAFT_MOE_N, 128, dtype="fp8_w8a8"
        )
        is None
    )
    assert (
        module.get_config(
            DRAFT_MOE_E, DRAFT_MOE_N, 128, device_name="BW200B"
        )
        is None
    )


def test_prefill_sized_batches_keep_vllms_own_tiling():
    """The large-M entries reproduce vLLM's default config unchanged."""

    pytest.importorskip("vllm")
    from vllm.model_executor.layers.fused_moe.fused_moe import get_default_config

    table = _config_table()

    for m, config in table.items():
        if m <= TUNED_MAX_M:
            continue
        expected = get_default_config(
            m, DRAFT_MOE_E, DRAFT_MOE_N, DRAFT_MOE_K, DRAFT_MOE_TOPK, None
        )
        # The launch site forces SPLIT_K=1, so the table does not carry it.
        expected.pop("SPLIT_K")
        assert config == expected, m


def test_unset_folder_is_armed_with_aiter_configs():
    """The plugin points vLLM at AITER only when nothing was chosen."""

    folder = _aiter_folder()
    environ: dict[str, str] = {}

    configure_tuned_moe_config_folder(environ)

    assert environ[TUNED_CONFIG_FOLDER_ENV] == str(folder)


def test_a_user_selected_folder_is_preserved():
    """Tuning another shape stays a user-side, one-variable operation."""

    environ = {TUNED_CONFIG_FOLDER_ENV: "/opt/tuned"}

    configure_tuned_moe_config_folder(environ)

    assert environ[TUNED_CONFIG_FOLDER_ENV] == "/opt/tuned"


def test_an_empty_user_selection_still_arms_the_aiter_configs():
    """An empty value counts as unset, matching how vLLM reads the variable."""

    folder = _aiter_folder()
    environ = {TUNED_CONFIG_FOLDER_ENV: ""}

    configure_tuned_moe_config_folder(environ)

    assert environ[TUNED_CONFIG_FOLDER_ENV] == str(folder)


def test_missing_aiter_leaves_vllm_lookup_untouched(monkeypatch):
    """Without AITER the plugin must not substitute a local tuning table."""

    from vllm_hcu import moe_config_env

    monkeypatch.setattr(moe_config_env, "_aiter_package_dir", lambda: None)
    environ: dict[str, str] = {}

    moe_config_env.configure_tuned_moe_config_folder(environ)

    assert TUNED_CONFIG_FOLDER_ENV not in environ


def test_package_import_arms_the_folder_before_vllm_reads_it():
    """The folder is set during plugin discovery, not at the first MoE call."""

    pytest.importorskip("vllm")
    folder = _aiter_folder()

    env = dict(os.environ)
    env.pop(TUNED_CONFIG_FOLDER_ENV, None)
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(REPOSITORY), env.get("PYTHONPATH")) if part
    )

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import os, vllm_hcu; "
            f"assert os.environ[{TUNED_CONFIG_FOLDER_ENV!r}] == {str(folder)!r}",
        ],
        cwd=REPOSITORY,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr
