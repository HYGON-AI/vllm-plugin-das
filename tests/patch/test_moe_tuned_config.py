# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Contract tests for the tuned MoE configs shipped with the plugin.

The shipped table is a frozen measurement: it was tuned on real Step5 routing
and re-validated end to end, so these tests pin its scope rather than its
values.  Retuning a shape means replacing a file and the expectations here.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from vllm_hcu.moe_config_env import (
    MOE_CONFIG_FOLDER,
    TUNED_CONFIG_FOLDER_ENV,
    configure_tuned_moe_config_folder,
)


REPOSITORY = Path(__file__).resolve().parents[2]

# The MTP draft-layer MoE of Step5: 352 routed experts over EP8, so E=44 is the
# local expert count, N is `moe_intermediate_size` and K is `hidden_size`.
DRAFT_MOE_E = 44
DRAFT_MOE_N = 1536
DRAFT_MOE_K = 4096
DRAFT_MOE_TOPK = 8
TUNED_DEVICE_NAME = "BW1100"

# Decode-sized M only.  vLLM resolves a batch to the nearest key, so the
# entries above the tuned range have to stay in the file: dropping them would
# spread the decode tile over prefill batches.
TUNED_MAX_M = 256
TUNED_TILE = {
    "BLOCK_SIZE_M": 16,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 128,
    "GROUP_SIZE_M": 1,
    "num_warps": 2,
    "num_stages": 1,
}
DEFAULT_BLOCK_M_FLOOR = 64

DRAFT_MOE_CONFIG = (
    Path(MOE_CONFIG_FOLDER)
    / f"E={DRAFT_MOE_E},N={DRAFT_MOE_N},device_name={TUNED_DEVICE_NAME}.json"
)
# The deployment reports both spellings for the same HCU part, so the table
# ships the same content under both; they differ in the file name only.
ALIAS_DEVICE_NAMES = ("BW1100", "BW1101")
CONFIG_FILE_NAME_RE = re.compile(
    r"^E=\d+,N=\d+,device_name=[^,]+(,dtype=[^,]+)?(,block_shape=\[\d+,\d+\])?\.json$"
)


def _config_table() -> dict[int, dict[str, int]]:
    return {
        int(key): value
        for key, value in json.loads(DRAFT_MOE_CONFIG.read_text()).items()
    }


def test_shipped_configs_are_named_after_the_shape_they_tune():
    """One file per (shape, device) key, the way vLLM resolves them."""

    names = sorted(os.listdir(MOE_CONFIG_FOLDER))

    assert names, "the plugin must ship at least one tuned MoE config"
    for name in names:
        assert CONFIG_FILE_NAME_RE.match(name), name
    assert DRAFT_MOE_CONFIG.name in names


def test_device_name_aliases_are_byte_identical():
    """The alias exists only to tolerate the two reported device names."""

    alias = (
        Path(MOE_CONFIG_FOLDER)
        / f"E={DRAFT_MOE_E},N={DRAFT_MOE_N},device_name={ALIAS_DEVICE_NAMES[1]}.json"
    )

    assert alias.is_file()
    assert alias.read_bytes() == DRAFT_MOE_CONFIG.read_bytes()


def test_the_decode_range_carries_the_tuned_tile():
    """The measured tile is confined to the decode-sized M entries."""

    table = _config_table()

    assert {m for m, config in table.items() if config == TUNED_TILE} == {
        m for m in table if m <= TUNED_MAX_M
    }
    for m, config in table.items():
        if m <= TUNED_MAX_M:
            assert config == TUNED_TILE, m
        else:
            assert config["BLOCK_SIZE_M"] >= DEFAULT_BLOCK_M_FLOOR, m


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


@pytest.mark.parametrize("device_name", ALIAS_DEVICE_NAMES)
def test_each_alias_is_the_key_vllm_looks_up_on_that_device(device_name):
    """Each shipped alias matches vLLM's own rule for the name it covers."""

    pytest.importorskip("vllm")
    from vllm.model_executor.layers.fused_moe.fused_moe import get_config_file_name
    from vllm.utils.platform_utils import get_device_name_as_file_name

    reported = get_device_name_as_file_name()
    if reported != device_name:
        pytest.skip(f"this host reports {reported}")

    assert get_config_file_name(DRAFT_MOE_E, DRAFT_MOE_N, None) == (
        f"E={DRAFT_MOE_E},N={DRAFT_MOE_N},device_name={device_name}.json"
    )


def test_unset_folder_is_armed_with_the_shipped_configs():
    """The plugin points vLLM at its configs only when nothing was chosen."""

    environ: dict[str, str] = {}

    configure_tuned_moe_config_folder(environ)

    assert environ[TUNED_CONFIG_FOLDER_ENV] == MOE_CONFIG_FOLDER


def test_a_user_selected_folder_is_preserved():
    """Tuning another shape stays a user-side, one-variable operation."""

    environ = {TUNED_CONFIG_FOLDER_ENV: "/opt/tuned"}

    configure_tuned_moe_config_folder(environ)

    assert environ[TUNED_CONFIG_FOLDER_ENV] == "/opt/tuned"


def test_an_empty_user_selection_still_arms_the_shipped_configs():
    """An empty value counts as unset, matching how vLLM reads the variable."""

    environ = {TUNED_CONFIG_FOLDER_ENV: ""}

    configure_tuned_moe_config_folder(environ)

    assert environ[TUNED_CONFIG_FOLDER_ENV] == MOE_CONFIG_FOLDER


def test_package_import_arms_the_folder_before_vllm_reads_it():
    """The folder is set during plugin discovery, not at the first MoE call."""

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
            f"assert os.environ[{TUNED_CONFIG_FOLDER_ENV!r}] == {MOE_CONFIG_FOLDER!r}",
        ],
        cwd=REPOSITORY,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr
