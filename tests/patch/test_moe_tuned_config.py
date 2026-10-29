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


@pytest.fixture
def production_selector(monkeypatch):
    """Expose vLLM's real nearest-M selector with AITER's folder selected."""

    pytest.importorskip("vllm")
    from vllm.platforms import current_platform

    if not current_platform.is_rocm():
        pytest.skip("AITER's Step5 table targets HCU/ROCm defaults")

    from vllm.model_executor.layers import fused_moe as fused_moe_package
    from vllm.model_executor.layers.fused_moe import fused_moe as fused_moe_impl

    # try_get_optimal_moe_config() imports this override at call time.
    monkeypatch.setattr(
        fused_moe_package, "get_config", lambda: None, raising=False
    )
    monkeypatch.setenv(TUNED_CONFIG_FOLDER_ENV, str(_aiter_folder()))
    fused_moe_impl.get_moe_configs.cache_clear()
    try:
        yield (
            fused_moe_impl.try_get_optimal_moe_config,
            fused_moe_impl.get_default_config,
            fused_moe_impl.get_moe_configs,
        )
    finally:
        fused_moe_impl.get_moe_configs.cache_clear()


def test_real_selector_boundary_keeps_target_and_native_fallback(
    production_selector,
):
    """The real vLLM selector must not leak the decode tile past M=256."""

    try_get_config, get_default_config, get_moe_configs = production_selector
    table = get_moe_configs(
        DRAFT_MOE_E, DRAFT_MOE_N, None, 0, 0
    )
    assert table is not None

    w1_shape = (DRAFT_MOE_E, 2 * DRAFT_MOE_N, DRAFT_MOE_K)
    w2_shape = (DRAFT_MOE_E, DRAFT_MOE_K, DRAFT_MOE_N)

    def selected(m: int) -> dict[str, int]:
        return try_get_config(
            w1_shape, w2_shape, DRAFT_MOE_TOPK, None, m
        )

    for m in (16, 32, 64, 128, TUNED_MAX_M):
        assert selected(m) == TUNED_TILE, m

    expected_keys = {
        257: 257,
        300: 257,
        384: 257,
        385: 512,
        512: 512,
        513: 513,
        600: 513,
        768: 513,
        769: 1024,
        4096: 4096,
        5675: 5675,
        5676: 5676,
        6000: 5676,
        6144: 5676,
        6145: 5676,
        8192: 8192,
        65536: 32768,
    }
    for m, expected_key in expected_keys.items():
        actual_key = min(table, key=lambda bucket: abs(bucket - m))
        assert actual_key == expected_key, (m, actual_key)
        expected = get_default_config(
            m, DRAFT_MOE_E, DRAFT_MOE_N, DRAFT_MOE_K, DRAFT_MOE_TOPK, None
        )
        expected.pop("SPLIT_K")
        assert selected(m) == expected, m


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
    """The plugin treats an empty value as unset and arms the AITER configs."""

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
