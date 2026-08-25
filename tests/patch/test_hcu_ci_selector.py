# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Portable contracts for the HCU CI change selector."""

from __future__ import annotations

import builtins
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

CI_SCRIPTS = (
    Path(__file__).resolve().parents[2] / ".github" / "scripts" / "hcu_ci"
)
if str(CI_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(CI_SCRIPTS))

from select_hcu_tests import (  # noqa: E402
    DEFAULT_CONFIG,
    _load_config,
    select_jobs,
    validate_config,
)
from hcu_ci_preflight import PreflightError, run_preflight  # noqa: E402


def _config() -> dict:
    return _load_config(DEFAULT_CONFIG)


def test_selector_configuration_is_valid() -> None:
    jobs = validate_config(_config())
    assert "accuracy-gfx936" in jobs
    assert "deepseek-tp-ep" in jobs


def test_docs_only_change_does_not_select_hardware() -> None:
    jobs, groups, fallback = select_jobs(
        _config(),
        ["docs/runtime_patch_architecture_v0251.md"],
    )
    assert jobs == []
    assert groups == ["docs-only"]
    assert fallback is False


def test_moe_change_selects_kernel_and_tp_ep_jobs() -> None:
    jobs, groups, fallback = select_jobs(
        _config(),
        ["vllm_hcu/model_executor/layers/fused_moe/aiter_runtime.py"],
    )
    assert {job["id"] for job in jobs}.issuperset(
        {"accuracy-gfx936", "qwen35-tp-ep"}
    )
    assert "moe" in groups
    assert fallback is False


def test_deepseek_runtime_change_selects_gfx938_accuracy() -> None:
    jobs, groups, fallback = select_jobs(
        _config(),
        ["vllm_hcu/model_executor/layers/deepseek_v4_attention.py"],
    )
    assert "accuracy-gfx938" in {job["id"] for job in jobs}
    assert "deepseek" in groups
    assert fallback is False


def test_model_runtime_change_selects_text_vl_and_pooling_models() -> None:
    jobs, groups, fallback = select_jobs(
        _config(),
        ["tests/integration/model_runtime.py"],
    )
    assert {job["id"] for job in jobs}.issuperset(
        {
            "qwen35-smoke",
            "qwen25-models",
            "qwen3-pooling",
        }
    )
    assert "model-tests" in groups
    assert fallback is False


def test_protocol_change_selects_protocol_server_job() -> None:
    jobs, groups, fallback = select_jobs(
        _config(),
        ["tests/integration/server/test_qwen3_protocol_features.py"],
    )
    assert {job["id"] for job in jobs} == {"qwen3-protocol"}
    assert "qwen3-protocol-tests" in groups
    assert fallback is False


def test_pooling_server_change_selects_pooling_job() -> None:
    jobs, groups, fallback = select_jobs(
        _config(),
        ["tests/integration/server/test_qwen3_pooling_server.py"],
    )
    assert {job["id"] for job in jobs} == {"qwen3-pooling"}
    assert "pooling-tests" in groups
    assert fallback is False


def test_protocol_helper_change_selects_all_server_consumers() -> None:
    jobs, groups, fallback = select_jobs(
        _config(),
        ["tests/integration/server/openai_server.py"],
    )
    assert {job["id"] for job in jobs} == {
        "qwen25-models",
        "qwen3-pooling",
        "qwen3-protocol",
    }
    assert "protocol-tests" in groups
    assert fallback is False


def test_mamba_change_selects_real_mamba_smoke() -> None:
    jobs, groups, fallback = select_jobs(
        _config(),
        ["vllm_hcu/model_executor/layers/mamba_runtime.py"],
    )
    assert {job["id"] for job in jobs} == {"mamba-smoke"}
    assert "mamba" in groups
    assert fallback is False


def test_unknown_production_change_uses_conservative_fallback() -> None:
    jobs, groups, fallback = select_jobs(
        _config(),
        ["vllm_hcu/new_runtime_area.py"],
    )
    assert {job["id"] for job in jobs} == {
        "accuracy-gfx936",
        "contract-hcu-gfx936",
        "integration-smoke-gfx938",
    }
    assert groups == ["conservative-fallback"]
    assert fallback is True


def test_accuracy_mode_adds_all_accuracy_jobs() -> None:
    jobs, groups, fallback = select_jobs(
        _config(),
        ["docs/accuracy-notes.md"],
        accuracy=True,
    )
    assert {job["id"] for job in jobs}.issuperset(
        {
            "qwen35-gsm8k",
            "qwen3-8b-gsm8k",
            "qwen3-vl-mmmu",
            "deepseek-gsm8k",
        }
    )
    assert "docs-only" in groups
    assert "accuracy-hcu" in groups
    assert fallback is False


def test_preflight_hides_runtime_dependency_import_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__
    private_backend_name = "".join(("A", "M", "D"))

    def fail_torch_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError(
                f"{private_backend_name} backend package is unavailable"
            )
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_torch_import)

    with pytest.raises(
        PreflightError,
        match="^HCU runtime dependency initialization failed\\.$",
    ) as error:
        run_preflight(
            expected_arch="gfx936",
            required_cards=1,
            requirements=[],
        )

    assert error.value.__suppress_context__ is True
    assert private_backend_name not in str(error.value)


def test_preflight_hides_device_inspection_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    private_link_name = "".join(("XG", "MI"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "device_count",
        lambda: (_ for _ in ()).throw(
            RuntimeError(f"{private_link_name} backend query failed")
        ),
    )

    with pytest.raises(
        PreflightError,
        match="^HCU device inspection failed\\.$",
    ) as error:
        run_preflight(
            expected_arch="gfx936",
            required_cards=1,
            requirements=[],
        )

    assert error.value.__suppress_context__ is True
    assert private_link_name not in str(error.value)
