# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

"""Lightweight global pytest configuration for vLLM-HCU tests.

This module must stay safe to import without initializing an HCU context,
installing runtime patches, loading models, or starting child processes.
Domain-specific fixtures belong in ``tests/fixtures`` and should be requested
only by tests that need them.
"""

from __future__ import annotations

import os
from pathlib import Path

# Test modules import vLLM targets during collection.  Keep entry-point plugin
# activation explicit so collection cannot register replacement modules that a
# later isolation fixture then removes while leaving vLLM custom-op state live.
os.environ.setdefault("VLLM_PLUGINS", "__disabled__")

import pytest

from tests.fixtures.artifacts import EnvironmentFingerprint
from tests.fixtures.resources import TestResources


def _environment_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("vllm-hcu resources")
    group.addoption(
        "--model-root",
        type=Path,
        default=os.environ.get("VLLM_HCU_TEST_MODEL_ROOT"),
        help="Root directory for model paths declared relative to test configs.",
    )
    group.addoption(
        "--dataset-root",
        type=Path,
        default=os.environ.get("VLLM_HCU_TEST_DATASET_ROOT"),
        help="Root directory for dataset paths declared relative to test configs.",
    )
    group.addoption(
        "--model-config",
        type=Path,
        default=None,
        help="Optional model test YAML configuration.",
    )
    group.addoption(
        "--allow-model-download",
        action="store_true",
        default=_environment_flag("VLLM_HCU_TEST_ALLOW_DOWNLOAD"),
        help="Allow model/dataset download when a local resource is unavailable.",
    )
    group.addoption(
        "--strict-test-resources",
        action="store_true",
        default=_environment_flag("VLLM_HCU_TEST_STRICT_RESOURCES"),
        help="Fail instead of skip when a selected model/dataset is unavailable.",
    )


@pytest.fixture(scope="session")
def hcu_test_resources(pytestconfig: pytest.Config) -> TestResources:
    """Resolve model and dataset roots without touching accelerator state."""
    return TestResources(
        model_root=pytestconfig.getoption("--model-root"),
        dataset_root=pytestconfig.getoption("--dataset-root"),
        model_config=pytestconfig.getoption("--model-config"),
        allow_download=pytestconfig.getoption("--allow-model-download"),
        strict=pytestconfig.getoption("--strict-test-resources"),
    )


@pytest.fixture(scope="session")
def hcu_environment_fingerprint() -> EnvironmentFingerprint:
    """Return a CPU-safe base environment fingerprint.

    Device and extension versions are intentionally populated by hardware
    fixtures later, after a test explicitly requests a live HCU.
    """
    return EnvironmentFingerprint.collect_base()
