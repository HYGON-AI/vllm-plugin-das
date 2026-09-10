# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Regression coverage for Qwen4-Exp's lazy package import order."""

from __future__ import annotations

import os
import subprocess
import sys


def test_qwen4_exp_config_import_after_hcu_platform_activation() -> None:
    env = os.environ.copy()
    env["VLLM_PLUGINS"] = "hcu"
    code = """
import importlib

from vllm.platforms import current_platform
from vllm_hcu.patch.worker import prepare_worker_patches

assert current_platform.device_type
prepare_worker_patches()
config_module = importlib.import_module(
    "vllm.transformers_utils.configs.qwen4_exp"
)
assert config_module.Qwen4ExpTextConfig is not None
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stdout + result.stderr
