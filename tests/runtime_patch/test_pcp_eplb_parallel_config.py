# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Regression coverage for PCP-only expert-parallel load balancing."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[2]
TARGET_VLLM_ROOT = Path(
    os.environ.get("VLLM_V0251_SOURCE_ROOT", REPOSITORY.parent / "vllm_0251")
).resolve()
if not (TARGET_VLLM_ROOT / "vllm" / "__init__.py").is_file():
    raise RuntimeError(
        f"VLLM_V0251_SOURCE_ROOT does not contain vllm: {TARGET_VLLM_ROOT}"
    )


def test_real_v0251_parallel_config_accepts_pcp_only_eplb() -> None:
    script = r'''
from types import SimpleNamespace

from pydantic import ValidationError
import vllm.config.parallel as parallel_module

from vllm_hcu.patch import apply_platform_patches, patch_report
from vllm_hcu.patch.platform.core_fix import patch_parallel_config

apply_platform_patches()
parallel_module.current_platform = SimpleNamespace(
    device_count=lambda: 0,
    is_cuda=lambda: False,
    is_cuda_alike=lambda: True,
    is_tpu=lambda: False,
    use_custom_allreduce=lambda: False,
)

assert patch_parallel_config.apply_to_module(parallel_module) is False
patch_record = patch_report()["patches"][patch_parallel_config.PATCH_ID]
assert patch_record["status"] == "applied"

pcp_only = parallel_module.ParallelConfig(
    enable_eplb=True,
    enable_expert_parallel=True,
    prefill_context_parallel_size=2,
)
assert pcp_only.tensor_parallel_size == 1
assert pcp_only.prefill_context_parallel_size == 2
assert pcp_only.data_parallel_size == 1
assert pcp_only.world_size == 2

try:
    parallel_module.ParallelConfig(
        enable_eplb=True,
        enable_expert_parallel=True,
    )
except ValidationError as error:
    assert "EPLB requires" in str(error)
else:
    raise AssertionError("EPLB without TP, PCP, or DP must remain invalid")

try:
    parallel_module.ParallelConfig(
        enable_eplb=True,
        enable_expert_parallel=False,
        prefill_context_parallel_size=2,
    )
except ValidationError as error:
    assert "enable_expert_parallel must be True" in str(error)
else:
    raise AssertionError("EPLB without expert parallelism must remain invalid")
'''
    env = dict(os.environ)
    env["VLLM_PLUGINS"] = "__disabled__"
    env["PYTHONPATH"] = os.pathsep.join(
        (str(TARGET_VLLM_ROOT), str(REPOSITORY))
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
