# SPDX-License-Identifier: Apache-2.0

from pathlib import Path


def test_explicit_aiter_linear_backend_normalizes_upstream_env_contract():
    source = (
        Path(__file__).parents[2]
        / "vllm_hcu/patch/platform/core_fix/patch_vllm_config.py"
    ).read_text()
    branch = source.split(
        'if getattr(kernel_config, "linear_backend", None) == "aiter":', 1
    )[1]
    assert 'os.environ["VLLM_ROCM_USE_AITER"] = "1"' in branch
    assert 'os.environ["VLLM_ROCM_USE_AITER_LINEAR"] = "1"' in branch
