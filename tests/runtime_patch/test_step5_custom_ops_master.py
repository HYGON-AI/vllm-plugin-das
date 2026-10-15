# SPDX-License-Identifier: Apache-2.0
"""The custom-op master must dominate every optional Step5 child switch."""

import os
import subprocess
import sys
from types import ModuleType

import pytest


_MODEL_CHILDREN = (
    "VLLM_HCU_STEP5_FUSE_QKVG",
    "VLLM_HCU_STEP5_BF16_ROUTER",
    "VLLM_HCU_STEP5_FUSED_ROUTER_SHARED",
    "VLLM_HCU_STEP5_FP8_DENSE",
    "VLLM_HCU_STEP5_FP8_BF16_MOE",
    "VLLM_HCU_STEP5_SWA_FUSED_QK",
    "VLLM_HCU_STEP5_FUSED_GATE",
    "STEP5_SKINNY_LM_HEAD",
    "VLLM_HCU_TRITON_TOKEN_FP8_QUANT",
)

_RUNTIME_CHILDREN = (
    "VLLM_HCU_EP_FUSED_MOE",
    "VLLM_HCU_EP_FUSED_COMBINE",
    "VLLM_HCU_HOST_SEQ_LENS",
    "VLLM_HCU_SWA_DECODE",
)


@pytest.mark.parametrize(
    ("master", "child", "expected"),
    [
        ("0", "1", (False,) * 9),
        ("1", "1", (True,) * 9),
        ("1", "0", (False,) * 9),
        (
            "1",
            None,
            (True, True, True, False, False, False, False, True, False),
        ),
    ],
)
def test_step5_import_time_features_obey_custom_ops_master(
    master, child, expected
):
    """Dropping a master check on any model-local provider fails this matrix."""

    env = os.environ.copy()
    env["VLLM_HCU_USE_CUSTOM_OPS"] = master
    for name in _MODEL_CHILDREN:
        if child is None:
            env.pop(name, None)
        else:
            env[name] = child
    script = f"""
from vllm_hcu import hcu_platform_register_model
hcu_platform_register_model()
from vllm_hcu.models import step5_text
from vllm_hcu.models import step5_swa_fused
from vllm_hcu.model_executor.layers import skinny_lm_head
from vllm_hcu.ops import token_fp8_quant

actual = (
    step5_text.FUSE_QKVG,
    step5_text.BF16_ROUTER,
    step5_text.FUSED_ROUTER_SHARED,
    step5_text.FP8_DENSE,
    step5_text.FP8_BF16_MOE,
    step5_swa_fused.SWA_FUSED_QK,
    step5_swa_fused.FUSED_GATE,
    skinny_lm_head._ENABLED,
    token_fp8_quant.ENABLED,
)
assert actual == {expected!r}, actual
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    ("master", "child", "expected"),
    [
        ("0", "1", (False,) * 4),
        ("1", "1", (True,) * 4),
        ("1", "0", (False,) * 4),
        ("1", None, (False, False, True, True)),
    ],
)
def test_step5_runtime_features_obey_custom_ops_master(master, child, expected):
    """Runtime fast paths must use the same effective master/child state."""

    env = os.environ.copy()
    env["VLLM_HCU_USE_CUSTOM_OPS"] = master
    for name in _RUNTIME_CHILDREN:
        if child is None:
            env.pop(name, None)
        else:
            env[name] = child
    script = f"""
import importlib
import vllm
from vllm_hcu import hcu_platform_register_model
hcu_platform_register_model()

deep_gemm_moe = importlib.import_module(
    "vllm.model_executor.layers.fused_moe.experts.deep_gemm_moe"
)
moe_runner = importlib.import_module(
    "vllm.model_executor.layers.fused_moe.runner.moe_runner"
)
hcu_model_runner_v2 = importlib.import_module("vllm_hcu.v1.hcu_model_runner_v2")
flash_attn = importlib.import_module("vllm_hcu.v1.attention.backends.flash_attn")

actual = (
    deep_gemm_moe._EP_FUSED_MOE,
    moe_runner._EP_FUSED_COMBINE,
    hcu_model_runner_v2._USE_HOST_SEQ_LENS,
    flash_attn._SWA_DECODE_ENABLED,
)
assert actual == {expected!r}, actual
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_step5_splitting_ops_ignore_enabled_children_when_master_is_off(
    monkeypatch,
):
    """Master-off must not register graph splits for disabled fused ops."""
    from vllm_hcu.patch.platform.core_fix import patch_step5_splitting_ops as adapter

    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "0")
    monkeypatch.setenv("VLLM_HCU_STEP5_FUSED_GATE", "1")
    monkeypatch.setenv("VLLM_HCU_STEP5_SWA_FUSED_QK", "1")

    target = ModuleType(adapter.TARGET_MODULE)

    class CompilationConfig:
        def __init__(self):
            self.splitting_ops = None

        def set_splitting_ops_for_v1(self, all2all_backend, data_parallel_size):
            del all2all_backend, data_parallel_size
            self.splitting_ops = ["vllm::baseline"]

    target.CompilationConfig = CompilationConfig
    adapter.apply_to_module(target)
    config = CompilationConfig()
    config.set_splitting_ops_for_v1("allgather_reducescatter", 8)

    assert config.splitting_ops == ["vllm::baseline"]


def test_token_fp8_wrapper_uses_original_without_importing_provider_when_master_off(
    monkeypatch,
):
    """Master-off must branch before importing the token-FP8 provider."""
    from vllm_hcu.patch.worker.framework_opt import patch_step5_token_fp8 as adapter
    import vllm_hcu.ops as ops

    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "0")
    monkeypatch.setenv("VLLM_HCU_TRITON_TOKEN_FP8_QUANT", "1")
    forbidden = ModuleType("vllm_hcu.ops.token_fp8_quant")

    def fail_can_use(_value):
        pytest.fail("master-off imported the token-FP8 provider")

    forbidden.can_use = fail_can_use
    monkeypatch.setitem(sys.modules, forbidden.__name__, forbidden)
    monkeypatch.setattr(ops, "token_fp8_quant", forbidden, raising=False)

    target = ModuleType(adapter.TARGET_MODULE)
    calls = []

    def native(A, A_scale, per_act_token, block_shape=None):
        calls.append((A, A_scale, per_act_token, block_shape))
        return "native"

    target._fp8_quantize = native
    adapter.apply_to_module(target)

    marker = object()
    assert target._fp8_quantize(marker, None, True, None) == "native"
    assert calls == [(marker, None, True, None)]
