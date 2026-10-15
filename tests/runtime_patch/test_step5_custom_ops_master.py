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


@pytest.mark.parametrize(
    ("master", "expected"),
    [
        (
            "0",
            (
                False,
                False,
                frozenset(),
                frozenset(),
                "off",
                False,
                False,
            ),
        ),
        (
            "1",
            (
                True,
                True,
                frozenset({"decode"}),
                frozenset({"prefill"}),
                "full",
                True,
                True,
            ),
        ),
    ],
)
def test_step4_dsa_optional_paths_obey_custom_ops_master(master, expected):
    """DSA remains present, but every alternate provider/fast path is baseline."""

    env = os.environ.copy()
    env.update(
        {
            "VLLM_HCU_USE_CUSTOM_OPS": master,
            "VLLM_STEP4_DSA_SPEC_INDEXER_PER_REQUEST": "1",
            "VLLM_STEP4_DSA_SPEC_INDEXER_FP8_DOT": "1",
            "VLLM_STEP4_DSA_BOLTOPS": "decode",
            "VLLM_STEP4_DSA_FLASH_ATTN": "prefill",
            "VLLM_STEP4_DSA_FP8_MATH": "full",
            "VLLM_STEP4_DSA_SPEC_FUSED_SUMMARIES": "1",
            "STEP5_DSA_MERGED_PROJ": "1",
        }
    )
    script = f"""
import builtins
import vllm

native_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name in {"boltops.step4_dsa", "flash_attn.flash_attn_interface"}:
        raise AssertionError("custom-op test imported optional provider " + name)
    return native_import(name, *args, **kwargs)
builtins.__import__ = guarded_import

from vllm_hcu.model_executor.layers import step4_dsa as dsa

actual = (
    dsa._SPEC_INDEXER_PER_REQUEST,
    dsa._SPEC_INDEXER_FP8_DOT,
    dsa.BOLTOPS_STAGES,
    dsa.FLASH_ATTN_DSA_STAGES,
    dsa._fp8_math,
    dsa._SPEC_FUSED_SUMMARIES,
    dsa._MERGED_PROJ,
)
assert actual == {expected!r}, actual
if dsa._fp8_math == "off":
    assert (dsa.DECODE_QK_FP8, dsa.DECODE_PV_FP8) == (False, False)
    assert (dsa.PREFILL_QK_FP8, dsa.PREFILL_PV_FP8) == (False, False)
else:
    assert (dsa.DECODE_QK_FP8, dsa.DECODE_PV_FP8) == (True, True)
    assert (dsa.PREFILL_QK_FP8, dsa.PREFILL_PV_FP8) == (True, True)
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
    ("overrides", "message"),
    [
        ({"VLLM_STEP4_DSA_BOLTOPS": "bogus"}, "unknown stage"),
        ({"VLLM_STEP4_DSA_FLASH_ATTN": "bogus"}, "unknown stage"),
        ({"VLLM_STEP4_DSA_FP8_MATH": "bogus"}, "expected one of"),
        (
            {
                "VLLM_STEP4_DSA_BOLTOPS": "decode",
                "VLLM_STEP4_DSA_FLASH_ATTN": "decode",
            },
            "both claim",
        ),
    ],
)
def test_step4_dsa_rejects_invalid_requested_values_with_master_off(
    overrides, message
):
    """The master disables providers, but must not hide misspelled requests."""

    env = os.environ.copy()
    env.update(
        {
            "VLLM_HCU_USE_CUSTOM_OPS": "0",
            "VLLM_STEP4_DSA_BOLTOPS": "",
            "VLLM_STEP4_DSA_FLASH_ATTN": "",
            "VLLM_STEP4_DSA_FP8_MATH": "off",
            **overrides,
        }
    )
    result = subprocess.run(
        [sys.executable, "-c", "import vllm_hcu.model_executor.layers.step4_dsa"],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode != 0
    assert message in result.stderr


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


@pytest.mark.parametrize(
    ("fused_gate", "swa_fused_qk", "expected"),
    [
        ("1", "0", ["vllm::baseline"]),
        ("0", "1", ["vllm::baseline", "vllm::step5_swa_qk_norm_rope_cache"]),
        (
            "1",
            "1",
            [
                "vllm::baseline",
                "vllm::step5_swa_attention_gate_quant",
                "vllm::step5_swa_qk_norm_rope_cache",
            ],
        ),
    ],
)
def test_step5_splitting_ops_match_registered_provider_matrix(
    monkeypatch, fused_gate, swa_fused_qk, expected
):
    from vllm_hcu.patch.platform.core_fix import patch_step5_splitting_ops as adapter

    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.setenv("VLLM_HCU_STEP5_FUSED_GATE", fused_gate)
    monkeypatch.setenv("VLLM_HCU_STEP5_SWA_FUSED_QK", swa_fused_qk)

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

    assert config.splitting_ops == expected


@pytest.mark.parametrize(
    ("fused_gate", "swa_fused_qk", "expected"),
    [
        ("1", "0", {"step5_gate_quant_fp8"}),
        (
            "1",
            "1",
            {
                "step5_swa_qk_norm_rope_cache",
                "step5_gate_quant_fp8",
                "step5_swa_attention_gate_quant",
            },
        ),
    ],
)
def test_step5_enabled_swa_provider_registers_during_model_import(
    fused_gate, swa_fused_qk, expected
):
    """Enabled ops must register at startup, before tracing or graph capture."""

    env = os.environ.copy()
    env.update(
        {
            "VLLM_HCU_USE_CUSTOM_OPS": "1",
            "VLLM_HCU_STEP5_FUSED_GATE": fused_gate,
            "VLLM_HCU_STEP5_SWA_FUSED_QK": swa_fused_qk,
        }
    )
    script = f"""
import vllm.utils.torch_utils as torch_utils

registered = []
native_register = torch_utils.direct_register_custom_op
def record_step5(**kwargs):
    if kwargs["op_name"].startswith("step5_"):
        registered.append(kwargs["op_name"])
    return native_register(**kwargs)
torch_utils.direct_register_custom_op = record_step5

from vllm_hcu.models import step5_text
assert set(registered) == {expected!r}, registered
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_step5_swa_ops_are_not_registered_when_master_is_off():
    """Disabled Step5 providers must not mutate the global custom-op registry."""

    env = os.environ.copy()
    env.update(
        {
            "VLLM_HCU_USE_CUSTOM_OPS": "0",
            "VLLM_HCU_STEP5_FUSED_GATE": "1",
            "VLLM_HCU_STEP5_SWA_FUSED_QK": "1",
        }
    )
    script = """
import sys
import vllm.utils.torch_utils as torch_utils

registered = []
native_register = torch_utils.direct_register_custom_op
def record_step5(**kwargs):
    if kwargs["op_name"].startswith("step5_"):
        registered.append(kwargs["op_name"])
    return native_register(**kwargs)
torch_utils.direct_register_custom_op = record_step5
from vllm_hcu.models import step5_text
assert "vllm_hcu.models.step5_swa_fused" not in sys.modules
from vllm_hcu.models import step5_swa_fused

assert step5_swa_fused.SWA_FUSED_QK is False
assert step5_swa_fused.FUSED_GATE is False
assert registered == [], registered
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_flash_attention_does_not_import_swa_provider_when_master_is_off():
    """Master-off must select the baseline before importing the Triton provider."""

    env = os.environ.copy()
    env.update(
        {
            "VLLM_HCU_USE_CUSTOM_OPS": "0",
            "VLLM_HCU_SWA_DECODE": "1",
        }
    )
    script = """
import builtins

native_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == "vllm_hcu.v1.attention.backends.swa_decode":
        raise AssertionError("master-off imported the optional SWA provider")
    return native_import(name, *args, **kwargs)
builtins.__import__ = guarded_import

from vllm_hcu.v1.attention.backends import flash_attn
assert flash_attn._SWA_DECODE_ENABLED is False
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


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
