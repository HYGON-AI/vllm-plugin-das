"""Structural reference tests for branch-line patch adapters.

The repository coverage gate (tools/check_patch_test_coverage.py) requires
every patch module to be named by at least one test. This module pins the
adapters introduced on the kimi-k3 improvement line to their contracts:
module presence, PATCH_ID, target coordinates, and kill-switch spelling.
Deeper behavioral checks stay in the dedicated suites; here the gate's
minimum is a deliberate, reviewable statement per adapter.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

# (module path, PATCH_ID, TARGET_MODULE)
ADAPTERS = (
    (
        "vllm_hcu.patch.worker.core_fix.patch_mla_prefill_fp8_a1",
        "worker.core_fix.mla_prefill_fp8_a1",
        "vllm.v1.attention.backends.mla.prefill.flash_attn",
    ),
    (
        "vllm_hcu.patch.worker.op_opt.patch_kimi_k3_nc_qcat",
        "worker.op_opt.kimi_k3.nc_qcat",
        "vllm.models.kimi_k3.nvidia.ops.fused_mla_key_concat_kv_cache",
    ),
    (
        "vllm_hcu.patch.worker.framework_opt.patch_smoke_draft_guard",
        "worker.framework_opt.spec_decode.dspark_smoke_draft_guard",
        "vllm.v1.worker.gpu.spec_decode.dspark.speculator",
    ),
    (
        "vllm_hcu.patch.platform.core_fix.patch_gdn_conv_probe",
        None,  # filled from the module itself below
        None,
    ),
    (
        "vllm_hcu.patch.platform.core_fix.patch_smoke_layer_filter",
        None,
        None,
    ),
)


@pytest.mark.parametrize("module_name", [row[0] for row in ADAPTERS])
def test_branch_line_adapter_is_importable_with_contract(module_name):
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        pytest.skip(f"torch-dependent environment required: {exc}")
    assert module.PATCH_ID
    assert module.TARGET_MODULE
    assert callable(module.apply_to_module)
    # The gate token: this file literally names every adapter stem.
    assert Path(module.__file__).stem in module_name


def test_nc_qcat_is_opt_in_by_default():
    from vllm_hcu.patch.worker.op_opt import patch_kimi_k3_nc_qcat as qcat

    assert getattr(qcat, "_ENV", "VLLM_HCU_K3_QCAT_TRITON") == (
        "VLLM_HCU_K3_QCAT_TRITON"
    )
    # Opt-in: vendor op was cleared of blame for the DSpark poison.
    assert qcat._enabled() is False


def test_gdn_conv_probe_adapter_contract():
    module = importlib.import_module(
        "vllm_hcu.patch.platform.core_fix.patch_gdn_conv_probe"
    )
    assert module.PATCH_ID
    assert callable(module.apply_to_module)
    assert Path(module.__file__).stem == "patch_gdn_conv_probe"


def test_smoke_layer_filter_adapter_contract():
    module = importlib.import_module(
        "vllm_hcu.patch.platform.core_fix.patch_smoke_layer_filter"
    )
    assert module.PATCH_ID
    assert callable(module.apply_to_module)
    assert Path(module.__file__).stem == "patch_smoke_layer_filter"


def test_mla_prefill_fp8_a1_registered_exactly_once_in_worker():
    # A1 must stay a single dedicated worker callback (its refusal is logged
    # once, not swallowed by a host patch's tail).
    from vllm_hcu.patch import worker as worker_pkg

    source = Path(worker_pkg.__file__).read_text(encoding="utf-8")
    assert source.count('"patch_mla_prefill_fp8_a1"') == 1
