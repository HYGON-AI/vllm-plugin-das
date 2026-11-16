from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from packaging.version import Version


pytestmark = pytest.mark.hcu

_RESULT_PREFIX = "HCU_GRAPH_RESULT="
_PROBE = r'''
import json
import os
import warnings

import torch
import torch.nn.functional as functional
import vllm
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.platforms import current_platform
from vllm.plugins import load_general_plugins

load_general_plugins()

from hcu_vllm_das import rms_norm as rms_module
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm


def rms_reference(value, weight, epsilon):
    value_float = value.float()
    return (
        value_float
        * torch.rsqrt(value_float.square().mean(dim=-1, keepdim=True) + epsilon)
        * weight.float()
    )


torch.manual_seed(20261029)
config = VllmConfig()
with set_current_vllm_config(config):
    silu = SiluAndMul()
    rms = RMSNorm(256).to(device="cuda", dtype=torch.bfloat16)
    rms.weight.data.normal_(mean=1.0, std=0.1)

    def graph(silu_input, norm_input, residual):
        silu_output = silu(silu_input)
        ordinary = rms(norm_input)
        fused_input = norm_input.clone()
        fused, updated_residual = rms(fused_input, residual)
        return silu_output, ordinary, fused, updated_residual

    compiled = torch.compile(graph, dynamic=True)
    first_silu = torch.randn((8, 512), device="cuda", dtype=torch.bfloat16)
    first_norm = torch.randn((8, 256), device="cuda", dtype=torch.bfloat16)
    first_residual = torch.randn_like(first_norm)
    second_silu = first_silu + 0.125
    second_norm = first_norm - 0.25
    second_residual = first_residual + 0.375

    silu_expected = (
        functional.silu(second_silu[..., :256].float())
        * second_silu[..., 256:].float()
    )
    ordinary_expected = rms_reference(
        second_norm,
        rms.weight.data,
        rms.variance_epsilon,
    )
    summed_expected = second_norm.float() + second_residual.float()
    fused_expected = rms_reference(
        summed_expected,
        rms.weight.data,
        rms.variance_epsilon,
    )
    expected = (
        silu_expected,
        ordinary_expected,
        fused_expected,
        summed_expected,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        compiled(first_silu.clone(), first_norm.clone(), first_residual.clone())
        outputs = compiled(
            second_silu.clone(),
            second_norm.clone(),
            second_residual.clone(),
        )
        torch.cuda.synchronize()
    max_errors = [
        float((actual.float() - reference).abs().max().item())
        for actual, reference in zip(outputs, expected, strict=True)
    ]

report = {
    "mode": os.environ["HCU_GRAPH_PROBE_MODE"],
    "vllm_version": vllm.__version__,
    "vllm_file": vllm.__file__,
    "plugin_file": rms_module.__file__,
    "platform": f"{type(current_platform).__module__}.{type(current_platform).__name__}",
    "custom_ops": config.compilation_config.custom_ops,
    "silu_class": type(silu).__name__,
    "rms_class": type(rms).__name__,
    "silu_forward": silu._forward_method.__name__,
    "rms_forward": rms._forward_method.__name__,
    "max_errors": max_errors,
    "output_sums": [float(value.float().sum().item()) for value in outputs],
    "alias_warnings": [
        str(item.message)
        for item in caught
        if "may not alias any inputs" in str(item.message)
    ],
}
print("HCU_GRAPH_RESULT=" + json.dumps(report, sort_keys=True))
'''


def _live_requested() -> bool:
    return os.environ.get("HCU_VLLM032_GRAPH_LIVE") == "1"


def _run_probe(mode: str) -> dict[str, object]:
    environment = dict(os.environ)
    environment["HCU_GRAPH_PROBE_MODE"] = mode
    environment["VLLM_HCU_USE_CUSTOM_OPS"] = "1" if mode == "on" else "0"
    environment["VLLM_HCU_USE_CUSTOM_RMS_NORM"] = "1"
    result = subprocess.run(
        [sys.executable, "-c", _PROBE],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        cwd=Path("/tmp"),
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    result_lines = [
        line
        for line in result.stdout.splitlines()
        if line.startswith(_RESULT_PREFIX)
    ]
    assert len(result_lines) == 1, result.stdout + result.stderr
    return json.loads(result_lines[0].removeprefix(_RESULT_PREFIX))


@pytest.mark.skipif(
    not _live_requested(),
    reason="set HCU_VLLM032_GRAPH_LIVE=1 for installed-wheel Graph validation",
)
def test_installed_operator_on_off_graph_replay_uses_expected_paths() -> None:
    optimized = _run_probe("on")
    official = _run_probe("off")

    for report in (optimized, official):
        version = Version(report["vllm_version"])
        assert version.epoch == 0 and version.release[:2] == (0, 32)
        assert "site-packages/vllm/__init__.py" in report["vllm_file"]
        assert "site-packages/hcu_vllm_das/rms_norm.py" in report["plugin_file"]
        assert report["platform"] == "hcu_adapter.bridges.vllm.HcuPlatform"
        assert "none" in report["custom_ops"]
        assert report["silu_class"] == "SiluAndMul"
        assert report["rms_class"] == "HcuRMSNorm"
        assert report["alias_warnings"] == []
        assert all(error <= 0.16 for error in report["max_errors"])

    assert optimized["silu_forward"] == "forward_native"
    assert optimized["rms_forward"] == "forward_hip"
    assert official["silu_forward"] == "forward_native"
    assert official["rms_forward"] == "forward_native"
    assert optimized["output_sums"] == pytest.approx(
        official["output_sums"],
        rel=4e-2,
        abs=4e-2,
    )
