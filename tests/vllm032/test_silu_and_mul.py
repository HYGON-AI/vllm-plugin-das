from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


PLUGIN_SOURCE_ROOT = (
    Path(__file__).resolve().parents[2] / "adapter_extension" / "src"
)
sys.path.insert(0, os.fspath(PLUGIN_SOURCE_ROOT))

from hcu_vllm_das.environment import custom_ops_enabled, silu_and_mul_enabled


def test_environment_switches_default_on_and_each_can_roll_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("VLLM_HCU_USE_CUSTOM_OPS", raising=False)
    monkeypatch.delenv("VLLM_HCU_USE_CUSTOM_SILU_AND_MUL", raising=False)
    assert custom_ops_enabled() is True
    assert silu_and_mul_enabled() is True

    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "0")
    assert custom_ops_enabled() is False
    assert silu_and_mul_enabled() is False

    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_SILU_AND_MUL", "false")
    assert custom_ops_enabled() is True
    assert silu_and_mul_enabled() is False


@pytest.mark.parametrize(
    ("name", "value"),
    (
        ("VLLM_HCU_USE_CUSTOM_OPS", "sometimes"),
        ("VLLM_HCU_USE_CUSTOM_SILU_AND_MUL", "enabled"),
    ),
)
def test_environment_switches_reject_ambiguous_values(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str
) -> None:
    monkeypatch.setenv(name, value)

    with pytest.raises(ValueError, match=name):
        silu_and_mul_enabled()


def run_fake_runtime(*, master: str = "1", child: str = "1") -> dict[str, object]:
    code = f"""
import importlib
import json
import os
import sys
from types import ModuleType, SimpleNamespace

os.environ["VLLM_HCU_USE_CUSTOM_OPS"] = {master!r}
os.environ["VLLM_HCU_USE_CUSTOM_SILU_AND_MUL"] = {child!r}
sys.path.insert(0, {os.fspath(PLUGIN_SOURCE_ROOT)!r})
events = []

class Tensor:
    def __init__(self, shape, dtype="bf16", device="hcu:0", tag="input"):
        self.shape = shape
        self.dtype = dtype
        self.device = device
        self.tag = tag

def empty(shape, *, dtype, device):
    events.append(["empty", list(shape), dtype, device])
    return Tensor(shape, dtype=dtype, device=device, tag="output")

torch = ModuleType("torch")
torch.Tensor = Tensor
torch.empty = empty
torch.ops = SimpleNamespace(vllm=SimpleNamespace())
sys.modules["torch"] = torch

lightop = ModuleType("lightop")
lightop.__path__ = []
activation = ModuleType("lightop.activation")
def silu_and_mul_opt(out, value):
    events.append(["lightop", out.tag, value.tag])
activation.silu_and_mul_opt = silu_and_mul_opt
lightop.activation = activation
sys.modules["lightop"] = lightop
sys.modules["lightop.activation"] = activation

vllm = ModuleType("vllm")
vllm.__path__ = []
model_executor = ModuleType("vllm.model_executor")
model_executor.__path__ = []
layers = ModuleType("vllm.model_executor.layers")
layers.__path__ = []
activation_module = ModuleType("vllm.model_executor.layers.activation")
utils = ModuleType("vllm.utils")
utils.__path__ = []
torch_utils = ModuleType("vllm.utils.torch_utils")

oot = {{}}
class SiluAndMul:
    @classmethod
    def register_oot(cls, replacement):
        oot[cls.__name__] = replacement
        events.append(["register_oot", replacement.__name__])
        return replacement

    def forward_cuda(self, value):
        events.append(["native", value.tag])
        return "native-result"

activation_module.SiluAndMul = SiluAndMul
registrations = []
def direct_register_custom_op(**kwargs):
    registrations.append(kwargs)
    setattr(torch.ops.vllm, kwargs["op_name"], kwargs["op_func"])
torch_utils.direct_register_custom_op = direct_register_custom_op

for name, module in (
    ("vllm", vllm),
    ("vllm.model_executor", model_executor),
    ("vllm.model_executor.layers", layers),
    ("vllm.model_executor.layers.activation", activation_module),
    ("vllm.utils", utils),
    ("vllm.utils.torch_utils", torch_utils),
):
    sys.modules[name] = module

first = importlib.import_module("hcu_vllm_das.silu_and_mul")
second = importlib.import_module("hcu_vllm_das.silu_and_mul")
value = Tensor((3, 16))
result = first.HcuSiluAndMul().forward_hip(value)
fake = registrations[0]["fake_impl"](value)
print(json.dumps({{
    "same_module": first is second,
    "registration_count": len(registrations),
    "op_name": registrations[0]["op_name"],
    "mutates_args": registrations[0]["mutates_args"],
    "oot": sorted(oot),
    "result": getattr(result, "tag", result),
    "result_shape": list(result.shape) if hasattr(result, "shape") else None,
    "fake_shape": list(fake.shape),
    "events": events,
}}))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout)


def test_enabled_operator_registers_once_and_calls_lightop() -> None:
    report = run_fake_runtime()

    assert report["same_module"] is True
    assert report["registration_count"] == 1
    assert report["op_name"] == "hcu_silu_and_mul_lightop"
    assert report["mutates_args"] == []
    assert report["oot"] == ["SiluAndMul"]
    assert report["result"] == "output"
    assert report["result_shape"] == [3, 8]
    assert ["lightop", "output", "input"] in report["events"]


@pytest.mark.parametrize(
    ("master", "child"),
    (("0", "1"), ("1", "0")),
)
def test_each_rollback_switch_uses_vllm_native_path(
    master: str, child: str
) -> None:
    report = run_fake_runtime(master=master, child=child)

    assert report["result"] == "native-result"
    assert ["native", "input"] in report["events"]
    assert not any(event[0] == "lightop" for event in report["events"])

