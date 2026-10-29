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


def run_fake_runtime(
    *,
    residual: bool = False,
    master: str = "1",
    child: str = "1",
    hidden_size: int = 256,
    input_width: int | None = None,
    batch_invariant: bool = False,
    has_weight: bool = True,
    variance_override: int | None = None,
    input_contiguous: bool = True,
    residual_contiguous: bool = True,
    weight_contiguous: bool = True,
    input_dtype: str = "bf16",
    residual_dtype: str = "bf16",
    weight_dtype: str = "bf16",
    residual_width: int | None = None,
    base_enabled: bool = False,
) -> dict[str, object]:
    width = hidden_size if input_width is None else input_width
    residual_width = width if residual_width is None else residual_width
    code = f"""
import importlib
import json
import os
import sys
from types import ModuleType, SimpleNamespace

os.environ["VLLM_HCU_USE_CUSTOM_OPS"] = {master!r}
os.environ["VLLM_HCU_USE_CUSTOM_RMS_NORM"] = {child!r}
sys.path.insert(0, {os.fspath(PLUGIN_SOURCE_ROOT)!r})
events = []

class Tensor:
    def __init__(self, shape, *, dtype="bf16", tag="tensor", contiguous=True):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = "hcu:0"
        self.tag = tag
        self._contiguous = contiguous
        self.data = self

    def is_contiguous(self):
        return self._contiguous

def empty_like(value):
    events.append(["empty_like", value.tag])
    return Tensor(value.shape, dtype=value.dtype, tag="ordinary-fake")

torch = ModuleType("torch")
torch.Tensor = Tensor
torch.bfloat16 = "bf16"
torch.empty_like = empty_like
torch.float16 = "fp16"
torch.ops = SimpleNamespace(vllm=SimpleNamespace())
sys.modules["torch"] = torch

lightop = ModuleType("lightop")
lightop.__path__ = []
norm = ModuleType("lightop.norm")
def rmsnorm_forward_autograd(value, weight, epsilon, training):
    events.append(
        ["ordinary-lightop", value.tag, weight.tag, epsilon, training]
    )
    return Tensor(value.shape, dtype=value.dtype, tag="ordinary-output")
def fused_add_rms_norm(value, residual_value, weight, epsilon):
    events.append(
        ["fused-lightop", value.tag, residual_value.tag, weight.tag, epsilon]
    )
    value.tag = "fused-output"
    residual_value.tag = "fused-residual"
    return "ignored-lightop-return"
norm.rmsnorm_forward_autograd = rmsnorm_forward_autograd
norm.fused_add_rms_norm = fused_add_rms_norm
lightop.norm = norm
sys.modules["lightop"] = lightop
sys.modules["lightop.norm"] = norm

vllm = ModuleType("vllm")
vllm.__path__ = []
vllm.envs = SimpleNamespace(VLLM_BATCH_INVARIANT={batch_invariant!r})
model_executor = ModuleType("vllm.model_executor")
model_executor.__path__ = []
layers = ModuleType("vllm.model_executor.layers")
layers.__path__ = []
layernorm = ModuleType("vllm.model_executor.layers.layernorm")
utils = ModuleType("vllm.utils")
utils.__path__ = []
torch_utils = ModuleType("vllm.utils.torch_utils")

oot = {{}}
class RMSNorm:
    @classmethod
    def register_oot(cls, replacement):
        oot[cls.__name__] = replacement
        events.append(["register_oot", replacement.__name__])
        return replacement

    @classmethod
    def enabled(cls):
        events.append(["base-enabled"])
        return {base_enabled!r}

    def __init__(self):
        self.hidden_size = {hidden_size!r}
        self.variance_epsilon = 1e-6
        self.training = False
        self.has_weight = {has_weight!r}
        self.pass_weight = {has_weight!r}
        self.pass_weight_add = {has_weight!r}
        self.variance_size_override = {variance_override!r}
        self.weight = Tensor(
            ({hidden_size!r},),
            dtype={weight_dtype!r},
            tag="weight",
            contiguous={weight_contiguous!r},
        )

    def forward_cuda(self, value, residual_value=None):
        events.append(
            [
                "native",
                value.tag,
                None if residual_value is None else residual_value.tag,
            ]
        )
        return "native-result"

layernorm.RMSNorm = RMSNorm
registrations = []
def direct_register_custom_op(**kwargs):
    registrations.append(kwargs)
    setattr(torch.ops.vllm, kwargs["op_name"], kwargs["op_func"])
torch_utils.direct_register_custom_op = direct_register_custom_op

for name, module in (
    ("vllm", vllm),
    ("vllm.model_executor", model_executor),
    ("vllm.model_executor.layers", layers),
    ("vllm.model_executor.layers.layernorm", layernorm),
    ("vllm.utils", utils),
    ("vllm.utils.torch_utils", torch_utils),
):
    sys.modules[name] = module

module = importlib.import_module("hcu_vllm_das.rms_norm")
layer = module.HcuRMSNorm()
try:
    effective_enabled = module.HcuRMSNorm.enabled()
    enabled_error = None
except ValueError as error:
    effective_enabled = None
    enabled_error = str(error)
value = Tensor(
    (3, {width!r}),
    dtype={input_dtype!r},
    tag="input",
    contiguous={input_contiguous!r},
)
residual_value = None
if {residual!r}:
    residual_value = Tensor(
        (3, {residual_width!r}),
        dtype={residual_dtype!r},
        tag="residual",
        contiguous={residual_contiguous!r},
    )
result = (
    layer.forward_hip(value, residual_value)
    if enabled_error is None
    else "not-run"
)
if isinstance(result, str):
    serialized_result = result
elif isinstance(result, tuple):
    serialized_result = [item.tag for item in result]
else:
    serialized_result = result.tag
ordinary_registration = next(
    item
    for item in registrations
    if item["op_name"] == "hcu_rmsnorm_forward_lightop"
)
fused_registration = next(
    item
    for item in registrations
    if item["op_name"] == "hcu_fused_add_rms_norm_lightop"
)
ordinary_fake = ordinary_registration["fake_impl"](
    value,
    layer.weight,
    layer.variance_epsilon,
    layer.training,
)
fused_fake = fused_registration["fake_impl"](
    value,
    residual_value if residual_value is not None else value,
    layer.weight,
    layer.variance_epsilon,
)
layer_events = list(events)
impl_probe_value = Tensor(
    (3, {width!r}),
    dtype={input_dtype!r},
    tag="impl-probe-input",
)
impl_probe_residual = Tensor(
    (3, {residual_width!r}),
    dtype={residual_dtype!r},
    tag="impl-probe-residual",
)
fused_impl = fused_registration["op_func"](
    impl_probe_value,
    impl_probe_residual,
    layer.weight,
    layer.variance_epsilon,
)
print(json.dumps({{
    "registration_names": [item["op_name"] for item in registrations],
    "ordinary_mutates": ordinary_registration["mutates_args"],
    "fused_mutates": fused_registration["mutates_args"],
    "oot": sorted(oot),
    "effective_enabled": effective_enabled,
    "enabled_error": enabled_error,
    "result": serialized_result,
    "fused_result_identity": (
        isinstance(result, tuple)
        and result[0] is value
        and result[1] is residual_value
    ),
    "ordinary_fake_shape": list(ordinary_fake.shape),
    "fused_impl_returns_none": fused_impl is None,
    "fused_fake_returns_none": fused_fake is None,
    "events": layer_events,
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


def test_ordinary_rmsnorm_registers_and_calls_lightop() -> None:
    report = run_fake_runtime()

    assert report["registration_names"] == [
        "hcu_rmsnorm_forward_lightop",
        "hcu_fused_add_rms_norm_lightop",
    ]
    assert report["ordinary_mutates"] == []
    assert report["fused_mutates"] == ["x", "residual"]
    assert report["oot"] == ["RMSNorm"]
    assert report["result"] == "ordinary-output"
    assert report["ordinary_fake_shape"] == [3, 256]
    assert any(event[0] == "ordinary-lightop" for event in report["events"])


def test_fused_add_rmsnorm_returns_mutated_inputs() -> None:
    report = run_fake_runtime(residual=True)

    assert report["result"] == ["fused-output", "fused-residual"]
    assert report["fused_result_identity"] is True
    assert report["fused_impl_returns_none"] is True
    assert report["fused_fake_returns_none"] is True
    assert any(event[0] == "fused-lightop" for event in report["events"])


@pytest.mark.parametrize(
    ("master", "child", "base_enabled", "expected"),
    (
        ("1", "1", False, True),
        ("0", "1", True, True),
        ("1", "0", True, True),
        ("0", "1", False, False),
    ),
)
def test_rmsnorm_oot_enablement_preserves_plugin_and_official_paths(
    master: str,
    child: str,
    base_enabled: bool,
    expected: bool,
) -> None:
    report = run_fake_runtime(
        master=master,
        child=child,
        base_enabled=base_enabled,
    )

    assert report["effective_enabled"] is expected


@pytest.mark.parametrize("master", ("1", "0"))
def test_invalid_rmsnorm_policy_fails_before_consulting_official_base(
    master: str,
) -> None:
    report = run_fake_runtime(
        master=master,
        child="enabled",
        base_enabled=True,
    )

    assert "VLLM_HCU_USE_CUSTOM_RMS_NORM" in report["enabled_error"]
    assert ["base-enabled"] not in report["events"]


@pytest.mark.parametrize(
    "overrides",
    (
        {"master": "0"},
        {"child": "0"},
        {"hidden_size": 32},
        {"batch_invariant": True},
        {"has_weight": False},
        {"variance_override": 128},
        {"input_contiguous": False},
        {"residual": True, "residual_contiguous": False},
        {"weight_contiguous": False},
        {"residual": True, "residual_width": 128},
        {"residual": True, "residual_dtype": "fp16"},
        {"weight_dtype": "fp16"},
        {"input_dtype": "fp32", "weight_dtype": "fp32"},
    ),
    ids=(
        "master-switch",
        "child-switch",
        "narrow-width",
        "batch-invariant",
        "weightless",
        "variance-override",
        "strided-input",
        "strided-residual",
        "strided-weight",
        "residual-shape",
        "residual-dtype",
        "weight-dtype",
        "unsupported-dtype",
    ),
)
def test_unsupported_rmsnorm_cases_use_vllm_native_path(
    overrides: dict[str, object],
) -> None:
    report = run_fake_runtime(**overrides)

    assert report["result"] == "native-result"
    assert any(event[0] == "native" for event in report["events"])
    assert not any(
        event[0] in {"ordinary-lightop", "fused-lightop"}
        for event in report["events"]
    )
