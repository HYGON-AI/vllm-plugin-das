from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


PLUGIN_SOURCE_ROOT = (
    Path(__file__).resolve().parents[2] / "adapter_extension" / "src"
)
sys.path.insert(0, os.fspath(PLUGIN_SOURCE_ROOT))

from hcu_adapter.extensions import ExtensionContext, ExtensionSpec
from hcu_vllm_das import extension


def context(version: str | None = "0.32.0") -> ExtensionContext:
    return ExtensionContext(adapter_version="0.1.0.dev0", vllm_version=version)


@pytest.fixture(autouse=True)
def reset_activation(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(extension, "_activated", False)


def successful_importer(calls: list[str]):
    class RMSNorm:
        register_oot = staticmethod(lambda value: value)

    class MoeQuantType:
        W8A8 = "int8_w8a8"

    class Int8Oracle:
        installs = 0

        @classmethod
        def install(cls):
            cls.installs += 1
            return cls.installs == 1

    def get_aiter_moe_config(
        M,
        E,
        N1,
        N2,
        K,
        top_k,
        block_size,
        dtype,
        quant_type,
        activation="silu",
        gated=None,
        spec_sol_type=None,
        use_shuffle=0,
    ):
        return False, None

    def aiter_moe(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        moe_config,
        inplace=False,
        activation="silu",
        w1_scale=None,
        w2_scale=None,
        w1_zp=None,
        w2_zp=None,
        a1_scale=None,
        a2_scale=None,
        block_shape=None,
        global_num_experts=-1,
        expert_map=None,
        routed_scaling_factor=1.0,
        use_weight_shuffle=False,
        output_dtype=None,
        gemm1_alpha=None,
        gemm1_limit=None,
    ):
        return hidden_states

    modules = {
        "lightop.norm": SimpleNamespace(
            fused_add_rms_norm=lambda x, residual, weight, eps: (x, residual),
            rmsnorm_forward_autograd=lambda x, weight, eps, training: x,
        ),
        "vllm.model_executor.layers.layernorm": SimpleNamespace(RMSNorm=RMSNorm),
        "vllm.utils.torch_utils": SimpleNamespace(
            direct_register_custom_op=lambda **kwargs: None
        ),
        "aiter.moe": SimpleNamespace(
            MoeQuantType=MoeQuantType,
            get_aiter_moe_config=get_aiter_moe_config,
            aiter_moe=aiter_moe,
        ),
        "hcu_vllm_das.rms_norm": SimpleNamespace(),
        "hcu_vllm_das.moe.int8_oracle": Int8Oracle,
    }

    def import_module(name: str):
        calls.append(name)
        return modules[name]

    import_module.int8_oracle = Int8Oracle
    return import_module


def test_factory_declares_the_adapter_api_and_migrated_capabilities() -> None:
    spec = extension.extension_spec()

    assert isinstance(spec, ExtensionSpec)
    assert spec.name == "das-hcu"
    assert spec.api_version == 1
    assert spec.priority == 100
    assert spec.capabilities == frozenset(
        {"lightop-rms-norm", "aiter-int8-moe"}
    )
    assert spec.activate is extension.activate_extension


def test_factory_does_not_claim_a_future_adapter_api() -> None:
    code = f"""
import sys
from types import ModuleType

adapter = ModuleType("hcu_adapter")
adapter.__path__ = []
extensions = ModuleType("hcu_adapter.extensions")
extensions.EXTENSION_API_VERSION = 2
extensions.ExtensionContext = object

class ExtensionSpec:
    def __init__(self, *, name, api_version, priority, capabilities, activate):
        self.name = name
        self.api_version = api_version
        self.priority = priority
        self.capabilities = capabilities
        self.activate = activate

extensions.ExtensionSpec = ExtensionSpec
sys.modules["hcu_adapter"] = adapter
sys.modules["hcu_adapter.extensions"] = extensions
sys.path.insert(0, {os.fspath(PLUGIN_SOURCE_ROOT)!r})

from hcu_vllm_das.extension import extension_spec
assert extension_spec().api_version == 1
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_discovery_and_factory_import_no_runtime_frameworks() -> None:
    code = f"""
import sys
sys.path.insert(0, {os.fspath(PLUGIN_SOURCE_ROOT)!r})
blocked = {{"torch", "vllm", "lightop", "aiter", "vllm_hcu"}}
before = set(sys.modules)
from hcu_vllm_das.extension import extension_spec
spec = extension_spec()
assert spec.name == "das-hcu"
introduced = (set(sys.modules) - before).intersection(blocked)
assert not introduced, introduced
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        env=dict(os.environ),
    )

    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "version",
    (
        "0.32.0",
        "0.32.0rc1",
        "0.32.0.dev7",
        "0.32.0.post1",
        "0.32.0+rocm723",
        "0.32.1",
        "0.32.7+das.1",
    ),
)
def test_activation_accepts_the_vllm_032_release_line(
    monkeypatch: pytest.MonkeyPatch, version: str
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(extension, "_import_module", successful_importer(calls))

    extension.activate_extension(context(version))

    assert calls == [
        "lightop.norm",
        "aiter.moe",
        "vllm.model_executor.layers.layernorm",
        "vllm.utils.torch_utils",
        "hcu_vllm_das.rms_norm",
        "hcu_vllm_das.moe.int8_oracle",
    ]


@pytest.mark.parametrize(
    "version",
    (None, "not-a-version", "1!0.32.0", "0.31.0", "0.31.9", "0.33.0"),
)
def test_activation_rejects_unsupported_metadata_before_importing_runtime(
    monkeypatch: pytest.MonkeyPatch, version: str | None
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(extension, "_import_module", successful_importer(calls))

    with pytest.raises(
        extension.DasExtensionError,
        match=r">=0\.32\.0,<0\.33\.0",
    ):
        extension.activate_extension(context(version))

    assert calls == []
    assert extension._activated is False


@pytest.mark.parametrize(
    ("replacement", "message"),
    (
        (
            SimpleNamespace(
                rmsnorm_forward_autograd=None,
                fused_add_rms_norm=lambda x, residual, weight, eps: (x, residual),
            ),
            "rmsnorm_forward_autograd",
        ),
        (
            SimpleNamespace(
                rmsnorm_forward_autograd=lambda x, weight, eps, training: x,
                fused_add_rms_norm=None,
            ),
            "fused_add_rms_norm",
        ),
    ),
)
def test_missing_lightop_rmsnorm_symbol_fails_before_product_imports(
    monkeypatch: pytest.MonkeyPatch,
    replacement: object,
    message: str,
) -> None:
    calls: list[str] = []
    base = successful_importer(calls)

    def changed_importer(name: str):
        if name == "lightop.norm":
            calls.append(name)
            return replacement
        return base(name)

    monkeypatch.setattr(extension, "_import_module", changed_importer)

    with pytest.raises(extension.DasExtensionError, match=message):
        extension.activate_extension(context())

    assert "hcu_vllm_das.rms_norm" not in calls
    assert extension._activated is False


@pytest.mark.parametrize(
    ("replacement", "message"),
    (
        (
            SimpleNamespace(
                MoeQuantType=SimpleNamespace(W8A8="int8_w8a8"),
                get_aiter_moe_config=None,
                aiter_moe=lambda *args, **kwargs: None,
            ),
            "get_aiter_moe_config",
        ),
        (
            SimpleNamespace(
                MoeQuantType=SimpleNamespace(W8A8="int8_w8a8"),
                get_aiter_moe_config=lambda *args, **kwargs: (False, None),
                aiter_moe=None,
            ),
            "aiter_moe",
        ),
        (
            SimpleNamespace(
                MoeQuantType=SimpleNamespace(W8A8="changed"),
                get_aiter_moe_config=lambda *args, **kwargs: (False, None),
                aiter_moe=lambda *args, **kwargs: None,
            ),
            "MoeQuantType.W8A8",
        ),
    ),
)
def test_missing_aiter_int8_symbol_fails_before_product_imports(
    monkeypatch: pytest.MonkeyPatch,
    replacement: object,
    message: str,
) -> None:
    calls: list[str] = []
    base = successful_importer(calls)

    def changed_importer(name: str):
        if name == "aiter.moe":
            calls.append(name)
            return replacement
        return base(name)

    monkeypatch.setattr(extension, "_import_module", changed_importer)

    with pytest.raises(extension.DasExtensionError, match=message):
        extension.activate_extension(context())

    assert "hcu_vllm_das.rms_norm" not in calls
    assert "hcu_vllm_das.moe.int8_oracle" not in calls
    assert extension._activated is False


@pytest.mark.parametrize(
    ("module_name", "callable_name", "replacement"),
    (
        (
            "aiter.moe",
            "get_aiter_moe_config",
            lambda M, E: (False, None),
        ),
        (
            "aiter.moe",
            "aiter_moe",
            lambda hidden_states, w1: hidden_states,
        ),
    ),
)
def test_changed_aiter_abi_fails_before_product_imports(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    callable_name: str,
    replacement: object,
) -> None:
    calls: list[str] = []
    base = successful_importer(calls)

    def changed_importer(name: str):
        module = base(name)
        if name == module_name:
            setattr(module, callable_name, replacement)
        return module

    monkeypatch.setattr(extension, "_import_module", changed_importer)

    with pytest.raises(extension.DasExtensionError, match="signature changed"):
        extension.activate_extension(context())

    assert "hcu_vllm_das.rms_norm" not in calls
    assert "hcu_vllm_das.moe.int8_oracle" not in calls
    assert extension._activated is False


@pytest.mark.parametrize(
    ("module_name", "replacement", "message"),
    (
        (
            "vllm.model_executor.layers.layernorm",
            SimpleNamespace(RMSNorm=SimpleNamespace()),
            "RMSNorm.register_oot",
        ),
        (
            "vllm.utils.torch_utils",
            SimpleNamespace(direct_register_custom_op=None),
            "direct_register_custom_op",
        ),
    ),
)
def test_changed_vllm_api_fails_before_capability_registration(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    replacement: object,
    message: str,
) -> None:
    calls: list[str] = []
    base = successful_importer(calls)

    def changed_importer(name: str):
        if name == module_name:
            calls.append(name)
            return replacement
        return base(name)

    monkeypatch.setattr(extension, "_import_module", changed_importer)

    with pytest.raises(extension.DasExtensionError, match=message):
        extension.activate_extension(context())

    assert "hcu_vllm_das.rms_norm" not in calls
    assert extension._activated is False


def test_repeated_and_concurrent_activation_registers_capability_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    importer = successful_importer(calls)
    monkeypatch.setattr(extension, "_import_module", importer)

    with ThreadPoolExecutor(max_workers=8) as pool:
        tuple(pool.map(extension.activate_extension, [context()] * 32))
    extension.activate_extension(context())

    assert calls.count("hcu_vllm_das.rms_norm") == 1
    assert calls.count("hcu_vllm_das.moe.int8_oracle") == 1
    assert importer.int8_oracle.installs == 1
    assert calls == [
        "lightop.norm",
        "aiter.moe",
        "vllm.model_executor.layers.layernorm",
        "vllm.utils.torch_utils",
        "hcu_vllm_das.rms_norm",
        "hcu_vllm_das.moe.int8_oracle",
    ]
