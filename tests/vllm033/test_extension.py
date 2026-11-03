from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import inspect
import math
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


def context(version: str | None = "0.33.0") -> ExtensionContext:
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
        "vllm.envs": SimpleNamespace(
            environment_variables={}, compile_factors=lambda: {}
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
        {"lightop-rms-norm"}
    )
    assert spec.activate is extension.activate_extension


def test_operator_activation_does_not_require_or_modify_int8_routing(monkeypatch):
    calls = []
    base = successful_importer(calls)

    def importer(name):
        if name == "aiter.moe" or name.startswith("hcu_vllm_das.moe"):
            raise AssertionError("Official INT8 routing must remain independent")
        return base(name)

    monkeypatch.setattr(extension, "_import_module", importer)
    extension.activate_extension(context())
    assert extension._activated is True
    assert "hcu_vllm_das.rms_norm" in calls


def test_extension_does_not_advertise_int8_without_public_registration():
    assert "aiter-int8-moe" not in extension.extension_spec().capabilities


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
        "0.33.0",
        "0.33.0rc1",
        "0.33.0.dev7",
        "0.33.0.post1",
        "0.33.0+rocm723",
        "0.33.1",
        "0.33.7+das.1",
    ),
)
def test_activation_accepts_the_vllm_033_release_line(
    monkeypatch: pytest.MonkeyPatch, version: str
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(extension, "_import_module", successful_importer(calls))

    extension.activate_extension(context(version))

    assert calls == [
        "lightop.norm",
        "vllm.model_executor.layers.layernorm",
        "vllm.utils.torch_utils",
        "vllm.envs",
        "hcu_vllm_das.rms_norm",
    ]


@pytest.mark.parametrize(
    "version",
    (
        None,
        "not-a-version",
        "1!0.33.0",
        "0.31.0",
        "0.31.9",
        "0.32.0",
        "0.32.9+das.1",
        "0.34.0",
    ),
)
def test_activation_rejects_unsupported_metadata_before_importing_runtime(
    monkeypatch: pytest.MonkeyPatch, version: str | None
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(extension, "_import_module", successful_importer(calls))

    with pytest.raises(
        extension.DasExtensionError,
        match=r">=0\.33\.0,<0\.34\.0",
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
def test_missing_aiter_int8_symbols_do_not_affect_rms_activation(
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

    extension.activate_extension(context())
    assert "aiter.moe" not in calls
    assert "hcu_vllm_das.moe.int8_oracle" not in calls
    assert extension._activated is True


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
def test_changed_aiter_abi_does_not_affect_rms_activation(
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

    extension.activate_extension(context())
    assert "aiter.moe" not in calls
    assert "hcu_vllm_das.moe.int8_oracle" not in calls
    assert extension._activated is True


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
    assert importer.int8_oracle.installs == 0
    assert calls == [
        "lightop.norm",
        "vllm.model_executor.layers.layernorm",
        "vllm.utils.torch_utils",
        "vllm.envs",
        "hcu_vllm_das.rms_norm",
    ]


@pytest.mark.parametrize("documentation", (
    "rmsnorm_forward_autograd(arg0: torch.Tensor, arg1: torch.Tensor, arg2: typing.SupportsFloat, arg3: bool) -> torch.Tensor",
    "rmsnorm_forward_autograd(*args, **kwargs)\nOverloaded function.\n\n"
    "1. rmsnorm_forward_autograd(a, b, epsilon, training, required_extra)\n"
    "2. rmsnorm_forward_autograd(a, b, epsilon, training, optional_extra=False)",
))
def test_uninspectable_native_rms_uses_documented_call_abi_before_publication(
    monkeypatch: pytest.MonkeyPatch, documentation: str,
) -> None:
    # A real C builtin without inspect metadata stands in for the pybind export.
    # Only its documentation is doubled; it must never be invoked by preflight.
    with pytest.raises(ValueError):
        inspect.signature(math.log)
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_RMS_NORM", "1")
    calls = []
    base = successful_importer(calls)
    getdoc = inspect.getdoc
    monkeypatch.setattr(inspect, "getdoc", lambda value: documentation if value is math.log else getdoc(value))

    def changed_importer(name):
        module = base(name)
        if name == "lightop.norm":
            module.rmsnorm_forward_autograd = math.log
        return module

    monkeypatch.setattr(extension, "_import_module", changed_importer)
    extension.activate_extension(context())
    assert extension._activated is True
    assert calls.index("lightop.norm") < calls.index("hcu_vllm_das.rms_norm")


@pytest.mark.parametrize("documentation", (
    None,
    "rmsnorm_forward_autograd(*args, **kwargs)",
    "rmsnorm_forward_autograd(a, b, epsilon)",
    "rmsnorm_forward_autograd(a, b, epsilon, training, required_extra)",
    "rmsnorm_forward_autograd(a, b, epsilon, *, training)",
    "another_function(a, b, epsilon, training)",
))
def test_uninspectable_native_rms_rejects_unproven_or_incompatible_doc_abi(
    monkeypatch: pytest.MonkeyPatch, documentation: str | None,
) -> None:
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_RMS_NORM", "1")
    calls = []
    base = successful_importer(calls)
    getdoc = inspect.getdoc
    monkeypatch.setattr(inspect, "getdoc", lambda value: documentation if value is math.log else getdoc(value))

    def changed_importer(name):
        module = base(name)
        if name == "lightop.norm":
            module.rmsnorm_forward_autograd = math.log
        return module

    monkeypatch.setattr(extension, "_import_module", changed_importer)
    with pytest.raises(extension.DasExtensionError, match="documented.*argument"):
        extension.activate_extension(context())
    assert calls == ["lightop.norm"]
    assert extension._activated is False
    assert "hcu_vllm_das.rms_norm" not in calls


def test_uninspectable_python_rms_is_not_treated_as_a_native_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class OpaquePythonCallable:
        """rmsnorm_forward_autograd(x, weight, epsilon, training) -> Tensor"""

        __signature__ = 42

        def __call__(self, x, weight, epsilon, training):
            raise AssertionError("preflight must not call providers")

    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_RMS_NORM", "1")
    calls = []
    base = successful_importer(calls)

    def changed_importer(name):
        module = base(name)
        if name == "lightop.norm":
            module.rmsnorm_forward_autograd = OpaquePythonCallable()
        return module

    monkeypatch.setattr(extension, "_import_module", changed_importer)
    with pytest.raises(extension.DasExtensionError, match="four-positional-argument") as caught:
        extension.activate_extension(context())
    assert calls == ["lightop.norm"]
    assert extension._activated is False
    assert "hcu_vllm_das.rms_norm" not in calls

    assert isinstance(caught.value.__cause__, TypeError)
    assert "__signature__" in str(caught.value.__cause__)


@pytest.mark.parametrize("name", ("rmsnorm_forward_autograd", "fused_add_rms_norm"))
@pytest.mark.parametrize("kind", (
    "required_fifth", "only_three", "keyword_only", "coroutine", "generator", "async_generator",
))
def test_rms_call_contract_rejects_before_factors_or_oot_publication(
    monkeypatch: pytest.MonkeyPatch, name: str, kind: str,
) -> None:
    def required_fifth(a, b, c, d, e):
        raise AssertionError("preflight must not call providers")

    def only_three(a, b, c):
        raise AssertionError("preflight must not call providers")

    def keyword_only(a, b, c, *, d):
        raise AssertionError("preflight must not call providers")

    async def coroutine(a, b, c, d):
        raise AssertionError("preflight must not call providers")

    def generator(a, b, c, d):
        raise AssertionError("preflight must not call providers")
        yield

    async def async_generator(a, b, c, d):
        raise AssertionError("preflight must not call providers")
        yield

    replacements = {
        "required_fifth": required_fifth, "only_three": only_three,
        "keyword_only": keyword_only, "coroutine": coroutine,
        "generator": generator, "async_generator": async_generator,
    }
    calls: list[str] = []
    base = successful_importer(calls)

    def changed_importer(module_name: str):
        module = base(module_name)
        if module_name == "lightop.norm":
            setattr(module, name, replacements[kind])
        return module

    monkeypatch.setattr(extension, "_import_module", changed_importer)
    with pytest.raises(extension.DasExtensionError, match="four-positional-argument|synchronous"):
        extension.activate_extension(context())

    assert extension._activated is False
    assert calls == ["lightop.norm"]


def test_inspectable_native_rms_cannot_use_documentation_to_hide_wrong_arity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inspect.signature(math.sin).bind(0)
    calls: list[str] = []
    base = successful_importer(calls)
    getdoc = inspect.getdoc
    documentation_reads = []

    def documentation(function):
        if function is math.sin:
            documentation_reads.append(function)
            return "rmsnorm_forward_autograd(x, weight, epsilon, training)"
        return getdoc(function)

    def changed_importer(name: str):
        module = base(name)
        if name == "lightop.norm":
            module.rmsnorm_forward_autograd = math.sin
        return module

    monkeypatch.setattr(inspect, "getdoc", documentation)
    monkeypatch.setattr(extension, "_import_module", changed_importer)
    with pytest.raises(extension.DasExtensionError, match="four-positional-argument"):
        extension.activate_extension(context())

    assert calls == ["lightop.norm"]
    assert documentation_reads == []
    assert extension._activated is False


def test_native_rms_documentation_defaults_and_annotations_are_not_evaluated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = "HCU_RMS_DOC_SHOULD_NOT_EXECUTE"
    monkeypatch.delenv(marker, raising=False)
    documentation = (
        "rmsnorm_forward_autograd(a, b, epsilon, training, "
        "optional_extra=__import__('os').environ.__setitem__('"
        + marker + "', '1')) -> __import__('os').environ.__setitem__('"
        + marker + "', '1')"
    )
    calls: list[str] = []
    base = successful_importer(calls)
    getdoc = inspect.getdoc
    monkeypatch.setattr(inspect, "getdoc", lambda function: documentation if function is math.log else getdoc(function))

    def changed_importer(name: str):
        module = base(name)
        if name == "lightop.norm":
            module.rmsnorm_forward_autograd = math.log
        return module

    monkeypatch.setattr(extension, "_import_module", changed_importer)
    extension.activate_extension(context())

    assert extension._activated is True
    assert marker not in os.environ
