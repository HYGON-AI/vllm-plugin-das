"""CPU contracts for DSV4.1 routing against the selected vLLM 0.33 source.

Run with the target Core and Adapter on PYTHONPATH, HCU_ADAPTER_HOST=none,
VLLM_PLUGINS=__disabled__, and all GPU visibility variables empty. Platform
identity and layer construction are doubles; the configuration classes,
quantization registry, exclusion helper and ModelConfig selector are real.
These checks do not establish checkpoint loading or GPU/model correctness.
"""
from __future__ import annotations

import copy
import importlib
import inspect
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
CORE_ROOT = Path(os.environ.get("VLLM_TARGET_ROOT", ROOT.parent / "vllm")).resolve()
PLUGIN_SOURCE = ROOT / "adapter_extension" / "src"
sys.path.insert(0, os.fspath(PLUGIN_SOURCE))
NAME = "hcu_dsv41_channel_fp8"
CAPABILITY = "dsv41-channel-fp8-routing"
SCHEMA = {"quant_method": "fp8", "is_per_channel": True,
          "activation_scheme": "dynamic"}


class HcuPlatform:
    device_name = "hcu"
    supported_quantization = None

    @staticmethod
    def is_rocm():
        return True

    @staticmethod
    def verify_quantization(name):
        return None


@pytest.fixture
def contracts(monkeypatch):
    activation = importlib.import_module("hcu_vllm_das.dsv41_activation")
    candidate = importlib.import_module(
        "hcu_vllm_das.quantization.dsv41_channel_fp8"
    )
    quant = importlib.import_module("vllm.model_executor.layers.quantization")
    model = importlib.import_module("vllm.config.model")
    from hcu_adapter.extensions import ExtensionContext

    assert Path(inspect.getsourcefile(model.ModelConfig)).resolve().is_relative_to(CORE_ROOT)
    # The official API has no unregister operation. Isolate its state only in
    # this test fixture; production must exclusively use its public functions.
    monkeypatch.setattr(quant, "QUANTIZATION_METHODS", list(quant.QUANTIZATION_METHODS))
    monkeypatch.setattr(quant, "_CUSTOMIZED_METHOD_TO_QUANT_CONFIG",
                        dict(quant._CUSTOMIZED_METHOD_TO_QUANT_CONFIG))
    # The real ModelConfig uses a LazyLoader that caches module attributes.
    # Point this isolated replay at the same real module whose registry state
    # is copied above; otherwise the fixture itself creates stale aliases.
    monkeypatch.setattr(model, "me_quant", quant)
    monkeypatch.setattr(activation, "_publication", None)
    monkeypatch.setattr(activation, "_failure", None)
    monkeypatch.setattr(candidate, "current_platform", HcuPlatform())
    monkeypatch.setattr(model, "current_platform", HcuPlatform())
    return SimpleNamespace(
        activation=activation, candidate=candidate, quant=quant, model=model,
        context=ExtensionContext(adapter_version="0.1.0", vllm_version="0.33.0"),
        cls=candidate.HcuDSV41ChannelFP8Config,
    )


def activate(c):
    c.activation.activate_extension(c.context)
    return c


def hf(model_type="deepseek_v41"):
    return SimpleNamespace(model_type=model_type, architectures=[])


def select(c, schema, model_config=None, user_quant=None):
    # Invoke the actual official selector, rather than duplicating its priority
    # loop or requiring model weights/devices for this configuration boundary.
    config = SimpleNamespace(
        quantization=user_quant,
        model_arch_config=SimpleNamespace(quantization_config=copy.deepcopy(schema)),
        hf_config=model_config or hf(),
        allow_deprecated_quantization=False,
    )
    c.model.ModelConfig._verify_quantization(config)
    return config.quantization


def layer(kind):
    import torch
    from vllm.model_executor.layers.linear import ReplicatedLinear
    from vllm.model_executor.layers.fused_moe import RoutedExperts

    cls = ReplicatedLinear if kind == "linear" else RoutedExperts
    obj = cls.__new__(cls)
    torch.nn.Module.__init__(obj)
    obj.moe_config = SimpleNamespace()
    return obj


def test_factory_is_framework_and_vendor_free():
    code = f"""
import sys
sys.path[:0] = {list(sys.path)!r}
import importlib.abc
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {{'torch', 'vllm', 'lightop', 'aiter', 'vllm_hcu'}}:
            raise AssertionError('discovery imported ' + fullname)
sys.meta_path.insert(0, Block())
from hcu_vllm_das.dsv41_activation import extension_spec
spec = extension_spec()
assert spec.api_version == 1
assert spec.capabilities == frozenset({{{CAPABILITY!r}}})
assert spec.check_health is not None
assert not set(sys.modules).intersection({{'torch', 'vllm', 'lightop', 'aiter', 'vllm_hcu'}})
"""
    result = subprocess.run([sys.executable, "-S", "-B", "-c", code],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("version", ["0.33.0", "0.33.1", "0.33.0rc1",
                                     "0.33.0.dev1", "0.33.0+das.test"])
def test_supported_release_registers_distinct_method(contracts, version):
    from hcu_adapter.extensions import ExtensionContext
    c = contracts
    c.activation.activate_extension(ExtensionContext("0.1.0", version))
    assert c.quant.get_quantization_config(NAME) is c.cls
    assert c.activation.extension_spec().capabilities == frozenset({CAPABILITY})
    c.activation.check_health()


@pytest.mark.parametrize("version", [None, "bad", "0.32.0", "0.34.0", "1!0.33.0"])
def test_wrong_release_rejects_before_runtime_import_or_registration(contracts, monkeypatch, version):
    from hcu_adapter.extensions import ExtensionContext
    c = contracts
    calls = []
    monkeypatch.setattr(c.activation, "_import_module", lambda name: calls.append(name))
    with pytest.raises(c.activation.Dsv41ActivationError, match="0.33"):
        c.activation.activate_extension(ExtensionContext("0.1.0", version))
    assert not calls
    assert NAME not in c.quant.QUANTIZATION_METHODS


def test_activation_is_idempotent_and_does_not_overwrite_a_foreign_owner(contracts, monkeypatch):
    c = contracts
    original = c.quant.register_quantization_config
    calls = []

    def register(quantization):
        calls.append(quantization)
        return original(quantization)

    monkeypatch.setattr(c.quant, "register_quantization_config", register)
    activate(c)
    c.activation.activate_extension(c.context)
    assert calls == [NAME]
    assert c.quant.QUANTIZATION_METHODS.count(NAME) == 1


def test_foreign_registration_is_rejected_without_mutation(contracts):
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config
    c = contracts
    c.quant.register_quantization_config(NAME)(Fp8Config)
    before = list(c.quant.QUANTIZATION_METHODS)
    with pytest.raises(c.activation.Dsv41ActivationError, match="owner|registered"):
        c.activation.activate_extension(c.context)
    assert c.quant.get_quantization_config(NAME) is Fp8Config
    assert c.quant.QUANTIZATION_METHODS == before


def test_partial_publication_is_commit_fatal_and_retry_refuses(contracts, monkeypatch):
    from hcu_adapter.extensions import ExtensionCommitError
    c = contracts

    def broken(quantization):
        def publish(cls):
            c.quant.QUANTIZATION_METHODS.append(quantization)
            raise RuntimeError("published then failed")
        return publish

    monkeypatch.setattr(c.quant, "register_quantization_config", broken)
    with pytest.raises(ExtensionCommitError):
        c.activation.activate_extension(c.context)
    with pytest.raises(ExtensionCommitError):
        c.activation.activate_extension(c.context)


def test_changed_registration_abi_rejects_before_publication(contracts, monkeypatch):
    c = contracts
    monkeypatch.setattr(c.quant, "register_quantization_config",
                        lambda quantization, extra: None)
    with pytest.raises(c.activation.Dsv41ActivationError, match="ABI|signature|contract"):
        c.activation.activate_extension(c.context)
    assert NAME not in c.quant.QUANTIZATION_METHODS


@pytest.mark.parametrize("model_type", ["deepseek_v41", "deepseek_v41_text"])
@pytest.mark.parametrize("method", ["fp8", "deepseek_v4_fp8"])
def test_official_model_config_selects_the_custom_override(contracts, model_type, method):
    c = activate(contracts)
    schema = dict(SCHEMA, quant_method=method)
    assert select(c, schema, hf(model_type)) == NAME


def test_nested_v41_text_config_is_recognized(contracts):
    c = activate(contracts)
    config = SimpleNamespace(model_type="wrapper", text_config=hf(), architectures=[])
    assert c.cls.override_quantization_method(SCHEMA, None, config) == NAME


@pytest.mark.parametrize("model_type", ["llama", "deepseek_v4", "glm5next", "qwen3"])
def test_other_models_keep_the_official_selector_result(contracts, model_type):
    c = contracts
    before = select(c, SCHEMA, hf(model_type))
    activate(c)
    assert select(c, SCHEMA, hf(model_type)) == before
    assert c.cls.override_quantization_method(SCHEMA, None, hf(model_type)) is None


@pytest.mark.parametrize("changes", [
    {"is_per_channel": False}, {"is_per_channel": "true"},
    {"weight_block_size": [128, 128]}, {"quant_method": "quark"},
    {"quant_method": "mxfp8"}, {"activation_scheme": "static"},
])
def test_non_channel_schemes_are_not_captured(contracts, changes):
    c = activate(contracts)
    schema = dict(SCHEMA, **changes)
    assert c.cls.override_quantization_method(schema, None, hf()) is None
    with pytest.raises(ValueError, match="Channel-FP8|channel"):
        c.cls.from_config(schema)


@pytest.mark.parametrize("choice", ["quark", "modelopt", "compressed-tensors"])
def test_incompatible_explicit_choices_are_not_overridden(contracts, choice):
    c = activate(contracts)
    assert c.cls.override_quantization_method(SCHEMA, choice, hf()) is None


@pytest.mark.parametrize("rocm,name", [(False, "cpu"), (False, "cuda"), (True, "rocm")])
def test_non_hcu_platforms_are_not_auto_selected(contracts, monkeypatch, rocm, name):
    c = activate(contracts)
    monkeypatch.setattr(c.candidate, "current_platform", SimpleNamespace(
        device_name=name, is_rocm=lambda: rocm))
    assert c.cls.override_quantization_method(SCHEMA, None, hf()) is None
    with pytest.raises(ValueError, match="HCU|hcu"):
        c.cls.from_config(SCHEMA)


def test_native_configuration_policy_and_input_are_preserved(contracts):
    c = activate(contracts)
    schema = dict(SCHEMA, ignored_layers=["model.layers.0.skip"], store_dtype=None)
    before = copy.deepcopy(schema)
    config = c.cls.from_config(schema)
    assert isinstance(config, c.candidate.DeepseekV4FP8Config)
    assert config.ignored_layers == schema["ignored_layers"]
    assert config.activation_scheme == "dynamic"
    assert config.weight_block_size is None
    assert config.get_name() == NAME
    assert schema == before
    implicit = c.cls.from_config({"quant_method": "fp8", "is_per_channel": True})
    assert implicit.activation_scheme == "dynamic"


def test_native_symbols_and_oracle_are_not_rebound(contracts):
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import CompressedTensorsConfig
    from vllm.model_executor.layers.fused_moe.oracle import int8 as oracle
    c = contracts
    before = (Fp8Config.get_quant_method, c.candidate.DeepseekV4FP8Config.get_quant_method,
              CompressedTensorsConfig.get_quant_method, oracle.Int8MoeBackend,
              oracle.select_int8_moe_backend, oracle.convert_to_int8_moe_kernel_format)
    activate(c)
    after = (Fp8Config.get_quant_method, c.candidate.DeepseekV4FP8Config.get_quant_method,
             CompressedTensorsConfig.get_quant_method, oracle.Int8MoeBackend,
             oracle.select_int8_moe_backend, oracle.convert_to_int8_moe_kernel_format)
    assert all(a is b for a, b in zip(before, after))


@pytest.mark.parametrize("kind", ["linear", "experts"])
def test_matching_layers_delegate_to_official_ct_channel_scheme(contracts, monkeypatch, kind):
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import CompressedTensorsConfig
    c = activate(contracts)
    calls = []
    sentinel = object()

    def native_ct(config, obj, prefix):
        calls.append((config, obj, prefix))
        return sentinel

    monkeypatch.setattr(CompressedTensorsConfig, "get_quant_method", native_ct)
    config = c.cls.from_config(SCHEMA)
    obj = layer(kind)
    assert config.get_quant_method(obj, "model.layers.0.proj") is sentinel
    delegated = calls[0][0]
    assert delegated.quant_format == "float-quantized"
    assert delegated.target_scheme_map["Linear"]["weights"].strategy == "channel"
    assert delegated.target_scheme_map["Linear"]["input_activations"].strategy == "token"
    assert delegated.target_scheme_map["Linear"]["input_activations"].dynamic is True


def test_wo_a_stays_bf16_without_ct_selection(contracts, monkeypatch):
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import CompressedTensorsConfig
    c = activate(contracts)
    monkeypatch.setattr(CompressedTensorsConfig, "get_quant_method",
                        lambda *args: pytest.fail("wo_a entered CT"))
    config = c.cls.from_config(SCHEMA)
    assert isinstance(config.get_quant_method(layer("linear"), "model.layers.0.attn.wo_a"),
                      UnquantizedLinearMethod)


@pytest.mark.parametrize("mode,ignored,prefix", [
    ("exact", ["model.layers.0.skip"], "model.layers.0.skip"),
    ("substring", ["layers.0"], "model.layers.0.proj"),
    ("suffix", ["skip"], "model.layers.0.skip"),
])
def test_native_linear_exclusions_run_before_ct(contracts, monkeypatch, mode, ignored, prefix):
    c = activate(contracts)
    sentinel = object()
    monkeypatch.setattr(c.candidate.DeepseekV4FP8Config, "get_quant_method",
                        lambda self, obj, name: sentinel)
    config = c.cls.from_config(dict(SCHEMA, ignored_layers=ignored))
    config.ignored_layers_match_mode = mode
    assert config.get_quant_method(layer("linear"), prefix) is sentinel


def test_expert_exact_exclusion_is_preserved(contracts, monkeypatch):
    c = activate(contracts)
    sentinel = object()
    monkeypatch.setattr(c.candidate.DeepseekV4FP8Config, "get_quant_method",
                        lambda self, obj, name: sentinel)
    config = c.cls.from_config(dict(SCHEMA, ignored_layers=["model.layers.0.ffn.experts.0.w1"]))
    assert config.get_quant_method(layer("experts"), "model.layers.0.ffn.experts") is sentinel


def test_mixed_precision_fused_shards_still_reject(contracts):
    c = activate(contracts)
    config = c.cls.from_config(dict(SCHEMA, ignored_layers=["model.layers.0.q_proj"]))
    config.packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}
    with pytest.raises(ValueError, match="some but not all shards"):
        config.get_quant_method(layer("linear"), "model.layers.0.qkv_proj")


def test_ct_mapping_follows_native_mapper_updates_and_is_instance_local(contracts, monkeypatch):
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import CompressedTensorsConfig
    c = activate(contracts)
    calls = []
    monkeypatch.setattr(CompressedTensorsConfig, "get_quant_method",
                        lambda config, obj, prefix: calls.append(config))
    first = c.cls.from_config(SCHEMA)
    second = c.cls.from_config(SCHEMA)
    first.get_quant_method(layer("linear"), "model.layers.0.proj")
    first.packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}
    first.get_quant_method(layer("linear"), "model.layers.0.proj")
    second.get_quant_method(layer("linear"), "model.layers.0.proj")
    assert calls[1].packed_modules_mapping == first.packed_modules_mapping
    assert calls[2] is not calls[1]
    assert calls[2].packed_modules_mapping == {}


def test_unrelated_layer_keeps_native_method(contracts, monkeypatch):
    import torch
    c = activate(contracts)
    sentinel = object()
    monkeypatch.setattr(c.candidate.DeepseekV4FP8Config, "get_quant_method",
                        lambda self, obj, name: sentinel)
    assert c.cls.from_config(SCHEMA).get_quant_method(torch.nn.Module(), "embedding") is sentinel


def test_changed_registry_owner_is_observable(contracts):
    from hcu_adapter.extensions import ExtensionCommitError
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config
    c = activate(contracts)
    c.quant.register_quantization_config(NAME)(Fp8Config)
    with pytest.raises(ExtensionCommitError):
        c.activation.check_health()
