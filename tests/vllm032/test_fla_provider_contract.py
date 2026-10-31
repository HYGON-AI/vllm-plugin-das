"""CPU-only alias, ABI, dispatch and real target AOT cache contracts.

Set VLLM_TARGET_ROOT to the fixed target checkout for source contracts. Provider
source discovery uses find paths only, never imports BoltOPs/Triton/Torch.
No tensor storage, kernel, device query or model is used by this file.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, fields, is_dataclass
from collections.abc import Mapping, Sequence, Set
import enum
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import pathlib
import sys
from types import ModuleType, SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "adapter_extension/src/hcu_vllm_das/fla"


@dataclass(frozen=True)
class Device:
    type: str = "cuda"
    index: int = 0


class Tensor:
    def __init__(self, shape, dtype="torch.bfloat16", *, contiguous=True, device=Device(), strides=None):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = device
        self.contiguous = contiguous
        self.strides = strides

    def is_contiguous(self):
        return self.contiguous

    def stride(self, axis):
        if self.strides is not None:
            return self.strides[axis]
        strides, step = [], 1 if self.contiguous else 2
        for dim in reversed(self.shape):
            strides.insert(0, step)
            step *= dim
        return strides[axis]

    def item(self):
        raise AssertionError("dispatch must not inspect tensor contents")

    cpu = numpy = tolist = item


@pytest.fixture
def runtime(monkeypatch):
    # Load only this new package, independent of activation and other migrations.
    name = "_fla_contract_package"
    spec = importlib.util.spec_from_file_location(
        name, PACKAGE / "__init__.py", submodule_search_locations=[str(PACKAGE)],
    )
    package = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, package)
    try:
        spec.loader.exec_module(package)
        p = sys.modules[name + ".provider"]
        monkeypatch.delenv("VLLM_HCU_USE_CUSTOM_OPS", raising=False)
        monkeypatch.delenv("VLLM_HCU_USE_CUSTOM_AITER_FLA", raising=False)
        events = []
        result = (object(), object())

        def official_packed(mixed_qkv, a, b, A_log, dt_bias, scale, initial_state,
                            out, ssm_state_indices, use_qk_l2norm_in_kernel=False):
            events.append(("official", p.PACKED, locals().copy()))
            return result

        def official_sigmoid(A_log, a, b, dt_bias, q, k, v, beta=1.0, threshold=20.0,
                             scale=None, initial_state=None, inplace_final_state=True,
                             cu_seqlens=None, ssm_state_indices=None,
                             num_accepted_tokens=None, use_qk_l2norm_in_kernel=False,
                             is_kda=False):
            events.append(("official", p.SIGMOID, locals().copy()))
            return result

        def vendor_packed(mixed_qkv, a, b, A_log, dt_bias, scale, initial_state,
                          out, ssm_state_indices, use_qk_l2norm_in_kernel=False,
                          null_state_index=-1, kernel_cfg=None):
            events.append(("vendor", p.PACKED, locals().copy()))
            if fail[0]:
                raise RuntimeError("selected provider failed")
            return result

        def vendor_sigmoid(A_log, a, b, dt_bias, q, k, v, beta=1.0, threshold=20.0,
                           scale=None, initial_state=None, inplace_final_state=True,
                           cu_seqlens=None, ssm_state_indices=None,
                           num_accepted_tokens=None, use_qk_l2norm_in_kernel=False,
                           is_kda=False, null_state_index=0, kernel_cfg=None):
            events.append(("vendor", p.SIGMOID, locals().copy()))
            if fail[0]:
                raise RuntimeError("selected provider failed")
            return result

        fail = [False]
        originals = {p.PACKED: official_packed, p.SIGMOID: official_sigmoid}
        vendor = ModuleType("boltops.fla.gdn")
        vendor.__file__ = __file__
        setattr(vendor, p.PACKED, vendor_packed)
        setattr(vendor, p.SIGMOID, vendor_sigmoid)
        consumer = ModuleType(p.CONSUMER)
        modules = {"boltops.fla.gdn": vendor, p.CONSUMER: consumer}
        for key, original in originals.items():
            setattr(consumer, key, original)
            owner = ModuleType(p._OWNERS[key])
            setattr(owner, key, original)
            modules[p._OWNERS[key]] = owner
        imports = []

        def import_module(key):
            imports.append(key)
            return modules[key]

        monkeypatch.setattr(p, "import_module", import_module)
        envs = ModuleType("vllm.envs")
        envs.environment_variables = {}
        envs.compile_factors = lambda: {k: getter() for k, getter in envs.environment_variables.items()}
        envs.VLLM_BATCH_INVARIANT = False
        envs.VLLM_USE_MEGA_AOT_ARTIFACT = False
        envs._is_envs_cache_enabled = lambda: False
        yield SimpleNamespace(
            package=package, p=p, events=events, result=result, vendor=vendor,
            consumer=consumer, originals=originals, envs=envs, fail=fail,
            imports=imports, modules=modules,
        )
    finally:
        sys.modules.pop(name + ".provider", None)


def activate(rt, monkeypatch):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_AITER_FLA", "1")
    rt.p.register_compile_factors(rt.envs)
    assert rt.p.install(architecture="gfx936")


@pytest.mark.parametrize("master,family,invariant,expected", [
    (None, None, False, False), ("1", None, False, False),
    (None, "1", False, False), ("0", "1", False, False),
    ("true", "yes", False, True), ("1", "1", True, False),
])
def test_public_is_requested_is_effective_and_lazy(
    runtime, monkeypatch, master, family, invariant, expected,
):
    runtime.p.register_compile_factors(runtime.envs)
    for name, value in [("VLLM_HCU_USE_CUSTOM_OPS", master),
                        ("VLLM_HCU_USE_CUSTOM_AITER_FLA", family)]:
        if value is not None:
            monkeypatch.setenv(name, value)
    runtime.envs.VLLM_BATCH_INVARIANT = invariant
    assert runtime.package.is_requested() is expected
    assert runtime.imports == []
    assert runtime.p._installation is None


def test_public_is_requested_rejects_invalid_policy_before_runtime_imports(runtime, monkeypatch):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_AITER_FLA", "invalid")
    with pytest.raises(ValueError, match="must be a boolean"):
        runtime.package.is_requested()
    assert runtime.imports == []


def packed_inputs(batch=2, hv=8, heads=4):
    return dict(
        mixed_qkv=Tensor((batch, 2 * heads * 128 + hv * 128)),
        a=Tensor((batch, hv)), b=Tensor((batch, hv)),
        A_log=Tensor((hv,), "torch.float32"),
        dt_bias=Tensor((hv,), "torch.float32"), scale=128**-0.5,
        initial_state=Tensor((32, hv, 128, 128), "torch.float32"),
        out=Tensor((batch, 1, hv, 128)),
        ssm_state_indices=Tensor((batch,), "torch.int32"),
        use_qk_l2norm_in_kernel=True,
    )


def sigmoid_inputs(*, varlen=True, accepted=True, hv=12):
    # Real Qwen-style flattening: q/k [1,total,H,K], gates [total,HV].
    batch, tokens, count = (1, 6, 2) if varlen else (2, 1, 2)
    return dict(
        A_log=Tensor((hv,), "torch.float32"), a=Tensor((batch * tokens, hv)),
        b=Tensor((batch * tokens, hv)), dt_bias=Tensor((hv,), "torch.float32"),
        q=Tensor((batch, tokens, 4, 128)), k=Tensor((batch, tokens, 4, 128)),
        v=Tensor((batch, tokens, hv, 128)), beta=1.5, threshold=18.0,
        scale=0.125, initial_state=Tensor((32, hv, 128, 128), "torch.float32"),
        inplace_final_state=True,
        cu_seqlens=Tensor((count + 1,), "torch.int32") if varlen else None,
        ssm_state_indices=Tensor((count, 4), "torch.int32") if accepted else Tensor((count,), "torch.int64"),
        num_accepted_tokens=Tensor((count,), "torch.int32") if accepted else None,
        use_qk_l2norm_in_kernel=True, is_kda=False,
    )


def assert_forwarded(rt, name, values, route):
    assert getattr(rt.consumer, name)(**values) is rt.result
    kind, actual_name, forwarded = rt.events[-1]
    assert (kind, actual_name) == (route, name)
    for key, value in values.items():
        assert forwarded[key] is value
    if route == "vendor":
        assert forwarded["null_state_index"] == 0
        assert forwarded["kernel_cfg"] is None


@pytest.mark.parametrize("master,family", [(None, None), ("1", None), ("0", "1"), (None, "1")])
def test_default_off_never_imports_or_mounts(runtime, monkeypatch, master, family):
    if master is not None:
        monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", master)
    if family is not None:
        monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_AITER_FLA", family)
    assert runtime.p.install(architecture="gfx936") is False
    assert not runtime.imports
    assert json.loads(runtime.p.effective_compile_factor())["route"] == "official"


def test_requires_compile_factor_before_any_mount(runtime, monkeypatch):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_AITER_FLA", "1")
    with pytest.raises(runtime.p.FlaInstallError, match="compile factors"):
        runtime.p.install(architecture="gfx936")
    assert not runtime.imports


@pytest.mark.parametrize("architecture", [None, "gfx942", "gfx928", "native"])
def test_unsupported_architecture_leaves_official(runtime, monkeypatch, architecture):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_AITER_FLA", "1")
    runtime.p.register_compile_factors(runtime.envs)
    assert not runtime.p.install(architecture=architecture)
    assert not runtime.imports


def test_only_two_qwen_aliases_are_mounted_and_restored(runtime, monkeypatch):
    activate(runtime, monkeypatch)
    for name, original in runtime.originals.items():
        wrapper = getattr(runtime.consumer, name)
        assert wrapper is not original
        assert getattr(runtime.modules[runtime.p._OWNERS[name]], name) is original
        assert inspect.signature(wrapper) == inspect.signature(original)
        assert runtime.p._signature_matches(wrapper, name, vendor=False)
        assert wrapper.__wrapped__ is original
    assert runtime.p.install(architecture="gfx936")
    assert runtime.p.uninstall()
    assert not runtime.p.uninstall()
    for name, original in runtime.originals.items():
        assert getattr(runtime.consumer, name) is original


@pytest.mark.parametrize("case", ["import", "symbol", "signature", "default", "kind"])
def test_provider_unavailable_or_abi_drift_never_partially_mounts(runtime, monkeypatch, case):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_AITER_FLA", "1")
    runtime.p.register_compile_factors(runtime.envs)
    if case == "import":
        monkeypatch.setattr(runtime.p, "import_module", lambda _: (_ for _ in ()).throw(ImportError("optional unavailable")))
    elif case == "symbol":
        delattr(runtime.vendor, runtime.p.SIGMOID)
    elif case == "signature":
        setattr(runtime.vendor, runtime.p.SIGMOID, lambda *args, **kwargs: None)
    elif case == "default":
        function = getattr(runtime.vendor, runtime.p.SIGMOID)
        function.__defaults__ = function.__defaults__[:-2] + (-1, None)
    else:
        original = getattr(runtime.vendor, runtime.p.SIGMOID)
        sig = inspect.signature(original)
        original.__signature__ = sig.replace(parameters=[
            p.replace(kind=inspect.Parameter.KEYWORD_ONLY) for p in sig.parameters.values()
        ])
    assert runtime.p.install(architecture="gfx936") is False
    assert not runtime.p._installation
    for name, original in runtime.originals.items():
        assert getattr(runtime.consumer, name) is original


def test_unexpected_import_failure_is_not_hidden(runtime, monkeypatch):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_AITER_FLA", "1")
    runtime.p.register_compile_factors(runtime.envs)
    monkeypatch.setattr(runtime.p, "import_module", lambda _: (_ for _ in ()).throw(RuntimeError("broken import")))
    with pytest.raises(RuntimeError, match="broken import"):
        runtime.p.install(architecture="gfx936")


@pytest.mark.parametrize("case", ["signature", "alias"])
def test_official_abi_or_foreign_alias_rejected_atomically(runtime, monkeypatch, case):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_AITER_FLA", "1")
    runtime.p.register_compile_factors(runtime.envs)
    if case == "signature":
        runtime.originals[runtime.p.SIGMOID].__defaults__ = (None,) * 10
    else:
        setattr(runtime.consumer, runtime.p.SIGMOID, lambda: None)
    before = vars(runtime.consumer).copy()
    with pytest.raises(runtime.p.FlaInstallError):
        runtime.p.install(architecture="gfx936")
    assert vars(runtime.consumer) == before


@pytest.mark.parametrize("hv", [8, 12])
def test_packed_forwards_every_argument_and_preserves_output_state_objects(runtime, monkeypatch, hv):
    activate(runtime, monkeypatch)
    values = packed_inputs(hv=hv)
    # Fake provider's sentinel pair tests transparent return, not GPU numerics.
    assert_forwarded(runtime, runtime.p.PACKED, values, "vendor")


def test_packed_row_pitched_gate_views_are_supported_without_copy(runtime, monkeypatch):
    activate(runtime, monkeypatch)
    values = packed_inputs()
    for key in ("a", "b"):
        values[key] = Tensor((2, 8), contiguous=False, strides=(16, 1))
    values["mixed_qkv"] = Tensor((2, 2048), contiguous=False, strides=(4096, 1))
    assert_forwarded(runtime, runtime.p.PACKED, values, "vendor")


@pytest.mark.parametrize("varlen,accepted", [(False, False), (True, True)])
def test_sigmoid_preserves_full_metadata_and_nondefault_scalars(runtime, monkeypatch, varlen, accepted):
    activate(runtime, monkeypatch)
    values = sigmoid_inputs(varlen=varlen, accepted=accepted)
    assert_forwarded(runtime, runtime.p.SIGMOID, values, "vendor")


def test_sigmoid_single_token_varlen_and_default_options(runtime, monkeypatch):
    activate(runtime, monkeypatch)
    values = sigmoid_inputs(varlen=False, accepted=False)
    values.update(q=Tensor((1, 2, 4, 128)), k=Tensor((1, 2, 4, 128)),
                  v=Tensor((1, 2, 12, 128)), cu_seqlens=Tensor((3,), "torch.int32"))
    for key in ("beta", "threshold", "scale", "inplace_final_state",
                "num_accepted_tokens", "use_qk_l2norm_in_kernel", "is_kda"):
        values.pop(key)
    assert_forwarded(runtime, runtime.p.SIGMOID, values, "vendor")


@pytest.mark.parametrize("name", ["packed", "sigmoid"])
def test_off_or_batch_invariant_after_install_calls_original(runtime, monkeypatch, name):
    activate(runtime, monkeypatch)
    key = runtime.p.PACKED if name == "packed" else runtime.p.SIGMOID
    values = packed_inputs() if name == "packed" else sigmoid_inputs()
    runtime.envs.VLLM_BATCH_INVARIANT = True
    assert_forwarded(runtime, key, values, "official")
    runtime.envs.VLLM_BATCH_INVARIANT = False
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_AITER_FLA", "0")
    assert_forwarded(runtime, key, values, "official")


@pytest.mark.parametrize("case", [
    "cpu", "other_device", "fp16", "state_dtype", "parameter_dtype", "state_layout",
    "out_shape", "gate_shape", "head_ratio", "indices_dtype", "indices_rank",
    "noncontiguous", "empty", "grid", "scale", "norm", "head_dim", "no_state",
])
def test_packed_unsupported_metadata_calls_exact_original(runtime, monkeypatch, case):
    activate(runtime, monkeypatch)
    a = packed_inputs()
    if case == "cpu":
        for t in a.values():
            if isinstance(t, Tensor):
                t.device = Device("cpu")
    elif case == "other_device": a["out"].device = Device(index=1)
    elif case == "fp16": a["mixed_qkv"].dtype = "torch.float16"
    elif case == "state_dtype": a["initial_state"].dtype = "torch.bfloat16"
    elif case == "parameter_dtype": a["A_log"].dtype = "torch.bfloat16"
    elif case == "state_layout": a["initial_state"].contiguous = False
    elif case == "out_shape": a["out"].shape = (2, 8, 128)
    elif case == "gate_shape": a["a"].shape = (1, 8)
    elif case == "head_ratio": a["mixed_qkv"].shape = (2, 2 * 3 * 128 + 8 * 128)
    elif case == "indices_dtype": a["ssm_state_indices"].dtype = "torch.float32"
    elif case == "indices_rank": a["ssm_state_indices"].shape = (2, 1)
    elif case == "noncontiguous": a["mixed_qkv"].contiguous = False
    elif case == "empty": a = packed_inputs(batch=0)
    elif case == "grid": a = packed_inputs(batch=8192)
    elif case == "scale": a["scale"] = float("nan")
    elif case == "norm": a["use_qk_l2norm_in_kernel"] = 1
    elif case == "head_dim": a["initial_state"].shape = (32, 8, 64, 128)
    elif case == "no_state": a["initial_state"] = None
    assert_forwarded(runtime, runtime.p.PACKED, a, "official")


@pytest.mark.parametrize("case", [
    "kda", "non_inplace", "no_state", "state_dtype", "state_layout", "fp32_q",
    "qk_shape", "value_shape", "gate_shape", "parameter_shape", "missing_indices",
    "accepted_shape", "accepted_dtype", "accepted_without_varlen", "index_layout",
    "cu_dtype", "cu_layout", "cu_rank", "varlen_batch", "index_count",
    "one_index_spec", "one_index_multitoken", "zero_index_width", "grid",
    "scale", "beta", "threshold", "other_device",
])
def test_sigmoid_unsupported_contract_calls_exact_original(runtime, monkeypatch, case):
    activate(runtime, monkeypatch)
    a = sigmoid_inputs()
    if case == "kda": a["is_kda"] = True
    elif case == "non_inplace": a["inplace_final_state"] = False
    elif case == "no_state": a["initial_state"] = None
    elif case == "state_dtype": a["initial_state"].dtype = "torch.bfloat16"
    elif case == "state_layout": a["initial_state"].contiguous = False
    elif case == "fp32_q": a["q"].dtype = "torch.float32"
    elif case == "qk_shape": a["k"].shape = (1, 6, 3, 128)
    elif case == "value_shape": a["v"].shape = (1, 6, 12, 64)
    elif case == "gate_shape": a["b"].shape = (6, 4)
    elif case == "parameter_shape": a["dt_bias"].shape = (4,)
    elif case == "missing_indices": a["ssm_state_indices"] = None
    elif case == "accepted_shape": a["num_accepted_tokens"].shape = (1,)
    elif case == "accepted_dtype": a["num_accepted_tokens"].dtype = "torch.float32"
    elif case == "accepted_without_varlen": a["cu_seqlens"] = None
    elif case == "index_layout": a["ssm_state_indices"].contiguous = False
    elif case == "cu_dtype": a["cu_seqlens"].dtype = "torch.float32"
    elif case == "cu_layout": a["cu_seqlens"].contiguous = False
    elif case == "cu_rank": a["cu_seqlens"].shape = (1, 3)
    elif case == "varlen_batch":
        for key in ("q", "k"): a[key].shape = (2, 3, 4, 128)
        a["v"].shape = (2, 3, 12, 128)
    elif case == "index_count": a["ssm_state_indices"].shape = (1, 4)
    elif case == "one_index_spec": a["ssm_state_indices"].shape = (2,)
    elif case == "one_index_multitoken":
        a["ssm_state_indices"].shape = (2,)
        a["num_accepted_tokens"] = None
    elif case == "zero_index_width": a["ssm_state_indices"].shape = (2, 0)
    elif case == "grid":
        a["cu_seqlens"].shape = (6001,)
        a["ssm_state_indices"].shape = (6000, 4)
        a["num_accepted_tokens"].shape = (6000,)
    elif case == "scale": a["scale"] = 0
    elif case == "beta": a["beta"] = -1
    elif case == "threshold": a["threshold"] = float("inf")
    elif case == "other_device": a["num_accepted_tokens"].device = Device(index=1)
    assert_forwarded(runtime, runtime.p.SIGMOID, a, "official")


@pytest.mark.parametrize("name", ["packed", "sigmoid"])
def test_selected_provider_error_propagates_without_fallback(runtime, monkeypatch, name):
    activate(runtime, monkeypatch)
    runtime.fail[0] = True
    key = runtime.p.PACKED if name == "packed" else runtime.p.SIGMOID
    with pytest.raises(RuntimeError, match="selected provider failed"):
        getattr(runtime.consumer, key)(**(packed_inputs() if name == "packed" else sigmoid_inputs()))
    assert len(runtime.events) == 1 and runtime.events[0][0] == "vendor"


def test_uninstall_does_not_overwrite_foreign_owner_or_partially_restore(runtime, monkeypatch):
    activate(runtime, monkeypatch)
    packed = getattr(runtime.consumer, runtime.p.PACKED)
    setattr(runtime.consumer, runtime.p.SIGMOID, lambda: None)
    with pytest.raises(runtime.p.FlaInstallError, match="partial restore"):
        runtime.p.uninstall()
    assert getattr(runtime.consumer, runtime.p.PACKED) is packed


def _target_root():
    explicit = os.environ.get("VLLM_TARGET_ROOT")
    if explicit is None:
        pytest.skip("set VLLM_TARGET_ROOT for real target source/ABI/AOT contracts")
    root = Path(explicit)
    assert (root / "vllm/envs.py").is_file()
    return root


def _functions(path, names, namespace):
    nodes = [n for n in ast.parse(path.read_text()).body
             if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    code = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
    exec(compile(code, str(path), "exec"), namespace)


def test_real_target_and_installed_provider_signature_contracts(runtime):
    root = _target_root()
    # Locate files without find_spec(), which could execute parent initializers.
    bases = [Path(p) for p in sys.path if p]
    provider_root = next((p / "boltops/fla/gdn" for p in bases if (p / "boltops/fla/gdn/__init__.py").is_file()), None)
    assert provider_root is not None, "real BoltOPs package must be present for this gate"
    for name, filename in [(runtime.p.PACKED, "fused_recurrent.py"),
                           (runtime.p.SIGMOID, "fused_sigmoid_gating.py")]:
        for vendor, path in [
            (False, root / "vllm/third_party/flash_linear_attention/ops" / filename),
            (True, provider_root / "triton" / filename),
        ]:
            namespace = {}
            _functions(path, (name,), namespace)
            assert runtime.p._signature_matches(namespace[name], name, vendor=vendor)
    public = ast.parse((provider_root / "__init__.py").read_text())
    mapping = next(ast.literal_eval(n.value) for n in public.body
                   if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)
                   and n.target.id == "_LAZY_EXPORTS")
    for name in (runtime.p.PACKED, runtime.p.SIGMOID):
        assert mapping[name] == (".triton", name)
    triton_init = ast.parse((provider_root / "triton/__init__.py").read_text())
    assert all(any(isinstance(n, ast.ImportFrom) and n.module == filename
                   and any(a.name == name and a.asname is None for a in n.names)
                   for n in triton_init.body)
               for name, filename in [(runtime.p.PACKED, "fused_recurrent"),
                                      (runtime.p.SIGMOID, "fused_sigmoid_gating")])


@pytest.fixture
def actual_cache(runtime, monkeypatch):
    root = _target_root()
    utils = dict(
        hashlib=hashlib, json=json, enum=enum, pathlib=pathlib,
        Mapping=Mapping, Sequence=Sequence, Set=Set,
        is_dataclass=is_dataclass, fields=fields,
        torch=SimpleNamespace(dtype=type("UnusedDtype", (), {})),
    )
    _functions(root / "vllm/config/utils.py", ("normalize_value", "hash_factors"), utils)
    for name in ("vllm", "vllm.config", "vllm.config.utils"):
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    sys.modules["vllm.config.utils"].normalize_value = utils["normalize_value"]
    envs = runtime.envs
    namespace = dict(
        environment_variables=envs.environment_variables,
        os=SimpleNamespace(getenv=lambda _: None),
        logger=SimpleNamespace(warning=lambda *args: pytest.fail("compile factor was skipped")),
    )
    _functions(root / "vllm/envs.py", ("compile_factors",), namespace)
    envs.compile_factors = namespace["compile_factors"]
    aot = dict(envs=envs, hash_factors=utils["hash_factors"])
    _functions(root / "vllm/compilation/caching.py", ("aot_compile_hash_factors",), aot)
    config = SimpleNamespace(compute_hash=lambda: "fixed-config-hash")

    def keys():
        factors = envs.compile_factors()
        assert runtime.p.COMPILE_FACTOR in factors
        return utils["hash_factors"](factors), aot["aot_compile_hash_factors"](config)

    runtime.p.register_compile_factors(envs)
    return keys


def test_effective_route_changes_real_compile_and_aot_factors(runtime, monkeypatch, actual_cache):
    off = actual_cache()
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_AITER_FLA", "1")
    unavailable = actual_cache()
    assert runtime.p.install(architecture="gfx936")
    on = actual_cache()
    assert off[0] != unavailable[0] != on[0]
    assert off[1][0] != unavailable[1][0] != on[1][0]
    assert on[1][1] == off[1][1] == "fixed-config-hash"
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "true")
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_AITER_FLA", "yes")
    assert actual_cache() == on
    runtime.envs.VLLM_BATCH_INVARIANT = True
    assert actual_cache() != on


def test_provider_config_identity_changes_both_cache_keys(runtime, monkeypatch, actual_cache):
    activate(runtime, monkeypatch)
    before = actual_cache()
    runtime.p._installation.identity["provider"][runtime.p.PACKED]["config"] = "changed-config-bytes"
    after = actual_cache()
    assert before[0] != after[0] and before[1][0] != after[1][0]


def test_foreign_alias_changes_factor_and_blocks_reinstall(runtime, monkeypatch, actual_cache):
    activate(runtime, monkeypatch)
    before = actual_cache()
    setattr(runtime.consumer, runtime.p.PACKED, lambda: None)
    assert actual_cache() != before
    assert json.loads(runtime.p.effective_compile_factor())["route"] == "foreign-alias"
    with pytest.raises(runtime.p.FlaInstallError, match="ownership"):
        runtime.p.install(architecture="gfx936")


def test_registration_ownership_invalid_settings_and_late_registration(runtime, monkeypatch):
    runtime.p.register_compile_factors(runtime.envs)
    runtime.p.register_compile_factors(runtime.envs)
    runtime.envs.environment_variables[runtime.p.COMPILE_FACTOR] = lambda: "foreign"
    with pytest.raises(runtime.p.FlaInstallError, match="another owner"):
        runtime.p.register_compile_factors(runtime.envs)
    runtime.envs.environment_variables.clear()
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_AITER_FLA", "invalid")
    assert "invalid" in json.loads(runtime.p.effective_compile_factor())
    with pytest.raises(ValueError):
        runtime.p.register_compile_factors(runtime.envs)
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_AITER_FLA", "0")
    runtime.envs._is_envs_cache_enabled = lambda: True
    with pytest.raises(runtime.p.FlaInstallError, match="caches env"):
        runtime.p.register_compile_factors(runtime.envs)
