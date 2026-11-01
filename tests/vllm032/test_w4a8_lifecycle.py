"""Real Adapter/official Worker boundaries; CPU model and transfer providers only."""

# Explicit source-phase import; never insert a vendor or vLLM source path.
# ruff: noqa: E402
from __future__ import annotations

import gc
import ast
import base64
import csv
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
import signal
import subprocess
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from types import SimpleNamespace as NS
from weakref import WeakKeyDictionary, ref

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
WHEEL_ROOT = os.environ.get("QNT009_WHEEL_ROOT")
PACKAGE_ROOT = (
    Path(WHEEL_ROOT).resolve() if WHEEL_ROOT else ROOT / "adapter_extension/src"
)
sys.path.insert(0, str(PACKAGE_ROOT))
from hcu_adapter import model_load as adapter
from hcu_adapter.bridges import vllm_model_load as adapter_host
from hcu_adapter.bridges.vllm_worker import Worker
from hcu_vllm_das.w4a8 import config as api
from hcu_vllm_das.w4a8 import lifecycle as owner
from vllm import envs
from vllm.compilation.caching import aot_compile_hash_factors
from vllm.config import DeviceConfig, ParallelConfig, VllmConfig
from vllm.config.utils import hash_factors
from vllm.model_executor.layers import quantization as registry
from vllm.v1.worker import gpu_model_runner as v1, gpu_worker
from vllm.v1.worker.gpu import model_runner as v2
from vllm.v1.worker.gpu.spec_decode.mtp.speculator import MTPSpeculator

if WHEEL_ROOT:
    # Explicit isolated wheel target; no Plugin source path is inserted.
    assert Path(owner.__file__).resolve().is_relative_to(PACKAGE_ROOT)
    dist = importlib.metadata.distribution("hcu-vllm-das-plugin")
    records = {
        row[0]: row[1:] for row in csv.reader(dist.read_text("RECORD").splitlines())
    }
    for name in ("__init__.py", "config.py", "runtime.py", "lifecycle.py"):
        path = Path("hcu_vllm_das/w4a8") / name
        digest, size = records[str(path)]
        content = (PACKAGE_ROOT / path).read_bytes()
        assert (
            digest
            == "sha256="
            + base64.urlsafe_b64encode(hashlib.sha256(content).digest())
            .rstrip(b"=")
            .decode()
        )
        assert len(content) == int(size)


@dataclass
class Metadata:
    quantization: str = "compressed-tensors"
    model: str = "CPU-model"

    def compute_hash(self):
        return self.quantization


class Selected:
    native = None
    _processed = True


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    assert not torch.cuda.is_initialized()
    envs.disable_envs_cache()
    for name, value in dict(
        _installed=False,
        _commit_failure=None,
        _installing_error_type=None,
        _pid=os.getpid(),
        _fork_error=None,
        _envs=None,
        _selected={},
        _alias_configs=[],
        _workers=WeakKeyDictionary(),
        _frozen=None,
        _frozen_rows=None,
        _setting=None,
        _file_hashes={},
    ).items():
        monkeypatch.setattr(owner, name, value)
    monkeypatch.delenv("VLLM_HCU_DAS_W4A8_NATIVE", raising=False)
    monkeypatch.delenv("VLLM_ELASTIC_EP_SCALE_UP_LAUNCH", raising=False)
    monkeypatch.setattr(
        envs, "environment_variables", envs.environment_variables.copy()
    )
    # Reset the tested Adapter module as a unit; owner code uses only
    # public registration/dispatch APIs, never private registry snapshots.
    importlib.reload(adapter)
    importlib.reload(adapter_host)
    monkeypatch.setattr(
        registry, "QUANTIZATION_METHODS", registry.QUANTIZATION_METHODS.copy()
    )
    monkeypatch.setattr(registry, "_CUSTOMIZED_METHOD_TO_QUANT_CONFIG", {})
    from vllm.platforms import current_platform

    monkeypatch.setattr(
        current_platform,
        "supported_quantization",
        current_platform.supported_quantization.copy(),
    )

    def forbidden(*args, **kwargs):
        pytest.fail("CPU lifecycle queried/initialized GPU")

    for name in ("get_device_properties", "current_device", "init", "_lazy_init"):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    monkeypatch.setattr(
        gpu_worker.Worker,
        "_maybe_get_memory_pool_context",
        lambda *a, **kw: nullcontext(),
    )
    monkeypatch.setattr(
        gpu_worker.Worker, "_scoped_allocator_max_split", lambda *a, **kw: nullcontext()
    )
    monkeypatch.setattr(gpu_worker, "has_ec_transfer", lambda: False)
    monkeypatch.setattr(gpu_worker, "set_torch_threads_for_runtime", lambda: None)
    yield
    envs.disable_envs_cache()
    assert not torch.cuda.is_initialized()


def config(role="target"):
    cfg = VllmConfig(
        device_config=DeviceConfig("cpu"),
        parallel_config=ParallelConfig(
            worker_cls=adapter_host.POST_MODEL_LOAD_WORKER,
            distributed_executor_backend="uni",
        ),
    )
    cfg.model_config = Metadata(
        "slimquant_w4a8" if role == "target" else "compressed-tensors"
    )
    cfg.speculative_config = NS(
        draft_model_config=Metadata(
            "slimquant_w4a8" if role == "draft" else "compressed-tensors"
        ),
        compute_hash=lambda: "CPU-draft",
    )
    return cfg


def loaded_worker(monkeypatch, version, role, events, *, selection=False):
    cfg = config(role)
    target, draft = torch.nn.Module(), torch.nn.Module()
    module = v1 if version == 1 else v2
    runner = object.__new__(module.GPUModelRunner)
    runner.vllm_config, runner.model_config, runner.model = (
        cfg,
        cfg.model_config,
        target,
    )
    if version == 1:
        runner.drafter = NS(model=draft)
    else:
        runner.speculator = object.__new__(MTPSpeculator)
        runner.speculator.model = draft
    method = Selected()

    def load_model(**kwargs):
        if selection:
            owner.record("cpu.experts", method)

    runner.load_model = load_model
    worker = object.__new__(Worker)
    worker.vllm_config, worker.model_runner = cfg, runner
    worker.speculative_config = cfg.speculative_config
    worker.load_model()  # Real Adapter -> official Worker -> postload/freeze/ownership.

    def reload(*args, **kwargs):
        events.append("reload-provider")

    monkeypatch.setattr(module, "reload_weights", reload)
    runner.reset_lora_state = lambda: events.append("reset-lora")
    runner.reset_encoder_cache = lambda: events.append("reset-encoder")
    runner.reset_mm_cache = lambda: events.append("reset-mm")
    worker.weight_transfer_engine = NS(
        supports_draft_weight_update=True,
        set_weight_update_target=lambda *a: events.append("draft-retarget"),
        start_weight_update=lambda: events.append("engine-start"),
        reset_weight_update_target=lambda: events.append("engine-reset"),
    )
    worker._weight_update_active, worker._weight_update_is_draft = False, None
    return worker


def invoke(worker, operation):
    if operation == "reload":
        return worker.reload_weights(weights_path="CPU-new-weights")
    if operation == "target":
        return worker.start_weight_update()
    return worker.start_draft_weight_update()


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("role", ["none", "target", "draft"])
@pytest.mark.parametrize("operation", ["reload", "target", "draft"])
@pytest.mark.parametrize("override", [False, True])
def test_real_worker_ownership_blocks_before_official_mutation(
    monkeypatch, version, role, operation, override
):
    owner.install()
    events = []
    worker = loaded_worker(monkeypatch, version, role, events)
    assert (worker in owner._workers) == (role != "none")
    if override:
        worker.update_config(
            {
                "model_config": {
                    "quantization": "slimquant_w4a8"
                    if role == "none"
                    else "compressed-tensors"
                }
            }
        )
        worker.vllm_config.speculative_config.draft_model_config.quantization = (
            "compressed-tensors"
        )
        worker.model_runner.model = torch.nn.Module()
    config_before = worker.vllm_config.model_config
    envs.enable_envs_cache()
    if role == "none":
        assert invoke(worker, operation) is None
        assert events == (
            ["reload-provider", "reset-lora", "reset-encoder", "reset-mm"]
            if operation == "reload"
            else ["draft-retarget", "engine-start"]
            if operation == "draft"
            else ["engine-start"]
        )
    else:
        assert (
            owner._frozen is not None and owner._selected == {}
        )  # Dense/ignored alias still owned.
        with pytest.raises(
            adapter.ModelMutationError, match="W4A8 owns frozen models"
        ) as error:
            invoke(worker, operation)
        assert isinstance(error.value.__cause__, owner.W4LifecycleError)
        assert events == [] and not worker._weight_update_active
        assert worker._weight_update_is_draft is None
        assert worker.vllm_config.model_config is config_before


def test_other_ct_worker_loads_and_delegates_after_w4_freeze(monkeypatch):
    owner.install()
    protected = loaded_worker(monkeypatch, 1, "draft", [], selection=True)
    events = []
    default = loaded_worker(monkeypatch, 2, "none", events)
    envs.enable_envs_cache()
    default.reload_weights()
    assert events == ["reload-provider", "reset-lora", "reset-encoder", "reset-mm"]
    assert default not in owner._workers and protected in owner._workers


def test_ownership_and_selected_methods_do_not_retain_models_or_worker(monkeypatch):
    owner.install()
    worker = loaded_worker(monkeypatch, 2, "target", [], selection=True)
    weak_worker, weak_target, weak_draft = (
        ref(worker),
        ref(worker.get_model()),
        ref(worker.get_draft_model()),
    )
    assert all(isinstance(handle, ref) for handle in owner._workers[worker])
    method_handle = owner._selected["cpu.experts"]
    del worker
    gc.collect()
    assert weak_worker() is weak_target() is weak_draft() is method_handle() is None
    assert not owner._workers
    assert json.loads(owner.factor())["phase"] == "frozen"


def test_alias_selection_remembered_before_config_override(monkeypatch):
    owner.install()
    cfg = config()
    owner.require_selection(cfg)
    cfg.model_config.quantization = "compressed-tensors"
    target = torch.nn.Module()

    class WorkerDouble:
        vllm_config = cfg
        model_runner = NS(get_model=lambda: target, get_draft_model=lambda: None)

    worker = WorkerDouble()
    adapter_host.dispatch_post_model_load(worker, load_dummy_weights=False)
    assert worker in owner._workers and owner._frozen is not None


def test_loaded_model_alias_visible_when_config_replaced(monkeypatch):
    owner.install()
    cfg = config("none")
    model = torch.nn.Module()
    model.quant_config = api.SlimQuantW4A8Config({}, [], "pack-int4-nibble")

    class WorkerDouble:
        vllm_config = cfg
        model_runner = NS(get_model=lambda: model, get_draft_model=lambda: None)

    worker = WorkerDouble()
    adapter_host.dispatch_post_model_load(worker, load_dummy_weights=False)
    assert worker in owner._workers


CALLABLES = (
    "register_post_model_load",
    "require_post_model_load",
    "register_model_mutation_guard",
    "require_model_mutation_guard",
    "preflight_model_lifecycle_hooks",
    "register_model_lifecycle_hooks",
)


@pytest.mark.parametrize("name", CALLABLES)
@pytest.mark.parametrize(
    "failure",
    [
        "missing",
        "signature",
        "async",
        "async-partial",
        "async-callable",
        "generator",
        "async-generator",
    ],
)
def test_missing_abi_rejects_before_any_registry_or_provider_mutation(
    monkeypatch, name, failure
):
    module = NS(**{key: getattr(adapter_host, key) for key in CALLABLES})
    if failure == "missing":
        delattr(module, name)
    elif failure == "signature":
        setattr(module, name, lambda first, second, newly_required: None)
    elif failure.startswith("async"):

        async def incompatible(*args, **kwargs):
            pass

        class AsyncCallable:
            async def __call__(self, *args, **kwargs):
                pass

        async def asynchronous_generator(*args, **kwargs):
            yield None

        setattr(
            module,
            name,
            {
                "async": incompatible,
                "async-partial": partial(incompatible),
                "async-callable": AsyncCallable(),
                "async-generator": asynchronous_generator,
            }[failure],
        )
    else:

        def generator(*args, **kwargs):
            yield None

        setattr(module, name, generator)
    before = (
        dict(envs.environment_variables),
        registry.QUANTIZATION_METHODS.copy(),
        dict(registry._CUSTOMIZED_METHOD_TO_QUANT_CONFIG),
    )
    imports = []
    real_import = owner.import_module

    def importer(subject):
        if subject in ("hcu_adapter.extensions", "hcu_adapter.bridges.vllm"):
            return real_import(subject)
        imports.append(subject)
        assert subject == "hcu_adapter.bridges.vllm_model_load"
        return module

    monkeypatch.setattr(owner, "import_module", importer)
    with pytest.raises(
        owner.W4LifecycleError,
        match="Optional W4A8 requires Adapter postload.*mutation ABI",
    ):
        owner.install()
    assert imports == ["hcu_adapter.bridges.vllm_model_load"]
    assert before == (
        envs.environment_variables,
        registry.QUANTIZATION_METHODS,
        registry._CUSTOMIZED_METHOD_TO_QUANT_CONFIG,
    )
    assert not adapter._callbacks.callbacks and not adapter._mutation_guards.callbacks
    assert not owner._installed and owner._setting is None and owner._envs is None


def test_compatible_optional_abi_additions_keep_real_boundaries(monkeypatch):
    post_register, post_require = (
        adapter.register_post_model_load,
        adapter_host.require_post_model_load,
    )
    mutation_register, mutation_require = (
        adapter.register_model_mutation_guard,
        adapter_host.require_model_mutation_guard,
    )
    lifecycle_preflight = adapter.preflight_model_lifecycle_hooks
    lifecycle_register = adapter.register_model_lifecycle_hooks

    def register_post_model_load(owner, callback, *, future_option=False):
        return post_register(owner, callback)

    def require_post_model_load(vllm_config, *, future_option=None):
        return post_require(vllm_config)

    def register_model_mutation_guard(owner, callback, *, future_option=False):
        return mutation_register(owner, callback)

    def require_model_mutation_guard(vllm_config, *, future_option=None):
        return mutation_require(vllm_config)

    def preflight_model_lifecycle_hooks(
        *,
        post_model_load_owner,
        post_model_load_callback,
        model_mutation_owner,
        model_mutation_guard,
        future_option=None,
    ):
        return lifecycle_preflight(
            post_model_load_owner=post_model_load_owner,
            post_model_load_callback=post_model_load_callback,
            model_mutation_owner=model_mutation_owner,
            model_mutation_guard=model_mutation_guard,
        )

    def register_model_lifecycle_hooks(
        *,
        post_model_load_owner,
        post_model_load_callback,
        model_mutation_owner,
        model_mutation_guard,
        future_option=None,
    ):
        return lifecycle_register(
            post_model_load_owner=post_model_load_owner,
            post_model_load_callback=post_model_load_callback,
            model_mutation_owner=model_mutation_owner,
            model_mutation_guard=model_mutation_guard,
        )

    for function in (
        register_post_model_load,
        require_post_model_load,
        register_model_mutation_guard,
        require_model_mutation_guard,
        preflight_model_lifecycle_hooks,
        register_model_lifecycle_hooks,
    ):
        monkeypatch.setattr(adapter_host, function.__name__, function)
    assert owner.install()
    worker = loaded_worker(monkeypatch, 2, "target", [])
    with pytest.raises(adapter.ModelMutationError, match="W4A8 owns frozen models"):
        worker.reload_weights()


@pytest.mark.parametrize("kind", ["postload", "mutation", "env", "quant"])
def test_existing_owner_conflict_cannot_half_register(monkeypatch, kind):
    if kind == "postload":
        adapter.register_post_model_load(owner.OWNER, lambda context: None)
    elif kind == "mutation":
        adapter.register_model_mutation_guard(
            owner.MUTATION_OWNER, lambda context: None
        )
    elif kind == "env":
        envs.environment_variables[owner.FACTOR] = lambda: "other"
    else:
        registry._CUSTOMIZED_METHOD_TO_QUANT_CONFIG["slimquant_w4a8"] = object()
    before = (
        dict(adapter._callbacks.callbacks),
        dict(adapter._mutation_guards.callbacks),
        dict(envs.environment_variables),
        dict(registry._CUSTOMIZED_METHOD_TO_QUANT_CONFIG),
    )
    with pytest.raises(owner.W4LifecycleError):
        owner.install()
    assert before == (
        adapter._callbacks.callbacks,
        adapter._mutation_guards.callbacks,
        envs.environment_variables,
        registry._CUSTOMIZED_METHOD_TO_QUANT_CONFIG,
    )
    assert not owner._installed


@pytest.mark.parametrize("boundary", ["postload", "mutation"])
def test_require_boundaries_fail_before_any_native_selection(monkeypatch, boundary):
    owner.install()
    cfg = config()
    calls = []
    if boundary == "postload":
        cfg.parallel_config.worker_cls = "foreign.Worker"
    else:

        def unavailable(vllm_config):
            raise adapter.ModelMutationError("unsupported mutation seam")

        monkeypatch.setattr(adapter_host, "require_model_mutation_guard", unavailable)
    from vllm.config import set_current_vllm_config

    layer = api.RoutedExperts.__new__(api.RoutedExperts)
    torch.nn.Module.__init__(layer)
    monkeypatch.setattr(
        api.runtime, "prepare_native", lambda *a: calls.append("provider")
    )
    q = api.SlimQuantW4A8Config({}, [], "pack-int4-nibble")
    with (
        set_current_vllm_config(cfg),
        pytest.raises((adapter.PostModelLoadError, adapter.ModelMutationError)),
    ):
        q.get_quant_method(layer, "cpu.experts")
    assert not calls and not owner._selected


def keys(cfg):
    return hash_factors(envs.compile_factors()), aot_compile_hash_factors(cfg)


@pytest.mark.parametrize(
    "relative",
    [
        "plugin/config.py",
        "plugin/runtime.py",
        "plugin/lifecycle.py",
        "plugin/__init__.py",
        "vendor/moe.py",
        "vendor/jit/module_moe_c_w4a8_kernel.so",
        "vendor/moe_c_configs/tuned_moe_c_w4a8_channel.csv",
    ],
)
@pytest.mark.parametrize("frozen", [False, True])
def test_real_normal_and_aot_getters_detect_same_shape_byte_drift(
    monkeypatch, tmp_path, relative, frozen
):
    # Controlled files isolate byte identity from model geometry/configs. No
    # provider is imported/executed and no production/vendor files are edited.
    plugin, vendor = tmp_path / "plugin", tmp_path / "vendor"
    for name in ("__init__.py", "config.py", "runtime.py", "lifecycle.py"):
        path = plugin / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("implementation-A")
    for name in owner._VENDOR_FILES:
        path = vendor / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("provider-A")
    monkeypatch.setattr(owner, "_plugin_root", lambda: plugin)
    monkeypatch.setattr(owner, "_vendor_root", lambda: vendor)
    monkeypatch.setenv("VLLM_HCU_DAS_W4A8_NATIVE", "1")
    owner.install()
    cfg = (
        loaded_worker(monkeypatch, 2, "target", []).vllm_config if frozen else config()
    )
    before = keys(cfg)
    envs.enable_envs_cache()
    assert keys(cfg) == before
    changed = tmp_path / relative
    stamp = changed.stat()
    changed.write_text(changed.read_text().replace("-A", "-B"))
    os.utime(changed, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    after = keys(cfg)
    assert after[0] != before[0] and after[1] != before[1]
    assert ("invalid" in json.loads(owner.factor())) is frozen


def test_actual_public_vendor_and_all_four_production_hashes_without_import(
    monkeypatch,
):
    monkeypatch.setenv("VLLM_HCU_DAS_W4A8_NATIVE", "1")
    before = {
        name for name in sys.modules if name == "aiter" or name.startswith("aiter.")
    }
    identity = owner.implementation_identity()
    assert set(identity["plugin"]) == {
        "__init__.py",
        "config.py",
        "runtime.py",
        "lifecycle.py",
    }
    for name, digest in identity["plugin"].items():
        assert (
            digest
            == hashlib.sha256((owner._plugin_root() / name).read_bytes()).hexdigest()
        )
    for name in (
        "moe.py",
        "fused_moe_c.py",
        "ops/shuffle.py",
        "jit/module_moe_c_w4a8_kernel.so",
        "moe_c_configs/tuned_moe_c_w4a8_channel.csv",
    ):
        assert (
            identity["aiter"][name]
            == hashlib.sha256((owner._vendor_root() / name).read_bytes()).hexdigest()
        )
    assert before == {
        name for name in sys.modules if name == "aiter" or name.startswith("aiter.")
    }


def run_fork(callback):
    ctx = multiprocessing.get_context("fork")
    reader, writer = ctx.Pipe(duplex=False)

    def child():
        try:
            writer.send((True, callback()))
        except BaseException as error:
            writer.send((False, f"{type(error).__name__}: {error}"))
        finally:
            writer.close()

    process = ctx.Process(target=child)
    process.start()
    writer.close()
    try:
        assert reader.poll(25), "fork boundary hung"
        ok, result = reader.recv()
        process.join(5)
        assert process.exitcode == 0 and ok, result
        return result
    finally:
        if process.is_alive():
            process.terminate()
        process.join(5)
        reader.close()


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX fork")
def test_fork_poison_protects_w4_but_unowned_ct_uses_real_worker(monkeypatch):
    owner.install()
    protected_events, ct_events = [], []
    protected = loaded_worker(monkeypatch, 1, "draft", protected_events)
    default = loaded_worker(monkeypatch, 2, "none", ct_events)

    def child():
        for operation in ("reload", "target", "draft"):
            with pytest.raises(adapter.ModelMutationError, match="across fork"):
                invoke(protected, operation)
        assert not protected_events and not protected._weight_update_active
        assert "invalid" in json.loads(owner.factor())
        # Even changing default worker metadata to the alias cannot invent
        # loaded ownership, including in an inherited poisoned W4 process.
        default.update_config({"model_config": {"quantization": "slimquant_w4a8"}})
        for operation in ("reload", "target", "draft"):
            default._weight_update_active = False
            invoke(default, operation)
        return ct_events

    with owner._lock, adapter._callbacks.lock, adapter._mutation_guards.lock:
        assert run_fork(child) == [
            "reload-provider",
            "reset-lora",
            "reset-encoder",
            "reset-mm",
            "engine-start",
            "draft-retarget",
            "engine-start",
        ]
    assert not protected_events and not ct_events


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX fork")
def test_clean_fork_registration_can_load_and_protect_in_child(monkeypatch):
    owner.install()

    def child():
        assert owner._installed and owner._frozen is None
        worker = loaded_worker(monkeypatch, 2, "target", [], selection=True)
        assert owner._fork_error is None and worker in owner._workers
        with pytest.raises(adapter.ModelMutationError, match="W4A8 owns frozen models"):
            worker.start_weight_update()
        return json.loads(owner.factor())["phase"]

    with owner._lock, adapter._callbacks.lock, adapter._mutation_guards.lock:
        assert run_fork(child) == "frozen"
    assert owner._frozen is None and not owner._selected and not owner._workers


def test_owner_has_no_adapter_private_registry_access():
    text = Path(owner.__file__).read_text()
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            assert not (node.value.id == "adapter" and node.attr.startswith("_"))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if (
                isinstance(node.func.value, ast.Name)
                and node.func.value.id == "adapter"
            ):
                assert node.func.attr not in {
                    "register_post_model_load",
                    "register_model_mutation_guard",
                }
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            assert node.value not in {
                "_callbacks",
                "_mutation_guards",
                "_lifecycle_hooks",
            }


def test_atomic_registration_failure_does_not_publish_w4_factor_or_alias(monkeypatch):
    before = (
        dict(envs.environment_variables),
        list(registry.QUANTIZATION_METHODS),
        dict(registry._CUSTOMIZED_METHOD_TO_QUANT_CONFIG),
    )

    def fail(
        *,
        post_model_load_owner,
        post_model_load_callback,
        model_mutation_owner,
        model_mutation_guard,
    ):
        raise adapter.ModelMutationError("second hook unavailable")

    monkeypatch.setattr(adapter_host, "register_model_lifecycle_hooks", fail)
    with pytest.raises(adapter.ModelMutationError, match="second hook unavailable"):
        owner.install()
    assert before == (
        envs.environment_variables,
        registry.QUANTIZATION_METHODS,
        registry._CUSTOMIZED_METHOD_TO_QUANT_CONFIG,
    )
    assert not owner._installed and owner._envs is None and owner._setting is None


def test_owner_calls_only_public_pair_api_in_order(monkeypatch):
    calls = []
    real_import = owner.import_module
    preflight = adapter.preflight_model_lifecycle_hooks
    register = adapter.register_model_lifecycle_hooks

    def importer(subject):
        calls.append(subject)
        return real_import(subject)

    def check(**kwargs):
        calls.append("public-preflight")
        return preflight(**kwargs)

    def commit(**kwargs):
        calls.append("public-atomic-commit")
        assert owner.FACTOR not in envs.environment_variables
        assert "slimquant_w4a8" not in registry._CUSTOMIZED_METHOD_TO_QUANT_CONFIG
        return register(**kwargs)

    monkeypatch.setattr(owner, "import_module", importer)
    monkeypatch.setattr(adapter_host, "preflight_model_lifecycle_hooks", check)
    monkeypatch.setattr(adapter_host, "register_model_lifecycle_hooks", commit)
    assert owner.install()
    assert calls == [
        "hcu_adapter.extensions",
        "hcu_adapter.bridges.vllm",
        "hcu_adapter.extensions",
        "hcu_adapter.bridges.vllm",
        "hcu_adapter.bridges.vllm_model_load",
        "public-preflight",
        "public-atomic-commit",
    ]
    worker = loaded_worker(monkeypatch, 2, "target", [])
    with pytest.raises(adapter.ModelMutationError, match="W4A8 owns frozen models"):
        worker.reload_weights()


@pytest.mark.parametrize("after_insert", [False, True])
@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt])
def test_real_second_registration_failure_rolls_back_before_w4_publication(
    monkeypatch,
    after_insert,
    failure,
):
    adapter.register_post_model_load("unrelated", lambda context: None)
    adapter.register_model_mutation_guard("unrelated", lambda context: None)
    before = (
        dict(adapter._callbacks.callbacks),
        dict(adapter._mutation_guards.callbacks),
        dict(envs.environment_variables),
        list(registry.QUANTIZATION_METHODS),
        dict(registry._CUSTOMIZED_METHOD_TO_QUANT_CONFIG),
    )
    real_register = adapter.register_model_mutation_guard

    def broken(owner, callback):
        # Adapter's public atomic API calls the real split API internally.
        # Faults both before and after insertion must undo its first hook too.
        if after_insert:
            real_register(owner, callback)
        raise failure("second real registration failed")

    monkeypatch.setattr(adapter, "register_model_mutation_guard", broken)
    with pytest.raises(failure, match="second real registration failed"):
        owner.install()
    assert before == (
        adapter._callbacks.callbacks,
        adapter._mutation_guards.callbacks,
        envs.environment_variables,
        registry.QUANTIZATION_METHODS,
        registry._CUSTOMIZED_METHOD_TO_QUANT_CONFIG,
    )
    assert not owner._installed and owner._envs is None and owner._setting is None
    monkeypatch.setattr(adapter, "register_model_mutation_guard", real_register)
    assert owner.install()  # Failure leaves a clean, retryable declaration.


def test_preflight_is_not_reservation_atomic_commit_rechecks(monkeypatch):
    actual = adapter.register_model_lifecycle_hooks

    def other_guard(context):
        pass

    def race(**kwargs):
        adapter.register_model_mutation_guard(owner.MUTATION_OWNER, other_guard)
        return actual(**kwargs)

    monkeypatch.setattr(adapter_host, "register_model_lifecycle_hooks", race)
    with pytest.raises(adapter.ModelMutationError, match="already registered"):
        owner.install()
    assert owner.OWNER not in adapter._callbacks.callbacks
    assert adapter._mutation_guards.callbacks == {owner.MUTATION_OWNER: other_guard}
    assert owner.FACTOR not in envs.environment_variables
    assert "slimquant_w4a8" not in registry._CUSTOMIZED_METHOD_TO_QUANT_CONFIG
    assert not owner._installed


@pytest.mark.parametrize("missing", ["error", "classifier", "health"])
def test_independent_public_fatal_abi_missing_before_publication(monkeypatch, missing):
    import hcu_adapter.extensions as public
    import hcu_adapter.bridges.vllm as bridge

    module, name = {
        "error": (public, "ExtensionCommitError"),
        "classifier": (public, "is_fatal_extension_error"),
        "health": (bridge, "require_runtime_healthy"),
    }[missing]
    monkeypatch.delattr(module, name)
    with pytest.raises(owner.W4LifecycleError, match="public commit-fatal ABI"):
        owner.install()
    assert not owner._installed
    assert owner.FACTOR not in envs.environment_variables
    assert "slimquant_w4a8" not in registry._CUSTOMIZED_METHOD_TO_QUANT_CONFIG


def test_independent_public_fatal_abi_accepts_optional_arguments(monkeypatch):
    import hcu_adapter.extensions as public
    import hcu_adapter.bridges.vllm as bridge

    class CompatibleError(RuntimeError):
        def __init__(self, message, reports=(), *, future=None):
            super().__init__(message)
            self.reports = reports

    monkeypatch.setattr(public, "ExtensionCommitError", CompatibleError)
    monkeypatch.setattr(public, "is_fatal_extension_error", lambda error, *, future=None: False)
    monkeypatch.setattr(bridge, "require_runtime_healthy", lambda *, future=None: None)
    assert owner._preflight_adapter() is adapter_host
    assert owner.FACTOR not in envs.environment_variables


def test_independent_public_fatal_abi_rejects_async_health(monkeypatch):
    import hcu_adapter.bridges.vllm as bridge

    async def incompatible():
        return None

    monkeypatch.setattr(bridge, "require_runtime_healthy", incompatible)
    with pytest.raises(owner.W4LifecycleError, match="public commit-fatal ABI") as caught:
        owner.install()
    assert isinstance(caught.value.__cause__, TypeError)
    assert str(caught.value.__cause__) == "requires synchronous fatal ABI"
    assert not owner._installed


def test_independent_owner_latches_failure_after_atomic_hook_publication(monkeypatch):
    import hcu_adapter.extensions as public

    def broken_alias(*args, **kwargs):
        raise RuntimeError("injected alias publication failure")

    monkeypatch.setattr(registry, "register_quantization_config", broken_alias)
    with pytest.raises(public.ExtensionCommitError, match="after publication") as first:
        owner.install()
    assert public.is_fatal_extension_error(first.value)
    assert not owner._installed
    assert owner.FACTOR not in envs.environment_variables
    for operation in (owner.install, owner.require_selection, owner._preflight_adapter):
        with pytest.raises(public.ExtensionCommitError, match="after publication"):
            operation()
    owner._after_fork_child()
    with pytest.raises(public.ExtensionCommitError, match="after publication"):
        owner.install()


def test_independent_w4_preflight_never_imports_the_bundle():
    code = '''
import importlib.abc
import sys

class NoBundle(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "hcu_vllm_das.extension":
            raise AssertionError("independent W4 preflight imported bundle")

assert "hcu_vllm_das.extension" not in sys.modules
sys.meta_path.insert(0, NoBundle())
from hcu_vllm_das.w4a8 import lifecycle
lifecycle._preflight_adapter()
assert "hcu_vllm_das.extension" not in sys.modules
assert lifecycle._envs is None and not lifecycle._installed
print("public ABI preflight only; no bundle/provider/registry publication")
'''
    result = subprocess.run([sys.executable, "-B", "-c", code],
                            text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


def _bounded_fork_refusal(operation):
    """Real child; its alarm and parent's bounded wait prevent a hung CPU test."""
    import hcu_adapter.extensions as public

    pid = os.fork()
    if pid == 0:
        signal.alarm(3)
        try:
            operation()
        except BaseException as error:
            os._exit(0 if isinstance(error, public.ExtensionCommitError) else 2)
        os._exit(3)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        waited, status = os.waitpid(pid, os.WNOHANG)
        if waited:
            assert os.waitstatus_to_exitcode(status) == 0
            return
        time.sleep(.01)
    # An unreaped direct child cannot reuse its PID. Never signal any other PID.
    os.kill(pid, signal.SIGKILL)
    os.waitpid(pid, 0)
    pytest.fail("owned fork child timed out")


@pytest.mark.parametrize("render_error", [RuntimeError, KeyboardInterrupt])
@pytest.mark.parametrize("fork_in_render", [False, True])
def test_real_alias_failure_latches_before_unrenderable_error(
    monkeypatch, render_error, fork_in_render,
):
    """Real public Adapter pair/host and official registry; only late fault doubled."""
    import hcu_adapter.extensions as public

    if fork_in_render and not hasattr(os, "fork"):
        pytest.skip("POSIX fork required")
    original = registry.register_quantization_config
    seen, later, published = [], [], []

    class BadString(RuntimeError):
        def __str__(self):
            seen.append(owner._commit_failure is not None)
            if fork_in_render:
                _bounded_fork_refusal(owner.install)
            raise render_error("diagnostic cannot render")

    failure = BadString("real alias publication completed then faulted")

    def register(name):
        def publish(config):
            original(name)(config)
            published.append(registry._CUSTOMIZED_METHOD_TO_QUANT_CONFIG[name] is config)
            raise failure
        return publish

    monkeypatch.setattr(registry, "register_quantization_config", register)
    points = [
        NS(name="w4", load=lambda: lambda: public.ExtensionSpec(
            "w4", 1, 10, {"slimquant-w4a8"}, lambda context: owner.install())),
        NS(name="later", load=lambda: lambda: public.ExtensionSpec(
            "later", 1, 0, {"later"}, lambda context: later.append(True))),
    ]
    with pytest.raises(public.ExtensionCommitError) as caught:
        public.activate_extensions(public.ExtensionContext("cpu", "0.32.0"), points)
    assert published == [True] and seen and all(seen) and not later
    assert caught.value.reports[0].status.value == "failed"
    causes, current = [], caught.value
    while current is not None and all(current is not c for c in causes):
        causes.append(current)
        current = current.__cause__
    assert any(c is failure for c in causes)
    for operation in (owner.install, owner.require_selection, owner._preflight_adapter):
        with pytest.raises(public.ExtensionCommitError):
            operation()
    assert not owner._installed
    assert owner.FACTOR not in envs.environment_variables
    # A successful pair remains declared; failure is fatal, not rolled back.
    adapter.register_post_model_load(owner.OWNER, owner.after_model_load)
    with pytest.raises(adapter.PostModelLoadError, match="already registered"):
        adapter.register_post_model_load(owner.OWNER, lambda context: None)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires bounded POSIX fork")
@pytest.mark.parametrize("phase", ["prepare", "publish"])
def test_real_inflight_install_fork_does_not_retry_child(monkeypatch, phase):
    original_preflight = owner._preflight_adapter
    original_register = registry.register_quantization_config
    seen = []

    def check_child():
        def retry():
            # Restore the original entry functions only in the child; otherwise
            # a regression could recursively fork instead of trying its install.
            owner._preflight_adapter = original_preflight
            registry.register_quantization_config = original_register
            owner.install()
        _bounded_fork_refusal(retry)
        seen.append(phase)

    if phase == "prepare":
        def preflight():
            check_child()
            return original_preflight()
        monkeypatch.setattr(owner, "_preflight_adapter", preflight)
    else:
        def register(name):
            def publish(config):
                result = original_register(name)(config)
                check_child()
                return result
            return publish
        monkeypatch.setattr(registry, "register_quantization_config", register)
    assert owner.install()
    assert seen == [phase]
    assert owner._commit_failure is None and owner._installing_error_type is None


@pytest.mark.parametrize("wrapped", [False, True])
def test_fatal_preparation_stays_fatal_without_alias_publication(monkeypatch, wrapped):
    import hcu_adapter.extensions as public

    def preflight():
        fatal = public.ExtensionCommitError("dependency already published")
        if wrapped:
            try:
                raise fatal
            except public.ExtensionCommitError:
                raise RuntimeError("dependency wrapper") from None
        raise fatal

    monkeypatch.setattr(owner, "_preflight_adapter", preflight)
    with pytest.raises(public.ExtensionCommitError, match="preparation inherited a fatal failure"):
        owner.install()
    assert owner._commit_failure is not None and owner._installing_error_type is None
    assert owner.FACTOR not in envs.environment_variables
    assert "slimquant_w4a8" not in registry._CUSTOMIZED_METHOD_TO_QUANT_CONFIG
    with pytest.raises(public.ExtensionCommitError):
        owner.require_selection()


def test_clean_atomic_second_hook_failure_remains_optional_and_retryable(monkeypatch):
    import hcu_adapter.extensions as public

    original = adapter.register_model_mutation_guard

    def fail(*args, **kwargs):
        raise RuntimeError("clean second hook failure")

    monkeypatch.setattr(adapter, "register_model_mutation_guard", fail)
    point = NS(name="w4", load=lambda: lambda: public.ExtensionSpec(
        "w4", 1, 0, {"slimquant-w4a8"}, lambda context: owner.install()))
    reports = public.activate_extensions(public.ExtensionContext("cpu", "0.32.0"), [point])
    assert reports[0].status.value == "failed"
    assert owner._commit_failure is None and owner._installing_error_type is None
    assert owner.FACTOR not in envs.environment_variables
    assert "slimquant_w4a8" not in registry._CUSTOMIZED_METHOD_TO_QUANT_CONFIG
    # Real pair rollback leaves neither owner, then real retry succeeds.
    adapter.register_post_model_load(owner.OWNER, owner.after_model_load)
    monkeypatch.setattr(adapter, "register_model_mutation_guard", original)
    assert owner.install()


@pytest.mark.parametrize("stage", ["fatal-abi", "facade"])
@pytest.mark.parametrize("render_error", [RuntimeError, KeyboardInterrupt])
@pytest.mark.parametrize("fork_in_render", [False, True])
def test_actual_activation_preflight_preserves_unrenderable_fatal(
    monkeypatch, stage, render_error, fork_in_render,
):
    """Actual public activation/runtime host; dependency fault before any packing."""
    import hcu_adapter.extensions as public
    from hcu_adapter.bridges import vllm as host

    if fork_in_render and not hasattr(os, "fork"):
        pytest.skip("POSIX fork required")
    actions = host.ProcessLocalActions()
    original_import = owner.import_module
    seen, later = [], []

    class BadFatal(public.ExtensionCommitError):
        def __str__(self):
            seen.append(owner._commit_failure is not None)
            if fork_in_render:
                _bounded_fork_refusal(owner.install)
            raise render_error("unrenderable typed preflight failure")

    failure = BadFatal("dependency already failed")
    subject = ("hcu_adapter.bridges.vllm" if stage == "fatal-abi" else
               "hcu_adapter.bridges.vllm_model_load")

    def dependency(name):
        if name == subject:
            raise failure
        return original_import(name)

    monkeypatch.setattr(owner, "import_module", dependency)
    points = [
        NS(name="w4", load=lambda: owner.extension_spec),
        NS(name="later", load=lambda: lambda: public.ExtensionSpec(
            "later", 1, 0, {"later"}, lambda context: later.append(True))),
    ]
    context = public.ExtensionContext("cpu", "0.32.0")

    def boot():
        public.activate_extensions(context, points)

    with pytest.raises(host.LatchedVllmRuntimeError) as caught:
        actions.run_once("vllm-runtime", boot)
    assert seen and all(seen) and not later
    assert owner._commit_failure is not None
    assert owner._installing_error_type is None
    assert owner.FACTOR not in envs.environment_variables
    assert "slimquant_w4a8" not in registry._CUSTOMIZED_METHOD_TO_QUANT_CONFIG
    chain, current = [], caught.value
    while current is not None and all(current is not c for c in chain):
        chain.append(current)
        current = current.__cause__
    assert any(c is failure for c in chain)
    reports = next(c.reports for c in chain
                   if isinstance(c, public.ExtensionCommitError) and c.reports)
    assert len(reports) == 1 and reports[0].status.value == "failed"
    with pytest.raises(host.LatchedVllmRuntimeError):
        actions.require_runtime_healthy()
    with pytest.raises(host.LatchedVllmRuntimeError):
        actions.run_once("vllm-runtime", lambda: later.append(True))
    for operation in (owner.install, owner.require_selection,
                      lambda: owner.activate(context)):
        with pytest.raises(public.ExtensionCommitError):
            operation()
    monkeypatch.setattr(owner, "import_module", original_import)
    if fork_in_render:
        _bounded_fork_refusal(owner.install)


@pytest.mark.parametrize("stage", ["fatal-abi", "facade"])
def test_actual_activation_clean_preflight_stays_optional_without_rendering(
    monkeypatch, stage,
):
    import hcu_adapter.extensions as public
    from hcu_adapter.bridges import vllm as host

    original_import = owner.import_module
    rendered, later = [], []

    class CleanFailure(RuntimeError):
        def __str__(self):
            rendered.append(True)
            raise KeyboardInterrupt("clean diagnostic cannot render")

    failure = CleanFailure("no publication")
    subject = ("hcu_adapter.bridges.vllm" if stage == "fatal-abi" else
               "hcu_adapter.bridges.vllm_model_load")

    def dependency(name):
        if name == subject:
            raise failure
        return original_import(name)

    monkeypatch.setattr(owner, "import_module", dependency)
    points = [
        NS(name="w4", load=lambda: owner.extension_spec),
        NS(name="later", load=lambda: lambda: public.ExtensionSpec(
            "later", 1, 0, {"later"}, lambda context: later.append(True))),
    ]
    actions, reports = host.ProcessLocalActions(), []
    actions.run_once("vllm-runtime", lambda: reports.extend(
        public.activate_extensions(public.ExtensionContext("cpu", "0.32.0"), points)
    ))
    assert not rendered and later == [True]
    assert {report.entry_point_name: report.status.value for report in reports} == {
        "w4": "failed", "later": "active",
    }
    assert owner._commit_failure is None and owner._installing_error_type is None
    assert owner.FACTOR not in envs.environment_variables
    assert "slimquant_w4a8" not in registry._CUSTOMIZED_METHOD_TO_QUANT_CONFIG
    actions.require_runtime_healthy()
    monkeypatch.setattr(owner, "import_module", original_import)
    assert owner.install()


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires bounded POSIX fork")
def test_actual_activation_preflight_fork_refuses_child_but_not_clean_parent(monkeypatch):
    import hcu_adapter.extensions as public

    original = owner._preflight_adapter

    def preflight():
        def child_retry():
            owner._preflight_adapter = original
            owner.install()
        _bounded_fork_refusal(child_retry)
        raise owner.W4LifecycleError("clean dependency refusal after child receipt")

    monkeypatch.setattr(owner, "_preflight_adapter", preflight)
    point = NS(name="w4", load=lambda: owner.extension_spec)
    reports = public.activate_extensions(public.ExtensionContext("cpu", "0.32.0"), [point])
    assert reports[0].status.value == "failed"
    assert owner._commit_failure is None and owner._installing_error_type is None
    assert owner.FACTOR not in envs.environment_variables
    assert "slimquant_w4a8" not in registry._CUSTOMIZED_METHOD_TO_QUANT_CONFIG
