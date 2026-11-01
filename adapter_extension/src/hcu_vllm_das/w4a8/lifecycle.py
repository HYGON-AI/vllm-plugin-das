"""Independent W4 ownership, Adapter boundaries and ordinary/AOT identity."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import inspect
import json
import os
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
from threading import RLock
from weakref import WeakKeyDictionary, ref

FACTOR = "VLLM_HCU_DAS_W4A8_POLICY"
OWNER = "hcu_vllm_das.w4a8"
MUTATION_OWNER = OWNER + ".frozen-models"
_installed = False
_commit_failure = None
_installing_error_type = None
_pid = os.getpid()
_fork_error = None
_lock = RLock()
_envs = None
_selected = {}
_alias_configs = []
_workers = WeakKeyDictionary()
_frozen = None
_frozen_rows = None
_setting = None
_file_hashes = {}

# Read files without importing or initializing AITER. Versions are provenance,
# not admission rules; the selected route still checks the public callable ABI.
_VENDOR_FILES = (
    "__init__.py",
    "moe.py",
    "fused_moe_c.py",
    "ops/moe_c_op.py",
    "ops/quant.py",
    "ops/shuffle.py",
    "ops/triton/fused_moe.py",
    "ops/triton/moe_activation.py",
    "jit/core.py",
    "jit/utils/torch_guard.py",
    "moe_c_configs/tuned_moe_c_w4a8_channel.csv",
    "moe_c_configs/silu_configs/silu_config_summary.json",
    "jit/module_moe_c_w4a8_kernel.so",
    "jit/module_moe_c_activation.so",
    "jit/module_moe_c_sum.so",
    "jit/module_moe_c_align.so",
    "jit/module_quant.so",
)


class W4LifecycleError(RuntimeError):
    """The optional W4 owner cannot safely load, freeze or mutate models."""


def _require_commit_failure_abi():
    """Pure ABI validation; callers query health before taking owner locks."""
    try:
        api = import_module("hcu_adapter.extensions")
        error_type = getattr(api, "ExtensionCommitError", None)
        if not inspect.isclass(error_type) or not issubclass(error_type, RuntimeError):
            raise TypeError("missing public ExtensionCommitError type")
        inspect.signature(error_type).bind("commit failed", reports=())
        bridge = import_module("hcu_adapter.bridges.vllm")
        health = getattr(bridge, "require_runtime_healthy", None)
        for function, arguments in ((getattr(api, "is_fatal_extension_error", None), (RuntimeError(),)),
                                    (health, ())):
            if not callable(function):
                raise TypeError("missing public classifier/runtime health query")
            implementations = (function,) if inspect.isroutine(function) else (function, function.__call__)
            if any(check(value) for value in implementations for check in (
                inspect.iscoroutinefunction, inspect.isasyncgenfunction, inspect.isgeneratorfunction,
            )):
                raise TypeError("requires synchronous fatal ABI")
            inspect.signature(function).bind(*arguments)
        return api, health
    except Exception as error:
        raise W4LifecycleError(
            "Optional W4A8 requires Adapter public commit-fatal ABI: ExtensionCommitError, "
            "is_fatal_extension_error(error), require_runtime_healthy(); "
            f"preflight failed: {error}"
        ) from error


def _check_commit_failure():
    if _commit_failure is not None:
        error_type, message = _commit_failure
        raise error_type(message)


def native_requested():
    value = os.environ.get("VLLM_HCU_DAS_W4A8_NATIVE", "0")
    if value not in ("0", "1"):
        raise ValueError("VLLM_HCU_DAS_W4A8_NATIVE must be 0 or 1")
    return value == "1"


def _after_fork_child():
    global _pid, _lock, _fork_error, _commit_failure
    _lock = RLock()
    _pid = os.getpid()
    if _installing_error_type is not None and _commit_failure is None:
        _commit_failure = (
            _installing_error_type,
            "W4A8 inherited incomplete installation; use a fresh process",
        )
    if _selected or _frozen is not None or _workers:
        _fork_error = (
            "W4A8 inherited selected/frozen state across fork; spawn a fresh worker"
        )


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork_child)


@contextmanager
def _installation_attempt(error_type):
    global _installing_error_type
    _installing_error_type = error_type
    try:
        yield
    finally:
        _installing_error_type = None


def _raise_install_failure(error_type, error, *, published):
    global _commit_failure
    if _commit_failure is None:
        reason = (
            "W4A8 installation failed after publication" if published else
            "W4A8 preparation inherited a fatal failure"
        ) + "; use a fresh process"
        _commit_failure = (error_type, reason)
        try:
            detail = f"{reason}: {type(error).__name__}: {error}"
        except BaseException:
            detail = reason
        _commit_failure = (error_type, detail)
    stored_type, message = _commit_failure
    raise stored_type(message) from error


def W4_check_process():
    if os.getpid() != _pid:
        _after_fork_child()
    if _fork_error is not None:
        raise W4LifecycleError(_fork_error)


# Keep the source-stage diagnostic API used by the original CPU suite.
_check_process = W4_check_process


def _hash_file(path):
    try:
        stat = path.stat()
    except FileNotFoundError:
        return "missing"
    stamp = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    cached = _file_hashes.get(path)
    if cached is None or cached[0] != stamp:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        after = path.stat()
        if stamp != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise W4LifecycleError(f"W4A8 identity changed while hashing {path.name}")
        cached = (stamp, digest.hexdigest())
        _file_hashes[path] = cached
    return cached[1]


def _plugin_root():
    return Path(__file__).resolve().parent


def _vendor_root():
    # Top-level find_spec does not import AITER or execute its package init.
    spec = find_spec("aiter")
    return Path(spec.origin).resolve().parent if spec and spec.origin else None


def implementation_identity():
    root = _plugin_root()
    result = {
        "plugin": {
            name: _hash_file(root / name)
            for name in (
                "__init__.py",
                "config.py",
                "runtime.py",
                "lifecycle.py",
            )
        }
    }
    # Official INT8 fallback has no AITER dependency. Native requests include
    # public dispatch, math/packing, tuned config and all used native binaries.
    if native_requested() or any(
        method() is not None and method().native is not None
        for method in _selected.values()
    ):
        vendor = _vendor_root()
        result["aiter"] = (
            {name: _hash_file(vendor / name) for name in _VENDOR_FILES}
            if vendor is not None
            else {"availability": "missing"}
        )
    return result


def _selection_rows():
    rows = {}
    for prefix, handle in sorted(_selected.items()):
        method = handle()
        if method is None:
            if _frozen_rows is None:
                raise W4LifecycleError("W4A8 selected method vanished before freeze")
            rows[prefix] = _frozen_rows[prefix]
            continue
        plan = method.native
        rows[prefix] = {"route": "official-int8" if plan is None else "aiter-w4a8"}
        if plan is not None:
            rows[prefix].update(
                layout=plan.layout,
                configs_sha256=hashlib.sha256(
                    json.dumps(
                        [vars(config) for config in plan.configs],
                        sort_keys=True,
                        default=str,
                    ).encode()
                ).hexdigest(),
                shape=vars(plan.shape),
            )
    return rows


def factor():
    # Target vLLM can silently omit a raising getter. Invalid state is itself
    # hashed; ordinary and AOT getters both see real implementation byte drift.
    current = None
    try:
        W4_check_process()
        requested = native_requested()
        if _setting is not None and requested != _setting:
            raise W4LifecycleError("W4A8 native setting changed after activation")
        current = dict(
            abi=2,
            native=requested,
            phase="frozen" if _frozen is not None else "selecting",
            selections=_selection_rows(),
            identity=implementation_identity(),
        )
        result = json.dumps(current, sort_keys=True, default=str)
        if isinstance(_frozen, str) and result != _frozen:
            raise W4LifecycleError(
                "W4A8 implementation/route/config changed after freeze"
            )
        return result
    except Exception as error:
        return json.dumps(
            {"abi": 2, "invalid": str(error), "current": current},
            sort_keys=True,
            default=str,
        )


def _lifecycle_arguments():
    return dict(
        post_model_load_owner=OWNER,
        post_model_load_callback=after_model_load,
        model_mutation_owner=MUTATION_OWNER,
        model_mutation_guard=before_model_mutation,
    )


def _preflight_adapter():
    _check_commit_failure()
    _require_commit_failure_abi()
    subject = "hcu_adapter.bridges.vllm_model_load"
    try:
        module = import_module(subject)
        for name, arguments in (
            (
                "register_post_model_load",
                {"owner": OWNER, "callback": after_model_load},
            ),
            ("require_post_model_load", {"vllm_config": object()}),
            (
                "register_model_mutation_guard",
                {"owner": MUTATION_OWNER, "callback": before_model_mutation},
            ),
            ("require_model_mutation_guard", {"vllm_config": object()}),
            (
                "preflight_model_lifecycle_hooks",
                _lifecycle_arguments(),
            ),
            (
                "register_model_lifecycle_hooks",
                _lifecycle_arguments(),
            ),
        ):
            subject = f"hcu_adapter.bridges.vllm_model_load.{name}"
            function = getattr(module, name, None)
            if not callable(function):
                raise TypeError("missing callable")
            implementations = (
                (function,)
                if inspect.isroutine(function)
                else (function, function.__call__)
            )
            if any(
                check(implementation)
                for implementation in implementations
                for check in (
                    inspect.iscoroutinefunction,
                    inspect.isasyncgenfunction,
                    inspect.isgeneratorfunction,
                )
            ):
                raise TypeError("requires synchronous callable")
            signature = inspect.signature(function)
            signature.bind(**arguments)
            if name not in (
                "preflight_model_lifecycle_hooks",
                "register_model_lifecycle_hooks",
            ):
                signature.bind(*arguments.values())
        subject = "hcu_adapter.bridges.vllm_model_load.preflight_model_lifecycle_hooks"
        module.preflight_model_lifecycle_hooks(**_lifecycle_arguments())
        return module
    except Exception as error:
        raise W4LifecycleError(
            f"Optional W4A8 requires Adapter postload + mutation ABI with public atomic lifecycle hooks; {subject}: {error}"
        ) from error


def _uses_alias(config):
    models = [getattr(config, "model_config", None)]
    speculative = getattr(config, "speculative_config", None)
    if speculative is not None:
        models.append(getattr(speculative, "draft_model_config", None))
    return any(
        getattr(model, "quantization", None) == "slimquant_w4a8" for model in models
    )


def _remembered(config):
    return any(handle() is config for handle in _alias_configs)


def require_selection(config=None):
    _check_commit_failure()
    _, health = _require_commit_failure_abi()
    health()
    W4_check_process()
    if not _installed or _envs.environment_variables.get(FACTOR) is not factor:
        raise W4LifecycleError(
            "Activate the independent W4A8 extension before selection"
        )
    if _frozen is not None or _envs._is_envs_cache_enabled():
        raise W4LifecycleError("W4A8 selections frozen; recreate worker before loading")
    if native_requested() != _setting:
        raise W4LifecycleError("W4A8 native setting changed after activation")
    if config is None:
        from vllm.config import get_current_vllm_config

        config = get_current_vllm_config()
    adapter = _preflight_adapter()
    adapter.require_post_model_load(config)
    adapter.require_model_mutation_guard(config)
    if not _uses_alias(config):
        raise W4LifecycleError(
            "W4A8 selection requires target/draft slimquant_w4a8 alias"
        )
    if not _remembered(config):
        _alias_configs.append(ref(config))


def record(prefix, method):
    require_selection()
    if prefix in _selected:
        raise W4LifecycleError(f"W4A8 duplicate layer selection: {prefix}")
    _selected[prefix] = ref(method)


def _loaded_models(worker):
    runner = worker.model_runner
    target, draft = runner.get_model(), runner.get_draft_model()
    if target is None:
        raise W4LifecycleError("Cannot protect an unloaded W4A8 worker")
    return (target,) if draft is None else (target, draft)


def _loaded_alias(models):
    from .config import (
        SlimQuantW4A8Config,
        SlimQuantW4A8LinearMethod,
        SlimQuantW4A8MoEMethod,
    )

    return any(
        isinstance(getattr(module, "quant_config", None), SlimQuantW4A8Config)
        or isinstance(
            getattr(module, "quant_method", None),
            (SlimQuantW4A8MoEMethod, SlimQuantW4A8LinearMethod),
        )
        for model in models
        for module in model.modules()
    )


def after_model_load(context):
    global _frozen, _frozen_rows
    config = context.vllm_config
    # Unrelated CT workers must delegate even after another W4 worker froze,
    # or in a fork poisoned only for the inherited W4 owner.
    alias = _uses_alias(config) or _remembered(config)
    models = None
    if not alias:
        models = _loaded_models(context.worker)
        if not _loaded_alias(models):
            return
    W4_check_process()
    adapter = _preflight_adapter()
    adapter.require_post_model_load(config)
    adapter.require_model_mutation_guard(config)
    if not _installed or _envs.environment_variables.get(FACTOR) is not factor:
        raise W4LifecycleError("Activate W4A8 before successful-load freeze")
    if _frozen is not None or _envs._is_envs_cache_enabled():
        raise W4LifecycleError("W4A8 selections frozen; recreate worker before loading")
    if native_requested() != _setting:
        raise W4LifecycleError("W4A8 native setting changed after activation")
    if any(
        handle() is None or not handle()._processed for handle in _selected.values()
    ):
        raise W4LifecycleError("W4A8 selected layer missed post-load processing")
    models = _loaded_models(context.worker) if models is None else models
    identities = tuple(ref(model) for model in models)
    with _lock:
        if context.worker in _workers:
            raise W4LifecycleError("W4A8 worker already owns frozen models")
        _frozen_rows = _selection_rows()
        _frozen = True
        value = factor()
        if "invalid" in json.loads(value):
            _frozen = None
            raise W4LifecycleError(
                f"W4A8 cannot freeze invalid effective policy: {json.loads(value)['invalid']}"
            )
        _frozen = value
        _workers[context.worker] = identities


def before_model_mutation(context):
    # Ownership outlives config/model replacement; weak refs retain no models.
    # Return before process checks for workers outside this independent owner.
    with _lock:
        if context.worker not in _workers:
            return
    W4_check_process()
    raise W4LifecycleError(
        f"W4A8 owns frozen models: {context.operation}/{context.role} requires a fresh worker"
    )


def install():
    global _installed, _envs, _setting
    _check_commit_failure()
    fatal, health = _require_commit_failure_abi()
    health()
    W4_check_process()
    # All required signatures are checked before importing the quant registry or
    # config/provider modules, registering callbacks, or changing env getters.
    with _lock, _installation_attempt(fatal.ExtensionCommitError):
        _check_commit_failure()
        if _installed:
            if native_requested() != _setting:
                raise W4LifecycleError("W4A8 native setting changed after activation")
            return False
        published = False
        try:
            adapter = _preflight_adapter()
            import vllm.envs as envs
            from vllm.model_executor.layers import quantization as registry
            from .config import SlimQuantW4A8Config

            setting = native_requested()
            if envs._is_envs_cache_enabled():
                raise W4LifecycleError("Register W4A8 before env cache initialization")
            previous = envs.environment_variables.get(FACTOR)
            if previous is not None and previous is not factor:
                raise W4LifecycleError("Another owner registered the W4A8 compile factor")
            previous = registry._CUSTOMIZED_METHOD_TO_QUANT_CONFIG.get("slimquant_w4a8")
            if previous is not None and previous is not SlimQuantW4A8Config:
                raise W4LifecycleError("Another owner registered slimquant_w4a8")
            # Public preflight is not a lease. Atomic publication precedes
            # aliases/factors, and a clean atomic failure leaves both untouched.
            adapter.register_model_lifecycle_hooks(**_lifecycle_arguments())
            published = True
            registry.register_quantization_config("slimquant_w4a8")(SlimQuantW4A8Config)
            envs.environment_variables[FACTOR] = factor
            _setting, _envs, _installed = setting, envs, True
            return True
        except BaseException as error:
            if published or fatal.is_fatal_extension_error(error):
                _raise_install_failure(fatal.ExtensionCommitError, error, published=published)
            raise


def activate(context):
    _check_commit_failure()
    _, health = _require_commit_failure_abi()
    health()
    from packaging.version import Version

    version = Version(context.vllm_version)
    if version.epoch != 0 or version.release[:2] != (0, 32):
        raise W4LifecycleError("QNT009 requires the vLLM 0.32 release line")
    _preflight_adapter()
    from hcu_adapter.bridges.vllm import detect_hcu

    if not detect_hcu().detected:
        raise W4LifecycleError("QNT009 requires a detected HCU host")
    install()


def extension_spec():
    from hcu_adapter.extensions import ExtensionSpec

    return ExtensionSpec(
        name="das-hcu-w4a8",
        api_version=1,
        # The legacy bundle installs the INT8 oracle at priority 100, before
        # this owner's CT config import binds the consumer's oracle aliases.
        # With no bundle installed, this entry still activates independently.
        priority=90,
        capabilities=frozenset({"slimquant-w4a8"}),
        activate=activate,
    )
