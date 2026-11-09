"""Explicit independent publication through the public INT8 backend registry."""

from dataclasses import dataclass
from importlib import import_module
import os
import sys
from pathlib import Path
from threading import RLock

from hcu_adapter.extensions import ExtensionCommitError, ExtensionSpec

from .extension import _require_vllm_033_release_line

CAPABILITY = 'hcu-aiter-int8-moe'
FACTOR = 'VLLM_HCU_DAS_INT8_ASM_IDENTITY'
REGISTRY = 'vllm.model_executor.layers.fused_moe.oracle.int8_backend_registry'
_lock = RLock()
_published = None
_failure = False
_publishing = False


@dataclass(frozen=True)
class _Publication:
    pid: int
    backend: object
    identity: object
    bindings: tuple


def _after_fork():
    global _lock, _published, _failure
    _lock = RLock()
    _published = None
    _failure = _failure or _publishing


if hasattr(os, 'register_at_fork'):
    os.register_at_fork(after_in_child=_after_fork)


def _fatal(error=None):
    global _failure
    _failure = True
    raise ExtensionCommitError('INT8 ASM publication is unhealthy; restart the process') from error


def require_published():
    with _lock:
        if _failure:
            _fatal()
        if _published is None or _published.pid != os.getpid():
            raise RuntimeError('INT8 ASM publication is not active in this process')
        try:
            if import_module(REGISTRY).get_int8_moe_backend('aiter') is not _published.backend:
                _fatal()
            if import_module('vllm.envs').environment_variables.get(FACTOR) is not compile_factor:
                _fatal()
            if any(getattr(module, name) is not function
                   for module, name, function in _published.bindings):
                _fatal()
        except BaseException as error:
            _fatal(error)


def check_health():
    """Adapter callback: read only, no host lifecycle/activation/repair."""
    require_published()


def compile_factor():
    require_published()
    try:
        _published.identity.validate()
    except BaseException as error:
        _fatal(error)
    return 'int8-asm-v1:gfx936:' + _published.identity.digest


def _prepare_identity(native, moe):
    from .moe.int8_identity import FrozenFiles
    module = "aiter.fused_moe_asm_wna16"
    if module in sys.modules:
        raise RuntimeError("INT8 ASM must activate before vendor MoE config consumers; restart with a fresh spawned process")
    asm = import_module(module)
    jit = import_module('aiter.jit.core')
    # Prepare actual existing native modules before model capture and bind their
    # contents plus ASM objects/configs; missing ABI refuses before publication.
    modules = [native, moe, asm, jit, import_module('hcu_vllm_das.moe.int8_experts'),
               import_module('hcu_vllm_das.moe.fused_moe_aiter'),
               import_module('hcu_vllm_das.moe.int8_asm'),
               import_module('hcu_vllm_das.moe.int8_identity'),
               import_module(__name__)]
    modules += [jit.get_module(name) for name in ('module_cpp_api', 'module_moe_asm')]
    files = {module.__name__: module.__file__ for module in modules}
    root = Path(jit.get_asm_dir())
    objects = sorted(root.rglob('*.co')) + sorted(root.rglob('*.hsaco'))
    if not objects:
        raise RuntimeError('INT8 ASM native objects are unavailable')
    files.update({'asm/' + str(path.relative_to(root)): path for path in objects})
    path = asm.get_csv_path(asm.MoeQuantType.INT8_W8A8_C)
    files['channel-raw-config'] = path
    bindings = tuple((module, name, getattr(module, name)) for module, name in (
        (native, 'moe_sorting_fwd'), (native, 'asm_fmoe_a8'),
        (moe, 'get_aiter_moe_config'), (asm, 'get_moe_asm_solution'),
        (asm, 'get_csv_path'), (jit, 'get_module'),
    ))
    if not all(callable(function) for _, _, function in bindings):
        raise RuntimeError('INT8 ASM native API is unavailable')
    identity = FrozenFiles.capture(files)
    if not asm.load_and_cache_csv_for_quant(asm.MoeQuantType.INT8_W8A8_C, 0):
        raise RuntimeError("INT8 ASM channel config is unavailable")
    identity.validate()
    return identity, bindings


def activate(context):
    global _published, _publishing
    _require_vllm_033_release_line(context.vllm_version)
    if CAPABILITY not in context.requested_capabilities:
        raise RuntimeError('INT8 ASM requires explicit hcu-aiter-int8-moe capability')
    with _lock:
        if _failure or _publishing:
            _fatal()
        if _published is not None:
            check_health()
            return
        if "aiter.fused_moe_asm_wna16" in sys.modules:
            raise RuntimeError("INT8 ASM must activate before vendor MoE config consumers; restart with a fresh spawned process")
    from hcu_adapter.bridges.vllm import detect_hcu
    detection = detect_hcu()
    if not detection.detected or detection.architecture != 'gfx936':
        raise RuntimeError('INT8 ASM is currently validated for real gfx936 HCU only')
    with _lock:
        if _failure or _publishing:
            _fatal()
        if _published is not None:
            check_health()
            return
        _publishing = True
        try:
            registry = import_module(REGISTRY)
            if registry.get_int8_moe_backend('aiter') is not None:
                raise RuntimeError('INT8 backend aiter already has an owner')
            envs = import_module('vllm.envs')
            prior = envs.environment_variables.get(FACTOR)
            if prior is not None and prior is not compile_factor:
                raise RuntimeError('INT8 compile factor already has an owner')
            from .moe.int8_experts import HcuAiterInt8Experts, convert_weights
            native = import_module('aiter')
            moe = import_module('aiter.moe')
            identity, bindings = _prepare_identity(native, moe)
            backend = registry.register_int8_moe_backend('aiter', HcuAiterInt8Experts, convert_weights)
            envs.environment_variables[FACTOR] = compile_factor
            _published = _Publication(os.getpid(), backend, identity, bindings)
            check_health()
        except BaseException as error:
            _fatal(error)
        finally:
            _publishing = False


def extension_spec():
    return ExtensionSpec(
        name='das-hcu-aiter-int8', api_version=1, priority=80,
        capabilities=frozenset({CAPABILITY}), activate=activate, check_health=check_health,
    )
