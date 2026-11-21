"""Explicit independent publication through the public INT8 backend registry."""

import ctypes
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
_CHIP_INFO = 'aiter.jit.utils.chip_info'
_SUPPORTED_ARCHITECTURES = frozenset({'gfx936', 'gfx938'})
_lock = RLock()
_published = None
_failure = False
_publishing = False
_getenv = ctypes.PyDLL(None).getenv
_getenv.argtypes = (ctypes.c_char_p,)
_getenv.restype = ctypes.c_char_p


@dataclass(frozen=True)
class _Publication:
    pid: int
    backend: object
    identity: object
    bindings: tuple
    architecture: str
    chip_info: object
    asm_dir: bytes


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
            if (sys.modules.get(_CHIP_INFO) is not _published.chip_info
                    or _published.chip_info.get_gfx() != _published.architecture):
                _fatal()
            # The C++ loader reads getenv during lazy module loading, including
            # the trailing separator. Python's os.environ can be out of date.
            if _getenv(b'AITER_ASM_DIR') != _published.asm_dir:
                _fatal()
            _published.identity.validate_paths()
        except BaseException as error:
            _fatal(error)
        return _published.architecture


def check_health():
    """Adapter callback: read only, no host lifecycle/activation/repair."""
    require_published()


def compile_factor():
    require_published()
    try:
        _published.identity.validate()
    except BaseException as error:
        _fatal(error)
    return 'int8-asm-v1:' + _published.architecture + ':' + _published.identity.digest


def _asm_objects(root):
    def fail(error):
        raise error

    ancestors = {}
    for directory_name, _, files in os.walk(root, onerror=fail, followlinks=True):
        directory = Path(directory_name)
        stat = directory.stat()
        target = stat.st_dev, stat.st_ino
        lineage = ancestors.get(directory.parent, ())
        if target in lineage:
            raise RuntimeError('INT8 ASM object tree contains a directory cycle')
        ancestors[directory] = (*lineage, target)
        yield from (directory / name for name in files if name.endswith(('.co', '.hsaco')))


def _prepare_identity(native, moe, architecture):
    from .moe.int8_identity import FrozenFiles
    module = "aiter.fused_moe_asm_wna16"
    if module in sys.modules:
        raise RuntimeError("INT8 ASM must activate before vendor MoE config consumers; restart with a fresh spawned process")
    chip_info = import_module(_CHIP_INFO)
    read_architecture = getattr(chip_info, 'get_gfx', None)
    if not callable(read_architecture):
        raise RuntimeError('INT8 ASM plan architecture API is unavailable')
    provider_architecture = read_architecture()
    if provider_architecture != architecture:
        raise RuntimeError(
            f'INT8 ASM plan architecture {provider_architecture!r} differs from real HCU {architecture!r}'
        )
    asm = import_module(module)
    jit = import_module('aiter.jit.core')
    # Prepare actual existing native modules before model capture and bind their
    # contents plus ASM objects/configs; missing ABI refuses before publication.
    modules = [native, moe, asm, jit, chip_info, import_module('hcu_vllm_das.moe.int8_experts'),
               import_module('hcu_vllm_das.moe.fused_moe_aiter'),
               import_module('hcu_vllm_das.moe.int8_asm'),
               import_module('hcu_vllm_das.moe.int8_routes'),
               import_module('hcu_vllm_das.moe.int8_identity'),
               import_module(__name__)]
    modules += [jit.get_module(name) for name in ('module_cpp_api', 'module_moe_asm')]
    files = {module.__name__: module.__file__ for module in modules}
    asm_dir = _getenv(b'AITER_ASM_DIR')
    if (not asm_dir or not os.path.isabs(asm_dir) or not asm_dir.endswith(b'/')
            or asm_dir != os.fsencode(jit.get_asm_dir())):
        raise RuntimeError('INT8 ASM native loader directory differs from the vendor object root')
    root = Path(os.fsdecode(asm_dir))
    objects = sorted(_asm_objects(root))
    if not objects:
        raise RuntimeError('INT8 ASM native objects are unavailable')
    files.update({'asm/' + str(path.relative_to(root)): path for path in objects})
    path = asm.get_csv_path(asm.MoeQuantType.INT8_W8A8_C)
    files['channel-raw-config'] = path
    bindings = tuple((module, name, getattr(module, name)) for module, name in (
        (native, 'moe_sorting_fwd'), (native, 'asm_fmoe_a8'),
        (moe, 'get_aiter_moe_config'), (asm, 'get_moe_asm_solution'),
        (asm, 'get_csv_path'), (jit, 'get_module'),
        (chip_info, 'get_gfx'),
    ))
    if not all(callable(function) for _, _, function in bindings):
        raise RuntimeError('INT8 ASM native API is unavailable')
    identity = FrozenFiles.capture(files, watch_paths=(root,))
    if not asm.load_and_cache_csv_for_quant(asm.MoeQuantType.INT8_W8A8_C, 0):
        raise RuntimeError("INT8 ASM channel config is unavailable")
    identity.validate()
    return identity, bindings, chip_info, asm_dir


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
    if not detection.detected or detection.architecture not in _SUPPORTED_ARCHITECTURES:
        raise RuntimeError('INT8 ASM requires real gfx936 or gfx938 HCU')
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
            identity, bindings, chip_info, asm_dir = _prepare_identity(native, moe, detection.architecture)
            backend = registry.register_int8_moe_backend('aiter', HcuAiterInt8Experts, convert_weights)
            envs.environment_variables[FACTOR] = compile_factor
            _published = _Publication(os.getpid(), backend, identity, bindings, detection.architecture, chip_info, asm_dir)
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
