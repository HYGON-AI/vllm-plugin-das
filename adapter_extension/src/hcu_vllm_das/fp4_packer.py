# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Startup-only, opt-in 033 E2M1 rebinding. Discovery is framework-free.

The implementation belongs to the Plugin; Adapter supplies config, kill
switch and runtime bookkeeping. Stock/kill in a fresh process preserves the
exact original objects. Live rollback is refused because JIT/graph caches
may already refer to the published implementation.
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import os
import textwrap
from importlib import import_module
from pathlib import Path
from threading import RLock

from hcu_adapter import hcu_env, registry
from hcu_adapter.extensions import ExtensionCommitError, ExtensionSpec
from hcu_adapter.kill_switch import op_disabled

from .dsv41_activation import _require_release

OP = 'vllm_fp4_packer'
VARIABLE = 'HCU_VLLM_FP4_PACKER_IMPL'
FACTOR = 'VLLM_HCU_DAS_FP4_PACKER_LOWERING'
SYMBOL = '_fp32x2_to_fp4x2'
CAPABILITY = 'hcu-e2m1-packer-lowering'
# anchor-baseline(③): vLLM 0.33 tree 5afd482929 keeps the E2M1 definition,
# adds a V4 caller and a specialized V4.1 caller, and re-exports via common.ops.
OWNERS = (
    'vllm.models.deepseek_v4.common.ops.fused_indexer_q',
    'vllm.models.deepseek_v4.common.ops.fused_compress_quant_cache',
    'vllm.models.deepseek_v41.common.ops',
    'vllm.models.deepseek_v41.common.ops.indexer_k_store',
    'vllm.models.deepseek_v41.common.ops.fused_compress_quant_cache',
    'vllm.models.deepseek_v41.nvidia.specialized_decode.attention.indexer_k_store',
)
# An AST audit accepts comments/formatting changes, not changed instructions,
# operand order, purity, dtype, pack, saturation, rounding or signature.
_AUDITED = '''def _fp32x2_to_fp4x2(x_lo, x_hi):
    return tl.inline_asm_elementwise(
        "\\n        {\\n            .reg .b8 tmp;\\n            cvt.rn.satfinite.e2m1x2.f32 tmp, $1, $2;\\n            cvt.u32.u8 $0, tmp;\\n        }\\n        ",
        constraints="=r,f,f", args=[x_hi, x_lo], dtype=tl.uint32,
        is_pure=True, pack=1,
    ).to(tl.uint8)
'''
_state = None
_failure = None
_publishing = False
_lock = RLock()


class Fp4PackerCompatibilityError(RuntimeError):
    """An opt-in cannot safely bind the audited packer."""


def _mode():
    if op_disabled(OP):
        return 'stock'
    value, _ = hcu_env.resolve(VARIABLE, host='vllm')
    if value not in ('stock', 'triton'):
        raise Fp4PackerCompatibilityError(f'{VARIABLE} must be stock or triton')
    return value


def _function_tree(source):
    tree = ast.parse(textwrap.dedent(source))
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise ValueError('expected a single function')
    function = tree.body[0]
    function.decorator_list = []
    # Whitespace inside the assembly string is immaterial; tokens are not.
    for node in ast.walk(function):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = ' '.join(node.value.split())
    return ast.dump(function, include_attributes=False)


def _is_audited_source(source):
    """True only for the exact audited instruction sequence and operands."""
    try:
        return _function_tree(source) == _function_tree(_AUDITED)
    except (SyntaxError, ValueError):
        return False


def _is_audited_inline_asm(value):
    function = getattr(value, 'fn', None)
    if not callable(value) or not inspect.isfunction(function):
        return False
    try:
        return _is_audited_source(inspect.getsource(function))
    except (OSError, TypeError):
        return False


def _require_cold(modules):
    """Do not replace dependencies of already hashed/compiled Triton callers."""
    for module in modules:
        values = list(vars(module).values())
        for item in tuple(values):
            if inspect.isclass(item) and item.__module__ == module.__name__:
                values.extend(vars(item).values())
        for value in values:
            if isinstance(value, staticmethod):
                value = value.__func__
            fn = getattr(value, 'fn', None)
            if not inspect.isfunction(fn) or fn.__globals__ is not vars(module):
                continue
            if not {SYMBOL, '_quantize_mxfp4_pair'}.intersection(fn.__code__.co_names):
                continue
            fields = vars(value)
            if fields.get('hash') is not None or fields.get('device_caches'):
                raise Fp4PackerCompatibilityError('FP4 caller already hashed/compiled; use a fresh process')


def _fatal(message):
    global _failure
    if _failure is None:
        _failure = message
    raise ExtensionCommitError(_failure)


def _verify():
    if _state['mode'] != _mode():
        raise Fp4PackerCompatibilityError('FP4 policy changed; start a fresh process for stock/kill rollback')
    if _state['mode'] == 'stock':
        return
    for name, module, original in _state['originals']:
        if import_module(name) is not module or getattr(module, SYMBOL, None) is not _state['replacement']:
            raise Fp4PackerCompatibilityError('FP4 binding ownership changed')
    if _state['envs'].environment_variables.get(FACTOR) is not policy_factor:
        raise Fp4PackerCompatibilityError('FP4 compilation factor ownership changed')
    record = registry.get(OP)
    if record is None or record['implementation'] != 'triton' or record['source'] != __name__:
        raise Fp4PackerCompatibilityError('FP4 runtime registration changed')


def policy_factor():
    return _state['factor'] if _state is not None else 'stock'


def check_health():
    with _lock:
        if _failure is not None:
            raise ExtensionCommitError(_failure)
        if _state is None:
            if _mode() != 'stock':
                raise Fp4PackerCompatibilityError('FP4 lowering requested before activation')
            return
        if _state['pid'] != os.getpid():
            raise Fp4PackerCompatibilityError('activate FP4 lowering in the child first')
        try:
            _verify()
        except BaseException as error:
            try:
                _fatal('FP4 lowering health failed; start a fresh process')
            except ExtensionCommitError as refusal:
                raise refusal from error


def activate_extension(context):
    global _state, _publishing
    with _lock:
        if _failure is not None:
            raise ExtensionCommitError(_failure)
        _require_release(context.vllm_version)
        if _publishing:
            _fatal('recursive FP4 lowering activation')
        if _state is not None:
            try:
                _verify()
            except BaseException as error:
                try:
                    _fatal('FP4 lowering retry failed; start a fresh process')
                except ExtensionCommitError as refusal:
                    raise refusal from error
            _state['pid'] = os.getpid()
            return
        mode = _mode()
        if mode == 'stock':
            _state = dict(mode=mode, pid=os.getpid())
            return
        platform = import_module('vllm.platforms').current_platform
        if not (platform.is_rocm() and platform.device_name == 'hcu'
                and hcu_env.arch_name() == 'gfx938'):
            raise Fp4PackerCompatibilityError('FP4 lowering candidate requires real HCU gfx938')
        modules = tuple(import_module(name) for name in OWNERS)
        original = getattr(modules[0], SYMBOL, None)
        if not _is_audited_inline_asm(original):
            raise Fp4PackerCompatibilityError('FP4 definition is not the audited inline assembly')
        originals = tuple(zip(OWNERS, modules, (getattr(m, SYMBOL, None) for m in modules)))
        if any(value is not original for _, _, value in originals):
            raise Fp4PackerCompatibilityError('FP4 caller alias differs from audited definition')
        _require_cold(modules)
        envs = import_module('vllm.envs')
        if FACTOR in envs.environment_variables:
            raise Fp4PackerCompatibilityError('FP4 compilation factor has another owner')
        if registry.get(OP) is not None:
            raise Fp4PackerCompatibilityError('FP4 runtime registration has another owner')
        kernel = import_module('hcu_vllm_das.fp4_packer_triton')
        replacement = kernel.fp32x2_to_fp4x2
        digest = hashlib.sha256()
        for path in (Path(__file__), Path(kernel.__file__)):
            digest.update(path.read_bytes())
        digest.update(inspect.getsource(original.fn).encode())
        versions = (context.vllm_version, import_module('torch').__version__,
                    import_module('triton').__version__)
        digest.update(repr(versions).encode())
        factor = 'e2m1-rne-v1:' + digest.hexdigest()
        # All ABI checks finish before publication. A failure during publication
        # is commit-fatal; never pretend a dirty interpreter was rolled back.
        _publishing = True
        try:
            with registry.registration_transaction():
                registry.register(op=OP, impl='triton', target=OWNERS[0] + '.' + SYMBOL,
                                  mechanism=registry.MECH_REBIND, source=__name__,
                                  detail='033 E2M1 only; six audited aliases; native/model gates pending')
                for _, module, _ in originals:
                    setattr(module, SYMBOL, replacement)
                envs.environment_variables[FACTOR] = policy_factor
                _state = dict(mode=mode, pid=os.getpid(), originals=originals,
                              replacement=replacement, envs=envs, factor=factor)
                _verify()
        except BaseException as error:
            try:
                _fatal('FP4 lowering publication failed; start a fresh process')
            except ExtensionCommitError as refusal:
                raise refusal from error
        finally:
            _publishing = False


def _after_fork():
    global _lock, _failure
    _lock = RLock()
    if _publishing and _failure is None:
        _failure = 'fork inherited incomplete FP4 lowering publication'


if hasattr(os, 'register_at_fork'):
    os.register_at_fork(after_in_child=_after_fork)


def extension_spec():
    capabilities = frozenset({CAPABILITY}) if _mode() == 'triton' else frozenset()
    return ExtensionSpec(name='das-hcu-fp4-packer', api_version=1, priority=120,
                         capabilities=capabilities, activate=activate_extension,
                         check_health=check_health)
