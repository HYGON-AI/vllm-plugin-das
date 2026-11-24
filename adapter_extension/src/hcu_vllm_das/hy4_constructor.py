# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Private child factories while retaining Core's decorated constructor."""
from types import CellType, CodeType, FunctionType


def _names(code):
    names = set(code.co_names)
    for value in code.co_consts:
        if isinstance(value, CodeType):
            names.update(_names(value))
    return names


def _copy(function, namespace, closure):
    result = FunctionType(function.__code__, namespace, function.__name__,
                          function.__defaults__, closure)
    result.__kwdefaults__ = function.__kwdefaults__
    result.__annotations__ = function.__annotations__
    result.__dict__.update(function.__dict__)
    result.__module__ = function.__module__
    result.__qualname__ = function.__qualname__
    result.__doc__ = function.__doc__
    return result


def bind_constructor(function, substitutions):
    """Bind only reviewed child names, never mutate a native global/closure.

    HY4's compiled backbone closes over ``old_init`` in the Core decorator.
    Keep that wrapper bytecode and every other closure cell intact; replacing
    the class's constructor with its undecorated body would bypass compilation
    initialization even when forward still looks native.
    """
    if not isinstance(function, FunctionType) or not substitutions:
        raise RuntimeError('HY4 constructor anchor requires a Python function and bindings')
    names = _names(function.__code__)
    if all(name in names and name in function.__globals__ for name in substitutions):
        namespace = dict(function.__globals__)
        namespace.update(substitutions)
        return _copy(function, namespace, function.__closure__)
    if (function.__module__ == 'vllm.compilation.decorators'
            and 'old_init' in function.__code__.co_freevars):
        position = function.__code__.co_freevars.index('old_init')
        original = function.__closure__[position].cell_contents
        if original is function:
            raise RuntimeError('HY4 constructor anchor contains a recursive wrapper')
        bound = bind_constructor(original, substitutions)
        closure = list(function.__closure__)
        closure[position] = CellType(bound)
        return _copy(function, function.__globals__, tuple(closure))
    raise RuntimeError('HY4 native child constructor anchor changed')
