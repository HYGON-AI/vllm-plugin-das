# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Native constructor composition, preserving the Core compile wrapper."""
import inspect
from types import FunctionType

import pytest

from hcu_vllm_das.hy4_constructor import bind_constructor


def test_direct_and_nested_factory_lookups_are_private():
    class Original:
        pass
    class Owned:
        pass
    namespace = {'Child': Original}
    exec('def native(value=3):\n return Child(), (lambda: Child()), value\n', namespace)
    native = namespace['native']
    bound = bind_constructor(native, {'Child': Owned})
    first, factory, value = bound()
    assert type(first) is Owned and type(factory()) is Owned and value == 3
    assert type(native()[0]) is Original
    assert bound.__code__ is native.__code__
    assert inspect.signature(bound) == inspect.signature(native)
    with pytest.raises(RuntimeError, match='anchor'):
        bind_constructor(native, {'Missing': Owned})


def test_compile_wrapper_and_other_closure_cells_are_retained():
    namespace = {'Child': lambda: 'original'}
    exec('def native():\n return Child()\n', namespace)
    old_init = namespace['native']
    marker = []
    def wrapper():
        marker.append('wrapper')
        return old_init()
    wrapper.__module__ = 'vllm.compilation.decorators'
    bound = bind_constructor(wrapper, {'Child': lambda: 'owned'})
    assert bound() == 'owned'
    assert marker == ['wrapper']
    assert wrapper() == 'original'
    assert bound.__code__ is wrapper.__code__
    cells = dict(zip(wrapper.__code__.co_freevars, wrapper.__closure__))
    other = dict(zip(bound.__code__.co_freevars, bound.__closure__))
    assert cells['marker'] is other['marker']
    assert cells['old_init'] is not other['old_init']


def test_foreign_wrapper_is_not_unwrapped_by_guessing():
    namespace = {'Child': object}
    exec('def native():\n return Child()\n', namespace)
    old_init = namespace['native']
    def foreign():
        return old_init()
    with pytest.raises(RuntimeError, match='anchor'):
        bind_constructor(foreign, {'Child': object})


def test_actual_core_hy4_compile_wrapper_stays_present():
    pytest.importorskip('vllm', reason='requires actual installed Core')
    from vllm.models.hy_v4.amd.model import HYV4Model
    native = HYV4Model.__init__
    replacement = object()
    bound = bind_constructor(native, {'HYV4DecoderLayer': replacement})
    assert native.__module__ == 'vllm.compilation.decorators'
    assert bound.__code__ is native.__code__
    assert inspect.signature(bound) == inspect.signature(native)
    before = dict(zip(native.__code__.co_freevars, native.__closure__))
    after = dict(zip(bound.__code__.co_freevars, bound.__closure__))
    for name in before:
        if name != 'old_init':
            assert before[name] is after[name]
    assert after['old_init'].cell_contents.__globals__['HYV4DecoderLayer'] is replacement
    assert before['old_init'].cell_contents.__globals__['HYV4DecoderLayer'] is not replacement
