# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Native ByteLevel stream checks; no model, Torch or device imports."""
from types import SimpleNamespace

import pytest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers

from hcu_vllm_das.hy4_detokenizer import choose_prompt_tail, install


def tokenizer():
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    result = Tokenizer(models.BPE({c: i for i, c in enumerate(alphabet)}, []))
    result.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    result.decoder = decoders.ByteLevel()
    return result


def test_three_byte_boundary_has_bounded_work():
    t = tokenizer()
    ids = t.encode('汉' * 28333).ids
    tail = choose_prompt_tail(t, ids, True)
    assert len(tail) == 33
    assert t.decode(tail) == '汉' * 11


@pytest.mark.parametrize('text', ['汉', 'abc', '😀', 'a汉😀', '\ufffd'])
def test_native_stream_continuations_match_complete_prompt(text):
    t = tokenizer()
    ids = t.encode(text * 80).ids
    shortened = decoders.DecodeStream(ids=choose_prompt_tail(t, ids, True))
    original = decoders.DecodeStream(ids=ids)
    for token in t.encode('汉文🙂 x').ids:
        assert shortened.step(t, token) == original.step(t, token)


def test_budget_and_fallback_keep_original_identity():
    class AlwaysReplacement:
        decoder = decoders.ByteLevel()
        def __init__(self):
            self.sizes = []
        def decode(self, ids, **kwargs):
            self.sizes.append(len(ids))
            return '\ufffd'
    t = AlwaysReplacement()
    ids = [0] * 85000
    assert choose_prompt_tail(t, ids, True) is ids
    assert len(t.sizes) == 8 and sum(t.sizes) == 396


def test_constructor_is_idempotent_and_does_not_mutate_request():
    t = tokenizer()
    class Detokenizer:
        def __init__(self, tokenizer, request):
            self.tokenizer = tokenizer._tokenizer
            self.skip_special_tokens = True
            self.stream = decoders.DecodeStream(ids=request.prompt_token_ids)
    module = SimpleNamespace(FastIncrementalDetokenizer=Detokenizer)
    original = Detokenizer.__init__
    assert install(module)
    wrapper = Detokenizer.__init__
    assert not install(module)
    assert Detokenizer.__init__ is wrapper and wrapper is not original
    ids = t.encode('汉' * 100).ids
    request = SimpleNamespace(prompt_token_ids=ids)
    result = Detokenizer(SimpleNamespace(_tokenizer=t), request)
    assert request.prompt_token_ids is ids
    reference = decoders.DecodeStream(ids=ids)
    for token in t.encode('next 汉').ids:
        assert result.stream.step(t, token) == reference.step(t, token)


def test_replaced_stream_is_not_overwritten():
    class ForeignStream:
        pass
    class Detokenizer:
        def __init__(self, tokenizer, request):
            self.tokenizer = tokenizer._tokenizer
            self.skip_special_tokens = True
            self.stream = ForeignStream()
    module = SimpleNamespace(FastIncrementalDetokenizer=Detokenizer)
    install(module)
    result = Detokenizer(SimpleNamespace(_tokenizer=tokenizer()), SimpleNamespace(prompt_token_ids=[1] * 100))
    assert type(result.stream) is ForeignStream


def test_extension_default_off_does_not_import_core(monkeypatch):
    import importlib
    from hcu_adapter.extensions import ExtensionContext
    extension = importlib.reload(importlib.import_module('hcu_vllm_das.hy4_frontend'))
    monkeypatch.delenv('VLLM_HCU_ENABLE_BYTELEVEL_DETOKENIZER_TAIL', raising=False)
    monkeypatch.setattr(extension, 'import_module', lambda name: pytest.fail(name))
    extension.activate(ExtensionContext('0.5.22', '0.33.0'))
    extension.check_health()


def test_extension_rejects_invalid_flag_before_publication(monkeypatch):
    import importlib
    from hcu_adapter.extensions import ExtensionContext
    extension = importlib.reload(importlib.import_module('hcu_vllm_das.hy4_frontend'))
    monkeypatch.setenv('VLLM_HCU_ENABLE_BYTELEVEL_DETOKENIZER_TAIL', 'typo')
    with pytest.raises(ValueError, match='0 or 1'):
        extension.activate(ExtensionContext('0.5.22', '0.33.0'))


def test_required_capability_cannot_pass_with_disabled_provider(monkeypatch):
    import importlib
    from hcu_adapter.extensions import ExtensionContext
    extension = importlib.reload(importlib.import_module('hcu_vllm_das.hy4_frontend'))
    monkeypatch.setenv('VLLM_HCU_ENABLE_BYTELEVEL_DETOKENIZER_TAIL', '0')
    with pytest.raises(ValueError, match='requires'):
        extension.activate(ExtensionContext('0.5.22', '0.33.0',
            requested_capabilities={'bytelevel-prompt-tail'}))


def bind_host(extension, context):
    from hcu_adapter.extensions import activate_extensions, ExtensionCommitError
    from hcu_adapter.infra.runtime_lifecycle import ProcessLocalLifecycle
    host = ProcessLocalLifecycle(runtime_action='runtime', error_type=ExtensionCommitError)
    def start():
        reports = activate_extensions(context,
            entry_points=[SimpleNamespace(name='hy4', load=lambda: extension.extension_spec)],
            health_check_sink=host.record_extension_health)
        host.record_extension_reports(reports)
    host.run_once('runtime', start)
    return host


def test_extension_owns_publication_and_latches_drift(monkeypatch):
    import importlib
    from hcu_adapter.extensions import ExtensionContext, ExtensionCommitError
    extension = importlib.reload(importlib.import_module('hcu_vllm_das.hy4_frontend'))
    class Detokenizer:
        def __init__(self, tokenizer, request):
            pass
    module = SimpleNamespace(FastIncrementalDetokenizer=Detokenizer)
    monkeypatch.setenv('VLLM_HCU_ENABLE_BYTELEVEL_DETOKENIZER_TAIL', '1')
    monkeypatch.setattr(extension, 'import_module', lambda name: module)
    context = ExtensionContext('0.5.22', '0.33.1+vendor')
    host = bind_host(extension, context)
    constructor = Detokenizer.__init__
    extension.activate(context)
    assert Detokenizer.__init__ is constructor
    Detokenizer.__init__ = lambda *args: None
    with pytest.raises(ExtensionCommitError):
        host.require_extension_healthy('bytelevel-prompt-tail')
    Detokenizer.__init__ = constructor
    with pytest.raises(ExtensionCommitError):
        host.require_extension_healthy('bytelevel-prompt-tail')


@pytest.mark.parametrize('value', ['0', 'bad'])
def test_extension_cannot_recover_from_published_policy_drift(monkeypatch, value):
    import importlib
    from hcu_adapter.extensions import ExtensionContext, ExtensionCommitError
    extension = importlib.reload(importlib.import_module('hcu_vllm_das.hy4_frontend'))
    class Detokenizer:
        def __init__(self, tokenizer, request):
            pass
    module = SimpleNamespace(FastIncrementalDetokenizer=Detokenizer)
    monkeypatch.setenv('VLLM_HCU_ENABLE_BYTELEVEL_DETOKENIZER_TAIL', '1')
    monkeypatch.setattr(extension, 'import_module', lambda name: module)
    context = ExtensionContext('0.5.22', '0.33.0')
    host = bind_host(extension, context)
    monkeypatch.setenv('VLLM_HCU_ENABLE_BYTELEVEL_DETOKENIZER_TAIL', value)
    with pytest.raises(ExtensionCommitError):
        host.require_extension_healthy('bytelevel-prompt-tail')
    monkeypatch.setenv('VLLM_HCU_ENABLE_BYTELEVEL_DETOKENIZER_TAIL', '1')
    with pytest.raises(ExtensionCommitError):
        host.require_extension_healthy('bytelevel-prompt-tail')


@pytest.mark.parametrize('prefix', ['汉', 'abc', '😀', '\ufffd'])
@pytest.mark.parametrize('stop', [None, 'END', '汉'])
@pytest.mark.parametrize('minimum', [0, 3])
@pytest.mark.parametrize('include', [False, True])
def test_target_033_real_constructor_and_stop_updates(prefix, stop, minimum, include):
    """Execute the pinned Core class bodies, isolating imports, not their logic."""
    import ast
    import os
    import sys
    from abc import ABC, abstractmethod
    from bisect import bisect_left
    from pathlib import Path
    import tokenizers
    core = os.environ.get('VLLM_TARGET_ROOT')
    if not core:
        pytest.skip('set VLLM_TARGET_ROOT to replay the real 0.33 source')
    source = Path(core) / 'vllm/v1/engine/detokenizer.py'
    parsed = ast.parse(source.read_text(encoding='utf-8'))
    names = {'IncrementalDetokenizer', 'BaseIncrementalDetokenizer',
             'FastIncrementalDetokenizer', 'check_stop_strings'}
    nodes = [n for n in parsed.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    assert {n.name for n in nodes} == names
    nodes.insert(0, ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0))
    replay = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    namespace = dict(sys=sys, ABC=ABC, abstractmethod=abstractmethod, bisect_left=bisect_left,
                     tokenizers=tokenizers, INVALID_PREFIX_ERR_MSG='Invalid prefix encountered')
    exec(compile(replay, str(source), 'exec'), namespace)
    cls = namespace['FastIncrementalDetokenizer']
    t = tokenizer()
    request = SimpleNamespace(request_id='test', prompt_token_ids=t.encode(prefix * 100).ids,
        sampling_params=SimpleNamespace(stop=stop, min_tokens=minimum,
            include_stop_str_in_output=include, skip_special_tokens=True,
            spaces_between_special_tokens=True))
    baseline = cls(SimpleNamespace(_tokenizer=t), request)
    install(SimpleNamespace(FastIncrementalDetokenizer=cls))
    candidate = cls(SimpleNamespace(_tokenizer=t), request)
    tokens = t.encode('hello 汉END world').ids
    for i in range(0, len(tokens), 3):
        batch = tokens[i:i + 3]
        assert candidate.update(batch, False) == baseline.update(batch, False)
        assert candidate.output_token_ids == baseline.output_token_ids
        assert candidate.get_next_output_text(False, True) == baseline.get_next_output_text(False, True)
    assert candidate.get_next_output_text(True, False) == baseline.get_next_output_text(True, False)
