# SPDX-License-Identifier: Apache-2.0
"""CPU publication contracts with a synthetic registry; no kernel claims."""
from concurrent.futures import ThreadPoolExecutor
import os
from threading import RLock
from types import SimpleNamespace

import pytest

from hcu_adapter.extensions import ExtensionCommitError, ExtensionContext
from hcu_vllm_das import slimquant_activation as activation


@pytest.fixture
def registry(monkeypatch):
    state = dict(owner=None, calls=0, missing=False, commit_failure=False)
    config = type('FormatConfig', (), {})

    def get(name):
        assert name == activation.METHOD
        if state['owner'] is None:
            raise ValueError('unknown method')
        return state['owner']

    def register(name):
        assert name == activation.METHOD
        def publish(cls):
            state['calls'] += 1
            state['owner'] = cls
            if state['commit_failure']:
                raise RuntimeError('failure after registry write')
            return cls
        return publish

    public = SimpleNamespace(get_quantization_config=get,
                             register_quantization_config=register)

    def load(name):
        if name == 'vllm.models.qwen4_exp.common.ngram_embedding':
            return SimpleNamespace(Qwen4ExpPLEPerRowEmbeddingMethod=
                                   None if state['missing'] else object)
        if name == 'hcu_vllm_das.slimquant':
            return SimpleNamespace(SlimQuantConfig=config)
        assert name == 'vllm.model_executor.layers.quantization'
        return public

    monkeypatch.setattr(activation, '_published', None)
    monkeypatch.setattr(activation, '_failure', False)
    monkeypatch.setattr(activation, '_activating', False)
    monkeypatch.setattr(activation, '_lock', RLock())
    monkeypatch.setattr(activation, 'import_module', load)
    return state, config


def context(required=True):
    return ExtensionContext('0.1.0', '0.33.0', requested_capabilities=
                            {activation.CAPABILITY} if required else ())


def test_explicit_capability_required_before_publication(registry):
    state, _ = registry
    with pytest.raises(RuntimeError, match='explicit'):
        activation.activate(context(False))
    assert state['calls'] == 0


def test_missing_core_api_refuses_before_publication_and_allows_clean_retry(registry):
    state, config = registry
    state['missing'] = True
    with pytest.raises(RuntimeError, match='public per-row PLE Core API'):
        activation.activate(context())
    assert state['calls'] == 0 and not activation._failure
    state['missing'] = False
    activation.activate(context())
    assert state['owner'] is config


def test_foreign_owner_is_never_replaced(registry):
    state, _ = registry
    state['owner'] = object()
    with pytest.raises(RuntimeError, match='another owner'):
        activation.activate(context())
    assert state['calls'] == 0


def test_concurrent_publication_is_idempotent(registry):
    state, config = registry
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(activation.activate, [context()] * 8))
    activation.check_health()
    assert state['owner'] is config and state['calls'] == 1
    assert activation.extension_spec().check_health is activation.check_health


def test_write_then_failure_permanently_refuses_reactivation(registry):
    state, _ = registry
    state['commit_failure'] = True
    with pytest.raises(ExtensionCommitError):
        activation.activate(context())
    state['commit_failure'] = False
    with pytest.raises(ExtensionCommitError):
        activation.activate(context())
    assert state['calls'] == 1 and not activation._activating


def test_owner_drift_is_reported_without_republication(registry):
    state, _ = registry
    activation.activate(context())
    state['owner'] = object()
    with pytest.raises(ExtensionCommitError, match='owner changed'):
        activation.check_health()
    with pytest.raises(ExtensionCommitError):
        activation.activate(context())
    assert state['calls'] == 1


@pytest.mark.skipif(not hasattr(os, 'fork'), reason='requires actual CPU fork')
@pytest.mark.parametrize('in_flight', [False, True])
def test_fork_preserves_idle_publication_and_refuses_in_flight_child(registry, in_flight):
    import signal
    activation.activate(context())
    activation._activating = in_flight
    pid = os.fork()
    if pid == 0:
        signal.alarm(3)
        try:
            activation.activate(context())
            activation.check_health()
        except ExtensionCommitError:
            os._exit(0 if in_flight else 2)
        except BaseException:
            os._exit(3)
        os._exit(4 if in_flight else 0)
    _, status = os.waitpid(pid, 0)
    activation._activating = False
    assert os.waitstatus_to_exitcode(status) == 0
    activation.check_health()
