# SPDX-License-Identifier: Apache-2.0

import os
import pickle
from unittest import TestCase, mock

from hcu_vllm_das.model_runtime_activation import (
    ModelRuntimeActivation,
    ModelRuntimeConfig,
)

_activation = None


class _Config:
    def __init__(self, name):
        self.name = name

    @classmethod
    def from_dict(cls, values, **kwargs):
        return cls(values['name']), kwargs


class _OwnedConfig(ModelRuntimeConfig, _Config):
    @classmethod
    def request_runtime(cls):
        _activation.request()


class ModelRuntimeActivationTests(TestCase):
    def test_activation_runs_only_for_a_matching_config(self):
        calls = []
        runtime = ModelRuntimeActivation(calls.append)
        with mock.patch.dict(globals(), _activation=runtime):
            runtime.bind('adapter-context')
            _Config.from_dict({'name': 'other'})
            self.assertEqual(calls, [])
            cfg, unused = _OwnedConfig.from_dict({'name': 'owned'}, keep=True)
            self.assertEqual((cfg.name, unused), ('owned', {'keep': True}))
            _OwnedConfig.from_dict({'name': 'owned again'})
            self.assertEqual(calls, ['adapter-context'])


    def test_serialized_model_requests_activation_before_runtime_binding(self):
        payload = pickle.dumps(_OwnedConfig('owned'))
        calls = []
        runtime = ModelRuntimeActivation(calls.append)
        with mock.patch.dict(globals(), _activation=runtime):
            restored = pickle.loads(payload)
            self.assertEqual(restored.name, 'owned')
            self.assertEqual(calls, [])
            runtime.bind('child-context')
            self.assertEqual(calls, ['child-context'])


    def test_already_bound_process_activates_on_deserialization(self):
        calls = []
        runtime = ModelRuntimeActivation(calls.append)
        with mock.patch.dict(globals(), _activation=runtime):
            runtime.bind('child-context')
            pickle.loads(pickle.dumps(_OwnedConfig('owned')))
            self.assertEqual(calls, ['child-context'])

    def test_spawn_default_preserves_an_explicit_method(self):
        key = 'VLLM_WORKER_MULTIPROC_METHOD'
        for explicit in (None, 'fork'):
            with self.subTest(explicit=explicit), mock.patch.dict(os.environ):
                if explicit is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = explicit
                runtime = ModelRuntimeActivation(lambda _: None, spawn_by_default=True)
                runtime.bind('adapter-context')
                self.assertEqual(os.environ.get(key), explicit)
                runtime.request()
                self.assertEqual(os.environ[key], explicit or 'spawn')

    def _check_after_fork(self, runtime, check):
        if not hasattr(os, 'fork'):
            runtime._after_fork()
            check()
            return
        pid = os.fork()
        if pid == 0:
            status = 1
            try:
                check()
                status = 0
            finally:
                os._exit(status)
        _, status = os.waitpid(pid, 0)
        self.assertEqual(status, 0)

    def test_fork_discards_the_parent_host_context(self):
        calls = []
        runtime = ModelRuntimeActivation(calls.append)
        runtime.bind('parent')
        def check():
            runtime.request()
            self.assertEqual(calls, [])
            runtime.bind('child')
            self.assertEqual(calls, ['child'])
        self._check_after_fork(runtime, check)

    def test_fork_rejects_an_inherited_publication(self):
        runtime = ModelRuntimeActivation(lambda _: None)
        runtime.bind('parent')
        runtime.request()
        def check():
            with self.assertRaisesRegex(RuntimeError, 'use spawn'):
                runtime.request()
            with self.assertRaisesRegex(RuntimeError, 'use spawn'):
                runtime.bind('child')
        self._check_after_fork(runtime, check)
