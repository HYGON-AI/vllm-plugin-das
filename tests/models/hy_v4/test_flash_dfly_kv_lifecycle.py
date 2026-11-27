# SPDX-License-Identifier: Apache-2.0
"""Reuse lifecycle regression, not whole-model/Graph acceptance.

Default tests only inspect production AST. Native lane explicitly opts in with
HYV4_KV_LIFECYCLE_CPU=1, hides all GPUs, and imports actual Torch/Core types and
Core's grouping helper. Exact reuse methods are extracted without importing the
model/plugin tree. Model construction, allocation/binding and discovery are
explicit fixtures. HYV4_KV_OLD_SOURCE pins an unchanged old-source control.
"""
from __future__ import annotations
import ast
from copy import deepcopy
import os
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

SOURCE = Path(os.environ.get('HYV4_KV_SOURCE', str(
    Path(__file__).resolve().parents[3] /
    'vllm_hcu/models/hy_v4_flash/nvidia/dfly_target_reuse.py')))
METHODS = ('_set_storage_aliases', 'validate_groups', 'validate_storage')


def tree(path=SOURCE):
    return next(n for n in ast.parse(path.read_text(encoding='utf-8-sig')).body
                if isinstance(n, ast.ClassDef) and n.name == 'HYV4DFlyTargetReuse')


def load_methods(path, config_type):
    cls = tree(path)
    cls.bases = []
    cls.decorator_list = []
    cls.body = [deepcopy(n) for n in cls.body
                if isinstance(n, ast.FunctionDef) and n.name in METHODS]
    if {n.name for n in cls.body} != set(METHODS):
        raise AssertionError('Production reuse interface changed')
    namespace = {'KVCacheConfig': config_type}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])),
                 str(path), 'exec'), namespace)
    return namespace[cls.name]


class SourceContracts(unittest.TestCase):
    def test_storage_validation_does_not_disable_allocation_alias(self):
        method = next(n for n in tree().body if isinstance(n, ast.FunctionDef)
                      and n.name == 'validate_storage')
        self.assertNotIn('self._set_storage_aliases(False)', ast.unparse(method))

    def test_storage_and_group_guards_are_preserved(self):
        source = ast.unparse(tree())
        for check in ('a.numel() == 0', 'a.data_ptr() != b.data_ptr()',
                      'a.shape != b.shape', 'a.stride() != b.stride()',
                      'a.dtype != b.dtype', 'groups[draft] != groups.get(target)'):
            with self.subTest(check=check):
                self.assertIn(check, source)


@unittest.skipUnless(os.environ.get('HYV4_KV_LIFECYCLE_CPU') == '1',
                     'requires explicit real Torch/Core CPU lane')
class TorchCoreLifecycle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for mask in ('HIP_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES',
                     'ROCR_VISIBLE_DEVICES', 'GPU_DEVICE_ORDINAL',
                     'NVIDIA_VISIBLE_DEVICES'):
            if os.environ.get(mask) != '-1':
                raise RuntimeError(f'{mask} must be -1')
        import torch
        from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, MLAAttentionSpec
        from vllm.v1.worker.utils import add_kv_sharing_layers_to_kv_cache_groups
        cls.torch = torch
        cls.config_type = KVCacheConfig
        cls.group_type = KVCacheGroupSpec
        cls.spec = MLAAttentionSpec(block_size=64, num_kv_heads=1,
                                   head_size=8, dtype=torch.bfloat16)
        cls.add = staticmethod(add_kv_sharing_layers_to_kv_cache_groups)
        cls.new = load_methods(SOURCE, KVCacheConfig)
        old = Path(os.environ['HYV4_KV_OLD_SOURCE'])
        cls.old = load_methods(old, KVCacheConfig)

    def fixture(self, reuse_type):
        a = self.torch.zeros((2, 64, 8), dtype=self.torch.bfloat16)
        source = NS(layer_name='target', kv_cache=a)
        draft = NS(layer_name='draft', kv_cache=a,
                   kv_sharing_target_layer_name=None)
        reuse = reuse_type()
        reuse.layer_pairs = [(NS(mla_attn=source), NS(mla_attn=draft))]
        reuse.aliases = {'draft': 'target'}
        reuse._set_storage_aliases(True)
        return reuse, source, draft

    def phase(self, reuse, source, draft):
        config = self.config_type(2, [], [self.group_type(['target'], self.spec)])
        # Discovery fixture reads the same module marker as Core's owned route.
        target = draft.kv_sharing_target_layer_name
        aliases = {'draft': target} if target is not None else {}
        self.add(aliases, config.kv_cache_groups)
        reuse.validate_groups(config)
        # No real allocator/model binding in this focused AST lane.
        source.kv_cache = self.torch.zeros_like(source.kv_cache)
        draft.kv_cache = source.kv_cache
        reuse.validate_storage()

    def test_old_source_reproduces_second_initialization_failure(self):
        reuse, source, draft = self.fixture(self.old)
        self.phase(reuse, source, draft)
        self.assertIsNone(draft.kv_sharing_target_layer_name)
        with self.assertRaisesRegex(ValueError, 'must share a KV group'):
            self.phase(reuse, source, draft)

    def test_new_source_preserves_profile_final_and_reinitialization(self):
        reuse, source, draft = self.fixture(self.new)
        for name in ('profile', 'final', 'reinitialize'):
            with self.subTest(phase=name):
                self.phase(reuse, source, draft)
                self.assertEqual(draft.kv_sharing_target_layer_name, 'target')

    def test_non_alias_cache_is_still_rejected(self):
        reuse, source, draft = self.fixture(self.new)
        draft.kv_cache = source.kv_cache.clone()
        with self.assertRaisesRegex(ValueError, 'same local layout'):
            reuse.validate_storage()

    def test_wrong_group_is_still_rejected(self):
        reuse, source, draft = self.fixture(self.new)
        config = self.config_type(2, [], [self.group_type(['target'], self.spec),
                                        self.group_type(['draft'], self.spec)])
        with self.assertRaisesRegex(ValueError, 'must share a KV group'):
            reuse.validate_groups(config)

    def tearDown(self):
        self.assertFalse(self.torch.cuda.is_initialized())


if __name__ == '__main__':
    unittest.main(verbosity=2)
