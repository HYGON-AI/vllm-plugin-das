"""Stdlib-only contracts for files containing CPU and HCU cases together."""
import importlib.util
import ast
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    'v033_mixed_tier_runner', ROOT / 'tools/run_v033_ci.py')
RUNNER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RUNNER
SPEC.loader.exec_module(RUNNER)


class MixedTierContracts(unittest.TestCase):
    def test_hcu_named_modules_declare_hardware_marker(self):
        for path in sorted((ROOT / 'tests/vllm033').glob('*_hcu.py')):
            tree = ast.parse(path.read_text())
            values = [node.value for node in tree.body if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == 'pytestmark'
                              for target in node.targets)]
            with self.subTest(module=path.name):
                self.assertTrue(any(
                    isinstance(node, ast.Attribute) and ast.unparse(node) == 'pytest.mark.hcu'
                    for value in values for node in ast.walk(value)))

    def test_both_tiers_discover_a_mixed_test_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tests = root / 'tests/vllm033'
            tests.mkdir(parents=True)
            (tests / 'test_mixed.py').write_text(
                'def test_cpu(): pass\n\n@pytest.mark.hcu\ndef test_device(): pass\n')
            (tests / 'test_native_hcu.py').write_text('pytestmark = pytest.mark.hcu\n')
            (tests / 'test_cpu_only.py').write_text('import sys\nsys.path.insert(0, "/source")\n')
            with patch.object(RUNNER, 'ROOT', root):
                for tier in ('portable', 'device'):
                    self.assertIn('tests/vllm033/test_mixed.py', RUNNER.test_files(tier))
                self.assertNotIn('tests/vllm033/test_cpu_only.py', RUNNER.test_files('device'))

    def run_tier(self, tier):
        with patch.object(sys, 'argv', ['run_v033_ci.py', '--tier', tier]), \
             patch.object(RUNNER, 'test_files', return_value=['tests/vllm033/test_mixed.py']), \
             patch.object(RUNNER, '_installed_vllm', return_value=('0.33.0', '/installed')), \
             patch.object(RUNNER, '_vllm_release', return_value=(0, 33, 0)), \
             patch.object(RUNNER, '_run_public_selector_cases'), \
             patch.object(RUNNER, 'validate_result'), \
             patch.object(RUNNER.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0)) as run:
            self.assertEqual(RUNNER.main(), 0)
            return next(call for call in run.call_args_list if 'pytest' in call.args[0])

    def test_portable_filters_device_markers_in_mixed_files(self):
        call = self.run_tier('portable')
        command = call.args[0]
        self.assertIn('-m', command[4:])
        self.assertEqual(command[command.index('-m', 4) + 1], 'not hcu')

    def test_device_filters_hcu_and_enables_slimquant_opt_in(self):
        call = self.run_tier('device')
        command = call.args[0]
        self.assertIn('-m', command[4:])
        self.assertEqual(command[command.index('-m', 4) + 1], 'hcu')
        self.assertEqual(call.kwargs['env'].get('HCU_SLIMQUANT_LIVE'), '1')

    def test_installed_tiers_require_record_origin_guard(self):
        for tier in ('portable', 'device'):
            self.assertEqual(self.run_tier(tier).kwargs['env'].get(
                'HCU_V033_VERIFY_INSTALLED'), '1')

    def test_source_discards_inherited_installed_guard(self):
        with patch.dict(os.environ, {'HCU_V033_VERIFY_INSTALLED': '1',
                                     'VLLM_TARGET_ROOT': '/source'}):
            self.assertEqual(self.run_tier('source').kwargs['env'].get(
                'HCU_V033_VERIFY_INSTALLED'), '0')


if __name__ == '__main__':
    unittest.main()
