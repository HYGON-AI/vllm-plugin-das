"""Real interpreter resolution under parent-path pollution; stdlib only.

Fixture distributions contain metadata and an import-refusing module. These
tests prove CI identity resolution only, never framework or device support.
"""
import importlib.metadata
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import venv


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('_v033_ci_environment_runner',
                                             ROOT / 'tools/run_v033_ci.py')
runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)


def fake_distribution(site, version, package=True):
    info = site / ('vllm-' + version + '.dist-info')
    info.mkdir(parents=True)
    (info / 'METADATA').write_text('Metadata-Version: 2.1\nName: vllm\nVersion: ' + version + '\n')
    if package:
        module = site / 'vllm'
        module.mkdir()
        (module / '__init__.py').write_text("raise AssertionError('Probe must not import vLLM')\n")


class EnvironmentTests(unittest.TestCase):
    def test_probe_uses_child_installation_in_both_version_directions(self):
        for parent_version, child_version in [('0.32.9', '0.33.0'), ('0.33.0', '0.32.9')]:
            with self.subTest(parent=parent_version, child=child_version), \
                 tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                child = root / 'child'
                venv.EnvBuilder(with_pip=False).create(child)
                child_site = child / 'lib' / ('python%d.%d' % sys.version_info[:2]) / 'site-packages'
                parent_site = root / 'parent'
                fake_distribution(child_site, child_version)
                fake_distribution(parent_site, parent_version)
                environment = dict(os.environ, PYTHONNOUSERSITE='1', HCU_ADAPTER_HOST='none')
                environment.pop('PYTHONPATH', None)
                with patch.object(sys, 'path', [str(parent_site), *sys.path]), \
                     patch.dict(os.environ, PYTHONPATH=str(parent_site)), \
                     patch.object(runner.sys, 'executable', str(child / 'bin/python')):
                    self.assertEqual(importlib.metadata.version('vllm'), parent_version)
                    version, source = runner._installed_vllm(environment)
                self.assertEqual(version, child_version)
                self.assertEqual(source, str(child_site))

    def test_probe_rejects_metadata_without_a_package(self):
        with tempfile.TemporaryDirectory() as tmp:
            child = Path(tmp) / 'child'
            venv.EnvBuilder(with_pip=False).create(child)
            site = child / 'lib' / ('python%d.%d' % sys.version_info[:2]) / 'site-packages'
            fake_distribution(site, '0.33.0', package=False)
            environment = dict(os.environ, PYTHONNOUSERSITE='1', HCU_ADAPTER_HOST='none')
            environment.pop('PYTHONPATH', None)
            with patch.object(runner.sys, 'executable', str(child / 'bin/python')):
                with self.assertRaisesRegex(ValueError, 'importable vLLM'):
                    runner._installed_vllm(environment)


if __name__ == '__main__':
    unittest.main()
