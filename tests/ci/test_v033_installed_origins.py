"""Real stdlib subprocess import-policy tests; packages are CPU-only fixtures."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


GUARD = Path(__file__).resolve().parents[2] / 'tests/vllm033/conftest.py'


class InstalledOrigins(unittest.TestCase):
    def probe(self, mode):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            site = root / 'installed'
            source = root / 'source'
            for distribution, package in [('vllm', 'vllm'), ('sgl_kernel', 'hcu_adapter'),
                                          ('hcu_vllm_das_plugin', 'hcu_vllm_das')]:
                module = site / package
                module.mkdir(parents=True)
                (module / '__init__.py').write_text('LABEL = "installed"\n')
                info = site / (distribution + '-1.0.dist-info')
                info.mkdir()
                (info / 'METADATA').write_text('Name: ' + distribution.replace('_', '-') + '\nVersion: 1.0\n')
                (info / 'RECORD').write_text(package + '/__init__.py,,\n' + info.name + '/METADATA,,\n')
            foreign = source / 'hcu_vllm_das'
            foreign.mkdir(parents=True)
            (foreign / '__init__.py').write_text('LABEL = "source"\n')
            (foreign / 'foreign.py').write_text('LABEL = "source"\n')
            script = '''
import importlib.util, os, sys
os.environ['HCU_V033_VERIFY_INSTALLED'] = '1'
sys.path.insert(0, SITE)
spec = importlib.util.spec_from_file_location('installed_guard', GUARD)
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)
if MODE == 'shadow-before': sys.path.insert(0, SOURCE)
try:
    guard.pytest_sessionstart(None)
    if MODE == 'foreign-after':
        spec = importlib.util.spec_from_file_location('hcu_vllm_das.foreign', FOREIGN)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    if MODE == 'path-after': sys.path.insert(0, SOURCE)
    guard.pytest_collection_finish(None)
except RuntimeError as error:
    assert MODE in ('shadow-before', 'foreign-after'), str(error)
    assert 'not owned' in str(error), str(error)
else:
    assert MODE in ('clean', 'path-after'), 'unowned source was accepted'
    import hcu_vllm_das
    assert hcu_vllm_das.LABEL == 'installed'
'''
            preamble = '\n'.join(name + ' = ' + repr(value) for name, value in dict(
                SITE=str(site), SOURCE=str(source), GUARD=str(GUARD),
                FOREIGN=str(foreign / 'foreign.py'), MODE=mode).items())
            result = subprocess.run([sys.executable, '-S', '-B', '-c', preamble + script],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_clean_installed_import(self):
        self.probe('clean')

    def test_source_before_session_is_rejected(self):
        self.probe('shadow-before')

    def test_foreign_submodule_after_session_is_rejected(self):
        self.probe('foreign-after')

    def test_source_path_addition_keeps_locked_installed_package(self):
        self.probe('path-after')
