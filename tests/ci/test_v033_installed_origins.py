"""Real stdlib subprocess import-policy tests; packages are CPU-only fixtures."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


GUARD = Path(__file__).resolve().parents[2] / 'tests/vllm033/conftest.py'


class InstalledOrigins(unittest.TestCase):
    def probe(self, mode, layout="legacy"):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            site = root / 'installed'
            source = root / 'source'
            packages=[('vllm','vllm'),('hcu_vllm_das_plugin','hcu_vllm_das')]
            packages += [('sgl_kernel','hcu_adapter')] if layout=='legacy' else [('sgl_kernel','sgl_kernel'),('hcu_adapter','hcu_adapter')]
            if mode == 'pure-kernel':
                packages.remove(('hcu_adapter', 'hcu_adapter'))
            if mode == 'host-only':
                packages.remove(('sgl_kernel', 'sgl_kernel'))
            for distribution, package in packages:
                module = site / package
                module.mkdir(parents=True)
                (module / '__init__.py').write_text('LABEL = "installed"\n')
                info = site / (distribution + '-1.0.dist-info')
                info.mkdir()
                (info / 'METADATA').write_text('Name: ' + distribution.replace('_', '-') + '\nVersion: 1.0\n')
                (info / 'RECORD').write_text(package + '/__init__.py,,\n' + info.name + '/METADATA,,\n')
            if mode=='duplicate':
                record=site/'sgl_kernel-1.0.dist-info/RECORD'
                record.write_text(record.read_text()+'hcu_adapter/__init__.py,,\n')
            if mode == 'broken-modern':
                (site / 'hcu_adapter-1.0.dist-info/RECORD').write_text('')
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
    if MODE == 'broken-modern':
        assert 'hcu-adapter has no installed RECORD' in str(error), str(error)
    else:
        assert MODE in ('shadow-before', 'foreign-after', 'duplicate', 'pure-kernel'), str(error)
        assert 'not owned' in str(error), str(error)
else:
    assert MODE in ('clean', 'path-after', 'host-only'), 'unowned source was accepted'
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

    def test_modern_split_adapter_is_owned_by_host_dist(self):
        self.probe('clean','split')

    def test_split_guard_keeps_source_shadow_refusal(self):
        self.probe('shadow-before','split')

    def test_split_guard_refuses_duplicate_host_record_owners(self):
        self.probe('duplicate','split')

    def test_pure_kernel_does_not_supply_host(self):
        self.probe('pure-kernel', 'split')

    def test_broken_modern_host_is_not_hidden_by_kernel_fallback(self):
        self.probe('broken-modern', 'split')

    def test_host_only_metadata_supports_installed_guard(self):
        self.probe('host-only', 'split')
