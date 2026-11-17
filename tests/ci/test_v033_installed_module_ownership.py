"""CPU-only ownership negatives for installed Adapter compatibility aliases.

Run with `python -B -m unittest discover -s tests/ci -v`. No framework,
provider or device imports are needed; the real guard reads explicit module
objects and independent installed-file manifests in a temporary directory.
"""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType
import unittest


ROOT = Path(__file__).resolve().parents[2]


class InstalledModuleOwnership(unittest.TestCase):
    def setUp(self):
        spec = spec_from_file_location('_tested_installed_module_ownership',
                                      ROOT / 'tests/vllm033/conftest.py')
        self.guard = module_from_spec(spec)
        spec.loader.exec_module(self.guard)
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.site = Path(self.temp.name) / 'site-packages'
        self.modules = {}
        self.guard._OWNERS = {prefix: set() for prefix in
                               ('vllm', 'hcu_adapter', 'hcu_vllm_das')}
        for name in self.guard._OWNERS:
            self.add_module(name, self.site / name / '__init__.py', name)

    def add_module(self, name, path, owner=None):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# independent file\n')
        module = ModuleType(name)
        module.__file__ = str(path)
        self.modules[name] = module
        if owner:
            self.guard._OWNERS[owner].add(path.resolve())
        return module

    def add_alias(self, alias, target):
        facade = self.site.joinpath(*alias.split('.')).with_suffix('.py')
        facade.parent.mkdir(parents=True, exist_ok=True)
        facade.write_text('# installed compatibility facade\n')
        self.guard._OWNERS['hcu_vllm_das'].add(facade.resolve())
        path = self.site.joinpath(*target.split('.')).with_suffix('.py')
        module = self.add_module(target, path, 'hcu_adapter')
        self.modules[alias] = module
        return facade, module

    def check(self):
        self.guard._verify_installed_modules(self.modules)

    def test_all_declared_aliases_accept_the_exact_installed_adapter_owner(self):
        for alias, target in self.guard._ADAPTER_ALIASES.items():
            with self.subTest(alias=alias):
                self.add_alias(alias, target)
                self.check()

    def test_regular_package_modules_still_require_their_own_record(self):
        self.add_module('hcu_vllm_das.rms_norm',
                        self.site / 'hcu_vllm_das/rms_norm.py', 'hcu_vllm_das')
        self.check()
        self.modules['hcu_vllm_das.rms_norm'].__file__ = str(self.site / 'foreign.py')
        with self.assertRaisesRegex(RuntimeError, 'hcu_vllm_das.rms_norm'):
            self.check()

    def test_undeclared_plugin_alias_to_owned_adapter_module_is_rejected(self):
        target = self.add_module('hcu_adapter.other', self.site / 'hcu_adapter/other.py', 'hcu_adapter')
        self.modules['hcu_vllm_das.unrelated'] = target
        with self.assertRaisesRegex(RuntimeError, 'hcu_vllm_das.unrelated'):
            self.check()

    def test_missing_facade_record_is_rejected(self):
        alias, target = next(iter(self.guard._ADAPTER_ALIASES.items()))
        facade, _ = self.add_alias(alias, target)
        self.guard._OWNERS['hcu_vllm_das'].remove(facade.resolve())
        with self.assertRaisesRegex(RuntimeError, alias):
            self.check()

    def test_missing_adapter_record_is_rejected(self):
        alias, target = next(iter(self.guard._ADAPTER_ALIASES.items()))
        _, module = self.add_alias(alias, target)
        self.guard._OWNERS['hcu_adapter'].remove(Path(module.__file__).resolve())
        with self.assertRaisesRegex(RuntimeError, 'not owned'):
            self.check()

    def test_missing_canonical_adapter_binding_is_rejected(self):
        alias, target = next(iter(self.guard._ADAPTER_ALIASES.items()))
        self.add_alias(alias, target)
        del self.modules[target]
        with self.assertRaisesRegex(RuntimeError, alias):
            self.check()

    def test_different_module_object_at_the_same_path_is_rejected(self):
        alias, target = next(iter(self.guard._ADAPTER_ALIASES.items()))
        _, module = self.add_alias(alias, target)
        foreign = ModuleType(target)
        foreign.__file__ = module.__file__
        self.modules[alias] = foreign
        with self.assertRaisesRegex(RuntimeError, alias):
            self.check()

    def test_alias_to_a_different_owned_adapter_module_is_rejected(self):
        alias, target = next(iter(self.guard._ADAPTER_ALIASES.items()))
        self.add_alias(alias, target)
        other = self.add_module('hcu_adapter.other', self.site / 'hcu_adapter/other.py', 'hcu_adapter')
        self.modules[alias] = other
        with self.assertRaisesRegex(RuntimeError, alias):
            self.check()

    def test_same_name_and_bytes_from_a_foreign_checkout_are_rejected(self):
        alias, target = next(iter(self.guard._ADAPTER_ALIASES.items()))
        _, module = self.add_alias(alias, target)
        original = Path(module.__file__)
        foreign = self.site.parent / 'checkout/hcu_adapter' / original.name
        foreign.parent.mkdir(parents=True)
        foreign.write_bytes(original.read_bytes())
        module.__file__ = str(foreign)
        with self.assertRaisesRegex(RuntimeError, 'not owned'):
            self.check()

    def test_recorded_but_wrong_adapter_file_is_rejected(self):
        alias, target = next(iter(self.guard._ADAPTER_ALIASES.items()))
        _, module = self.add_alias(alias, target)
        other = self.add_module('hcu_adapter.other', self.site / 'hcu_adapter/other.py', 'hcu_adapter')
        module.__file__ = other.__file__
        with self.assertRaisesRegex(RuntimeError, alias):
            self.check()

    def test_changed_canonical_module_name_is_rejected(self):
        alias, target = next(iter(self.guard._ADAPTER_ALIASES.items()))
        _, module = self.add_alias(alias, target)
        module.__name__ = 'hcu_adapter.unexpected'
        with self.assertRaisesRegex(RuntimeError, alias):
            self.check()

    def test_declared_alias_without_a_file_is_rejected(self):
        for alias, target in self.guard._ADAPTER_ALIASES.items():
            with self.subTest(alias=alias):
                _, module = self.add_alias(alias, target)
                del module.__file__
                with self.assertRaisesRegex(RuntimeError, 'not owned'):
                    self.check()
                # Keep every previous alias valid for the next subtest.
                self.add_alias(alias, target)

    def test_declared_alias_cannot_bypass_identity_using_a_plugin_record(self):
        alias, target = next(iter(self.guard._ADAPTER_ALIASES.items()))
        facade, _ = self.add_alias(alias, target)
        self.modules[alias] = ModuleType(alias)
        self.modules[alias].__file__ = str(facade)
        with self.assertRaisesRegex(RuntimeError, alias):
            self.check()


if __name__ == '__main__':
    unittest.main()
