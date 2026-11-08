"""Opt-in RECORD ownership checks for installed CI; no provider computation."""
from importlib import import_module, metadata
import json
import os
from pathlib import Path
import sys


_OWNERS = {}


def _verify_installed_modules():
    for prefix, owned in _OWNERS.items():
        if prefix not in sys.modules:
            raise RuntimeError(prefix + ' is not owned by the installed distribution')
        for name, module in tuple(sys.modules.items()):
            if name != prefix and not name.startswith(prefix + '.'):
                continue
            origin = getattr(module, '__file__', None)
            if origin is None and name != prefix:
                continue  # Some Core dispatcher/namespace modules have no file.
            if not origin or Path(origin).resolve() not in owned:
                raise RuntimeError(name + ' is not owned by the installed distribution: ' + str(origin))


def pytest_sessionstart(session):
    _OWNERS.clear()
    if os.environ.get('HCU_V033_VERIFY_INSTALLED') != '1':
        return
    for distribution, prefix in [('vllm', 'vllm'), ('sgl-kernel', 'hcu_adapter'),
                                 ('hcu-vllm-das-plugin', 'hcu_vllm_das')]:
        import_module(prefix)  # Pin package __path__ before test-module imports.
        installed = metadata.distribution(distribution)
        if not installed.files or not installed.read_text('RECORD'):
            raise RuntimeError(distribution + ' has no installed RECORD')
        direct_url = installed.read_text('direct_url.json')
        if direct_url and json.loads(direct_url).get('dir_info', {}).get('editable'):
            raise RuntimeError(distribution + ' must not be editable in installed CI')
        # Snapshot the installed manifest before a unit fixture changes sys.path.
        _OWNERS[prefix] = frozenset(Path(installed.locate_file(item)).resolve()
                                   for item in installed.files)
    _verify_installed_modules()


def pytest_collection_finish(session):
    _verify_installed_modules()


def pytest_sessionfinish(session, exitstatus):
    _verify_installed_modules()
