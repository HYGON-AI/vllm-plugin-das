"""Run the v0.33 source or installed-device tier; never invoke legacy suites.

The source tier is intentionally limited to AST/packaging, not the entire
portable/lifecycle matrix. Device results must contain no skips or xfails.
Installed tiers default to one Inductor compiler thread; explicit overrides
remain available, and torch.compile and Graph coverage are unchanged.
"""
from __future__ import annotations
import argparse
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
SOURCE_TESTS = ['tests/vllm033/test_vllm_033_contract.py',
                'tests/vllm033/test_package_boundary.py']
PUBLIC_SELECTOR_TEST = 'tests/vllm033/test_flash_public_selector.py'
PUBLIC_SELECTOR_CASES = ('normal', 'optional', 'required', 'commit',
                         'foreign-before', 'duplicate', 'clear', 'foreign',
                         'config', 'fork', 'spawn')


def _has_hcu_contracts(path):
    """Read marker declarations without importing portable test modules."""
    tree = ast.parse(path.read_text(), filename=str(path))
    def marked(value):
        return any(isinstance(node, ast.Attribute)
                   and ast.unparse(node) == 'pytest.mark.hcu'
                   for node in ast.walk(value))
    for owner in [tree, *(node for node in ast.walk(tree) if isinstance(node, ast.ClassDef))]:
        for node in owner.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == 'pytestmark'
                for target in node.targets
            ) and marked(node.value):
                return True
    return any(any(marked(decorator) for decorator in node.decorator_list)
               for node in ast.walk(tree)
               if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)))


def test_files(tier):
    if tier == 'source':
        return SOURCE_TESTS
    return [p.relative_to(ROOT).as_posix()
            for p in sorted((ROOT / 'tests/vllm033').glob('test_*.py'))
            if p.relative_to(ROOT).as_posix() != PUBLIC_SELECTOR_TEST
            and (_has_hcu_contracts(p) if tier == 'device'
                 else not p.name.endswith('_hcu.py'))]


def _vllm_release(raw_version):
    """Use the installed Plugin's PEP 440 dependency for installed tiers."""
    # Keep source-tier and stdlib-only selection imports dependency-free.
    from packaging.version import Version

    version = Version(raw_version)
    if version.epoch != 0:
        raise ValueError('vLLM version must have an epoch of zero')
    return version.release


def _run_public_selector_cases(environment):
    for case in PUBLIC_SELECTOR_CASES:
        subprocess.run([sys.executable, '-B', str(ROOT / PUBLIC_SELECTOR_TEST), case],
                       env=environment, cwd=ROOT, check=True)


def _installed_vllm(environment):
    """Resolve metadata and source in the same interpreter/env/cwd as pytest.

    Querying a spec does not import the framework. A tagged record tolerates
    unrelated startup logging without accepting ambiguous probe output.
    """
    marker = 'HCU_V033_INSTALLED_PROBE='
    script = '''\
import importlib.metadata
import importlib.util
import json
spec = importlib.util.find_spec('vllm')
print('HCU_V033_INSTALLED_PROBE=' + json.dumps({
    'version': importlib.metadata.version('vllm'),
    'origin': None if spec is None else spec.origin,
}))
'''
    result = subprocess.run([sys.executable, '-B', '-c', script], env=environment,
                            cwd=ROOT, check=True, stdout=subprocess.PIPE, text=True)
    records = [line[len(marker):] for line in result.stdout.splitlines()
               if line.startswith(marker)]
    if len(records) != 1:
        raise ValueError('installed vLLM probe must return exactly one identity')
    identity = json.loads(records[0])
    if not identity['origin']:
        raise ValueError('installed image must provide an importable vLLM package')
    return identity['version'], str(Path(identity['origin']).resolve().parent.parent)


def validate_result(path):
    cases = list(ET.parse(path).getroot().iter('testcase'))
    if not cases or any(c.find(tag) is not None for c in cases
                        for tag in ('skipped', 'failure', 'error')):
        raise ValueError('empty, skipped/xfail, failed or errored v0.33 tier; not a pass')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tier', choices=('source', 'portable', 'device'), required=True)
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--junit', type=Path, default=Path('v033-results.xml'))
    args = parser.parse_args()
    files = test_files(args.tier)
    if args.list:
        print(json.dumps(files)); return 0
    if not files:
        raise ValueError('v0.33 suite is empty')
    environment = dict(os.environ)
    environment['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] = '1'
    environment['VLLM_PLUGINS'] = 'hcu_adapter'
    environment.pop('PYTHONPATH', None)
    environment['PYTHONNOUSERSITE'] = '1'
    if args.tier == 'source':
        environment['HCU_V033_VERIFY_INSTALLED'] = '0'
    if args.tier in {'source', 'portable'}:
        environment.update(HCU_ADAPTER_HOST='none', HIP_VISIBLE_DEVICES='-1',
                           ROCR_VISIBLE_DEVICES='-1', CUDA_VISIBLE_DEVICES='-1')
        if args.tier == 'source' and not environment.get('VLLM_TARGET_ROOT'):
            raise ValueError('source tier requires VLLM_TARGET_ROOT')
    if args.tier in {'portable', 'device'}:
        environment['HCU_V033_VERIFY_INSTALLED'] = '1'
        # Small CI suites do not need the default compiler sidecar pool. Its
        # interpreter startup can still run Adapter bootstrap during pytest exit.
        environment.setdefault('TORCHINDUCTOR_COMPILE_THREADS', '1')
        # Image must already contain ABI-matched Core, Adapter and vendor wheels.
        # Install only this independent Plugin project, not repository-root setup.py.
        if args.tier == 'device':
            environment.update(HCU_ADAPTER_HOST='vllm', VLLM_PLUGINS='hcu_adapter',
                           VLLM033_HCU_TEST='1', HCU_VLLM033_RMSNORM_LIVE='1',
                           HCU_VLLM033_GRAPH_LIVE='1', HCU_VLLM033_AITER_INT8_LIVE='1',
                           HCU_SLIMQUANT_LIVE='1')
        installed_version, installed_root = _installed_vllm(environment)
        if _vllm_release(installed_version)[:2] != (0, 33):
            raise ValueError('installed image must provide epoch-zero vLLM 0.33')
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps',
                        '--no-build-isolation', '--force-reinstall',
                        str(ROOT / 'adapter_extension')], env=environment, cwd=ROOT, check=True)
        if args.tier == 'portable':
            environment['VLLM_TARGET_ROOT'] = installed_root
            _run_public_selector_cases(environment)
    # A file may contain portable contracts and marked HCU numerics together.
    # Select actual pytest markers rather than inferring hardware from its name.
    marker_args = [] if args.tier == 'source' else [
        '-m', 'hcu' if args.tier == 'device' else 'not hcu']
    result = subprocess.run([sys.executable, '-m', 'pytest', '-q', *files, *marker_args,
        '--strict-markers', '--junitxml', str(args.junit.resolve()),
        '-o', 'xfail_strict=true', '-o', 'cache_dir=.pytest_cache/v033'], cwd=ROOT, env=environment)
    if result.returncode:
        return result.returncode
    validate_result(args.junit)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
