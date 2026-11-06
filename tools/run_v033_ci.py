"""Run the v0.33 source or installed-device tier; never invoke legacy suites.

The source tier is intentionally limited to AST/packaging, not the entire
portable/lifecycle matrix. Device results must contain no skips or xfails.
Installed tiers default to one Inductor compiler thread; explicit overrides
remain available, and torch.compile and Graph coverage are unchanged.
"""
from __future__ import annotations
import argparse
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


def test_files(tier):
    if tier == 'source':
        return SOURCE_TESTS
    return [p.relative_to(ROOT).as_posix()
            for p in sorted((ROOT / 'tests/vllm033').glob('test_*.py'))
            if p.name.endswith('_hcu.py') == (tier == 'device')
            and p.relative_to(ROOT).as_posix() != PUBLIC_SELECTOR_TEST]


def _vllm_release(raw_version):
    """Parse the release tuple without requiring a third-party package."""
    if '!' in raw_version:
        raise ValueError('vLLM version must have an epoch of zero')
    release = raw_version.split('+', 1)[0].split('-', 1)[0]
    parts = release.split('.')
    if len(parts) < 2 or any(not part.isdigit() for part in parts):
        raise ValueError(f'invalid vLLM release: {raw_version!r}')
    return tuple(int(part) for part in parts)


def _run_public_selector_cases(environment):
    for case in PUBLIC_SELECTOR_CASES:
        subprocess.run([sys.executable, '-B', str(ROOT / PUBLIC_SELECTOR_TEST), case],
                       env=environment, cwd=ROOT, check=True)


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
    if args.tier in {'source', 'portable'}:
        environment.update(HCU_ADAPTER_HOST='none', HIP_VISIBLE_DEVICES='-1',
                           ROCR_VISIBLE_DEVICES='-1', CUDA_VISIBLE_DEVICES='-1')
        if args.tier == 'source' and not environment.get('VLLM_TARGET_ROOT'):
            raise ValueError('source tier requires VLLM_TARGET_ROOT')
    if args.tier in {'portable', 'device'}:
        # Small CI suites do not need the default compiler sidecar pool. Its
        # interpreter startup can still run Adapter bootstrap during pytest exit.
        environment.setdefault('TORCHINDUCTOR_COMPILE_THREADS', '1')
        # Image must already contain ABI-matched Core, Adapter and vendor wheels.
        # Install only this independent Plugin project, not repository-root setup.py.
        from importlib.metadata import version
        if _vllm_release(version('vllm'))[:2] != (0, 33):
            raise ValueError('device image must provide epoch-zero vLLM 0.33')
        if args.tier == 'device':
            environment.update(HCU_ADAPTER_HOST='vllm', VLLM_PLUGINS='hcu_adapter',
                           VLLM033_HCU_TEST='1', HCU_VLLM033_RMSNORM_LIVE='1',
                           HCU_VLLM033_GRAPH_LIVE='1', HCU_VLLM033_AITER_INT8_LIVE='1')
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps',
                        '--no-build-isolation', '--force-reinstall',
                        str(ROOT / 'adapter_extension')], env=environment, cwd=ROOT, check=True)
        if args.tier == 'portable':
            from importlib.util import find_spec
            spec = find_spec('vllm')
            if spec is None or not spec.origin:
                raise ValueError('portable image must provide an importable vLLM package')
            environment['VLLM_TARGET_ROOT'] = str(Path(spec.origin).parent.parent)
            _run_public_selector_cases(environment)
    result = subprocess.run([sys.executable, '-m', 'pytest', '-q', *files,
        '--strict-markers', '--junitxml', str(args.junit.resolve()),
        '-o', 'xfail_strict=true', '-o', 'cache_dir=.pytest_cache/v033'], cwd=ROOT, env=environment)
    if result.returncode:
        return result.returncode
    validate_result(args.junit)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
