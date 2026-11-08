"""Release-scoped CI contracts; stdlib only, no GPU/framework imports."""
import json
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / '.github/scripts/hcu_ci'))
from select_v033_tests import select_jobs
from select_hcu_tests import _load_config, _git_changed_paths, DEFAULT_CONFIG, select_jobs as legacy_select

EXPECTED = {'v033-source-contract', 'v033-device-gfx936'}


class SelectionTests(unittest.TestCase):
    def ids(self, paths, **kwargs):
        result = select_jobs(paths, **kwargs)
        self.assertEqual({j['id'] for j in result['jobs']}, EXPECTED)
        return result

    def test_original_zero_selection_counterexamples(self):
        for path in ['adapter_extension/src/hcu_vllm_das/attention/e5m2_scales.py',
                     'adapter_extension/src/hcu_vllm_das/environment.py']:
            with self.subTest(path=path):
                old, _, fallback = legacy_select(_load_config(DEFAULT_CONFIG), [path])
                self.assertEqual(old, [])
                self.assertFalse(fallback)
                new = self.ids([path])
                self.assertFalse(new['fallback'])

    def test_every_delivered_python_file_selects_both_tiers(self):
        paths = list((ROOT / 'adapter_extension/src').rglob('*.py'))
        self.assertTrue(paths)
        for path in paths:
            with self.subTest(path=path):
                self.ids([path.relative_to(ROOT).as_posix()])

    def test_all_unknown_executable_paths_fallback(self):
        for path in ['new/provider.py', 'new/native.cc', 'future/BUILD',
                     'docs/probe.py', 'requirements-new.txt', 'new/policy.json']:
            with self.subTest(path=path):
                result = self.ids([path])
                # requirements files are explicitly mapped; the rest are unknown.
                if not path.startswith('requirements'):
                    self.assertTrue(result['fallback'])
                    self.assertEqual(result['unmapped_paths'], [path])

    def test_docs_empty_and_full(self):
        self.assertEqual(select_jobs([])['jobs'], [])
        self.assertTrue(select_jobs(['docs/guide.md'])['docs_only'])
        self.assertEqual(select_jobs(['README.md'])['jobs'], [])
        self.ids(['README.md'], full=True)
        self.ids([], accuracy=True)

    def test_ci_changes_expand_and_mixed_docs_do_not_hide_code(self):
        self.ids(['.github/scripts/hcu_ci/select_hcu_tests.py'])
        self.ids(['README.md', 'unknown.py'])
        self.ids(['tests/vllm033/test_rms_compile_cache.py'])

    def test_cli_routes_release_and_exports_real_jobs(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = Path(directory) / 'paths'; output = Path(directory) / 'outputs'
            paths.write_text('adapter_extension/src/hcu_vllm_das/environment.py\n')
            result = subprocess.run([sys.executable, '-B', str(ROOT / '.github/scripts/hcu_ci/select_hcu_tests.py'),
                '--release', 'v0.33.0', '--paths-from', str(paths), '--github-output', str(output)],
                capture_output=True, text=True, check=True)
            self.assertEqual({j['id'] for j in json.loads(result.stdout)['jobs']}, EXPECTED)
            self.assertIn('has_jobs=true', output.read_text())

    def test_cli_uses_explicit_release_config_and_rejects_invalid_config(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / 'config.json'
            paths = Path(directory) / 'paths'
            paths.write_text('custom-note.txt\n')
            original = json.loads((ROOT / '.github/configs/hcu-test-map-v033.json').read_text())
            original['docs_only_patterns'] = ['custom-note.txt']
            config.write_text(json.dumps(original))
            command = [sys.executable, '-B', str(ROOT / '.github/scripts/hcu_ci/select_hcu_tests.py'),
                       '--release', 'v0.33.0', '--config', str(config), '--paths-from', str(paths)]
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            self.assertTrue(json.loads(result.stdout)['docs_only'])
            self.assertEqual(json.loads(result.stdout)['jobs'], [])
            config.write_text('{}')
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn('invalid v0.33 release job/fallback contract', result.stderr)
            self.assertEqual(result.stdout, '')

    def test_release_diff_includes_deleted_paths_without_changing_legacy(self):
        with patch('select_hcu_tests.subprocess.run') as run:
            run.return_value = subprocess.CompletedProcess([], 0, 'removed.py\n', '')
            _git_changed_paths('base', 'head', include_deleted=True)
            self.assertIn('--diff-filter=ACDMRTUXB', run.call_args.args[0])
            self.assertIn('--no-renames', run.call_args.args[0])
            _git_changed_paths('base', 'head')
            self.assertIn('--diff-filter=ACMRTUXB', run.call_args.args[0])
            self.assertNotIn('--no-renames', run.call_args.args[0])

    def test_result_gate_rejects_skip_xfail_empty_failure_and_error(self):
        sys.path.insert(0, str(ROOT / 'tools'))
        from run_v033_ci import validate_result, test_files
        self.assertTrue(test_files('source'))
        self.assertEqual(set(test_files('device')), {
            p.relative_to(ROOT).as_posix() for p in (ROOT / 'tests/vllm033').glob('*_hcu.py')
        } | {'tests/vllm033/test_slimquant.py'})
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / 'junit.xml'
            for text in ['<testsuites/>', '<testsuite><testcase><skipped/></testcase></testsuite>',
                         '<testsuite><testcase><failure/></testcase></testsuite>',
                         '<testsuite><testcase><error/></testcase></testsuite>']:
                report.write_text(text)
                with self.assertRaises(ValueError):
                    validate_result(report)
            report.write_text('<testsuite><testcase name="real"/></testsuite>')
            validate_result(report)

    def test_workflow_is_release_scoped_and_keeps_cpu_device_separate(self):
        source = (ROOT / '.github/workflows/hcu-pr-ci-v033.yml').read_text()
        self.assertIn('--release v0.33.0', source)
        self.assertIn('v0.33.0', source)
        self.assertIn('HCU_CI_IMAGE_V033', source)
        self.assertNotIn('HCU_CI_IMAGE_V0281', source)
        self.assertNotIn('pull_request_target', source)
        for job in EXPECTED:
            self.assertIn(job + ':', source)

    def test_source_runner_does_not_discover_a_legacy_platform_plugin(self):
        sys.path.insert(0, str(ROOT / 'tools'))
        import run_v033_ci
        with patch.dict('os.environ', {'VLLM_PLUGINS': 'legacy_platform',
                                      'PYTHONPATH': '/old/vllm', 'PYTHONNOUSERSITE': '0',
                                      'VLLM_TARGET_ROOT': str(ROOT)}), \
             patch.object(sys, 'argv', ['run_v033_ci.py', '--tier', 'source']), \
             patch.object(run_v033_ci, 'validate_result'), \
             patch.object(run_v033_ci.subprocess, 'run') as run:
            run.return_value = subprocess.CompletedProcess([], 0)
            self.assertEqual(run_v033_ci.main(), 0)
            self.assertEqual(run.call_count, 1)
            for call in run.call_args_list:
                self.assertEqual(call.kwargs['env']['VLLM_PLUGINS'], 'hcu_adapter')
                self.assertEqual(call.kwargs['env']['HCU_ADAPTER_HOST'], 'none')
                self.assertNotIn('PYTHONPATH', call.kwargs['env'])
                self.assertEqual(call.kwargs['env']['PYTHONNOUSERSITE'], '1')

    def test_docs_under_mapped_directory_are_docs_only(self):
        result = select_jobs(['adapter_extension/README.md'])
        self.assertTrue(result['docs_only'])
        self.assertEqual(result['jobs'], [])

    def test_public_selector_is_run_as_explicit_portable_cases(self):
        sys.path.insert(0, str(ROOT / 'tools'))
        from run_v033_ci import PUBLIC_SELECTOR_CASES, PUBLIC_SELECTOR_TEST, test_files
        self.assertEqual(len(PUBLIC_SELECTOR_CASES), 11)
        self.assertNotIn(PUBLIC_SELECTOR_TEST, test_files('portable'))


if __name__ == '__main__':
    unittest.main()
