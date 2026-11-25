"""Exercise the hardware entry's authorization and child-result gate without GPUs."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
ENTRY = ROOT / 'tests/vllm033/test_hisparse_owner_gpu_lifecycle_hcu.py'


class OwnerLifecycleEntryTests(unittest.TestCase):
    def setUp(self):
        # Import-only marker double; the real entry/result-gate body executes.
        marker = ModuleType('pytest')
        marker.mark = SimpleNamespace(hcu=object())
        spec = importlib.util.spec_from_file_location('_owner_lifecycle_entry_test', ENTRY)
        self.entry = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'pytest': marker}):
            spec.loader.exec_module(self.entry)

    def run_entry(self, xml, code=0):
        def child(command, **kwargs):
            report = Path(command[command.index('--junitxml') + 1])
            report.write_text(xml)
            return subprocess.CompletedProcess(command, code, 'child output', '')
        with tempfile.TemporaryDirectory() as directory, \
             patch.dict(os.environ, {'VLLM033_HCU_TEST': '1',
                                     'HCU_OWNER_GPU_LIFECYCLE': '',
                                     'HIP_VISIBLE_DEVICES': 'assigned',
                                     'PYTHONPATH': '/unrelated/checkout'}), \
             patch.object(self.entry.subprocess, 'run', side_effect=child) as run:
            self.entry.test_complete_hisparse_owner_gpu_lifecycle(Path(directory))
            return run.call_args

    def test_missing_authorization_never_starts_a_child(self):
        with patch.dict(os.environ, {'VLLM033_HCU_TEST': '',
                                     'HCU_OWNER_GPU_LIFECYCLE': ''}), \
             patch.object(self.entry.subprocess, 'run') as run:
            with self.assertRaises(AssertionError):
                self.entry.test_complete_hisparse_owner_gpu_lifecycle(Path('/unused'))
            run.assert_not_called()

    def test_runs_full_original_cohort_and_preserves_assigned_visibility(self):
        call = self.run_entry('<testsuite><testcase name="passed"/></testsuite>')
        command = call.args[0]
        self.assertEqual(command[:6], [sys.executable, '-I', '-B', '-m', 'pytest', '-q'])
        self.assertEqual(Path(command[-1]), ENTRY.with_name('_hisparse_owner_gpu_lifecycle.py'))
        self.assertNotIn('-k', command)
        self.assertNotIn('--deselect', command)
        self.assertEqual(call.kwargs['env']['HIP_VISIBLE_DEVICES'], 'assigned')
        self.assertNotIn('PYTHONPATH', call.kwargs['env'])
        self.assertEqual(call.kwargs['timeout'], 300)

    def test_exit_zero_does_not_hide_failed_skipped_or_empty_children(self):
        for xml in ['<testsuites/>',
                    '<testsuite><testcase><failure/></testcase></testsuite>',
                    '<testsuite><testcase><error/></testcase></testsuite>',
                    '<testsuite><testcase><skipped/></testcase></testsuite>']:
            with self.subTest(xml=xml), self.assertRaises(AssertionError):
                self.run_entry(xml)

    def test_nonzero_exit_does_not_accept_a_green_child_report(self):
        with self.assertRaises(AssertionError):
            self.run_entry('<testsuite><testcase name="passed"/></testsuite>', code=1)


if __name__ == '__main__':
    unittest.main()
