"""Exercise native component launch environments and the unchanged no-skip gate."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('_component_ci_runner', ROOT / 'tools/run_v033_ci.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
COMPONENT = 'tests/vllm033/test_slimquant_packed_hcu.py'
CAPABILITY = 'hcu-slimquant-w4a8'
EXPECTED = [{'HCU_PACKED_LIVE': '1'}]


class ComponentLaunchTests(unittest.TestCase):
    def invoke(self, tier='device', *, component_only=False, outcome='pass'):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        output = Path(temporary.name) / 'v033-device.xml'
        calls = []
        files = [COMPONENT] + ([] if component_only else ['tests/vllm033/test_regular_hcu.py'])
        def run(command, *, env, **kwargs):
            if 'pytest' not in command:
                return subprocess.CompletedProcess(command, 0)
            calls.append((list(command), dict(env)))
            report = Path(command[command.index('--junitxml') + 1])
            suite = ET.Element('testsuite')
            case = ET.SubElement(suite, 'testcase', name=str(len(calls)))
            if COMPONENT in command:
                self.assertEqual(env.get('HCU_ADAPTER_HOST'), 'vllm')
                self.assertEqual(env.get('VLLM_PLUGINS'), 'hcu_adapter')
                self.assertEqual(set(env.get('HCU_ADAPTER_VLLM_REQUIRED_CAPABILITIES', '').split(',')),
                                 {'existing-owner', CAPABILITY})
                for key, value in EXPECTED[len(calls) - 1].items():
                    self.assertEqual(env.get(key), value)
                if outcome == 'skip':
                    ET.SubElement(case, 'skipped')
                elif outcome == 'failure':
                    ET.SubElement(case, 'failure')
            else:
                self.assertEqual(env['HCU_ADAPTER_VLLM_REQUIRED_CAPABILITIES'], 'existing-owner')
                for key in EXPECTED[0]:
                    self.assertEqual(env.get(key), 'inherited')
            if not (outcome == 'missing' and COMPONENT not in command):
                ET.ElementTree(suite).write(report)
            return subprocess.CompletedProcess(command, 7 if outcome == 'failure' and COMPONENT in command else 0)
        overrides = {key: 'inherited' for key in EXPECTED[0]}
        overrides.update(HCU_ADAPTER_VLLM_REQUIRED_CAPABILITIES='existing-owner', HCU_ADAPTER_HOST='none')
        with patch.dict(os.environ, overrides),              patch.object(sys, 'argv', ['runner', '--tier', tier, '--junit', str(output)]),              patch.object(runner, 'test_files', return_value=files),              patch.object(runner, '_installed_vllm', return_value=('0.33.0', '/installed')),              patch.object(runner, '_vllm_release', return_value=(0, 33, 0)),              patch.object(runner.subprocess, 'run', side_effect=run):
            code = runner.main()
        return code, calls, output

    def test_device_launch_requests_owner_before_startup_and_isolates_remaining_tests(self):
        code, calls, output = self.invoke()
        self.assertEqual(code, 0)
        self.assertEqual(len(calls), len(EXPECTED) + 1)
        self.assertEqual(sum(COMPONENT in command for command, _ in calls), len(EXPECTED))
        self.assertEqual(len(list(ET.parse(output).iter('testcase'))), len(calls))
        runner.validate_result(output)

    def test_component_only_selection_keeps_all_provider_results(self):
        code, calls, output = self.invoke(component_only=True)
        self.assertEqual(code, 0)
        self.assertEqual(len(calls), len(EXPECTED))
        self.assertEqual(len(list(ET.parse(output).iter('testcase'))), len(EXPECTED))

    def test_skipped_component_still_rejects_the_tier(self):
        with self.assertRaisesRegex(ValueError, 'skipped'):
            self.invoke(outcome='skip')

    def test_component_failure_retains_nonzero_exit_and_uploaded_failure(self):
        code, calls, output = self.invoke(outcome='failure')
        self.assertEqual(code, 7)
        self.assertEqual(len(calls), 1)
        self.assertIsNotNone(ET.parse(output).getroot().find('.//failure'))

    def test_gfx938_does_not_add_an_unselected_architecture_component(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / 'out.xml'
            with patch.object(runner.subprocess, 'run') as run:
                remaining, reports, code = runner.run_device_components(
                    ['tests/vllm033/gfx938/test_native.py'], {}, report)
            self.assertEqual(remaining, ['tests/vllm033/gfx938/test_native.py'])
            self.assertEqual((reports, code), ([], 0))
            run.assert_not_called()

    def test_success_without_remaining_report_cannot_borrow_component_results(self):
        with self.assertRaises(FileNotFoundError):
            self.invoke(outcome='missing')

    def test_actual_registry_selects_the_native_component(self):
        self.assertIn(COMPONENT, runner.test_files('device'))
        self.assertNotIn(COMPONENT, runner.test_files('device-gfx938'))


if __name__ == '__main__':
    unittest.main()
