"""Exercise the reused HCU authorization policy and actual release gate script."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import textwrap
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    'v033_authorization',
    ROOT / '.github/scripts/hcu_ci/check_hcu_ci_authorization.py')
AUTH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUTH)


class AuthorizationTests(unittest.TestCase):
    def authorize(self, author_permission, *, ready=False,
                  label_permission='write', api_failure=False):
        event = {'pull_request': {
            'number': 42, 'user': {'login': 'contributor'},
            'labels': [{'name': 'ready-hcu'}] if ready else []}}

        def github(url, token):
            if api_failure:
                raise AUTH.AuthorizationError('test API unavailable')
            if url.endswith('/collaborators/contributor/permission'):
                return {'permission': author_permission}
            if url.endswith('/collaborators/reviewer/permission'):
                return {'permission': label_permission}
            if '/issues/42/events?' in url:
                return [{'event': 'labeled', 'label': {'name': 'ready-hcu'},
                         'actor': {'login': 'reviewer'}}] if url.endswith('page=1') else []
            self.fail('Unexpected API path')

        with tempfile.TemporaryDirectory() as directory:
            event_path = Path(directory) / 'event.json'
            output_path = Path(directory) / 'outputs'
            event_path.write_text(json.dumps(event))
            environment = {'GITHUB_EVENT_PATH': str(event_path),
                           'GITHUB_OUTPUT': str(output_path),
                           'GITHUB_REPOSITORY': 'test/repository',
                           'GITHUB_API_URL': 'https://api.invalid',
                           'GITHUB_TOKEN': 'test-token'}
            with patch.dict(os.environ, environment, clear=True), \
                    patch.object(AUTH, '_github_json', side_effect=github), \
                    contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(io.StringIO()):
                result = AUTH.main()
            outputs = dict(line.split('=', 1) for line in
                           output_path.read_text().splitlines()) if output_path.exists() else {}
            return result, outputs

    def test_trusted_authors_are_auto_authorized_without_label(self):
        for permission in ['admin', 'maintain', 'write', 'triage']:
            with self.subTest(permission=permission):
                result, output = self.authorize(permission)
                self.assertEqual(result, 0)
                self.assertEqual(output['authorized'], 'true')
                self.assertEqual(output['ready'], 'auto')

    def test_untrusted_author_without_label_stays_locked(self):
        result, output = self.authorize('read')
        self.assertEqual(result, 0)
        self.assertEqual(output['authorized'], 'false')

    def test_trusted_label_actor_authorizes_external_author(self):
        result, output = self.authorize('read', ready=True)
        self.assertEqual(result, 0)
        self.assertEqual(output['authorized'], 'true')
        self.assertEqual(output['actor'], 'reviewer')

    def test_untrusted_label_actor_and_api_failure_fail_closed(self):
        for options in [{'ready': True, 'label_permission': 'read'},
                        {'api_failure': True}]:
            with self.subTest(options=options):
                result, output = self.authorize('read', **options)
                self.assertNotEqual(result, 0)
                self.assertNotEqual(output.get('authorized'), 'true')


class ReleaseGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = (ROOT / '.github/workflows/hcu-pr-ci-v033.yml').read_text()
        block = source.split('- name: Require each selected stage to pass\n', 1)[1]
        cls.script = textwrap.dedent(block.split('run: |\n', 1)[1])
        cls.good = dict(SELECT_RESULT='success', HAS_JOBS='true', SOURCE_RESULT='success',
                        AUTH_RESULT='success', AUTHORIZED='true',
                        IMAGE_RESULT='success', DEVICE_RESULT='success')

    def gate(self, **overrides):
        return subprocess.run(['bash', '--noprofile', '--norc', '-e', '-c', self.script],
                              env={**os.environ, **self.good, **overrides},
                              capture_output=True, text=True).returncode

    def test_only_completed_selected_pipeline_or_docs_only_pass(self):
        self.assertEqual(self.gate(), 0)
        self.assertEqual(self.gate(HAS_JOBS='false', SOURCE_RESULT='skipped',
                                  AUTH_RESULT='skipped', AUTHORIZED='',
                                  IMAGE_RESULT='skipped', DEVICE_RESULT='skipped'), 0)

    def test_failed_cancelled_skipped_or_unauthorized_selected_stage_fails(self):
        for field in ['SELECT_RESULT', 'SOURCE_RESULT', 'AUTH_RESULT',
                      'IMAGE_RESULT', 'DEVICE_RESULT']:
            for result in ['failure', 'cancelled', 'skipped', '']:
                with self.subTest(field=field, result=result):
                    self.assertNotEqual(self.gate(**{field: result}), 0)
        for values in [{'AUTHORIZED': 'false'}, {'AUTHORIZED': ''}, {'HAS_JOBS': ''},
                       {'HAS_JOBS': 'false', 'SELECT_RESULT': 'failure'}]:
            with self.subTest(values=values):
                self.assertNotEqual(self.gate(**values), 0)


if __name__ == '__main__':
    unittest.main()
