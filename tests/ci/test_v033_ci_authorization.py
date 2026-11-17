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
RESOLVER_SPEC = importlib.util.spec_from_file_location(
    'v033_run_resolver', ROOT / '.github/scripts/hcu_ci/resolve_v033_run.py')
RESOLVER = importlib.util.module_from_spec(RESOLVER_SPEC)
RESOLVER_SPEC.loader.exec_module(RESOLVER)


class AuthorizationTests(unittest.TestCase):
    def authorize(self, author_permission, *, ready=False,
                  label_permission='write', api_failure=False,
                  action='labeled', event_label='ready-hcu'):
        event = {'action': action, 'label': {'name': event_label},
                 'pull_request': {
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

    def test_stale_ready_label_does_not_authorize_a_later_head(self):
        for action in ['opened', 'synchronize', 'reopened', 'ready_for_review',
                       'unlabeled']:
            with self.subTest(action=action):
                result, output = self.authorize('read', ready=True, action=action)
                self.assertEqual(result, 0)
                self.assertEqual(output['authorized'], 'false')
                self.assertEqual(output['ready'], 'false')

    def test_unrelated_labeled_event_does_not_authorize(self):
        result, output = self.authorize('read', ready=True, event_label='bug')
        self.assertEqual(result, 0)
        self.assertEqual(output['authorized'], 'false')

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
        cls.good = dict(RESOLVE_RESULT='success', SELECT_RESULT='success', HAS_JOBS='true', SOURCE_RESULT='success',
                        AUTH_RESULT='success', AUTHORIZED='true',
                        IMAGE_RESULT='success', DEVICE_RESULT='success', GFX938_RESULT='success')

    def gate(self, **overrides):
        return subprocess.run(['bash', '--noprofile', '--norc', '-e', '-c', self.script],
                              env={**os.environ, **self.good, **overrides},
                              capture_output=True, text=True).returncode

    def test_only_completed_selected_pipeline_or_docs_only_pass(self):
        self.assertEqual(self.gate(), 0)
        self.assertEqual(self.gate(HAS_JOBS='false', SOURCE_RESULT='skipped',
                                  AUTH_RESULT='skipped', AUTHORIZED='',
                                  IMAGE_RESULT='skipped', DEVICE_RESULT='skipped', GFX938_RESULT='skipped'), 0)

    def test_failed_cancelled_skipped_or_unauthorized_selected_stage_fails(self):
        for field in ['RESOLVE_RESULT', 'SELECT_RESULT', 'SOURCE_RESULT', 'AUTH_RESULT',
                      'IMAGE_RESULT', 'DEVICE_RESULT', 'GFX938_RESULT']:
            for result in ['failure', 'cancelled', 'skipped', '']:
                with self.subTest(field=field, result=result):
                    self.assertNotEqual(self.gate(**{field: result}), 0)
        self.assertNotEqual(self.gate(HAS_JOBS='false', RESOLVE_RESULT='failure'), 0)
        for values in [{'AUTHORIZED': 'false'}, {'AUTHORIZED': ''}, {'HAS_JOBS': ''},
                       {'HAS_JOBS': 'false', 'SELECT_RESULT': 'failure'}]:
            with self.subTest(values=values):
                self.assertNotEqual(self.gate(**values), 0)


class ManualPrTests(unittest.TestCase):
    def setUp(self):
        self.head, self.base = 'a' * 40, 'b' * 40
        self.environment = dict(GITHUB_EVENT_NAME='workflow_dispatch',
            GITHUB_REPOSITORY='test/repository', GITHUB_REF='refs/heads/v0.33.0',
            GITHUB_SHA=self.base, GITHUB_ACTOR='reviewer')
        self.event = {'inputs': {'pr_number': '42', 'expected_head': self.head}}
        self.pull = dict(number=42, state='open', head={'sha': self.head},
            base={'sha': self.base, 'ref': 'v0.33.0',
                  'repo': {'full_name': 'test/repository'}},
            labels=[{'name': 'ready-hcu'}], user={'login': 'external'})
        self.permission = 'write'
        self.calls = []

    def github(self, path):
        self.calls.append(path)
        if path == '/repos/test/repository/pulls/42':
            return self.pull
        if path == '/repos/test/repository/collaborators/reviewer/permission':
            return {'permission': self.permission}
        self.fail('unexpected API request: ' + path)

    def resolve(self):
        return RESOLVER.resolve(self.event, self.environment, self.github)

    def test_explicit_trusted_dispatch_binds_external_pr_head(self):
        for permission in ['write', 'maintain', 'admin', 'triage']:
            self.permission = permission
            output = self.resolve()
            self.assertEqual(output, dict(head_sha=self.head, base_sha=self.base,
                pr_number='42', manual_authorized='true', release='v0.33.0'))
        self.assertNotIn('external/permission', '\n'.join(self.calls))

    def test_stale_head_is_rejected_before_authorization(self):
        self.pull['head']['sha'] = 'c' * 40
        with self.assertRaisesRegex(RESOLVER.AUTH.AuthorizationError, 'head moved'):
            self.resolve()
        self.assertEqual(self.calls, ['/repos/test/repository/pulls/42'])

    def test_persistent_label_cannot_authorize_untrusted_manual_actor(self):
        self.permission = 'read'
        with self.assertRaisesRegex(RESOLVER.AUTH.AuthorizationError, 'trusted dispatch actor'):
            self.resolve()

    def test_api_failure_emits_no_success(self):
        with patch.object(self, 'github', side_effect=RESOLVER.AUTH.AuthorizationError('unavailable')):
            with self.assertRaises(RESOLVER.AUTH.AuthorizationError):
                self.resolve()

    def test_ref_and_input_pair_are_validated_before_api(self):
        for number, head in [('', self.head), ('42', ''), ('42\nhead=x', self.head),
                             ('0', self.head), ('-1', self.head), ('042', self.head),
                             ('42', self.head[:12]), ('42', self.head + '\n'),
                             ('42', 'A' * 40)]:
            with self.subTest(number=number, head=head):
                self.event['inputs'] = dict(pr_number=number, expected_head=head)
                with self.assertRaises(RESOLVER.AUTH.AuthorizationError):
                    self.resolve()
        self.assertEqual(self.calls, [])
        self.environment['GITHUB_REF'] = 'refs/heads/v0.28.1-dev'
        with self.assertRaises(RESOLVER.AUTH.AuthorizationError):
            self.resolve()

    def test_closed_wrong_repository_or_release_is_rejected(self):
        for field, value in [('state', 'closed'), ('number', 41)]:
            previous = self.pull[field]
            self.pull[field] = value
            with self.assertRaises(RESOLVER.AUTH.AuthorizationError):
                self.resolve()
            self.pull[field] = previous
        self.pull['base']['ref'] = 'v0.28.1-dev'
        with self.assertRaises(RESOLVER.AUTH.AuthorizationError):
            self.resolve()
        self.pull['base']['ref'] = 'v0.33.0'
        self.pull['base']['repo']['full_name'] = 'foreign/repository'
        with self.assertRaises(RESOLVER.AUTH.AuthorizationError):
            self.resolve()

    def test_push_and_empty_dispatch_keep_target_and_do_not_authorize_pr(self):
        self.event = {'inputs': {'pr_number': '', 'expected_head': ''}}
        for event_name in ['push', 'workflow_dispatch']:
            self.environment['GITHUB_EVENT_NAME'] = event_name
            self.assertEqual(self.resolve(), dict(head_sha=self.base, base_sha='',
                pr_number='', manual_authorized='', release='v0.33.0'))
        self.assertEqual(self.calls, [])

    def test_pr_event_keeps_label_policy_instead_of_manual_authorization(self):
        self.environment['GITHUB_EVENT_NAME'] = 'pull_request_target'
        self.environment['GITHUB_REF'] = 'refs/heads/v0.28.1-dev'
        self.event = {'pull_request': self.pull, 'action': 'synchronize'}
        output = self.resolve()
        self.assertEqual(output['head_sha'], self.head)
        self.assertEqual(output['manual_authorized'], '')
        self.assertEqual(self.calls, ['/repos/test/repository/pulls/42'])


class WorkflowTargetTests(unittest.TestCase):
    def test_every_submitted_code_checkout_uses_validated_target(self):
        from test_v033_ci_image_policy import job_block
        source = (ROOT / '.github/workflows/hcu-pr-ci-v033.yml').read_text()
        for name in ['select', 'v033-source-contract', 'v033-device-gfx936', 'v033-device-gfx938']:
            with self.subTest(job=name):
                job = job_block(source, name)
                checkout = job.split('uses: actions/checkout@v4', 1)[1].split('\n      - ', 1)[0]
                self.assertIn('ref: ${{ needs.resolve-target.outputs.head_sha }}', checkout)
                self.assertNotIn('github.sha', checkout)
                self.assertNotIn('inputs.expected_head', checkout)
        resolver = job_block(source, 'resolve-target')
        self.assertIn("ref: ${{ github.event_name == 'pull_request_target' && 'v0.33.0' || github.sha }}", resolver)
        self.assertNotIn('pull_request.head.sha', resolver)
        authorization = job_block(source, 'authorize-device')
        self.assertIn('ref: ${{ needs.resolve-target.outputs.trusted_sha }}', authorization)
        self.assertIn('steps.manual-authorization.outputs.authorized', authorization)
        self.assertIn("needs.resolve-target.outputs.pr_number == ''", authorization)
        self.assertIn('github.event.pull_request.number || inputs.pr_number || github.ref', source)


if __name__ == '__main__':
    unittest.main()
