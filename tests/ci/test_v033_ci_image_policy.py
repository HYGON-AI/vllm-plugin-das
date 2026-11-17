"""Execute the actual image guard without Docker, registry access or devices."""
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import textwrap
import unittest

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / '.github/workflows/hcu-pr-ci-v033.yml'


def job_block(source, name):
    start = source.index('  ' + name + ':\n')
    following = re.search(r'\n  [\w-]+:\n', source[start + 1:])
    return source[start:start + 1 + following.start()] if following else source[start:]


class ImagePolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = WORKFLOW.read_text()
        cls.resolver = job_block(cls.source, 'resolve-device-image')
        cls.script = textwrap.dedent(cls.resolver.split('run: |\n', 1)[1])
        # Reserved test domain; no registry is contacted by these shell cases.
        cls.harbor = 'harbor-ci.invalid:5443'
        cls.repository = cls.harbor + '/jenkins/model_test_env/vllm'
        cls.digest = 'sha256:' + 'a' * 64

    def guard(self, image, harbor=None, runner_mode=''):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'outputs'
            environment = dict(os.environ, HCU_CI_IMAGE_V033=image,
                               HARBOR_SITE=self.harbor if harbor is None else harbor,
                               GITHUB_OUTPUT=str(output))
            if runner_mode is None:
                environment.pop('HCU_CI_RUNNER_MODE_V033', None)
            else:
                environment['HCU_CI_RUNNER_MODE_V033'] = runner_mode
            result = subprocess.run(
                ['bash', '--noprofile', '--norc', '-e', '-c', self.script],
                env=environment, capture_output=True, text=True, timeout=10)
            return result.returncode, output.read_text() if output.exists() else ''

    def test_only_configured_repository_digest_is_emitted(self):
        for harbor in [self.harbor, 'other-harbor.invalid']:
            for repository in ['jenkins/model_test_env/vllm', 'hcu/base_env/vllm']:
                with self.subTest(harbor=harbor, repository=repository):
                    image = harbor + '/' + repository + '@' + self.digest
                    code, output = self.guard(image, harbor)
                    self.assertEqual(code, 0)
                    self.assertEqual(dict(line.split('=', 1) for line in output.splitlines())['image'], image)

    def test_absent_whitespace_mutable_foreign_and_malformed_images_refuse(self):
        good = self.repository + '@' + self.digest
        images = ['', ' \t\n', self.repository + ':latest',
                  self.repository + ':v0.33@' + self.digest,
                  'foreign.invalid/jenkins/model_test_env/vllm@' + self.digest,
                  self.harbor + '/different/vllm@' + self.digest,
                  self.repository + '-other@' + self.digest,
                  self.repository + '@sha256:' + 'a' * 63,
                  self.repository + '@sha256:' + 'a' * 65,
                  self.repository + '@sha256:' + 'g' * 64,
                  self.repository + '@sha512:' + 'a' * 64,
                  'sha256:' + 'a' * 64,
                  good + '@' + self.digest,
                  ' ' + good, good + ' ', good + '\nimage=foreign']
        for image in images:
            for mode in [None, '', 'legacy', 'k8s']:
                with self.subTest(image=image, mode=mode):
                    code, output = self.guard(image, runner_mode=mode)
                    self.assertNotEqual(code, 0)
                    self.assertEqual(output, '')

    def test_missing_or_malformed_registry_refuses_even_matching_input(self):
        for harbor in ['', ' \t', 'https://harbor-ci.invalid',
                       'harbor-ci.invalid/path', 'harbor-ci.invalid\nforeign',
                       'user@harbor-ci.invalid']:
            with self.subTest(harbor=harbor):
                image = harbor + '/jenkins/model_test_env/vllm@' + self.digest
                code, output = self.guard(image, harbor)
                self.assertNotEqual(code, 0)
                self.assertEqual(output, '')

    def test_registry_variable_is_wired_and_pull_credentials_are_device_only(self):
        self.assertIn('HARBOR_SITE: ${{ vars.HARBOR_SITE }}', self.resolver)
        self.assertIn('HCU_CI_IMAGE_V033: ${{ vars.HCU_CI_IMAGE_V033 }}', self.resolver)
        device = job_block(self.source, 'v033-device-gfx936')
        container = device.split('    container:\n', 1)[1].split('    steps:\n', 1)[0]
        self.assertIn('image: ${{ needs.resolve-device-image.outputs.image }}', container)
        self.assertIn('credentials:\n', container)
        self.assertIn('username: ${{ secrets.HARBOR_USERNAME }}', container)
        self.assertIn('password: ${{ secrets.HARBOR_PASSWORD }}', container)
        self.assertNotIn('volumes:', container)
        self.assertIn('options: ${{ needs.resolve-device-image.outputs.container_options }}', container)
        self.assertNotIn('--privileged', container)
        for name in ['select', 'v033-source-contract']:
            with self.subTest(job=name):
                block = job_block(self.source, name)
                self.assertIn('runs-on: ubuntu-latest', block)
                for forbidden in ['secrets.', 'container:', '/dev/kfd', '/dev/dri',
                                  'HARBOR_USERNAME', 'HARBOR_PASSWORD',
                                  'needs: [select, authorize-device']:
                    self.assertNotIn(forbidden, block)
        self.assertIn('needs: [resolve-target, select]\n', job_block(self.source, 'v033-source-contract'))

    def test_container_packaging_trusts_only_the_checked_out_workspace(self):
        for name in ['v033-device-gfx936', 'v033-device-gfx938']:
            with self.subTest(job=name), tempfile.TemporaryDirectory() as directory:
                block = job_block(self.source, name)
                step = block.split('      - name: Trust checked-out workspace for package metadata\n', 1)[1]
                command = next(line.strip()[5:] for line in step.splitlines()
                               if line.strip().startswith('run: '))
                root = Path(directory)
                workspace, foreign = root / 'checkout with spaces', root / 'foreign'
                for path in [workspace, foreign]:
                    subprocess.run(['git', 'init', '-q', str(path)], check=True)
                environment = dict(os.environ, GITHUB_WORKSPACE=str(workspace),
                    GIT_CONFIG_GLOBAL=str(root / 'gitconfig'),
                    GIT_CONFIG_SYSTEM=os.devnull, GIT_TEST_ASSUME_DIFFERENT_OWNER='1')
                def status(path):
                    return subprocess.run(['git', '-C', str(path), 'status', '--porcelain'],
                        env=environment, capture_output=True, text=True, timeout=10)
                before = status(workspace)
                self.assertNotEqual(before.returncode, 0)
                self.assertIn('dubious ownership', before.stderr)
                subprocess.run(['bash', '--noprofile', '--norc', '-e', '-c', command],
                    env=environment, check=True, capture_output=True, timeout=10)
                self.assertEqual(status(workspace).returncode, 0)
                self.assertNotEqual(status(foreign).returncode, 0)
                configured = subprocess.run(['git', 'config', '--global', '--get-all', 'safe.directory'],
                    env=environment, check=True, capture_output=True, text=True, timeout=10)
                self.assertEqual(configured.stdout.splitlines(), [str(workspace)])


    def resolved(self, mode):
        image = self.repository + '@' + self.digest
        code, output = self.guard(image, runner_mode=mode)
        self.assertEqual(code, 0)
        outputs = dict(line.split('=', 1) for line in output.splitlines())
        self.assertEqual(set(outputs), {'image', 'runner_mode', 'gfx936_runner',
                                        'gfx938_runner', 'container_options'})
        self.assertEqual(outputs['image'], image)
        return outputs

    def test_explicit_legacy_keeps_existing_host_contract(self):
        outputs = self.resolved('legacy')
        self.assertEqual(outputs['runner_mode'], 'legacy')
        for arch, label in [('gfx936', 'bw1000'), ('gfx938', 'bw1100')]:
            self.assertEqual(json.loads(outputs[arch + '_runner']),
                             ['self-hosted', 'ci', label])
        self.assertEqual(outputs['container_options'],
                         '--device=/dev/kfd --device=/dev/dri --group-add video --shm-size=32g --volume=/opt/hyhal:/opt/hyhal:ro')

    def test_unset_empty_and_explicit_k8s_emit_pools_without_host_configuration(self):
        for mode in [None, '', 'k8s']:
            with self.subTest(mode=mode):
                outputs = self.resolved(mode)
                self.assertEqual(outputs['runner_mode'], 'k8s')
                self.assertEqual(json.loads(outputs['gfx936_runner']), 'hygon-bw1000-1card')
                self.assertEqual(json.loads(outputs['gfx938_runner']), 'hygon-bw1100-1card')
                self.assertEqual(outputs['container_options'], '')
                self.assertNotIn('self-hosted', outputs['gfx936_runner'] + outputs['gfx938_runner'])

    def test_invalid_runner_modes_fail_without_publishing_partial_outputs(self):
        image = self.repository + '@' + self.digest
        for mode in ['K8S', 'k8s ', ' k8s', 'k8s\nimage=foreign', 'custom-pool']:
            with self.subTest(mode=mode):
                code, output = self.guard(image, runner_mode=mode)
                self.assertNotEqual(code, 0)
                self.assertEqual(output, '')

    def test_both_device_jobs_consume_the_same_resolved_container_contract(self):
        self.assertIn('HCU_CI_RUNNER_MODE_V033: ${{ vars.HCU_CI_RUNNER_MODE_V033 }}', self.resolver)
        for arch in ['gfx936', 'gfx938']:
            block = job_block(self.source, 'v033-device-' + arch)
            self.assertIn('runs-on: ${{ fromJSON(needs.resolve-device-image.outputs.' + arch + '_runner) }}', block)
            container = block.split('    container:\n', 1)[1].split('    steps:\n', 1)[0]
            for expected in [
                'image: ${{ needs.resolve-device-image.outputs.image }}',
                'options: ${{ needs.resolve-device-image.outputs.container_options }}',
                'username: ${{ secrets.HARBOR_USERNAME }}',
                'password: ${{ secrets.HARBOR_PASSWORD }}',
            ]:
                self.assertIn(expected, container)
            self.assertNotIn('volumes:', container)
            for forbidden in ['/dev/kfd', '/dev/dri', '--shm-size', '--privileged', '/opt/hyhal:', 'docker ']:
                self.assertNotIn(forbidden, block)
            expected_timeout = 75 if arch == 'gfx936' else 45
            self.assertIn('timeout-minutes: ' + str(expected_timeout), block)

    def test_pr_code_jobs_do_not_persist_checkout_credentials(self):
        for name in ['select', 'v033-source-contract', 'v033-device-gfx936', 'v033-device-gfx938']:
            with self.subTest(job=name):
                block = job_block(self.source, name)
                self.assertIn('    permissions:\n      contents: read\n', block)
                for checkout in block.split('uses: actions/checkout@v4')[1:]:
                    # Stop at the next step so another checkout cannot satisfy
                    # the guard for this one (source checks out two repos).
                    step = checkout.split('\n      - ', 1)[0]
                    self.assertIn('persist-credentials: false', step)


if __name__ == '__main__':
    unittest.main()
