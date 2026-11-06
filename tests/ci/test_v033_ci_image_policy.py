"""Execute the actual image guard without Docker, registry access or devices."""
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

    def guard(self, image, harbor=None):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'outputs'
            environment = dict(os.environ, HCU_CI_IMAGE_V033=image,
                               HARBOR_SITE=self.harbor if harbor is None else harbor,
                               GITHUB_OUTPUT=str(output))
            result = subprocess.run(
                ['bash', '--noprofile', '--norc', '-e', '-c', self.script],
                env=environment, capture_output=True, text=True, timeout=10)
            return result.returncode, output.read_text() if output.exists() else ''

    def test_only_configured_repository_digest_is_emitted(self):
        for harbor in [self.harbor, 'other-harbor.invalid']:
            with self.subTest(harbor=harbor):
                image = harbor + '/jenkins/model_test_env/vllm@' + self.digest
                code, output = self.guard(image, harbor)
                self.assertEqual(code, 0)
                self.assertEqual(output, 'image=' + image + '\n')

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
            with self.subTest(image=image):
                code, output = self.guard(image)
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
        self.assertIn('/opt/hyhal:/opt/hyhal:ro', container)
        self.assertIn('--shm-size=32g', container)
        self.assertNotIn('--privileged', container)
        for name in ['select', 'v033-source-contract']:
            with self.subTest(job=name):
                block = job_block(self.source, name)
                self.assertIn('runs-on: ubuntu-latest', block)
                for forbidden in ['secrets.', 'container:', '/dev/kfd', '/dev/dri',
                                  'HARBOR_USERNAME', 'HARBOR_PASSWORD',
                                  'needs: [select, authorize-device']:
                    self.assertNotIn(forbidden, block)
        self.assertIn('needs: select\n', job_block(self.source, 'v033-source-contract'))

    def test_pr_code_jobs_do_not_persist_checkout_credentials(self):
        for name in ['select', 'v033-source-contract', 'v033-device-gfx936']:
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
