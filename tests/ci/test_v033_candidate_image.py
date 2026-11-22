"""Private delivery and real native-architecture refusal contracts."""
import base64
import ast
import builtins
import hashlib
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request
import zipfile

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('candidate', ROOT / '.github/scripts/hcu_ci/v033_candidate_image.py')
C = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(C)


def wheel_bytes(changed=None, omitted=None):
    payload = {'vllm/' + name: b'native-placeholder' for name in C.GPU_LIBRARIES}
    payload['vllm/__init__.py'] = b''
    payload['vllm-0.33.0.dist-info/METADATA'] = b'Name: vllm\nVersion: 0.33.0+candidate\n'
    record = 'vllm-0.33.0.dist-info/RECORD'
    rows = []
    for name, data in payload.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b'=').decode()
        if name != omitted:
            rows.append(f'{name},sha256={digest},{len(data)}\n')
    if changed:
        payload[changed] = b'tampered-after-record'
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w') as archive:
        for name, data in payload.items():
            archive.writestr(name, data)
        archive.writestr(record, ''.join(rows) + record + ',,\n')
    return stream.getvalue()


class CandidateTests(unittest.TestCase):
    def setUp(self):
        self.env = dict(GITHUB_REF='refs/heads/v0.33.0', GITHUB_EVENT_NAME='workflow_dispatch',
            PACKAGE_VERSION='0.33.0-ci-reviewed', CORE_SOURCE='a' * 40, WHEEL_SHA256='b' * 64,
            HARBOR_SITE='registry.invalid:5000',
            BASE_IMAGE='registry.invalid:5000/hcu/base_env/vllm@sha256:' + 'c' * 64,
            SOURCEFIND_PACKAGE_TOKEN='fixture-read-package-token')
        self.filename = 'vllm-0.33.0+candidate-cp312-cp312-linux_x86_64.whl'
        self.manifest = dict(source_head='a' * 40, wheel_sha256='b' * 64,
            full_native_rebuild=True, use_precompiled=False,
            requested_architectures=['gfx936', 'gfx938'], wheel_filename=self.filename)

    def test_trusted_pair_rejects_wrong_event_ref_registry_and_malformed_hashes(self):
        C.request_inputs(self.env)
        for key, bad in [('GITHUB_REF', 'refs/heads/feature'), ('GITHUB_EVENT_NAME', 'pull_request_target'),
                         ('PACKAGE_VERSION', '../../other'), ('CORE_SOURCE', 'a' * 12),
                         ('WHEEL_SHA256', 'b' * 64 + '\n'), ('HARBOR_SITE', 'https://registry.invalid'),
                         ('BASE_IMAGE', 'registry.invalid:5000/hcu/base_env/vllm:latest'),
                         ('BASE_IMAGE', 'foreign.invalid/hcu/base_env/vllm@sha256:' + 'c' * 64)]:
            with self.subTest(key=key, bad=bad), self.assertRaises(C.CandidateError):
                C.request_inputs(dict(self.env, **{key: bad}))

    def test_manifest_binds_source_hash_full_rebuild_architectures_and_filename(self):
        inputs = C.request_inputs(self.env)
        self.assertEqual(C.validate_manifest(self.manifest, inputs), self.filename)
        for key, bad in [('source_head', 'd' * 40), ('wheel_sha256', 'd' * 64),
                         ('full_native_rebuild', False), ('use_precompiled', True),
                         ('requested_architectures', ['gfx938']), ('wheel_filename', '../other.whl'),
                         ('wheel_filename', self.filename.replace('0.33.0', '0.28.1'))]:
            with self.subTest(key=key), self.assertRaises(C.CandidateError):
                C.validate_manifest(dict(self.manifest, **{key: bad}), inputs)

    def test_wheel_rejects_corrupt_unrecorded_and_wrong_hash_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / self.filename
            for data, valid in [(wheel_bytes(), True),
                                (wheel_bytes(changed='vllm/__init__.py'), False),
                                (wheel_bytes(omitted='vllm/__init__.py'), False)]:
                path.write_bytes(data)
                digest = hashlib.sha256(data).hexdigest()
                if valid:
                    self.assertEqual(C.verify_wheel(path, digest), 7)
                    with self.assertRaises(C.CandidateError):
                        C.verify_wheel(path, 'f' * 64)
                else:
                    with self.assertRaises(C.CandidateError):
                        C.verify_wheel(path, digest)

    def test_preparation_fetches_fixed_private_package_and_validates_download(self):
        data = wheel_bytes()
        digest = hashlib.sha256(data).hexdigest()
        calls = []
        def fetch(url, token, destination, maximum):
            calls.append((url, token, maximum))
            destination.write_bytes(json.dumps(dict(self.manifest, wheel_sha256=digest)).encode()
                if url.endswith('candidate-manifest.json') else data)
        with tempfile.TemporaryDirectory() as directory:
            report = C.prepare(Path(directory) / 'context', dict(self.env, WHEEL_SHA256=digest), fetch)
            self.assertEqual(report['record_entries'], 7)
            self.assertFalse(report['device_qualified'])
            self.assertEqual(len(calls), 2)
            self.assertTrue(all(url.startswith(C.PACKAGE_ROOT) for url, _, _ in calls))
            self.assertNotIn(self.env['SOURCEFIND_PACKAGE_TOKEN'], json.dumps(report))

    def test_builder_preparation_uses_only_host_stdlib_without_runtime_metadata(self):
        # The image builder may use an OS unsupported by setup-python. Its
        # preparation path needs neither Python 3.12 syntax nor runtime imports.
        source = (ROOT / '.github/scripts/hcu_ci/v033_candidate_image.py').read_text()
        ast.parse(source, feature_version=(3, 6))
        original_import = builtins.__import__
        def host_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == 'importlib.metadata' or (name == 'importlib' and 'metadata' in fromlist):
                raise ImportError('metadata belongs to the installed-runtime inspector')
            return original_import(name, globals, locals, fromlist, level)
        host = importlib.util.module_from_spec(SPEC)
        data = wheel_bytes()
        digest = hashlib.sha256(data).hexdigest()
        def fetch(url, token, destination, maximum):
            destination.write_bytes(json.dumps(dict(self.manifest, wheel_sha256=digest)).encode()
                if url.endswith('candidate-manifest.json') else data)
        with patch.object(builtins, '__import__', side_effect=host_import):
            SPEC.loader.exec_module(host)
            with tempfile.TemporaryDirectory() as directory:
                report = host.prepare(Path(directory) / 'context', dict(self.env, WHEEL_SHA256=digest), fetch)
                self.assertEqual(report['record_entries'], 7)
            with self.assertRaises(ImportError):
                host.inspect_image()

    def test_private_download_never_follows_redirects(self):
        request = Request(C.PACKAGE_ROOT + 'version/file')
        self.assertIsNone(C.NoRedirect().redirect_request(request, None, 302, 'Found', {}, 'https://foreign.invalid'))
        with tempfile.TemporaryDirectory() as directory, patch.object(C, 'build_opener') as opener:
            opener.return_value.open.side_effect = HTTPError(request.full_url, 302, 'Found', {}, None)
            with self.assertRaises(HTTPError):
                C.download(request.full_url, 'fixture-token', Path(directory) / 'out', 100)
            self.assertEqual(opener.return_value.open.call_count, 1)
            sent = opener.return_value.open.call_args.args[0]
            self.assertEqual(sent.get_header('Deploy-token'), 'fixture-token')
            self.assertNotIn('fixture-token', sent.full_url)

    def test_actual_architecture_report_rejects_single_architecture_image(self):
        binary = Path('native.so')
        for targets, valid in [(['gfx938'], False), (['gfx936'], False),
                               (['gfx936', 'gfx938'], True), ([], False)]:
            output = '\n'.join('hipv4-amdgcn-amd-amdhsa--' + a for a in targets)
            runner = lambda *args, **kwargs: SimpleNamespace(stdout=output)
            if valid:
                self.assertEqual(C.architectures(binary, runner), targets)
            else:
                with self.assertRaises(C.CandidateError):
                    C.architectures(binary, runner)

    def test_publisher_has_no_pr_entry_public_wheel_or_runtime_variable_write(self):
        source = (ROOT / '.github/workflows/hcu-ci-image-v033.yml').read_text()
        self.assertIn("if: github.event_name == 'workflow_dispatch' && github.ref == 'refs/heads/v0.33.0'", source)
        self.assertIn('ref: ${{ github.sha }}', source)
        self.assertIn('persist-credentials: false', source)
        self.assertIn('docker build --network=none', source)
        self.assertIn('v033-image-publication', source)
        self.assertIn('path: ${{ env.PUBLISH_ROOT }}/publication.json', source)
        for absent in ['pull_request:', 'pull_request_target:', 'HCU_CI_IMAGE_V033',
                       '--privileged', '/dev/kfd', '/dev/dri', 'contents: write', ':latest']:
            self.assertNotIn(absent, source)
        dockerfile = (ROOT / 'docker/Dockerfile.v033-ci-candidate').read_text()
        self.assertIn('--no-index --no-deps', dockerfile)
        self.assertNotIn('SOURCEFIND_PACKAGE_TOKEN', dockerfile)

    def test_first_dispatch_has_a_secret_free_release_registration_entry(self):
        from test_v033_ci_image_policy import job_block
        source = (ROOT / '.github/workflows/hcu-ci-image-v033.yml').read_text()
        events = source.split('permissions:\n', 1)[0]
        self.assertIn('branches: [v0.33.0]', events)
        self.assertIn("- '.github/workflows/hcu-ci-image-v033.yml'", events)
        registration = job_block(source, 'register-manual-entry')
        self.assertIn("if: github.event_name == 'push'", registration)
        self.assertIn('runs-on: ubuntu-latest', registration)
        for forbidden in ['secrets.', 'checkout@', 'docker', '/dev/', 'SOURCEFIND']:
            self.assertNotIn(forbidden, registration)


if __name__ == '__main__':
    unittest.main()
