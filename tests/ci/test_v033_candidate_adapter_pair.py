"""Exercise paired runtime preparation and real filesystem metadata inspection."""
import base64
import csv
import hashlib
from importlib import metadata
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import test_v033_candidate_image as fixtures

C = fixtures.C

BUILD = 'dtk2610.torch2130.2611172048.g1c9448'
HOST_VERSION = '0.5.22+' + BUILD
KERNEL_VERSION = '0.0.0+' + BUILD


def component_wheel(distribution, *, version=None, requirement=None, extra=None,
                    missing_native=False, metadata_name=None, omit=()):
    host = distribution == 'hcu-adapter'
    version = version or (HOST_VERSION if host else KERNEL_VERSION)
    namespace = distribution.replace('-', '_')
    info = namespace + '-' + version + '.dist-info'
    payload = {namespace + '/__init__.py': b'raise AssertionError("must not import during inspection")\n'}
    if host:
        payload.update({'hcu_adapter/' + name: b'# reviewed host interface\n' for name in C.ADAPTER_SEAMS})
        payload['hcu_adapter/kernels/pcp_dspark.py'] = b'# reviewed companion interface\n'
        payload['hcu_adapter/hcu_config.py'] = (
            b'BACKEND_CATALOG = {"vllm_pcp_q_rope": {"backends": '
            b'{"hcu": {"status": "candidate"}}}}\n')
        payload['sitecustomize.py'] = b'# reviewed bootstrap\n'
    elif not missing_native:
        payload['sgl_kernel/native_ops.cpython-312-x86_64-linux-gnu.so'] = b'native fixture, not device evidence'
    meta = 'Name: ' + (metadata_name or distribution) + '\nVersion: ' + version + '\n'
    if host:
        meta += 'Requires-Dist: ' + (requirement or 'sgl-kernel==' + KERNEL_VERSION) + '\n'
    payload[info + '/METADATA'] = meta.encode()
    payload.update(extra or {})
    for name in omit:
        payload.pop(name)
    record = info + '/RECORD'
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w') as archive:
        rows = []
        for name, data in payload.items():
            archive.writestr(name, data)
            digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b'=').decode()
            rows.append([name, 'sha256=' + digest, str(len(data))])
        raw = io.StringIO()
        writer = csv.writer(raw, lineterminator='\n')
        writer.writerows(rows + [[record, '', '']])
        archive.writestr(record, raw.getvalue())
    tag = 'py3-none-any' if host else 'cp312-cp312-linux_x86_64'
    return namespace + '-' + version + '-' + tag + '.whl', stream.getvalue()


class PairFixture:
    def setUp(self):
        fixture = fixtures.CandidateTests()
        fixture.setUp()
        self.env = fixture.env
        self.manifest = fixture.manifest
        self.core_name = fixture.filename
        self.core = fixtures.wheel_bytes()

    def inputs(self, host=None, kernel=None):
        host = host or component_wheel('hcu-adapter')
        kernel = kernel or component_wheel('sgl-kernel')
        env = dict(self.env, WHEEL_SHA256=hashlib.sha256(self.core).hexdigest(),
                   ADAPTER_SOURCE='d' * 40,
                   ADAPTER_WHEEL_SHA256=hashlib.sha256(host[1]).hexdigest(),
                   KERNEL_WHEEL_SHA256=hashlib.sha256(kernel[1]).hexdigest())
        pair = dict(source_head=env['ADAPTER_SOURCE'], adapter_wheel_filename=host[0],
                    adapter_wheel_sha256=env['ADAPTER_WHEEL_SHA256'],
                    kernel_wheel_filename=kernel[0], kernel_wheel_sha256=env['KERNEL_WHEEL_SHA256'])
        manifest = dict(self.manifest, wheel_sha256=env['WHEEL_SHA256'], adapter=pair)
        files = {self.core_name:self.core, host[0]:host[1], kernel[0]:kernel[1],
                 'candidate-manifest.json':json.dumps(manifest).encode()}
        return env, manifest, files

    def prepare(self, directory, host=None, kernel=None):
        env, manifest, files = self.inputs(host, kernel)
        self.calls = []
        def fetch(url, token, destination, maximum):
            self.calls.append((url, token, maximum))
            destination.write_bytes(files[destination.name])
        return C.prepare(directory / 'context', env, fetch)


class PairedPreparationTests(PairFixture, unittest.TestCase):
    def test_preparation_rejects_missing_dspark_and_unregistrable_pcp_q(self):
        cases = [
            component_wheel('hcu-adapter', omit=('hcu_adapter/kernels/pcp_dspark.py',)),
            component_wheel('hcu-adapter', omit=('hcu_adapter/hcu_config.py',)),
        ]
        for source in (
                b'BACKEND_CATALOG = {}\n',
                b'BACKEND_CATALOG = {"vllm_pcp_q_rope": {"backends": {"stock": {}}}}\n',
                b'BACKEND_CATALOG = {"vllm_pcp_q_rope": {"backends": {"hcu": {"status": "planned"}}}}\n',
                b'BACKEND_CATALOG = load_runtime_config()\n',
                b'BACKEND_CATALOG = {}\nBACKEND_CATALOG = {}\n'):
            cases.append(component_wheel('hcu-adapter', extra={'hcu_adapter/hcu_config.py': source}))
        for index, host in enumerate(cases):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as directory, \
                    self.assertRaises(C.CandidateError):
                self.prepare(Path(directory), host=host)

    def test_partial_or_malformed_dispatch_cannot_select_an_unreviewed_pair(self):
        env, manifest, _ = self.inputs()
        C.adapter_manifest(manifest, C.request_inputs(env))
        for key in C.PAIR_INPUTS:
            for bad in ('', 'g' * 64, 'a' * 40 + '\n'):
                with self.subTest(key=key, bad=bad), self.assertRaises(C.CandidateError):
                    C.request_inputs(dict(env, **{key:bad}))
        with self.assertRaises(C.CandidateError):
            C.adapter_manifest(manifest, C.request_inputs(self.env))
        with self.assertRaises(C.CandidateError):
            C.adapter_manifest(self.manifest, C.request_inputs(env))

    def test_source_hash_filename_and_python_abi_are_bound_to_the_pair(self):
        env, manifest, _ = self.inputs()
        changes = [('source_head', 'e' * 40), ('adapter_wheel_sha256', 'f' * 64),
                   ('kernel_wheel_sha256', 'f' * 64), ('adapter_wheel_filename', '../host.whl'),
                   ('kernel_wheel_filename', 'sgl_kernel-0.0.0-cp310-cp310-linux_x86_64.whl'),
                   ('adapter_wheel_filename', None)]
        for key, value in changes:
            with self.subTest(key=key), self.assertRaises(C.CandidateError):
                C.adapter_manifest(dict(manifest, adapter=dict(manifest['adapter'], **{key:value})), C.request_inputs(env))

    def test_complete_pair_uses_the_fixed_private_package_and_preserves_all_payloads(self):
        with tempfile.TemporaryDirectory() as directory:
            report = self.prepare(Path(directory))
            self.assertEqual({w['distribution'] for w in report['wheels']}, {'vllm','hcu-adapter','sgl-kernel'})
            self.assertEqual(len(self.calls), 4)
            self.assertTrue(all(url.startswith(C.PACKAGE_ROOT) for url, _, _ in self.calls))
            self.assertNotIn(self.env['SOURCEFIND_PACKAGE_TOKEN'], json.dumps(report))
            self.assertFalse(report['device_qualified'])
            for wheel in report['wheels']:
                with zipfile.ZipFile(Path(directory) / 'context/core' / wheel['filename']) as archive:
                    for name, digest in wheel['payload_sha256'].items():
                        self.assertEqual(hashlib.sha256(archive.read(name)).hexdigest(), digest)

    def test_pair_refuses_wrong_abi_mixed_builds_and_missing_exact_kernel_requirement(self):
        cases = [
            (component_wheel('hcu-adapter', version=HOST_VERSION.replace('torch2130','torch2110')), None),
            (None, component_wheel('sgl-kernel', version=KERNEL_VERSION.replace('2048','2049'))),
            (component_wheel('hcu-adapter', requirement='sgl-kernel>=0.0.0'), None),
            (component_wheel('hcu-adapter', requirement='sgl-kernel==' + KERNEL_VERSION + '; extra == "optional"'), None),
            (None, component_wheel('sgl-kernel', missing_native=True)),
            (None, component_wheel('sgl-kernel', metadata_name='another-package')),
        ]
        for index, (host, kernel) in enumerate(cases):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as directory, self.assertRaises(C.CandidateError):
                self.prepare(Path(directory), host, kernel)

    def test_pair_cannot_write_another_owner_or_escape_the_wheel(self):
        for distribution, extra in [
                ('hcu-adapter', {'vllm/__init__.py':b'foreign Core replacement'}),
                ('sgl-kernel', {'hcu_adapter/__init__.py':b'legacy duplicate host'}),
                ('sgl-kernel', {'sitecustomize.py':b'foreign bootstrap'}),
                ('hcu-adapter', {'../escape':b'outside wheel'}),
                ('hcu-adapter', {'/absolute':b'outside wheel'}),
        ]:
            wheel = component_wheel(distribution, extra=extra)
            with self.subTest(distribution=distribution, extra=list(extra)), tempfile.TemporaryDirectory() as directory, self.assertRaises(C.CandidateError):
                self.prepare(Path(directory), wheel if distribution == 'hcu-adapter' else None,
                             wheel if distribution == 'sgl-kernel' else None)


class InstalledPairTests(PairFixture, unittest.TestCase):
    # Use real PathDistribution objects over extracted fixture wheels, without
    # importing their package bodies or pretending native placeholders execute.
    def installed(self, directory, host=None):
        report = self.prepare(directory, host=host)
        site = directory / 'installed'
        site.mkdir()
        for wheel in (directory / 'context/core').glob('*.whl'):
            with zipfile.ZipFile(wheel) as archive:
                archive.extractall(site)
        owners = {d.metadata['Name'].replace('_','-'):d for d in metadata.distributions(path=[str(site)])}
        self.assertEqual(set(owners), {'vllm','hcu-adapter','sgl-kernel'})
        return site, report, owners

    def inspect(self, report, owners):
        with patch.object(metadata, 'distributions', return_value=list(owners.values())), \
             patch.object(metadata, 'distribution', side_effect=owners.__getitem__), \
             patch.object(C, 'architectures', return_value=['gfx936','gfx938']) as arch:
            result = C.inspect_image(report)
        self.assertEqual(arch.call_count, len(C.GPU_LIBRARIES) + 1)
        return result

    def rewrite_record(self, site, distribution):
        record, = site.glob(distribution.replace('-','_') + '-*.dist-info/RECORD')
        rows = []
        for name, digest, size in csv.reader(io.StringIO(record.read_text())):
            if digest:
                data = (site / name).read_bytes()
                digest = 'sha256=' + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b'=').decode()
                size = str(len(data))
            rows.append([name,digest,size])
        raw = io.StringIO()
        csv.writer(raw,lineterminator='\n').writerows(rows)
        record.write_text(raw.getvalue())

    def test_installed_byte_and_record_checks_accept_the_complete_prepared_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            site, report, owners = self.installed(Path(directory))
            result = self.inspect(report, owners)
            self.assertTrue(result['prepared_payload_verified'])
            self.assertEqual(set(result['installed_records']), set(owners))
            self.assertFalse(result['model_qualified'])

    def test_installed_legacy_catalog_refuses_before_native_inspection_with_valid_record(self):
        with tempfile.TemporaryDirectory() as directory:
            site, report, owners = self.installed(Path(directory))
            # Reproduce a self-consistent older image, not a corrupt installation.
            (site / 'hcu_adapter/hcu_config.py').write_text('BACKEND_CATALOG = {}\n')
            self.rewrite_record(site, 'hcu-adapter')
            with patch.object(metadata, 'distributions', return_value=list(owners.values())), \
                 patch.object(metadata, 'distribution', side_effect=owners.__getitem__), \
                 patch.object(C, 'architectures') as arch, \
                 self.assertRaisesRegex(C.CandidateError, 'vllm_pcp_q_rope/hcu'):
                C.inspect_image()
            arch.assert_not_called()

    def test_isolated_inspection_discovers_payload_without_bootstrap_or_pth(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            host = component_wheel('hcu-adapter', extra={
                'sitecustomize.py': b'raise RuntimeError("bootstrap executed")\n'})
            site, report, _ = self.installed(directory, host=host)
            # Either startup hook would fail if inspection executed it.
            (site / 'unexpected.pth').write_text('import sys; sys.exit(99)\n')
            expected = directory / 'expected.json'
            expected.write_text(json.dumps(report))
            wrapper = '''import importlib.util, sys, sysconfig
helper, packages, expected = sys.argv[1:]
sysconfig.get_path = lambda key: packages
spec = importlib.util.spec_from_file_location('candidate', helper)
candidate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(candidate)
candidate.architectures = lambda binary: ['gfx936', 'gfx938']
sys.argv = [helper, 'inspect', '--expected', expected]
candidate.main()
assert not any(name in sys.modules for name in ('torch','sitecustomize','hcu_adapter','vllm','sgl_kernel'))
'''
            # Only architecture extraction is doubled; metadata discovery,
            # RECORD and prepared-byte checks use actual filesystem payloads.
            result = subprocess.run([sys.executable, '-I', '-S', '-B', '-c', wrapper,
                str(Path(C.__file__).resolve()), str(site), str(expected)],
                capture_output=True, text=True, check=True)
            self.assertTrue(json.loads(result.stdout)['prepared_payload_verified'])

    def test_corrupt_installed_record_and_consistently_rewritten_record_are_both_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            site, report, owners = self.installed(Path(directory))
            (site / 'hcu_adapter/__init__.py').write_text('# mutated after preparation\n')
            with self.assertRaisesRegex(C.CandidateError, 'RECORD differs'):
                self.inspect(report, owners)
            self.rewrite_record(site, 'hcu-adapter')
            with self.assertRaisesRegex(C.CandidateError, 'prepared wheel'):
                self.inspect(report, owners)

    def test_missing_or_duplicate_prepared_owner_cannot_hide_a_partial_install(self):
        with tempfile.TemporaryDirectory() as directory:
            site, report, owners = self.installed(Path(directory))
            for wheels in (report['wheels'][:1], report['wheels'] + [report['wheels'][0]]):
                with self.subTest(wheels=len(wheels)), self.assertRaises(C.CandidateError):
                    self.inspect(dict(report, wheels=wheels), owners)
            with patch.object(metadata, 'distributions', return_value=[*owners.values(),owners['hcu-adapter']]), self.assertRaisesRegex(C.CandidateError, 'one distribution owner'):
                C.inspect_image(report)


if __name__ == '__main__':
    unittest.main()
