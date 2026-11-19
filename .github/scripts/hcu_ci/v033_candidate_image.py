# SPDX-License-Identifier: Apache-2.0
"""Prepare private runtime wheels and inspect a built release CI candidate."""
import argparse
import ast
import base64
import csv
from email.parser import BytesParser
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
from urllib.parse import quote
from urllib.request import HTTPRedirectHandler, Request, build_opener
import zipfile

PACKAGE_ROOT = ('https://developer.sourcefind.cn/codes/api/v4/projects/2457/'
                'packages/generic/vllm-ci-candidate/')
GPU_LIBRARIES = ('_C.abi3.so', '_C_stable_libtorch.abi3.so',
                 '_moe_C_stable_libtorch.abi3.so', '_rocm_C.abi3.so')
ADAPTER_SEAMS = ('kernels/pcp_fork.py', 'fp4_indexer.py', 'v41_fp8_indexer.py',
                 'ct_w4a8_registration.py', 'dsv41_activation.py',
                 'dsv41_channel_fp8.py', 'dsv41_dspark_registration.py',
                 'kernels/pcp_dspark.py', 'hcu_config.py')
PAIR_INPUTS = ('ADAPTER_SOURCE', 'ADAPTER_WHEEL_SHA256', 'KERNEL_WHEEL_SHA256')


class CandidateError(ValueError):
    pass


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def request_inputs(environment):
    if (environment.get('GITHUB_REF') != 'refs/heads/v0.33.0'
            or environment.get('GITHUB_EVENT_NAME') != 'workflow_dispatch'):
        raise CandidateError('publisher requires trusted v0.33.0 dispatch')
    values = {key: environment.get(key, '') for key in
              ('PACKAGE_VERSION', 'CORE_SOURCE', 'WHEEL_SHA256', 'BASE_IMAGE', 'HARBOR_SITE')}
    for key, pattern in [('PACKAGE_VERSION', r'[A-Za-z0-9][A-Za-z0-9._+-]{0,127}'),
                         ('CORE_SOURCE', r'[0-9a-f]{40}'), ('WHEEL_SHA256', r'[0-9a-f]{64}'),
                         ('HARBOR_SITE', r'[A-Za-z0-9.-]+(?::[0-9]+)?')]:
        if not re.fullmatch(pattern, values[key]):
            raise CandidateError('invalid ' + key)
    prefix = values['HARBOR_SITE'] + '/hcu/base_env/vllm@sha256:'
    if not values['BASE_IMAGE'].startswith(prefix) or not re.fullmatch(
            r'[0-9a-f]{64}', values['BASE_IMAGE'][len(prefix):]):
        raise CandidateError('base requires an immutable approved runtime repository digest')
    pair = {key: environment.get(key, '') for key in PAIR_INPUTS}
    if any(pair.values()):
        for key in PAIR_INPUTS:
            if not re.fullmatch(r'[0-9a-f]{40}' if key == 'ADAPTER_SOURCE'
                                else r'[0-9a-f]{64}', pair[key]):
                raise CandidateError('complete paired Adapter inputs required: ' + key)
        values.update(pair)
    return values


def validate_manifest(manifest, inputs):
    if (manifest.get('source_head') != inputs['CORE_SOURCE']
            or manifest.get('wheel_sha256') != inputs['WHEEL_SHA256']
            or manifest.get('full_native_rebuild') is not True
            or manifest.get('use_precompiled') is not False
            or set(manifest.get('requested_architectures', [])) != {'gfx936', 'gfx938'}):
        raise CandidateError('candidate source, wheel hash or native build contract differs')
    filename = manifest.get('wheel_filename', '')
    if not re.fullmatch(r'vllm-0\.33\.0[A-Za-z0-9._+\-]*-cp312-cp312-linux_x86_64\.whl', filename):
        raise CandidateError('candidate must supply a vLLM 0.33 CPython 3.12 Linux wheel')
    return filename


def wheel_payload(path, expected):
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise CandidateError('downloaded wheel checksum differs')
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or sum(x.file_size for x in archive.infolist()) > 4 * 1024**3:
            raise CandidateError('duplicate wheel members or excessive unpacked size')
        for name in names:
            parsed = PurePosixPath(name)
            if parsed.is_absolute() or '..' in parsed.parts or '\\' in name:
                raise CandidateError('unsafe wheel member')
        records = [name for name in names if name.endswith('.dist-info/RECORD')]
        if len(records) != 1:
            raise CandidateError('wheel must contain one RECORD')
        checked = {}
        for name, digest, size in csv.reader(io.StringIO(archive.read(records[0]).decode())):
            if name in checked:
                raise CandidateError('duplicate RECORD member')
            payload = archive.read(name)
            if not digest:
                if name != records[0]:
                    raise CandidateError('unhashed wheel payload')
                checked[name] = None
                continue
            actual = 'sha256=' + base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b'=').decode()
            if digest != actual or size != str(len(payload)):
                raise CandidateError('wheel RECORD checksum or size differs')
            checked[name] = hashlib.sha256(payload).hexdigest()
        if set(checked) != {name for name in names if not name.endswith('/')}:
            raise CandidateError('wheel payload missing from RECORD')
    return checked


def core_payload(path, expected):
    payload = wheel_payload(path, expected)
    if not all('vllm/' + library in payload for library in GPU_LIBRARIES):
        raise CandidateError('wheel lacks required native libraries')
    return payload


def verify_wheel(path, expected):
    return len(core_payload(path, expected))


def wheel_identity(path):
    with zipfile.ZipFile(path) as archive:
        names = [n for n in archive.namelist() if n.endswith('.dist-info/METADATA')]
        if len(names) != 1:
            raise CandidateError('wheel requires exactly one METADATA')
        return names[0].split('/')[0], BytesParser().parsebytes(archive.read(names[0]))


def adapter_manifest(manifest, inputs):
    pair = manifest.get('adapter')
    if 'ADAPTER_SOURCE' not in inputs:
        if pair is not None:
            raise CandidateError('Adapter manifest requires explicit paired dispatch inputs')
        return None
    if not isinstance(pair, dict) or pair.get('source_head') != inputs['ADAPTER_SOURCE']:
        raise CandidateError('paired Adapter source differs or is missing')
    for key, prefix, tag, input_key in [
            ('adapter', 'hcu_adapter', 'py3-none-any', 'ADAPTER_WHEEL_SHA256'),
            ('kernel', 'sgl_kernel', 'cp312-cp312-linux_x86_64', 'KERNEL_WHEEL_SHA256')]:
        filename = pair.get(key + '_wheel_filename', '')
        if (not isinstance(filename, str)
                or not re.fullmatch(prefix + r'-[A-Za-z0-9.!+_]+-' + tag + r'\.whl', filename)
                or pair.get(key + '_wheel_sha256') != inputs[input_key]):
            raise CandidateError('paired ' + key + ' filename or checksum differs')
    return pair


def adapter_catalog(source):
    """Check registration metadata without executing package or config code."""
    try:
        tree = ast.parse(source.decode('utf-8'))
        values = [node.value for node in tree.body if isinstance(node, ast.Assign)
                  and any(isinstance(target, ast.Name) and target.id == 'BACKEND_CATALOG'
                          for target in node.targets)]
        if len(values) != 1:
            raise ValueError('one static catalog required')
        catalog = ast.literal_eval(values[0])
    except (ValueError, SyntaxError, TypeError, UnicodeError) as error:
        raise CandidateError('Adapter requires a static BACKEND_CATALOG: ' + str(error))
    entry = catalog.get('vllm_pcp_q_rope') if isinstance(catalog, dict) else None
    backends = entry.get('backends') if isinstance(entry, dict) else None
    provider = backends.get('hcu') if isinstance(backends, dict) else None
    if (not isinstance(provider, dict)
            or provider.get('status') not in ('candidate', 'validated', 'production')):
        raise CandidateError('Adapter BACKEND_CATALOG lacks selectable vllm_pcp_q_rope/hcu')


def component_payload(path, digest, distribution):
    payload = wheel_payload(path, digest)
    info, metadata = wheel_identity(path)
    namespace = distribution.replace('-', '_')
    version = metadata.get('Version', '')
    if (metadata.get('Name', '').lower().replace('_', '-') != distribution
            or info != namespace + '-' + version + '.dist-info'
            or not version.partition('+')[2].startswith('dtk2610.torch2130.')):
        raise CandidateError('paired ' + distribution + ' identity or DTK/Torch ABI differs')
    tag = 'py3-none-any' if distribution == 'hcu-adapter' else 'cp312-cp312-linux_x86_64'
    if path.name != namespace + '-' + version + '-' + tag + '.whl':
        raise CandidateError('paired wheel filename disagrees with METADATA')
    for name in payload:
        if (not name.startswith((namespace + '/', info + '/'))
                and not (distribution == 'hcu-adapter' and name == 'sitecustomize.py')):
            raise CandidateError('paired wheel claims foreign payload: ' + name)
    if namespace + '/__init__.py' not in payload:
        raise CandidateError('paired wheel lacks its package')
    if distribution == 'hcu-adapter':
        if not all('hcu_adapter/' + name in payload for name in ADAPTER_SEAMS):
            raise CandidateError('paired Adapter lacks required interfaces')
        with zipfile.ZipFile(path) as archive:
            adapter_catalog(archive.read('hcu_adapter/hcu_config.py'))
    elif len([name for name in payload if name.startswith('sgl_kernel/native_ops.')
              and name.endswith('.so')]) != 1:
        raise CandidateError('paired kernel requires exactly one native_ops library')
    return payload, metadata


def wheel_report(path, payload, distribution, version):
    return dict(distribution=distribution, filename=path.name, version=version,
                payload_sha256={name: digest for name, digest in payload.items() if digest})


def download(url, token, destination, maximum):
    if not url.startswith(PACKAGE_ROOT):
        raise CandidateError('private artifact URL outside fixed project/package')
    request = Request(url, headers={'Deploy-Token': token, 'User-Agent': 'v033-private-ci-image'})
    # A redirect never forwards the package credential to another destination.
    with build_opener(NoRedirect).open(request, timeout=60) as response, destination.open('xb') as output:
        size = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > maximum:
                raise CandidateError('private artifact exceeds size bound')
            output.write(chunk)


def prepare(destination, environment, fetch=download):
    inputs = request_inputs(environment)
    token = environment.get('SOURCEFIND_PACKAGE_TOKEN', '')
    if not token:
        raise CandidateError('read_package_registry deploy token required')
    destination.mkdir()
    core = destination / 'core'
    core.mkdir()
    base = PACKAGE_ROOT + quote(inputs['PACKAGE_VERSION'], safe='') + '/'
    manifest_path = destination / 'candidate-manifest.json'
    fetch(base + 'candidate-manifest.json', token, manifest_path, 1024**2)
    manifest = json.loads(manifest_path.read_text())
    filename = validate_manifest(manifest, inputs)
    pair = adapter_manifest(manifest, inputs)
    wheel = core / filename
    fetch(base + quote(filename, safe=''), token, wheel, 2 * 1024**3)
    payload = core_payload(wheel, inputs['WHEEL_SHA256'])
    _, metadata = wheel_identity(wheel)
    if metadata.get('Name', '').lower() != 'vllm':
        raise CandidateError('Core METADATA identity differs')
    report = dict(core_source=inputs['CORE_SOURCE'], wheel_sha256=inputs['WHEEL_SHA256'],
                  wheel_filename=filename, record_entries=len(payload), base_image=inputs['BASE_IMAGE'],
                  wheels=[wheel_report(wheel, payload, 'vllm', metadata['Version'])],
                  device_qualified=False, model_qualified=False)
    if pair:
        prepared = {}
        for key, distribution, input_key in [
                ('adapter', 'hcu-adapter', 'ADAPTER_WHEEL_SHA256'),
                ('kernel', 'sgl-kernel', 'KERNEL_WHEEL_SHA256')]:
            path = core / pair[key + '_wheel_filename']
            fetch(base + quote(path.name, safe=''), token, path, 512 * 1024**2)
            component, metadata = component_payload(path, inputs[input_key], distribution)
            prepared[key] = metadata
            report['wheels'].append(wheel_report(path, component, distribution, metadata['Version']))
        adapter_version, kernel_version = [prepared[key]['Version'] for key in ('adapter', 'kernel')]
        requirements = [re.sub(r'\s', '', value) for value in prepared['adapter'].get_all('Requires-Dist', [])]
        if (adapter_version.partition('+')[2] != kernel_version.partition('+')[2]
                or 'sgl-kernel==' + kernel_version not in requirements):
            raise CandidateError('Adapter must require the exact matching native build')
        report['adapter_source'] = inputs['ADAPTER_SOURCE']
        report['adapter_wheel_sha256'] = inputs['ADAPTER_WHEEL_SHA256']
        report['kernel_wheel_sha256'] = inputs['KERNEL_WHEEL_SHA256']
    return report


def architectures(binary, runner=subprocess.run):
    result = runner(['/opt/dtk/aillvm/bin/llvm-objdump', '--offloading', str(binary)],
                    check=True, capture_output=True, text=True)
    targets = sorted(set(re.findall(r'hipv4-amdgcn-amd-amdhsa--(gfx[0-9a-f]+)', result.stdout)))
    if not {'gfx936', 'gfx938'} <= set(targets):
        raise CandidateError('native payload lacks gfx936/gfx938: ' + binary.name)
    return targets


def inspect_image(expected=None):
    # Installed-runtime inspection runs inside the CPython 3.12 image. The
    # download/RECORD preparation path must also load on Python 3.6 builders.
    from importlib import metadata

    for name in ('vllm', 'hcu-adapter', 'sgl-kernel'):
        owners = [d for d in metadata.distributions()
                  if re.sub(r'[-_.]+', '-', d.metadata.get('Name', '').lower()) == name]
        if len(owners) != 1:
            raise CandidateError('installed ' + name + ' requires one distribution owner')
    core = metadata.distribution('vllm')
    if not core.version.startswith('0.33.0+'):
        raise CandidateError('image does not contain a vLLM 0.33 candidate')
    adapter = metadata.distribution('hcu-adapter')
    kernel = metadata.distribution('sgl-kernel')
    records = {}
    for name, distribution in [('vllm', core), ('hcu-adapter', adapter), ('sgl-kernel', kernel)]:
        raw = distribution.read_text('RECORD')
        if not raw:
            raise CandidateError('installed ' + name + ' lacks RECORD')
        count = 0
        seen = set()
        for filename, digest, size in csv.reader(io.StringIO(raw)):
            if filename in seen:
                raise CandidateError('duplicate installed RECORD member')
            seen.add(filename)
            if digest:
                payload = Path(distribution.locate_file(filename)).read_bytes()
                actual = 'sha256=' + base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b'=').decode()
                if digest != actual or size != str(len(payload)):
                    raise CandidateError('installed ' + name + ' RECORD differs: ' + filename)
                count += 1
        records[name] = dict(verified=count, record_sha256=hashlib.sha256(raw.encode()).hexdigest())
    if expected is not None:
        owners = {'vllm': core, 'hcu-adapter': adapter, 'sgl-kernel': kernel}
        wheels = expected.get('wheels', [])
        wanted = {'vllm', 'hcu-adapter', 'sgl-kernel'} if expected.get('adapter_source') else {'vllm'}
        if len(wheels) != len(wanted) or {w['distribution'] for w in wheels} != wanted:
            raise CandidateError('prepared installed-owner set differs')
        for wheel in wheels:
            distribution = owners[wheel['distribution']]
            if distribution.version != wheel['version'] or not wheel['payload_sha256']:
                raise CandidateError('installed wheel identity differs')
            for filename, digest in wheel['payload_sha256'].items():
                payload = Path(distribution.locate_file(filename)).read_bytes()
                if hashlib.sha256(payload).hexdigest() != digest:
                    raise CandidateError('installed payload differs from prepared wheel: ' + filename)
    for name in ADAPTER_SEAMS:
        if not Path(adapter.locate_file('hcu_adapter/' + name)).is_file():
            raise CandidateError('image Adapter lacks ' + name)
    adapter_catalog(Path(adapter.locate_file('hcu_adapter/hcu_config.py')).read_bytes())
    native = {}
    for name in GPU_LIBRARIES:
        native['vllm/' + name] = architectures(Path(core.locate_file('vllm/' + name)))
    library, = [Path(kernel.locate_file(name)) for name in kernel.files
                if str(name).startswith('sgl_kernel/native_ops.') and str(name).endswith('.so')]
    native[str(library.name)] = architectures(library)
    if 'torch' in sys.modules:
        raise CandidateError('metadata inspection unexpectedly imported Torch')
    return dict(versions={'vllm': core.version, 'hcu-adapter': adapter.version,
                          'sgl-kernel': kernel.version}, embedded_architectures=native,
                installed_records=records, prepared_payload_verified=expected is not None,
                device_qualified=False, model_qualified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'inspect'])
    parser.add_argument('--context', type=Path)
    parser.add_argument('--expected', type=Path)
    args = parser.parse_args()
    if args.command == 'prepare' and args.context is None:
        parser.error('prepare requires --context')
    if args.command == 'inspect' and sys.flags.no_site:
        # The image uses system Python. Add its package directories without
        # executing .pth or sitecustomize (Adapter bootstrap probes Torch).
        import sysconfig
        for key in ('purelib', 'platlib'):
            path = sysconfig.get_path(key)
            if path not in sys.path:
                sys.path.append(path)
    expected = json.loads(args.expected.read_text()) if args.expected else None
    report = prepare(args.context, os.environ) if args.command == 'prepare' else inspect_image(expected)
    print(json.dumps(report, sort_keys=True))


if __name__ == '__main__':
    main()
