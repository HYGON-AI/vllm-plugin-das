# SPDX-License-Identifier: Apache-2.0
"""Prepare a private Core wheel and inspect a built release CI candidate."""
import argparse
import base64
import csv
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
                 'dsv41_channel_fp8.py', 'dsv41_dspark_registration.py')


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


def verify_wheel(path, expected):
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
        checked = set()
        for name, digest, size in csv.reader(io.StringIO(archive.read(records[0]).decode())):
            if name in checked:
                raise CandidateError('duplicate RECORD member')
            checked.add(name)
            payload = archive.read(name)
            if not digest:
                if name != records[0]:
                    raise CandidateError('unhashed wheel payload')
                continue
            actual = 'sha256=' + base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b'=').decode()
            if digest != actual or size != str(len(payload)):
                raise CandidateError('wheel RECORD checksum or size differs')
        if checked != {name for name in names if not name.endswith('/')}:
            raise CandidateError('wheel payload missing from RECORD')
        if not all('vllm/' + library in checked for library in GPU_LIBRARIES):
            raise CandidateError('wheel lacks required native libraries')
    return len(checked)


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
    wheel = core / filename
    fetch(base + quote(filename, safe=''), token, wheel, 2 * 1024**3)
    entries = verify_wheel(wheel, inputs['WHEEL_SHA256'])
    return dict(core_source=inputs['CORE_SOURCE'], wheel_sha256=inputs['WHEEL_SHA256'],
                wheel_filename=filename, record_entries=entries, base_image=inputs['BASE_IMAGE'],
                device_qualified=False, model_qualified=False)


def architectures(binary, runner=subprocess.run):
    result = runner(['/opt/dtk/aillvm/bin/llvm-objdump', '--offloading', str(binary)],
                    check=True, capture_output=True, text=True)
    targets = sorted(set(re.findall(r'hipv4-amdgcn-amd-amdhsa--(gfx[0-9a-f]+)', result.stdout)))
    if not {'gfx936', 'gfx938'} <= set(targets):
        raise CandidateError('native payload lacks gfx936/gfx938: ' + binary.name)
    return targets


def inspect_image():
    # Installed-runtime inspection runs inside the CPython 3.12 image. The
    # download/RECORD preparation path must also load on Python 3.6 builders.
    from importlib import metadata

    core = metadata.distribution('vllm')
    if not core.version.startswith('0.33.0+'):
        raise CandidateError('image does not contain a vLLM 0.33 candidate')
    adapter = metadata.distribution('hcu-adapter')
    kernel = metadata.distribution('sgl-kernel')
    for name in ADAPTER_SEAMS:
        if not Path(adapter.locate_file('hcu_adapter/' + name)).is_file():
            raise CandidateError('image Adapter lacks ' + name)
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
                device_qualified=False, model_qualified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'inspect'])
    parser.add_argument('--context', type=Path)
    args = parser.parse_args()
    if args.command == 'prepare' and args.context is None:
        parser.error('prepare requires --context')
    report = prepare(args.context, os.environ) if args.command == 'prepare' else inspect_image()
    print(json.dumps(report, sort_keys=True))


if __name__ == '__main__':
    main()
