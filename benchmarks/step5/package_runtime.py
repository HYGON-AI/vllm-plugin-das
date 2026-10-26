"""Export a complete pinned Git tree with explicit CP312/tokenizer runtime assets.

Never reconstruct a release by applying a Git delta to an unrelated old snapshot:
runtime assets supplied separately are hashed independently from the Git tree.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import tarfile


PREPARE = r'''"""Verify the pinned source tree, then extract it without overwriting other code."""
import argparse,hashlib,io,json,pathlib,sys,tarfile
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--verify-only',action='store_true')
args=parser.parse_args()
root=pathlib.Path(__file__).resolve().parent
manifest=json.loads((root/'bundle-manifest.json').read_text())
for name,digest in manifest['files'].items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
hashes=json.loads((root/'source-sha256.json').read_text())
origin=json.loads((root/'source-git-tree.json').read_text())
assert origin['commit']==manifest['source_commit']
members={}
with tarfile.open(root/'source-runtime.tar.gz') as archive:
    for member in archive:
        relative=pathlib.PurePosixPath(member.name)
        assert member.isfile() and not relative.is_absolute() and '..' not in relative.parts
        assert member.name not in members
        data=archive.extractfile(member).read()
        assert hashlib.sha256(data).hexdigest()==hashes[member.name],member.name
        expected=origin['files'].get(member.name)
        if expected is not None:
            actual=hashlib.sha1(b'blob '+str(len(data)).encode()+bytes([0])+data).hexdigest()
            assert actual==expected['git_blob'],'Git source mismatch: '+member.name
        members[member.name]=(data,member.mode)
assert set(members)==set(hashes)
assert set(origin['files']).issubset(members)
table=json.loads(members['assets/ep_table.json'][0])
key='E=22,N=3072,K=4096,topk=8'
assert isinstance(table.get(key),dict) and table[key],'Missing Step5 EP16/E22 configuration'
if not args.verify_only:
    assert sys.version_info[:2]==(3,12),'Extract for execution inside the matching Python3.12/DTK image'
    targets=[]
    for name,(data,mode) in members.items():
        target=root.joinpath(*pathlib.PurePosixPath(name).parts).resolve()
        assert root in target.parents
        if target.exists():
            assert target.is_file() and target.read_bytes()==data,'Refusing to overwrite different file: '+str(target)
        targets.append((target,data,mode))
    for target,data,mode in targets:
        if not target.exists():
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data);target.chmod(mode&0o777)
print(json.dumps(dict(source_commit=origin['commit'],git_files=len(origin['files']),archive_files=len(members),ep16_entries=len(table[key]),verify_only=args.verify_only,worker_extension_required=False)))
'''

SERVER = '''#!/usr/bin/env bash
set -euo pipefail
phase="${1:?Usage: server_step5.sh prefill|decode NODE_RANK}"
rank="${2:?NODE_RANK must be 0..3}"
case "$phase" in prefill|decode) ;; *) exit 2;; esac
case "$rank" in 0|1|2|3) ;; *) exit 2;; esac
bundle="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
export STEP5_SOURCE_DIR="$bundle/source"
export STEP5_EP_TABLE="${STEP5_EP_TABLE:-$bundle/assets/ep_table.json}"
export STEP5_TOKENIZER="${STEP5_TOKENIZER:-$bundle/assets/tokenizer-bytelevel}"
export STEP5_STATE_DIR="${STEP5_STATE_DIR:-/var/lib/xuwq_step5_native}"
export STEP5_CACHE_DIR="${STEP5_CACHE_DIR:-/var/cache/xuwq_step5}"
export STEP5_HEAD_IP="${STEP5_HEAD_IP:-172.20.2.22}"
export VLLM_HOST_IP="${VLLM_HOST_IP:-172.20.2.$((22+rank))}"
if [[ "$phase" == decode ]]; then
    if [[ "${STEP5_DECODE_PARALLEL:-dp4tp4}" == dp16tp1 ]]; then
        export STEP5_DECODE_PROFILE="${STEP5_DECODE_PROFILE:-dp16_native}"
        exec bash "$bundle/source/benchmarks/step5/launch_multinode.sh" decode "$rank"
    fi
    : "${STEP5_DECODE_PLACEMENT:?Set the external Decode placement file}"
    export STEP5_DECODE_PLACEMENT
    exec bash "$bundle/source/benchmarks/step5/launch_decode_itl6479.sh" "$rank"
fi
exec bash "$bundle/source/benchmarks/step5/launch_multinode.sh" prefill "$rank"
'''


def git(repo: Path, *args: str) -> bytes:
    return subprocess.check_output(['git', *args], cwd=repo)


def build_bundle(repo: Path, ref: str, output: Path, hcu_ops: Path, tokenizer: Path, ep_table: Path) -> dict:
    repo, output = repo.resolve(), output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f'Refusing to replace an existing bundle: {output}')
    if not re.fullmatch(r'hcu_ops\.cpython-312-[\w-]+\.so', hcu_ops.name):
        raise ValueError('Supply the validated CPython3.12 HCU extension')
    binary = hcu_ops.read_bytes()
    if binary[:4] != bytes([127])+b'ELF':
        raise ValueError('HCU extension is not an ELF shared library')
    commit = git(repo, 'rev-parse', f'{ref}^{{commit}}').decode().strip()
    expected = {}
    for row in git(repo, 'ls-tree', '-rz', commit).split(bytes([0])):
        if not row:
            continue
        metadata, name = row.split(b'\t', 1)
        mode, kind, oid = metadata.decode().split()
        if kind == 'blob':
            if mode not in ('100644','100755'):
                raise ValueError('Unsupported Git source mode: '+name.decode())
            expected['source/'+name.decode()] = {'git_blob': oid, 'mode': int(mode,8)&0o777}
    for name in expected:
        relative = PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsupported source path: '+name)
        if relative.suffix in ('.pt','.pth','.ckpt','.safetensors'):
            raise ValueError('Model and placement weights must remain external: '+name)
    # Read raw Git blobs. `git archive` can apply checkout EOL conversion on
    # Windows; release bytes must match the pinned objects on every host.
    oids = list(dict.fromkeys(info['git_blob'] for info in expected.values()))
    process = subprocess.run(['git','cat-file','--batch'],cwd=repo,
        input=('\n'.join(oids)+'\n').encode(),capture_output=True,check=True)
    payload, cursor, blobs = process.stdout, 0, {}
    for oid in oids:
        end = payload.index(bytes([10]),cursor)
        actual_oid, kind, size = payload[cursor:end].decode().split()
        if actual_oid != oid or kind != 'blob':
            raise ValueError('Unexpected Git object response: '+oid)
        size = int(size);cursor=end+1;data=payload[cursor:cursor+size];cursor+=size
        if payload[cursor:cursor+1] != bytes([10]):
            raise ValueError('Truncated Git object: '+oid)
        cursor+=1
        actual = hashlib.sha1(b'blob '+str(size).encode()+bytes([0])+data).hexdigest()
        if actual != oid:
            raise ValueError('Git object checksum mismatch: '+oid)
        blobs[oid]=data
    entries = {name:(blobs[info['git_blob']],info['mode']) for name,info in expected.items()}
    entries['assets/ep_table.json'] = (ep_table.read_bytes(),0o644)
    table = json.loads(entries['assets/ep_table.json'][0])
    key = 'E=22,N=3072,K=4096,topk=8'
    if not isinstance(table.get(key),dict) or not table[key]:
        raise ValueError('External table is missing Step5 EP16/E22 configurations')
    entries['source/vllm_hcu/'+hcu_ops.name] = (binary,0o644)
    if not (tokenizer/'tokenizer.json').is_file() or not (tokenizer/'tokenizer_config.json').is_file():
        raise ValueError('Supply the benchmark tokenizer directory')
    for path in sorted(tokenizer.iterdir()):
        if path.is_file() and path.suffix in ('.json','.jinja','.txt'):
            entries['assets/tokenizer-bytelevel/'+path.name] = (path.read_bytes(),0o644)
    output.mkdir(parents=True,exist_ok=True)
    with tarfile.open(output/'source-runtime.tar.gz','w:gz') as archive:
        for name,(data,mode) in sorted(entries.items()):
            member = tarfile.TarInfo(name);member.size=len(data);member.mode=mode
            archive.addfile(member,io.BytesIO(data))
    (output/'source-git-tree.json').write_text(json.dumps({'commit':commit,'files':expected},indent=2),encoding='utf-8')
    (output/'source-sha256.json').write_text(json.dumps({name:hashlib.sha256(data).hexdigest() for name,(data,_) in entries.items()},indent=2),encoding='utf-8')
    (output/'prepare_runtime.py').write_text(PREPARE,encoding='utf-8')
    (output/'server_step5.sh').write_bytes(SERVER.encode('utf-8'))
    (output/'README.txt').write_text(
        f'Pinned Step5 source: {commit}\n'
        'The source archive contains the complete Git tree, the supplied CP312 extension and tokenizer.\n'
        'Model weights and placement PT files are external. This bundle asserts no performance score.\n'
        'Inside the matching Python3.12/DTK image: python3 prepare_runtime.py\n'
        'Set STEP5_DECODE_PLACEMENT, then bash server_step5.sh decode NODE_RANK (0..3).\n'
        'For DP16TP1EP16 Prefill on four four-GPU hosts: bash server_step5.sh prefill NODE_RANK (0..3).\n'
        'Set STEP5_PREFILL_DP_SIZE=8 for the optional two-host EP8 layout.\n'
        'Prefill defaults: batch8192, context70000, localhost:8018, gpu_memory_utilization0.875.\n',encoding='utf-8')
    files = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file()}
    manifest = {'source_commit':commit,'source_export':'complete pinned Git object export',
                'git_files':len(expected),'ep16_entries':len(table[key]),'external_hcu_ops_sha256':hashlib.sha256(binary).hexdigest(),
                'worker_extension_required':False,'files':files}
    (output/'bundle-manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return manifest


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument('--ref',default='HEAD')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--hcu-ops',type=Path,required=True)
    parser.add_argument('--tokenizer',type=Path,required=True)
    parser.add_argument('--ep-table',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(build_bundle(args.repo,args.ref,args.output,args.hcu_ops,args.tokenizer,args.ep_table)))
