"""Large-M local MoE tile screen using captured routing marginals.

Token/expert associations and activations are synthetic. This preserves original
contiguous EP placement and captured expert counts, but is not model throughput.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import time

import numpy as np
import torch
import vllm
from vllm.plugins import load_general_plugins
from vllm_hcu.patch.worker import apply_worker_patches

torch.set_num_threads(1)
torch.cuda.set_device(0)
assert torch.cuda.mem_get_info()[0] > 32 * 1024**3, "Requires an idle GPU"
load_general_plugins()
apply_worker_patches()
from vllm_hcu.model_executor.layers.fused_moe import ep_fused_moe as ep

root = Path(os.environ['STEP5_PROOF_DIR'])
meta = json.loads((root / 'fixture.json').read_text())
fixture = root / 'fixture.npz'
assert hashlib.sha256(fixture.read_bytes()).hexdigest() == meta['fixture_sha256']
data = np.load(fixture)
ids_cpu, counts = data['topk_ids'], data['counts']
M, TOPK = ids_cpu.shape
assert (M, TOPK) == (65536, 8)
assert np.array_equal(np.bincount(ids_cpu.ravel(), minlength=352), counts)
assert np.all(np.diff(np.sort(ids_cpu, axis=1), axis=1) > 0)
assert not os.environ.get('VLLM_STEP5_EXPERT_PLACEMENT')

torch.manual_seed(20261021)
E, N, K, G = 22, 3072, 4096, 352
ep.shared_add_provider = None
ep.CHUNK = M
w1 = (torch.randn(E, N, K, device='cuda', dtype=torch.bfloat16) * 32).to(ep.FP8)
w2 = (torch.randn(E, K, N // 2, device='cuda', dtype=torch.bfloat16) * 32).to(ep.FP8)
s1 = torch.rand(E, N, 1, device='cuda') * .0002 + .0005
s2 = torch.rand(E, K, 1, device='cuda') * .0002 + .0005
aq = (torch.randn(M, K, device='cuda', dtype=torch.bfloat16) * 32).to(ep.FP8)
asc = torch.rand(M, 1, device='cuda') * .0002 + .0005
weights = torch.rand(M, TOPK, device='cuda')
weights /= weights.sum(1, keepdim=True)
ids = torch.from_numpy(ids_cpu).to('cuda')
mapping = torch.full((G,), -1, device='cuda', dtype=torch.int32)
start = meta['rank_expert_start']
mapping[start:start + E] = torch.arange(E, device='cuda', dtype=torch.int32)
out = torch.empty(M, K, device='cuda', dtype=torch.bfloat16)
reference = torch.empty_like(out)
rows = ((M * TOPK + E * 255 + 255) // 256) * 256
ws13 = torch.empty(max(rows * N * 2, M * TOPK * K * 2), device='cuda', dtype=torch.uint8)
ws2 = torch.empty(rows * (K + 4), device='cuda', dtype=torch.uint8)

base = copy.deepcopy(ep.pick_configs(w1, w2, TOPK, M))
variants = {'baseline': base}
for stages in (1, 3):
    cfg = copy.deepcopy(base)
    for part in cfg:
        part['num_stages'] = stages
    variants['stages' + str(stages)] = cfg
cfg = copy.deepcopy(base)
cfg[0]['GROUP_SIZE_M'], cfg[1]['GROUP_SIZE_M'] = 8, 4
variants['groups8-4'] = cfg
cfg = copy.deepcopy(base)
for part in cfg:
    part['BLOCK_SIZE_M'] = 128
variants['bm128'] = cfg
cfg = copy.deepcopy(base)
for part in cfg:
    part['BLOCK_SIZE_N'] = 128
    part['num_warps'] = 4
variants['bn128-warps4'] = cfg
cfg = copy.deepcopy(base)
cfg[0]['a_presort'] = False
variants['without-presort'] = cfg
cfg = copy.deepcopy(ep.pick_configs(w1, w2, TOPK, 16384))
cfg[0]['a_presort'] = True
cfg[0]['GROUP_SIZE_M'] = 4
variants['bm128-existing16k'] = cfg
if os.environ.get('STEP5_TILE_ONLY'):
    names = json.loads(os.environ['STEP5_TILE_ONLY'])
    assert names and names[0] == 'baseline' and set(names) <= set(variants)
    variants = {name: variants[name] for name in names}


def call(name):
    return ep.ep_fused_experts(
        aq, w1, w2, weights, ids, mapping, s1, s2, a1_scale=asc,
        global_num_experts=G, output_dtype=torch.bfloat16,
        routed_scaling_factor=3., configs=variants[name], out=out,
        workspace13=ws13, workspace2=ws2)


def error():
    exact = torch.equal(out, reference)
    if exact:
        return {'exact': True, 'relative_l2': 0., 'finite': True}
    delta = out.float() - reference.float()
    return dict(exact=False, relative_l2=float(delta.norm() / reference.float().norm().clamp_min(1e-12)),
                max_abs=float(delta.abs().max()), finite=bool(torch.isfinite(out).all()))


def cache_state():
    path = Path(os.environ['TRITON_CACHE_DIR'])
    return {str(p.relative_to(path)): (p.stat().st_size, p.stat().st_mtime_ns)
            for p in path.rglob('*') if p.is_file()}


result = dict(completed=False, fixture=meta, configs=variants, numerical={}, rejected={},
              samples_ms={name: [] for name in variants}, diagnostic_only=True,
              full_model_performance_validated=False, started=time.time(),
              input_dtype=str(aq.dtype), output_dtype=str(out.dtype),
              native_source=ep.__file__, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def save():
    (root / 'result.json').write_text(json.dumps(result, indent=2))


# Compile every shape before timing. Resource-only compile failures reject a
# candidate; other failures remain failures and are never hidden as slow runs.
accepted = []
for name in variants:
    try:
        call(name)
        torch.cuda.synchronize()
        if name == 'baseline':
            reference.copy_(out)
        result['numerical'][name] = {'original_input': error()}
        assert result['numerical'][name]['original_input']['finite']
        call(name)
        torch.cuda.synchronize()
        accepted.append(name)
        print(json.dumps(dict(warmed=name, numerical=result['numerical'][name])), flush=True)
    except Exception as exc:
        text = str(exc)
        if name == 'baseline' or not any(x in text.lower() for x in ['out of resource', 'shared memory']):
            raise
        result['rejected'][name] = text
    save()

cache_before = cache_state()
rng = random.Random(20261021)
for iteration in range(15):
    middle = [name for name in accepted if name != 'baseline']
    rng.shuffle(middle)
    for name in ['baseline', *middle, 'baseline']:
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record()
        call(name)
        end.record()
        end.synchronize()
        result['samples_ms'][name].append(begin.elapsed_time(end))
    save()
assert cache_state() == cache_before, 'JIT/cache writes occurred during timing'

# Reuse the exact routing counts with changed and zero activations. Only the
# zero case must be bit-exact; nonzero error remains an explicit measured field.
for amplitude in (0., .5):
    aq.copy_((torch.randn_like(aq, dtype=torch.bfloat16) * 32 * amplitude).to(aq.dtype))
    call('baseline')
    reference.copy_(out)
    for name in accepted:
        call(name)
        torch.cuda.synchronize()
        e = error()
        assert e['finite']
        if amplitude == 0:
            assert e['exact'], (name, e)
        result['numerical'][name]['amplitude' + str(amplitude)] = e
        save()

result['medians_ms'] = {name: statistics.median(result['samples_ms'][name]) for name in accepted}
result.update(completed=True, cache_updates_in_measurement=0, finished=time.time())
save()
print(json.dumps(dict(completed=True, medians_ms=result['medians_ms'], rejected=result['rejected'])), flush=True)
