"""Matched component timing; unchecked vendor timing is not a correct baseline."""
import argparse
import importlib.util
import json
from pathlib import Path
import statistics
import sys

import torch
import vllm
import vllm._C_stable_libtorch
from hcu_vllm_das.attention.flash_attn import _vendor_varlen_func

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError(args.output)
root = Path(__file__).resolve().parents[1]
expected = Path(sys.executable).absolute().parent.parent / (
    f'lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages'
)
if not Path(vllm.__file__).absolute().is_relative_to(expected):
    raise RuntimeError(f'Core imported outside environment: {vllm.__file__}')
path = root / 'tests/vllm033/test_flash_attention_hcu.py'
spec = importlib.util.spec_from_file_location('softcap_fixture', path)
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)
torch.manual_seed(453)
rows = []
with torch.inference_mode():
    for nq, nk in ((1, 128), (1, 2048), (128, 128), (512, 512)):
        h, kh, d, page = 32, 8, 128, 64
        q = torch.randn(nq, h, d, device='cuda', dtype=torch.bfloat16)
        k = torch.randn(nk, kh, d, device='cuda', dtype=q.dtype)
        v = torch.randn_like(k)
        packed = fixture._make_packed_cache(layout_name='LBHNC', num_blocks=nk // page,
            num_heads=kh, block_size=page, head_size=d, device=q.device, dtype=q.dtype)
        impl = fixture._make_impl('LBHNC', d)
        kc, vc = impl._get_kv_cache_views(packed)
        from types import SimpleNamespace
        one = torch.ones((), device='cuda')
        impl.do_kv_cache_update(SimpleNamespace(_k_scale=one, _v_scale=one), k, v,
                               packed, torch.arange(nk, device='cuda'))
        kwargs = dict(q=q, k=kc, v=vc, out=torch.empty_like(q),
            cu_seqlens_q=torch.tensor([0, nq], device='cuda', dtype=torch.int32),
            seqused_k=torch.tensor([nk], device='cuda', dtype=torch.int32),
            max_seqlen_q=nq, max_seqlen_k=nk, causal=True, softmax_scale=d ** -0.5,
            block_table=torch.arange(nk // page, device='cuda', dtype=torch.int32)[None],
            softcap=0.5, layout='bhsd', fa_version=2)
        expected = fixture._reference_attention(q, k, v, softcap=0.5)
        timings = {}
        errors = {}
        for name, call in (('core-fallback', impl._run_flash_attention),
                           ('unchecked-vendor', _vendor_varlen_func())):
            for _ in range(5):
                call(**kwargs)
            torch.cuda.synchronize()
            errors[name] = (kwargs['out'].float() - expected.float()).abs().max().item()
            samples = []
            for _ in range(3):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(100):
                    call(**kwargs)
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end) * 1000 / 100)
            timings[name] = {'us': samples, 'median_us': statistics.median(samples)}
        rows.append({'query_tokens': nq, 'kv_tokens': nk, 'heads': h, 'kv_heads': kh,
                     'head_dim': d, 'timings': timings, 'maxabs_vs_fp32': errors})
record = {'architecture': torch.cuda.get_device_properties(0).gcnArchName,
          'rows': rows, 'scope': 'component eager GPU event timing, 3 warmed repetitions; unchecked vendor is numerically wrong; no serving performance acceptance'}
with args.output.open('x') as stream:
    stream.write(json.dumps(record, indent=2) + '\n')
print(json.dumps(record, indent=2), flush=True)
