#!/usr/bin/env python3
"""Pre-implementation CPU packed-loading integration; HIPC vendor packer doubled."""
from __future__ import annotations
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import torch
from vllm.model_executor.layers.fused_moe import RoutedExperts

MODULE = 'hcu_vllm_das.quantization.ct_w4a8_weights'
try:
    candidate = importlib.import_module(MODULE)
except ModuleNotFoundError as error:
    if error.name != MODULE:
        raise
    candidate = None
RESULTS = []
E, H, I = 2, 512, 256


def required():
    assert candidate is not None, 'packed CT W4A8 production module is missing'
    return candidate


def case(name, callback):
    try:
        callback()
        RESULTS.append({'case': name, 'status': 'passed'})
    except Exception as error:
        RESULTS.append({'case': name, 'status': 'failed', 'error': f'{type(error).__name__}: {error}'})


def rejects(callback):
    try:
        callback()
    except (ValueError, RuntimeError, TypeError):
        return
    raise AssertionError('malformed packed contract accepted')


def ct_pack(q):
    values = (q.to(torch.int64) + 8).reshape(*q.shape[:-1], -1, 8)
    shifts = torch.arange(8, dtype=torch.int64) * 4
    return (values << shifts).sum(-1).to(torch.int32).contiguous()


def hipc_reference(q):
    pairs = q.to(torch.int16).reshape(*q.shape[:-1], -1, 2)
    return (((pairs[..., 0] & 15) << 4) | (pairs[..., 1] & 15)).to(torch.uint8).view(torch.int8)


def fixture():
    obj = RoutedExperts.__new__(RoutedExperts)
    torch.nn.Module.__init__(obj)
    obj.quant_config = SimpleNamespace(get_name=lambda: 'hcu_dsv41_ct_w4a8')
    obj.quant_method = SimpleNamespace(moe_kernel=None)
    obj.moe_config = SimpleNamespace(tp_rank=0, is_act_and_mul=True, tp_shard_with_padding=False,
                                    moe_parallel_config=SimpleNamespace(tp_size=1))
    obj.expert_map_manager = SimpleNamespace(map_global_to_local=lambda e: e if e < E else -1)
    obj._fused_shared_expert_quantizer = None
    owner = required().PackedCTWeights(obj, E, H, I, obj.weight_loader)
    generator = torch.Generator().manual_seed(42)
    logical = {}
    for expert in range(E):
        for shard, rows, columns in (('w1', I, H), ('w3', I, H), ('w2', H, I)):
            q = torch.randint(-8, 8, (rows, columns), generator=generator, dtype=torch.int8)
            scale = torch.full((rows, 1), float(expert + {'w1':1,'w3':3,'w2':5}[shard]))
            stem = 'w2' if shard == 'w2' else 'w13'
            for suffix, data in (('weight_packed', ct_pack(q)), ('weight_scale', scale),
                                 ('weight_shape', torch.tensor([rows, columns], dtype=torch.int64))):
                name = f'{stem}_{suffix}'
                assert getattr(obj, name).weight_loader(getattr(obj, name), data, name,
                           shard_id=shard, expert_id=expert, return_success=True) is True
            logical[expert, shard] = (q, scale)
    return obj, owner, logical


def converted_200():
    for seed in range(200):
        generator = torch.Generator().manual_seed(seed)
        shape = (seed % 3 + 1, seed % 7 + 1, (seed % 8 + 1) * 8)
        q = torch.randint(-8, 8, shape, generator=generator, dtype=torch.int8)
        packed = ct_pack(q)
        pointer = packed.data_ptr()
        result = required().convert_ct_int4_to_hipc_bytes_(packed)
        assert result.data_ptr() == pointer and result.dtype == torch.int8
        assert torch.equal(result, hipc_reference(q)), seed
        assert result.shape == (*shape[:-1], shape[-1] // 2)


def scales_200():
    for seed in range(200):
        source = torch.arange(1, (seed % 17 + 1) * 2 + 1, dtype=torch.float32).view(2, -1, 1)
        before = source.clone()
        derived = required().hipc_channel_scale(source)
        assert torch.equal(source, before) and torch.equal(derived, before / 16)
        assert source.data_ptr() != derived.data_ptr()
        assert derived.untyped_storage().nbytes() >= (source.numel() + 524288) * 4


def native_loading():
    obj, owner, logical = fixture()
    assert obj.w13_weight_packed.shape == (E, 2 * I, H // 8)
    assert obj.w2_weight_packed.shape == (E, H, I // 8)
    for expert in range(E):
        for shard in ('w1', 'w3', 'w2'):
            q, scale = logical[expert, shard]
            actual = obj.w2_weight_packed[expert] if shard == 'w2' else obj.w13_weight_packed[expert, 0:I] if shard == 'w1' else obj.w13_weight_packed[expert, I:]
            actual_scale = obj.w2_weight_scale[expert] if shard == 'w2' else obj.w13_weight_scale[expert, 0:I] if shard == 'w1' else obj.w13_weight_scale[expert, I:]
            assert torch.equal(actual, ct_pack(q)) and torch.equal(actual_scale, scale)
        assert torch.equal(obj.w13_weight_shape[expert], torch.tensor([I, H]))
        assert torch.equal(obj.w2_weight_shape[expert], torch.tensor([H, I]))
    return obj, owner, logical


def pack_and_ownership():
    obj, owner, logical = native_loading()
    pointers = (obj.w13_weight_packed.data_ptr(), obj.w2_weight_packed.data_ptr())
    scales = (obj.w13_weight_scale.clone(), obj.w2_weight_scale.clone())
    calls = []
    def pack(t): calls.append(t.data_ptr()); return t
    owner.pack(pack)
    assert len(calls) == 2 and owner.state == 'ready'
    assert pointers == (obj.w13_weight_packed.data_ptr(), obj.w2_weight_packed.data_ptr())
    assert obj.w13_weight_packed.dtype == obj.w2_weight_packed.dtype == torch.int8
    assert torch.equal(owner.scales[0], scales[0] / 16) and torch.equal(owner.scales[1], scales[1] / 16)
    assert torch.equal(obj.w13_weight_scale, scales[0]) and torch.equal(obj.w2_weight_scale, scales[1])
    for e in range(E):
        assert torch.equal(obj.w13_weight_packed[e, :I], hipc_reference(logical[e, 'w1'][0]))
        assert torch.equal(obj.w13_weight_packed[e, I:], hipc_reference(logical[e, 'w3'][0]))
        assert torch.equal(obj.w2_weight_packed[e], hipc_reference(logical[e, 'w2'][0]))
    owner.pack(pack)
    assert len(calls) == 2
    rejects(lambda: obj.w2_weight_packed.weight_loader(obj.w2_weight_packed, torch.empty(H, I//8, dtype=torch.int32), 'w2_weight_packed', shard_id='w2', expert_id=0))
    obj.w13_weight_scale.add_(1)
    rejects(lambda: owner.pack(pack))


def n32():
    obj, owner, _ = fixture()
    def view(t):
        return t.view(t.shape[0], t.shape[2] // 32, t.shape[1] // 32, 4, 32, 8)
    owner.pack(lambda t:t, view)
    for tensor, raw in zip(owner.weights, (obj.w13_weight_packed, obj.w2_weight_packed)):
        assert tensor.data_ptr() == raw.data_ptr() and tensor.ndim == 6


def failed_pack():
    _, owner, _ = fixture()
    rejects(lambda: owner.pack(lambda t:t.clone()))
    assert owner.state == 'failed'
    rejects(lambda: owner.pack(lambda t:t))


def invalid_view():
    _, owner, _ = fixture()
    rejects(lambda: owner.pack(lambda t:t, lambda t:t.clone()))
    assert owner.state == 'failed'


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('output', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    case('200_seed_bitwise_reference_and_storage', converted_200)
    case('200_seed_scale_domain_guard_and_immutability', scales_200)
    case('native_expert_shards_and_shape_metadata', native_loading)
    case('pack_ownership_scales_idempotence_reload_guard', pack_and_ownership)
    case('n32_storage_alias_and_layout', n32)
    case('packer_clone_failure_is_sticky', failed_pack)
    case('invalid_n32_view_failure_is_sticky', invalid_view)
    case('invalid_packed_dtype', lambda: rejects(lambda: required().convert_ct_int4_to_hipc_bytes_(torch.zeros(1,2,8,dtype=torch.int8))))
    case('invalid_packed_rank', lambda: rejects(lambda: required().convert_ct_int4_to_hipc_bytes_(torch.zeros(2,8,dtype=torch.int32))))
    case('invalid_packed_stride', lambda: rejects(lambda: required().convert_ct_int4_to_hipc_bytes_(torch.zeros(1,8,8,dtype=torch.int32).transpose(1,2))))
    case('invalid_hidden_alignment', lambda: rejects(lambda: required().PackedCTWeights(torch.nn.Module(),1,513,256,lambda *a,**kw:None)))
    case('invalid_intermediate_alignment', lambda: rejects(lambda: required().PackedCTWeights(torch.nn.Module(),1,512,257,lambda *a,**kw:None)))
    case('invalid_scale_dtype', lambda: rejects(lambda: required().hipc_channel_scale(torch.ones(1,2,1,dtype=torch.float16))))
    assert not torch.cuda.is_initialized()
    root = Path(__file__).resolve().parents[2]
    sha = lambda p: subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD'],text=True).strip()
    source = root/'adapter_extension/src/hcu_vllm_das/quantization/ct_w4a8_weights.py'
    report={'level':'CPU native packed expert loading; vendor HIPC pack/view doubled',
            'plugin':sha(root),'core':sha(root.parent/'vllm'),'adapter':sha(root.parent/'adapter-das'),
            'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest() if source.is_file() else None,
            'cuda_initialized':False,'cases':RESULTS,'passed':sum(r['status']=='passed' for r in RESULTS),'total':len(RESULTS)}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))
    return 0 if report['passed']==report['total'] else 1

if __name__=='__main__': raise SystemExit(main())
