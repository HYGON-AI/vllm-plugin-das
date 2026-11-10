"""CPU destination/dispatch contracts; native kernels below are explicit fixtures."""
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'adapter_extension/src'))

from hcu_vllm_das.moe import fused_moe_aiter as provider, int8_asm, int8_experts
from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig, FusedMoEParallelConfig, RoutingMethodType, int8_w8a8_moe_quant_config,
)
from vllm.model_executor.layers.quantization.utils import int8_utils


def arguments(dtype=torch.bfloat16):
    return dict(
        hidden_states=torch.ones((3, 8), dtype=dtype),
        w1=torch.ones((4, 12, 8), dtype=torch.int8),
        w2=torch.ones((4, 8, 6), dtype=torch.int8),
        topk_ids=torch.tensor([[0, 1]]*3, dtype=torch.int32),
        topk_weights=torch.ones((3, 2), dtype=torch.float32),
        w1_scale=torch.ones((4, 12, 1), dtype=torch.float32),
        w2_scale=torch.ones((4, 8, 1), dtype=torch.float32),
        expert_map=None, global_num_experts=4, activation='silu', output_dtype=dtype,
        moe_config=SimpleNamespace(solution_type='asm', config=dict(
            SOL_ID1=10010, SOL_ID2=20000, BLOCK_SIZE_M=16)),
        a1_scale=None, a2_scale=None, w1_zp=None, w2_zp=None,
    )


def cpu_stage_fixture(monkeypatch):
    """Execute real wrapper allocation/reduction flow, not native math."""
    seen = dict(reductions=[], stages=[])

    def sorting(ids, weights, sorted_ids, sorted_weights, experts, positions, count, *unused):
        count.fill_(int((ids >= 0).sum()))

    def gemm(output, *args):
        # count is arg6 after the output in the actual stage ABI.
        count, stage = args[6], args[-5]
        seen['stages'].append(stage)
        assert torch.count_nonzero(output) == 0
        if int(count):
            output.fill_(2 if stage == 0 else 4)

    def silu(output, first):
        n = first.shape[-1]//2
        output.copy_(F.silu(first[:, :n])*first[:, n:])

    def reduce(second, output):
        seen['reductions'].append(output)
        output.copy_(second.sum(dim=1))

    monkeypatch.setattr(int8_utils, 'per_token_quant_int8', lambda x: (
        x.to(torch.int8), torch.ones((x.shape[0], 1), dtype=torch.float32)))
    monkeypatch.setattr(torch.ops._C, 'silu_and_mul', silu)
    monkeypatch.setattr(ops, 'moe_sum', reduce)
    return SimpleNamespace(moe_sorting_fwd=sorting, asm_fmoe_a8=gemm), seen


@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16])
def test_native_reduction_reuses_compatible_core_destination(monkeypatch, dtype):
    native, seen = cpu_stage_fixture(monkeypatch)
    a = arguments(dtype)
    output = torch.full((3, 8), float('nan'), dtype=dtype)
    actual = int8_asm.execute(native=native, **a, output=output)
    assert actual is output
    assert seen['reductions'] == [output]
    assert torch.equal(output, torch.full_like(output, 8))
    assert seen['stages'] == [0, 1]


@pytest.mark.parametrize('kind', ['absent', 'shape', 'dtype', 'strided', 'unaligned', 'device'])
def test_incompatible_hint_keeps_allocated_result_without_touching_hint(monkeypatch, kind):
    native, seen = cpu_stage_fixture(monkeypatch)
    a = arguments()
    output = None
    if kind == 'shape':
        output = torch.full((3, 9), float('nan'), dtype=torch.bfloat16)
    elif kind == 'dtype':
        output = torch.full((3, 8), float('nan'), dtype=torch.float32)
    elif kind == 'strided':
        output = torch.full((8, 3), float('nan'), dtype=torch.bfloat16).T
    elif kind == 'unaligned':
        output = torch.full((25,), float('nan'), dtype=torch.bfloat16)[1:].view(3, 8)
        assert output.is_contiguous() and output.data_ptr()%16
    elif kind == 'device':
        output = torch.empty((3, 8), dtype=torch.bfloat16, device='meta')
    actual = int8_asm.execute(native=native, **a, **({} if kind == 'absent' else dict(output=output)))
    assert actual is not output
    assert seen['reductions'] == [actual]
    assert torch.equal(actual, torch.full_like(actual, 8))
    if output is not None and output.device.type == 'cpu':
        assert torch.isnan(output).all()


def test_zero_remote_routes_overwrite_poisoned_destination(monkeypatch):
    native, seen = cpu_stage_fixture(monkeypatch)
    a = arguments()
    a['expert_map'] = torch.full((8,), -1, dtype=torch.int32)
    a['global_num_experts'] = 8
    output = torch.full((3, 8), float('nan'), dtype=torch.bfloat16)
    actual = int8_asm.execute(native=native, **a, output=output)
    assert actual is output and seen['stages'] == [0, 1]
    assert torch.equal(output, torch.zeros_like(output))


@pytest.mark.parametrize('returns_destination', [True, False])
def test_expert_forwards_destination_and_copies_only_allocated_fallback(monkeypatch, returns_destination):
    monkeypatch.setattr(int8_experts, '_require_active', lambda: None)
    cfg = FusedMoEConfig(num_experts=4, experts_per_token=2, hidden_dim=8,
        intermediate_size=6, num_local_experts=4, num_logical_experts=4,
        activation=MoEActivation.SILU, device='cuda', in_dtype=torch.bfloat16,
        routing_method=RoutingMethodType.Default,
        moe_backend='aiter', moe_parallel_config=FusedMoEParallelConfig.make_no_parallel())
    a = arguments()
    quant = int8_w8a8_moe_quant_config(w1_scale=a['w1_scale'], w2_scale=a['w2_scale'],
        a1_scale=None, a2_scale=None, per_act_token_quant=True)
    expert = int8_experts.HcuAiterInt8Experts(cfg, quant)

    class CountCopy(torch.Tensor):
        def copy_(self, other, *args, **kwargs):
            self.copies += 1
            return super().copy_(other, *args, **kwargs)

    output = torch.zeros((3, 8), dtype=torch.bfloat16).as_subclass(CountCopy)
    output.copies = 0
    seen = []

    def execute(**kw):
        seen.append(kw)
        if returns_destination and kw.get('output') is output:
            return output.fill_(7)
        return torch.full((3, 8), 7, dtype=torch.bfloat16)

    monkeypatch.setattr(provider, 'apply_aiter_int8_moe', execute)
    empty = torch.empty(0)
    expert.apply(output, a['hidden_states'], a['w1'], a['w2'], a['topk_weights'],
        a['topk_ids'], MoEActivation.SILU, 4, None, None, None, empty, empty, None, False)
    assert seen[0].get('output') is output
    assert output.copies == (0 if returns_destination else 1)
    assert torch.equal(output, torch.full_like(output, 7))


def test_public_provider_forwards_hint_to_native_api(monkeypatch):
    a = arguments()
    output = torch.empty_like(a['hidden_states'])
    seen = []
    monkeypatch.setattr(provider, '_load_aiter_api', lambda: SimpleNamespace(
        quant_type_w8a8='int8', get_config=lambda *args, **kw: (True, a['moe_config']),
        aiter_moe=lambda **kw: seen.append(kw) or output))
    quant = SimpleNamespace(use_int8_w8a8=True, w1_scale=a['w1_scale'], w2_scale=a['w2_scale'])
    result = provider.apply_aiter_int8_moe(
        **{k: a[k] for k in ('hidden_states', 'w1', 'w2', 'topk_weights', 'topk_ids',
                             'expert_map', 'output_dtype', 'activation')},
        moe_config=SimpleNamespace(num_experts=4), quant_config=quant, asm_only=True, output=output)
    assert result is output and seen[0]['output'] is output


def test_optional_destination_does_not_leak_to_generic_vendor_api(monkeypatch):
    a = arguments()
    a['moe_config'].solution_type = 'ck'
    a['output'] = torch.empty_like(a['hidden_states'])
    seen = []
    vendor = SimpleNamespace(MoeQuantType=SimpleNamespace(W8A8='int8'),
        get_aiter_moe_config=lambda: None, aiter_moe=lambda **kw: seen.append(kw) or 'vendor')
    original = provider.import_module
    monkeypatch.setattr(provider, 'import_module', lambda name: vendor if name == 'aiter.moe' else original(name))
    assert provider._load_aiter_api().aiter_moe(**a) == 'vendor'
    assert 'output' not in seen[0]
