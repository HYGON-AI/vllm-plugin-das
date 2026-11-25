"""Actual native draft method, complete KV bytes and changing graph inputs."""
from types import SimpleNamespace

import pytest
import torch

from hcu_vllm_das import qkv_split as candidate
from vllm.models.deepseek_v41.amd.rocm import DeepseekV41ROCMAiterMLAAttention

pytestmark = [pytest.mark.hcu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="HCU required")]


@pytest.mark.parametrize("m", [1, 7, 17, 35, 42, 64, 129, 257])
@pytest.mark.parametrize("block", [32, 64])
@pytest.mark.parametrize("partial", [False, True])
@torch.inference_mode()
def test_native_draft_binding_q_and_complete_cache_bytes(m, block, partial):
    torch.manual_seed(23542 + m)
    owner = object.__new__(DeepseekV41ROCMAiterMLAAttention)
    torch.nn.Module.__init__(owner)
    owner.use_pcp = owner.kv_mxfp8 = False
    assert not owner.accepts_unnormed_unroped_query
    owner.n_local_heads = owner.padded_heads = 64
    owner.eps = 1e-6
    pages = (2 * m + block - 1) // block + 2
    reference_cache = torch.full((pages, block, 584), 17, device="cuda", dtype=torch.uint8)
    actual_cache = reference_cache.clone()
    owner.swa_cache_layer = SimpleNamespace(prefix="draft.swa", kv_cache=reference_cache)
    angles = torch.randn(1024, 32, device="cuda")
    cos_sin = torch.cat((angles.cos(), angles.sin()), dim=1)
    owner.rotary_emb = SimpleNamespace(cos_sin_cache=cos_sin)
    q = torch.randn(m, 64, 512, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(m, 512, device="cuda", dtype=torch.bfloat16)
    positions = torch.arange(m, device="cuda", dtype=torch.int64)
    slots = torch.arange(max(1, m // 2) if partial else m, device="cuda") * 2
    slots[1::3] = -1
    metadata = {"draft.swa": SimpleNamespace(block_size=block, slot_mapping=slots)}
    model = SimpleNamespace(layers=[SimpleNamespace(attn=owner)])
    original = owner._fused_qnorm_rope_kv_insert
    assert candidate.bind_dspark_qkv(model) == 1
    installed = owner._fused_qnorm_rope_kv_insert
    assert candidate.bind_dspark_qkv(model) == 1
    assert owner._fused_qnorm_rope_kv_insert is installed
    assert owner._hcu_native_qkv == original
    assert installed(q, kv, positions, None) is q

    def native():
        owner.swa_cache_layer.kv_cache = reference_cache
        return original(q, kv, positions, metadata)

    def fused():
        owner.swa_cache_layer.kv_cache = actual_cache
        return installed(q, kv, positions, metadata)

    snapshots = [t.clone() for t in (q, kv, slots, positions, cos_sin)]
    expected, actual = native(), fused()
    torch.testing.assert_close(actual.view(torch.uint8), expected.view(torch.uint8), rtol=0, atol=0)
    assert torch.equal(actual_cache, reference_cache)
    assert all(torch.equal(a, b) for a, b in zip((q, kv, slots, positions, cos_sin), snapshots))

    graphs, outputs = [], []
    for function in (native, fused):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            function()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = function()
        graphs.append(graph)
        outputs.append(output)
    for replay, magnitude in enumerate((.001, .25, 8.)):
        q.normal_().mul_(magnitude)
        kv.normal_().mul_(magnitude)
        positions.copy_((torch.arange(m, device="cuda") * (replay + 1) + replay) % 1024)
        slots.copy_(torch.arange(slots.numel(), device="cuda") * 2 + replay % 2)
        slots[replay::3] = -1
        angles.normal_()
        cos_sin.copy_(torch.cat((angles.cos(), angles.sin()), dim=1))
        reference_cache.fill_(17)
        actual_cache.fill_(17)
        for graph in graphs:
            graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(outputs[1].view(torch.uint8), outputs[0].view(torch.uint8), rtol=0, atol=0)
        assert torch.equal(actual_cache, reference_cache)


def test_draft_binding_keeps_foreign_overrides(monkeypatch):
    owner = object.__new__(DeepseekV41ROCMAiterMLAAttention)
    torch.nn.Module.__init__(owner)
    replacement = lambda *args: None
    owner._fused_qnorm_rope_kv_insert = replacement
    monkeypatch.setattr(candidate, "qualify_qkv_split", lambda _: True)
    assert candidate.bind_dspark_qkv(SimpleNamespace(layers=[SimpleNamespace(attn=owner)])) == 0
    assert owner._fused_qnorm_rope_kv_insert is replacement
