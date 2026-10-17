# SPDX-License-Identifier: Apache-2.0
from types import ModuleType, SimpleNamespace
import pytest
import torch
from vllm_hcu.patch.worker.core_fix import patch_deepseek_v4_inverse_rope as patch
from vllm_hcu.patch.worker.core_fix._common import PatchCompatibilityError


def target():
    class Attention:
        def _o_proj(self, o, positions):
            return ('original', o, positions)
    m = ModuleType(patch.TARGET_MODULE)
    m.DeepseekV4ROCMAiterMLAAttention = Attention
    return m, Attention


def test_registration_and_fallback(monkeypatch):
    m, cls = target()
    assert patch.apply_to_module(m)
    assert not patch.apply_to_module(m)
    monkeypatch.setattr(patch, '_eligible', lambda *args: False)
    o, pos = torch.ones(3, 2, 512), torch.arange(3)
    assert cls()._o_proj(o, pos) == ('original', o, pos)


def test_stale_marker_and_signature():
    m, cls = target()
    patch.apply_to_module(m)
    cls._o_proj = lambda *args: None
    with pytest.raises(PatchCompatibilityError):
        patch.apply_to_module(m)
    m, cls = target()
    cls._o_proj = lambda self, o: None
    with pytest.raises(PatchCompatibilityError):
        patch.apply_to_module(m)


@pytest.mark.parametrize('flag', ['enabled', 'master', 'child', 'grad', 'cpu', 'short', 'stride', 'fp32'])
def test_guards_keep_reference(monkeypatch, flag):
    monkeypatch.setenv('VLLM_HCU_USE_CUSTOM_OPS', '0' if flag == 'master' else '1')
    monkeypatch.setenv('VLLM_HCU_INVERSE_ROPE_SUFFIX', '0' if flag == 'child' else '1')
    monkeypatch.setattr(torch.version, 'hip', 'test')
    rows = 17 if flag == 'short' else 512
    layer = SimpleNamespace(rotary_emb=SimpleNamespace(cos_sin_cache=torch.empty(1024, 64)),
                            rope_head_dim=64, n_local_heads=2, padded_heads=2)
    o = SimpleNamespace(is_cuda=flag != 'cpu', dtype=torch.float32 if flag == 'fp32' else torch.bfloat16,
                        ndim=3, shape=(rows, 2, 512), device=torch.device('cpu'),
                        is_contiguous=lambda: flag != 'stride', stride=lambda d: {-1:1,1:512,0:1024}[d])
    scope = torch.enable_grad() if flag == 'grad' else torch.no_grad()
    with scope:
        assert patch._eligible(layer, o, torch.arange(rows)) == (flag == 'enabled')


def test_optimized_projection_keeps_weight_adapter_and_wo_b(monkeypatch):
    from vllm.v1.attention.ops import rocm_aiter_mla_sparse as ops
    from vllm_hcu.utils import deepseek_v4_inverse_rope as kernel
    m, cls = target()
    patch.apply_to_module(m)
    monkeypatch.setattr(patch, '_eligible', lambda *args: True)
    seen = []
    w = torch.randn(2, 7, 512, dtype=torch.bfloat16)
    monkeypatch.setattr(ops, '_get_cached_wo_a_bf16', lambda *args: w)
    def rotate(o, *args):
        seen.append(1)
        o[...,448:] *= -1
        return o
    monkeypatch.setattr(kernel, 'inverse_rope_suffix_inplace', rotate)
    layer = cls()
    layer.n_local_groups=2
    layer.o_lora_rank=7
    layer.rope_head_dim=64
    layer.rotary_emb=SimpleNamespace(cos_sin_cache=torch.empty(10,64))
    layer.wo_a=object()
    layer._wo_b_scale=None
    layer.wo_b=lambda z: z + 1
    o = torch.randn(3,2,512,dtype=torch.bfloat16)
    ref = o.clone();ref[...,448:] *= -1
    expected = torch.einsum('tgd,grd->tgr',ref,w).flatten(1)+1
    torch.testing.assert_close(layer._o_proj(o,torch.arange(3)),expected)
    torch.testing.assert_close(o,ref)
    assert seen == [1]
    layer._wo_b_scale='scale'
    layer.wo_b=SimpleNamespace(weight='weight')
    layer._bpre_attn_gemm=lambda *a: a
    result=layer._o_proj(o.clone(),torch.arange(3))
    assert result[0:2] == ('weight','scale') and result[3] is True


def test_kernel_does_not_specialize_on_token_count():
    from vllm_hcu.utils import deepseek_v4_inverse_rope as kernel
    params = {p.name: p.is_constexpr for p in kernel._inverse_rope_suffix_inplace_kernel.params}
    assert params['count'] is False
