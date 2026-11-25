# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for the V4.1 draft's query-free context KV writes."""

from types import SimpleNamespace

import pytest
import torch
from vllm.models.deepseek_v41.amd import dspark as native

from hcu_vllm_das.models.dsv41_dspark import HcuDSparkDeepseekV41ForCausalLM


def _draft(cache, cos_sin, *, kv_mxfp8=False, scale=None):
    def project(states):
        return torch.cat((torch.zeros_like(states[:, :2]), states), dim=1), None

    attn = SimpleNamespace(
        fused_wqa_wkv=project,
        q_lora_rank=2,
        kv_norm=lambda kv: kv.mul(2),
        n_local_heads=64,
        padded_heads=64,
        head_dim=512,
        eps=1e-6,
        kv_mxfp8=kv_mxfp8,
        _flashinfer_fp8_kv_scale=scale,
        _flashinfer_fp8_q_scale_inv=scale,
        rotary_emb=SimpleNamespace(cos_sin_cache=cos_sin),
        swa_cache_layer=SimpleNamespace(kv_cache=cache, block_size=64),
    )
    draft = HcuDSparkDeepseekV41ForCausalLM.__new__(HcuDSparkDeepseekV41ForCausalLM)
    torch.nn.Module.__init__(draft)
    draft.model = SimpleNamespace(layers=[SimpleNamespace(attn=attn)] * 2)
    return draft


def test_context_kv_passes_layer_slots_and_normalized_values(monkeypatch):
    calls = []
    monkeypatch.setattr(
        torch.ops._C, "fused_deepseek_v4_kv_rope_insert",
        lambda *args: calls.append(args), raising=False,
    )
    states = torch.randn(3, 512, dtype=torch.bfloat16)
    positions = torch.tensor([0, 1, 8])
    slots = torch.tensor([64, -1])
    cache = torch.zeros(2, 64, 528, dtype=torch.uint8)
    cos_sin = torch.zeros(9, 64)
    draft = _draft(cache, cos_sin, kv_mxfp8=True)

    draft.precompute_and_store_context_kv(states, positions, [None, slots])

    assert len(calls) == 1
    kv, target, actual_slots, actual_positions, rope, block_size, scale, mxfp8 = calls[0]
    torch.testing.assert_close(kv, states * 2, atol=0, rtol=0)
    assert kv.is_contiguous()
    assert target is cache and actual_slots is slots and actual_positions is positions
    assert rope is cos_sin and block_size == 64 and scale is None and mxfp8 is True


@pytest.mark.parametrize("profiling", [True, False])
def test_context_kv_uses_native_profile_or_missing_op_fallback(monkeypatch, profiling):
    calls = []
    monkeypatch.setattr(
        native.DSparkDeepseekV4ForCausalLM, "precompute_and_store_context_kv",
        lambda self, *args: calls.append(args),
    )
    if not profiling:
        monkeypatch.setattr(torch.ops._C, "fused_deepseek_v4_kv_rope_insert", None, raising=False)
    draft = _draft(None, None)
    mappings = None if profiling else [None, torch.tensor([0])]
    states, positions = object(), object()
    draft.precompute_and_store_context_kv(states, positions, mappings)
    assert calls == [(states, positions, mappings)]


@pytest.mark.hcu
@pytest.mark.parametrize(
    "dtype,width,mxfp8",
    [(torch.uint8, 584, False), (torch.uint8, 528, True),
     (torch.bfloat16, 512, False), (torch.float8_e4m3fn, 512, False)],
)
def test_context_kv_matches_native_cache_bytes_on_gpu(dtype, width, mxfp8):
    if not torch.cuda.is_available() or not torch.version.hip:
        pytest.skip("requires the ROCm vLLM fused cache operators")
    import vllm._C_stable_libtorch  # noqa: F401

    assert callable(torch.ops._C.fused_deepseek_v4_kv_rope_insert)

    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(6197)
    states = torch.randn(17, 512, device=device, generator=generator).bfloat16()
    angle = torch.randn(64, 32, device=device, generator=generator)
    cos_sin = torch.cat((angle.cos(), angle.sin()), dim=1)
    positions = torch.arange(17, device=device) * 3
    slots = torch.arange(12, device=device) * 7
    slots[::4] = -1
    # Include storage gaps between pages and verify they remain untouched.
    actual_backing = torch.zeros(2, 64 * width + 128, device=device, dtype=dtype)
    expected_backing = actual_backing.clone()
    scale = torch.tensor([0.25], device=device)
    draft = _draft(
        actual_backing[:, :64 * width].view(2, 64, width), cos_sin,
        kv_mxfp8=mxfp8, scale=scale,
    )
    actual_attn = draft.model.layers[0].attn
    expected_attn = SimpleNamespace(**vars(actual_attn))
    expected_attn.swa_cache_layer = SimpleNamespace(
        kv_cache=expected_backing[:, :64 * width].view(2, 64, width), block_size=64,
    )

    native._insert_context_kv(expected_attn, states * 2, positions, slots)
    draft.precompute_and_store_context_kv(states, positions, [slots, None])

    actual_bytes = actual_backing.view(torch.uint8)
    assert torch.count_nonzero(actual_bytes)
    assert torch.equal(actual_bytes, expected_backing.view(torch.uint8))
